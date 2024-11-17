import pickle
import time

import numpy as np
import torch
import tqdm
import torch.optim as optim

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import tools.attack_utils.pert_utils as pert_utils
from graphviz import Digraph
from torchviz import make_dot
from utils import loss_utils


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def register_hooks(var):
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input

        fn.register_hook(register_grad)

    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        try:
            grad_output = grad_output.data
        except:
            print('Fail to get grad')
            return True
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '(' + (', ').join(map(str, size)) + ')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))

        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (
            metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)


    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        pert = pert_utils.generate_pert_for_scenes(batch_dict)
        pert.requires_grad_(True)
        #########################################################
        best_loss = torch.tensor(1e+6).cuda()
        best_pert = None
        learning_rate = 1e-4
        optimizer = optim.Adam([batch_dict['pert']], lr=learning_rate)
        # points = batch_dict['points'].clone()


        for query in range(40):
            # batch_dict['points'] = pert_utils.add_pert_to_point(pert_voxel_sampled2, points, in_voxel_mask)
            pred_dicts, ret_dict = model(batch_dict)

            iou = ret_dict.get('iou3d_roi')
            scores = pred_dicts['pred_scores']
            score_and_iou = torch.cat((scores.view(-1, 1), iou), 1)

            _, adv_loss = loss_utils.adv_loss(pert, score_and_iou)
            per_loss = loss_utils.calculate_per_loss(pred_dicts, pert, k=1.5)
            total_loss = adv_loss + 0.1 * per_loss


            if total_loss < best_loss:
                best_loss = total_loss.detach()
                best_pred = pred_dicts
                best_ret = ret_dict
            optimizer.zero_grad()
            # get_dot = register_hooks(total_loss)
            total_loss.backward()
            optimizer.step()

            # print('sample_id :', batch_dict['frame_id'], 'query :', query + 1, 'total_loss :', total_loss.item(),
            #       'best_loss :', best_loss.item(), 'gt :', ret_dict['gt'], ret_dict['rcnn_0.3'], ret_dict["rcnn_0.5"],
            #       ret_dict["rcnn_0.7"])
            # print(score_and_iou)
        #########################################################

        pert = batch_dict['pert']
        pred_dicts = best_pred
        ret_dict = best_ret
        disp_dict = {}
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict





if __name__ == '__main__':
    pass
