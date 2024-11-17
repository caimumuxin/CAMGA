# @Time : 2020/11/19 上午11:11 
# @Author : CMMX
# @File : loss_utils.py 
# @Software: PyCharm
import torch
import pert_utils as pert_utils
import roipool3d_cuda

from utils.bbox_util import check_point_in_box


def add_score_to_pts(score, pts_mask, num_points):
    final_score = torch.zeros([len(pts_mask), num_points]).cuda()
    for idx1, i in enumerate(pts_mask):
        for idx2, j in enumerate(i):
            if j:
                final_score[idx1][idx2] = score[idx1]
            else:
                final_score[idx1][idx2] = 0
    final_score = torch.sum(final_score, dim=0)
    return final_score



def pts_mask_all_boxes(pts_mask, num_points):
    total_mask = torch.zeros(num_points)
    for idx1, i in enumerate(pts_mask):
        for idx2, j in enumerate(i):
            if j:
                total_mask[idx2] += 1
    return total_mask.bool()


def pts_in_boxes3d(pts, boxes3d):
    """
    :param pts: (N, 3) in rect-camera coords
    :param boxes3d: (M, 7)
    :return: boxes_pts_mask_list: (M), list with [(N), (N), ..]
    """
    if not pts.is_cuda:
        pts = pts.float().contiguous()
        boxes3d = boxes3d.float().contiguous()
        pts_flag = torch.LongTensor(torch.Size((boxes3d.size(0), pts.size(0))))  # (M, N)
        roipool3d_cuda.pts_in_boxes3d_cpu(pts_flag, pts, boxes3d)

        boxes_pts_mask_list = torch.empty(0)
        for k in range(0, boxes3d.shape[0]):
            cur_mask = pts_flag[k] > 0
            boxes_pts_mask_list = torch.cat((boxes_pts_mask_list, pts_flag[k]))
        return boxes_pts_mask_list.cuda()
    else:
        raise NotImplementedError


# TODO 两个损失函数，归因图生成代码

def calculate_per_loss(box, perturbations, k):
    M = box.shape[0]
    T = perturbations.shape[0]

    L_per = 0.0

    for m in range(M):
        o_xm, o_ym, o_zm, w_m, h_m, l_m, theta = box[m]
        l_kbox = k * torch.sqrt(w_m**2 + h_m**2 + l_m**2)

        for t in range(T):
            t_x, t_y, t_z, delta_x_t, delta_y_t, delta_z_t = perturbations[t]
            if check_point_in_box(perturbations[t][0:3], box[m]):
                l_t = torch.sqrt((t_x - o_xm)**2 + (t_y - o_ym)**2 + (t_z - o_zm)**2)
                L_per += (l_kbox / l_t) * (delta_x_t**2 + delta_y_t**2 + delta_z_t**2)

    return L_per.item()

def adv_loss(pert, score_and_iou):
    dis_loss = torch.norm(pert ** 2, float('inf'))
    adv_loss = 0
    for x in range(score_and_iou.shape[0]):
        for y in range(1, score_and_iou.shape[1]):
            if score_and_iou[x][0] > 0.5 and score_and_iou[x][y] > 0.5:
                adv_loss += -score_and_iou[x][y] * torch.log(1 - score_and_iou[x][0])
                # adv_loss += torch.log(1 - score_and_iou[x][y]) * torch.log(1 - score_and_iou[x][0])
    weight = 50
    total_loss = weight * dis_loss + adv_loss
    print('dis:', dis_loss, 'adv:', adv_loss)
    # total_loss = adv_loss
    return total_loss, adv_loss


def center_loss(pert, pred_dicts):
    # dis_loss = torch.norm(pert_voxel ** 2, float('inf'))
    weight = 0.1
    dis_loss = weight * torch.sum(pert ** 2)
    adv_loss = 0
    for i in range(len(pred_dicts)):
        flag = pred_dicts[i]['pred_scores'] > 0.4
        adv_loss += pred_dicts[i]['pred_scores'][flag].sum()

    # total_loss = dis_loss + adv_loss
    # total_loss = adv_loss
    total_loss = weight * dis_loss
    print('dis:', dis_loss, 'adv:', adv_loss)

    return total_loss