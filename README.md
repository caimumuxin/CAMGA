 # Contextual attribution maps guided transferable adversarial attack for 3D object detection


![alt text](./1.png)


Here we only provide the functions of contextual assessment and adversarial attack in 3D object detection. The whole project is based on 3D_Corruptions_AD and OCCAM. 
It should be built upon MMDetection3D and OpenPCDet with necessary modifications of its source code.

## Prerequisites
* Python (3.9)
* Pytorch (1.9.0)
* numpy
* MMDetection3D
* OpenPCDet

## Contextual assessment

### Implemented in MMDetection3D pipeline

In the MMDetection3D pipeline, we modified the `mmdetection3d/mmdet3d/datasets/pipelines/test_time_aug.py`, the modification examples are:
```python
@TRANSFORMS.register_module()
class CorruptionMethods(object):
    def __init__(self, corruption_severity_dict=None):


        if corruption_severity_dict is None:
            corruption_severity_dict = {
                'NoCor': 2,
            }
        self.corruption_severity_dict = corruption_severity_dict

    def __call__(self, results):
        """Call function to augment common corruptions.
        """

        if 'NoCor' in self.corruption_severity_dict:
            return results

        if 'NoCar_bbox' in self.corruption_severity_dict:
            pl = results[0]['points'].tensor
            data = []

            # data.append(results['gt_bboxes_3d'])
            if 'gt_bboxes_3d' in results[0]:
                data.append(results[0]['gt_bboxes_3d'])
            else:
                # waymo
                data.append(results[0]['eval_ann_info']['gt_bboxes_3d'])

            # gt_bbox3d = results[0]['eval_ann_info']['gt_bboxes_3d'].tensor
            # print(pl.shape, gt_bbox3d.shape)
            severity = self.corruption_severity_dict['NoCar_bbox']
            points_aug = Nocar_bbox(pl.numpy(), severity, data)
            # name = results[0]['lidar_path'].replace('velodyne_reduced', 'nocar' + str(severity))
            # points_aug.tofile(name)
            pl = torch.from_numpy(points_aug)
            results[0]['points'].tensor = pl

        return results

```

Then add 'CorruptionMethods' to the test pipeline, modify the corresponding config files in `mmdetection3d/configs/`.
Such as the config for PointPillars `configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py`, the modification examples are:

```python
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),

    dict(
            type='CorruptionMethods',
            corruption_severity_dict=
                {
                    'NoCar_bbox': 4,
                    # 'NoCor': 0
                },
        ),

    dict(type='Pack3DDetInputs', keys=['points'])
]


```



## CAMGA

### Implemented in OPENPCDet pipeline

use `CAMGA_attack.py` replace `tools/test.py`,and move `eval_utils/camga_attack_utils.py` to `tools/utils/`.
The main change is to transform the evaluation process to adversarial attacks.
```python
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

```



