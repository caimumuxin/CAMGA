import torch

def Nocar_bbox(pointcloud,severity,bbox):
    cor = 'nocar_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.nocar_util(cor, severity, bbox, pointcloud)
    return pointcloud

# to mmdetection3d/mmdet3d/datasets/pipelines/test_time_aug.py
# @TRANSFORMS.register_module()
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

