import numpy as np
import math
import os
import pyquaternion

# to mmdet3d/datasets/transforms/utils
'''
select bbox
'''


def check_point_in_box(pts, box, k=1):
    """
    	pts[x,y,z]
    	box[c_x,c_y,c_z,dx,dy,dz,heading]
    """

    shift_x = pts[0] - box[0]
    shift_y = pts[1] - box[1]
    shift_z = pts[2] - box[2]
    cos_a = math.cos(box[6])
    sin_a = math.sin(box[6])
    dx, dy, dz = box[3]*k, box[4]*k, box[5]*k
    local_x = shift_x * cos_a + shift_y * sin_a
    local_y = shift_y * cos_a - shift_x * sin_a
    if abs(shift_z) > dz / 1.0 or abs(local_x) > dx / 2.0 or abs(local_y) > dy / 2.0:
        return False
    return True



def img2velodyne(calib_dir, img_id, p):
    """
    :param calib_dir
    :param img_id
    :param velo_box: (n,8,4)
    :return: (n,4)
    """
    calib_txt = os.path.join(calib_dir, img_id) + '.txt'
    calib_lines = [line.rstrip('\n') for line in open(calib_txt, 'r')]
    for calib_line in calib_lines:
        if 'P2' in calib_line:
            P2 = calib_line.split(' ')[1:]
            P2 = np.array(P2, dtype='float').reshape(3, 4)
        elif 'R0_rect' in calib_line:
            R0_rect = np.zeros((4, 4))
            R0 = calib_line.split(' ')[1:]
            R0 = np.array(R0, dtype='float').reshape(3, 3)
            R0_rect[:3, :3] = R0
            R0_rect[-1, -1] = 1
        elif 'velo_to_cam' in calib_line:
            velo_to_cam = np.zeros((4, 4))
            velo2cam = calib_line.split(' ')[1:]
            velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
            velo_to_cam[:3, :] = velo2cam
            velo_to_cam[-1, -1] = 1

    pts_rect_hom = p
    pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_rect, velo_to_cam).T))

    return pts_lidar


'''
Corruptions
'''


#
# MAP = {
#     'density_dec_bbox': density,
#     'cutout_bbox': cutout,
#     'gaussian_noise_bbox': gaussian,
#     'uniform_noise_bbox': uniform,
#     'impulse_noise_bbox': impulse,
#     'scale_bbox': scale,
#     'shear_bbox': shear,
#     'rotation_bbox': rotation,
#     'moving_noise_bbox': moving_object,
# }


def pick_bbox(cor, slevel, data, pointcloud):
    xyz = pointcloud
    bboxes = data[0]
    flag1 = 0
    for box in bboxes:

        flag1 += 1
        pcd_1 = []
        pcd_2 = []
        x = float(box[0])
        y = float(box[1])
        z = float(box[2])
        x_size = float(box[3])
        y_size = float(box[4])
        z_size = float(box[5])
        angel = float(box[6])
        p3 = (x, y, z)
        gt_boxes = []
        gt_boxes.append(p3[0])
        gt_boxes.append(p3[1])
        gt_boxes.append(p3[2])
        gt_boxes.append(x_size)
        gt_boxes.append(y_size)
        gt_boxes.append(z_size)
        gt_boxes.append(angel)

        for a in xyz:
            flag = check_point_in_box(a, gt_boxes)
            if flag == True:
                pcd_2.append(a)
            else:
                pcd_1.append(a)
        pcd_2 = np.array(pcd_2)
        pcd_1 = np.array(pcd_1)
        # print(xyz.shape, pcd_1.shape, pcd_2.shape)
        if len(pcd_2) != 0:
            # if 'bbox' in cor:
            #     pcd_2 = MAP[cor](pcd_2, slevel, gt_boxes)
            # else:
            #     pcd_2 = MAP[cor](pcd_2, slevel)
            # if 'scale' in cor or 'shear' in cor or 'rotation' in cor:
            #     pcd_2 = MAP[cor](pcd_2, slevel, gt_boxes)
            if 'nocar' in cor:
                xyz = np.array(pcd_1)
                return xyz
            # else:
            #     pcd_2 = MAP[cor](pcd_2, slevel)
            xyz = np.append(pcd_2, pcd_1, axis=0)
    return xyz


def nocar_util(cor, slevel, data, pointcloud):
    xyz = np.array(pointcloud)
    boxscale = [0, 1.0, 1.2, 1.4, 1.6, 2.0]
    bboxes = data[0]
    flag1 = 0
    flag0 = np.array([False] * pointcloud.shape[0], dtype=bool)

    for box in bboxes:

        flag1 += 1
        flag_temp = []

        x = float(box[0])
        y = float(box[1])
        z = float(box[2])
        x_size = float(box[3]) * boxscale[slevel]
        y_size = float(box[4]) * boxscale[slevel]
        z_size = float(box[5]) * boxscale[slevel]
        print(z_size)
        angel = float(box[6])
        p3 = (x, y, z)
        gt_boxes = []
        gt_boxes.append(p3[0])
        gt_boxes.append(p3[1])
        gt_boxes.append(p3[2])
        gt_boxes.append(x_size)
        gt_boxes.append(y_size)
        gt_boxes.append(z_size)
        gt_boxes.append(angel)

        for a in xyz:
            flag = check_point_in_box(a, gt_boxes)
            flag_temp.append(flag)
        flag0 = flag0 | np.array(flag_temp)
    # print(len(flag0), flag0)
    # flag0 = ~flag0
    # print(xyz.shape, flag0.shape, np.sum(~flag0), np.sum(flag0), xyz[flag0].shape)

    return xyz[~flag0]
