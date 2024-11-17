# @Time : 2020/11/10 下午4:50 
# @Author : CMMX
# @File : pert_utils.py 
# @Software: PyCharm
import os
import numpy as np
import open3d as o3d
import torch

# import torch

voxel_size = [800, 40, 700]
voxel = [800, 40, 700, 3]
side = 0.1


# voxel_size = [400, 20, 350]
# voxel = [400, 20, 350, 3]
# side = 0.2
# voxel_size = [160, 8, 140]
# voxel = [160, 8, 140, 3]
# side = 0.5
# voxel_size = [70, 80, 4]
# voxel = [70, 80, 4, 3]
# side = 1


def if_point_in_voxel(point):
    """
     if point in main scene.(-40,40),(0,4),(0,70)
    :param point:
    :return:
    """
    if -40 < point[2] < 40 and 0 < point[3] < 4 and 0 < point[1] < 70:
        return True
    return False


def if_point_in_voxel2(point):
    """
     if point in main scene.(-40,40),(0,70),(0,4)
    :param point:
    :return:
    """
    if np.abs(point[1]) < 40 and -4 < point[2] < 4 and 0 < point[0] < 70:
        return True
    return False


def if_point_in_voxel3(point):
    """
     if point in main scene.(-40,40),(0,70),(0,4)
    :param point:
    :return:
    """
    if np.abs(point[1]) < 40 and -4 < point[2] < 4 and 0 < point[0] < 70:
        return True
    return False


def index_in_voxel(point):
    index_x = int((point[2] + 40) / side)
    index_y = int(point[1] / side)
    index_z = int(point[3] / side)
    return index_x, index_y, index_z


def index_in_voxel2(point):
    index_x = int(point[0] / side)
    index_y = int((point[1] + 40) / side)
    index_z = int(point[2] + 4 / side)
    return index_x, index_y, index_z


# def get_voxel_sample(pert, pointcloud):
#     """
#     add pert to per point in main scene.
#     :param pert: universal perturbations mask, (800,40,700,3)
#     :param pointcloud: input data, (N,5)   (0, x, y, z, r)
#     :return: pointcloud after perturbation
#     """
#     in_voxel_mask = []
#     pert_voxel_sampled = torch.empty(0)
#     pert_voxel_sampled_index = torch.empty(0)
#     for i in range(pointcloud.shape[0]):
#         p = pointcloud[i]
#         if if_point_in_voxel(p):
#             index_y, index_x, index_z = index_in_voxel(p)
#             index_temp = torch.tensor([index_x, index_y, index_z])
#             pert_temp = pert[index_x][index_y][index_z]
#             pert_voxel_sampled = torch.cat((pert_voxel_sampled, pert_temp))
#             # p += pert[index_x][index_y][index_z]
#             pert_voxel_sampled_index = torch.cat((pert_voxel_sampled_index, index_temp))
#             in_voxel_mask.append(True)
#         else:
#             in_voxel_mask.append(False)
#     pert_voxel_sampled = pert_voxel_sampled.view(-1, 3)
#     pert_voxel_sampled_index = pert_voxel_sampled_index.view(-1, 3).int()
#     # pert_voxel_sampled.requires_grad_(True)
#
#     return pert_voxel_sampled, pert_voxel_sampled_index, in_voxel_mask


def add_pert_to_point(pert_voxel_sampled, pointcloud, in_voxel_mask):
    pc_with_pert = pointcloud.clone()
    num = 0
    for i in range(pointcloud.shape[0]):
        if in_voxel_mask[i]:
            pc_with_pert[i][1:4] += pert_voxel_sampled[num]
            num += 1
    return pc_with_pert


def add_pert_to_point2(pert_voxel, pointcloud):
    for i in range(pointcloud.shape[0]):
        p = pointcloud[i]
        if if_point_in_voxel2(p):
            index_x, index_y, index_z = index_in_voxel2(p)
            pert_temp = pert_voxel[index_x][index_y][index_z]
            pointcloud[i, 0:3] = pointcloud[i, 0:3] + pert_temp
    return pointcloud


def add_pert_to_point3(pert_voxel, pointcloud):
    pc = []
    for i in range(pointcloud.shape[0]):
        p = pointcloud[i]
        if if_point_in_voxel3(p):
            pc.append(p)
    return pc


def add_pert_to_point4_LiDARPure(pert_voxel, pointcloud):
    xx = np.zeros((pointcloud.shape[0] * 4, 3))
    for i in range(pointcloud.shape[0]):
        p = pointcloud[i, 0:3]
        if if_point_in_voxel2(p):
            index_x, index_y, index_z = index_in_voxel2(p)
            pert_temp = pert_voxel[index_x][index_y][index_z]
            p1 = p + pert_temp
            p2 = p + 2 * pert_temp
            # p3 = p + 3 * pert_temp
            p4 = p - pert_temp
            p5 = p - 2 * pert_temp
            # p6 = p - 3 * pert_temp
            xx[i * 4] = p1
            xx[i * 4 + 1] = p2
            # xx[i * 2 + 2] = p3
            xx[i * 4 + 2] = p4
            xx[i * 4 + 3] = p5
            # xx[i * 2 + 5] = p6

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xx)
    pcd_down = pcd.random_down_sample(0.4)
    # pcd_down = pcd.farthest_point_down_sample(pointcloud.shape[0])
    xx = torch.from_numpy(np.asarray(pcd_down.points))
    # print(xx.shape)
    return xx


def add_pert_to_point5_LiDARPure_pillar(pert_voxel, pointcloud):
    xx = np.zeros((pointcloud.shape[0] * 4, 3))
    for i in range(pointcloud.shape[0]):
        p = pointcloud[i, 0:3]
        if if_point_in_voxel2(p):
            index_x, index_y, index_z = index_in_voxel2(p)
            pert_temp = pert_voxel[index_x][index_y][0]
            p1 = p + pert_temp
            p2 = p + 2 * pert_temp
            # p3 = p + 3 * pert_temp
            p4 = p - pert_temp
            p5 = p - 2 * pert_temp
            # p6 = p - 3 * pert_temp
            xx[i * 2] = p1
            xx[i * 2 + 1] = p2
            # xx[i * 2 + 2] = p3
            xx[i * 2 + 3] = p4
            xx[i * 2 + 4] = p5
            # xx[i * 2 + 5] = p6

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xx)
    pcd_down = pcd.random_down_sample(0.2)
    # pcd_down = pcd.farthest_point_down_sample(pointcloud.shape[0])
    xx = torch.from_numpy(np.asarray(pcd_down.points))
    # print(xx.shape)
    return xx

# def voxel_have_point(pointcloud):
#     """
#     :param pts: N
#     :return: the mask which voxel has pts
#     """
#     voxel_mask = torch.zeros(voxel_size)
#     for i in range(pointcloud.shape[0]):
#         p = pointcloud[i]
#         if if_point_in_voxel(p):
#             index_x, index_y, index_z = index_in_voxel(p)
#             voxel_mask[index_x][index_y][index_z] = 1
#     return voxel_mask


def generate_pert_for_scenes(batch):
    points = batch['points']
    num_points = points.shape[0]
    pert = torch.randn(num_points, 3)