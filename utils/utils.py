import numpy as np
import os
import trimesh.transformations as tra
from utils import sample
import torch
import h5py


def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))




def perturb_grasp(grasp, num, min_translation, max_translation, min_rotation,
                  max_rotation):
    """
      Self explanatory.
    """
    output_grasps = []
    for _ in range(num):
        sampled_translation = [
            np.random.uniform(lb, ub)
            for lb, ub in zip(min_translation, max_translation)
        ]
        sampled_rotation = [
            np.random.uniform(lb, ub)
            for lb, ub in zip(min_rotation, max_rotation)
        ]
        grasp_transformation = tra.euler_matrix(*sampled_rotation)
        grasp_transformation[:3, 3] = sampled_translation
        output_grasps.append(np.matmul(grasp, grasp_transformation))

    return output_grasps


def evaluate_grasps(grasp_tfs, obj_mesh):
    """
        Check the collision of the grasps and also heuristic quality for each
        grasp.
    """
    collisions, _ = sample.in_collision_with_gripper(
        obj_mesh,
        grasp_tfs,
        gripper_name='panda',
        silent=True,
    )
    qualities = sample.grasp_quality_point_contacts(
        grasp_tfs,
        collisions,
        object_mesh=obj_mesh,
        gripper_name='panda',
        silent=True,
    )

    return np.asarray(collisions), np.asarray(qualities)




def merge_pc_and_gripper_pc(pc,
                            gripper_pc,
                            instance_mode=0,
                            pc_latent=None,
                            gripper_pc_latent=None):
    """
    Merges the object point cloud and gripper point cloud and
    adds a binary auxilary feature that indicates whether each point
    belongs to the object or to the gripper.
    """

    pc_shape = pc.shape
    gripper_shape = gripper_pc.shape
    assert (len(pc_shape) == 3)
    assert (len(gripper_shape) == 3)
    assert (pc_shape[0] == gripper_shape[0])

    batch_size = pc.shape[0]

    if instance_mode == 1:
        assert pc_shape[-1] == 3
        latent_dist = [pc_latent, gripper_pc_latent]
        latent_dist = torch.cat(latent_dist, 1)

    l0_xyz = torch.cat((pc, gripper_pc), 1)
    labels = [
        torch.ones((pc.shape[1], 1), dtype=torch.float32),
        torch.zeros((gripper_pc.shape[1], 1), dtype=torch.float32)
    ]
    labels = torch.cat(labels, 0)
    labels = torch.expand_dims(labels, 0)
    labels = torch.tile(labels, [batch_size, 1, 1])

    if instance_mode == 1:
        l0_points = torch.cat([l0_xyz, latent_dist, labels], -1)
    else:
        l0_points = torch.cat([l0_xyz, labels], -1)

    return l0_xyz, l0_points



def get_equidistant_points(p1, p2, parts=10):
    return np.linspace((p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2]), parts+1)


def get_control_point_tensor(batch_size, use_torch=True, dense=False, device="cpu"):
    """
      Outputs a tensor of shape (batch_size x 6 x 3).
      use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.load('./gripper_control_points/panda.npy')[:, :3]
    if dense:
        points_between_fingers = get_equidistant_points(control_points[0, :], control_points[1, :])
        points_from_base_to_gripper = get_equidistant_points([0, 0, 0], (control_points[0, :]+control_points[1, :])/2.0)
        control_points = np.concatenate(([[0, 0, 0], [0, 0, 0]], points_between_fingers, points_from_base_to_gripper, control_points))

    else:
        control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                          control_points[1, :], control_points[-2, :],
                          control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0),
                             [batch_size, 1, 1])

    if use_torch:
        return torch.tensor(control_points).to(device)

    return control_points


def transform_control_points(gt_grasps, batch_size, mode='qt', device="cpu", dense=False):
    """
      Transforms canonical points using gt_grasps.
      mode = 'qt' expects gt_grasps to have (batch_size x 7) where each 
        element is catenation of quaternion and translation for each
        grasps.
      mode = 'rt': expects to have shape (batch_size x 4 x 4) where
        each element is 4x4 transformation matrix of each grasp.
    """
    assert (mode == 'qt' or mode == 'rt'), mode
    grasp_shape = gt_grasps.shape
    if mode == 'qt':
        assert (len(grasp_shape) == 2), grasp_shape
        assert (grasp_shape[-1] == 7), grasp_shape
        control_points = get_control_point_tensor(batch_size, device=device, dense=dense)
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps

        gt_grasps = torch.unsqueeze(input_gt_grasps,
                                    1).repeat(1, num_control_points, 1)

        gt_q = gt_grasps[:, :, :4]
        gt_t = gt_grasps[:, :, 4:]
        gt_control_points = qrot(gt_q, control_points)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert (len(grasp_shape) == 3), grasp_shape
        assert (grasp_shape[1] == 4 and grasp_shape[2] == 4), grasp_shape
        control_points = get_control_point_tensor(batch_size, device=device, dense=dense)
        shape = control_points.shape
        ones = torch.ones((shape[0], shape[1], 1), dtype=torch.float32)
        control_points = torch.cat((control_points, ones), -1)
        return torch.matmul(control_points, gt_grasps.permute(0, 2, 1))


def transform_control_points_numpy(gt_grasps, batch_size, mode='qt', dense=False):
    """
      Transforms canonical points using gt_grasps.
      mode = 'qt' expects gt_grasps to have (batch_size x 7) where each 
        element is catenation of quaternion and translation for each
        grasps.
      mode = 'rt': expects to have shape (batch_size x 4 x 4) where
        each element is 4x4 transformation matrix of each grasp.
    """
    assert (mode == 'qt' or mode == 'rt'), mode
    grasp_shape = gt_grasps.shape
    if mode == 'qt':
        assert (len(grasp_shape) == 2), grasp_shape
        assert (grasp_shape[-1] == 7), grasp_shape
        control_points = get_control_point_tensor(batch_size, use_torch=False, dense=dense)
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps
        gt_grasps = np.expand_dims(input_gt_grasps,
                                   1).repeat(num_control_points, axis=1)
        gt_q = gt_grasps[:, :, :4]
        gt_t = gt_grasps[:, :, 4:]

        gt_control_points = rotate_point_by_quaternion(control_points, gt_q)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert (len(grasp_shape) == 3), grasp_shape
        assert (grasp_shape[1] == 4 and grasp_shape[2] == 4), grasp_shape
        control_points = get_control_point_tensor(batch_size, use_torch=False, dense=dense)
        shape = control_points.shape
        ones = np.ones((shape[0], shape[1], 1), dtype=np.float32)
        control_points = np.concatenate((control_points, ones), -1)
        return np.matmul(control_points, np.transpose(gt_grasps, (0, 2, 1)))[:, :, :3]


def quaternion_mult(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def conj_quaternion(q):
    """
      Conjugate of quaternion q.
    """
    q_conj = q.clone()
    q_conj[:, :, 1:] *= -1
    return q_conj


def rotate_point_by_quaternion(point, q, device="cpu"):
    """
      Takes in points with shape of (batch_size x n x 3) and quaternions with
      shape of (batch_size x n x 4) and returns a tensor with shape of 
      (batch_size x n x 3) which is the rotation of the point with quaternion
      q. 
    """
    shape = point.shape
    q_shape = q.shape

    assert (len(shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (shape[-1] == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (len(q_shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[-1] == 4), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[1] == shape[1]), 'point shape = {} q shape = {}'.format(
        shape, q_shape)

    q_conj = conj_quaternion(q)
    r = torch.cat([
        torch.zeros(
            (shape[0], shape[1], 1), dtype=point.dtype).to(device), point
    ],
        dim=-1)
    final_point = quaternion_mult(quaternion_mult(q, r), q_conj)
    final_output = final_point[:, :,
                               1:]  # torch.slice(final_point, [0, 0, 1], shape)
    return final_output



def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)






def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)



def get_inlier_grasp_indices_with_control_points(grasps, query_point=torch.tensor([[0.0, 0.0, 0.0]]), threshold=0.4, device="cpu"):
    """This function returns all grasps whose distance between the mid of the finger tips and the query point is less than the threshold value. 

    Arguments:
        grasps are given as a tensor of [B,6,3] where B is the number of grasps, 6 is the number of control points, and 3 is their position
        in euclidean space.
        query_point is a 1x3 point in 3D space.
        threshold represents the maximum distance between a grasp and the query_point
    """
    grasp_centers = grasps.mean(1)
    query_point = query_point.to(device)
    distances = torch.cdist(grasp_centers, query_point)
    indices_with_distances_smaller_than_threshold = distances < threshold
    return grasps[indices_with_distances_smaller_than_threshold[:, 0]]




def read_h5_file(file):
    data = h5py.File(file, 'r')
    return data
