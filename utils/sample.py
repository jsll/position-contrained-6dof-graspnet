# -*- coding: utf-8 -*-
"""Helper classes and functions to sample grasps for a given object mesh."""

from __future__ import print_function

import numpy as np

from tqdm import tqdm

import trimesh
import trimesh.transformations as tra


class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.

        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        self.mesh = trimesh.load(filename)
        self.as_mesh()

        self.scale = 1.0

        # print(filename)
        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('object', self.mesh)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.

        :param scale
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def as_mesh(self):
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in self.mesh.geometry.values()])


class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder=''):
        """Create a Franka Panda parallel-yaw gripper object.

        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        fn_base = root_folder + 'gripper_models/panda_gripper/hand.stl'
        fn_finger = root_folder + 'gripper_models/panda_gripper/finger.stl'
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.ray_origins = []
        self.ray_directions = []
        for i in np.linspace(-0.01, 0.02, num_contact_points_per_finger):
            self.ray_origins.append(
                np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1])
            self.ray_origins.append(
                np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1])
            self.ray_directions.append(
                np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]])
            self.ray_directions.append(
                np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]])

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        self.standoff_range = np.array([max(self.finger_l.bounding_box.bounds[0, 2],
                                            self.base.bounding_box.bounds[1, 2]),
                                        self.finger_l.bounding_box.bounds[1, 2]])
        self.standoff_range[0] += 0.001

    def get_closing_rays(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.

        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix

        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return transform[:3, :].dot(
            self.ray_origins.T).T, transform[:3, :3].dot(self.ray_directions.T).T



def create_gripper(name, configuration=None, root_folder=''):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def in_collision_with_gripper(object_mesh, gripper_transforms, gripper_name, silent=False):
    """Check collision of object with gripper.

    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper

    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})

    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object('object', object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]
    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        min_distance.append(np.min([manager.min_distance_single(
            gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))

    return [d == 0 for d in min_distance], min_distance


def grasp_quality_point_contacts(transforms, collisions, object_mesh, gripper_name='panda', silent=False):
    """Grasp quality function

    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})

    Returns:
        list of float -- quality of grasps [0..1]
    """
    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    for p, colliding in tqdm(zip(transforms, collisions), total=len(transforms), disable=silent):
        if colliding:
            res.append(-1)
        else:
            ray_origins, ray_directions = gripper.get_closing_rays(p)
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False)

            if len(locations) == 0:
                res.append(0)
            else:
                # this depends on the width of the gripper
                valid_locations = np.linalg.norm(
                    ray_origins[index_ray]-locations, axis=1) < 2.0*gripper.q

                if sum(valid_locations) == 0:
                    res.append(0)
                else:
                    contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                    motion_normals = ray_directions[index_ray[valid_locations]]
                    dot_prods = (motion_normals * contact_normals).sum(axis=1)
                    res.append(np.cos(dot_prods).sum() / len(ray_origins))
    return res