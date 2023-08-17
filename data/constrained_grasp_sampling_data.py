import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
import glob
import copy
import pickle


class ConstrainedGraspSamplingData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        # self.get_mean_std()
        self.validate = opt.validate

    def __getitem__(self, index):
        path = self.paths[index]
        meta = {}
        try:
            point_clouds, clusters_with_grasps_per_point_cloud, grasps_per_clusters, prerendered_camera_poses, mesh_file, mesh_scale = self.read_grasp_file(
                path)
            if self.validate:
                sampled_point_cloud_indices, sampled_query_point_indices_with_grasps_per_point_cloud, sampled_grasps_per_query_point = self.sample_clusters_query_points_and_grasps(
                    1, point_clouds, clusters_with_grasps_per_point_cloud, grasps_per_clusters)
            else:
                sampled_point_cloud_indices, sampled_query_point_indices_with_grasps_per_point_cloud, sampled_grasps_per_query_point = self.sample_clusters_query_points_and_grasps(
                    self.opt.num_grasps_per_object, point_clouds, clusters_with_grasps_per_point_cloud, grasps_per_clusters)

        except NoPositiveGraspsException:
            return self.__getitem__(np.random.randint(0, self.size))

        if self.validate:
            gt_control_points = utils.transform_control_points_numpy(
                sampled_grasps_per_query_point, 1, mode='rt')[:, :, :3]
        else:
            gt_control_points = utils.transform_control_points_numpy(
                sampled_grasps_per_query_point, self.opt.num_grasps_per_object, mode='rt')[:, :, :3]

        pc = point_clouds[sampled_point_cloud_indices]
        features = self.create_constrained_features(
            pc.shape[0], pc.shape[1], sampled_query_point_indices_with_grasps_per_point_cloud)
        if self.validate:
            pc = np.tile(pc, (self.opt.num_grasps_per_object, 1, 1))
            features = np.tile(features, (self.opt.num_grasps_per_object, 1, 1))
            sampled_grasps_per_query_point = np.tile(sampled_grasps_per_query_point, (self.opt.num_grasps_per_object, 1, 1))
            gt_control_points = np.tile(gt_control_points, (self.opt.num_grasps_per_object, 1, 1))

            meta['mesh_file'] = np.array([mesh_file] *
                                         self.opt.num_grasps_per_object)
            meta['mesh_scale'] = np.array([mesh_scale] *
                                          self.opt.num_grasps_per_object)
            meta['camera_pose'] = prerendered_camera_poses[sampled_point_cloud_indices]
        meta['pc'] = pc
        meta['features'] = features

        meta['grasp_rt'] = sampled_grasps_per_query_point.reshape(
            len(sampled_grasps_per_query_point), -1)

        meta['target_cps'] = np.array(gt_control_points[:, :, :3])

        return meta

    def __len__(self):
        return self.size

    def create_constrained_features(self, batch_size,  points_per_pc, query_point_indices_to_grasp):
        features = np.zeros((batch_size, points_per_pc, 1))
        for i in range(batch_size):
            features[i, query_point_indices_to_grasp[i], 0] = 1
        return features

    def sample_clusters_query_points_and_grasps(self, num_samples, point_clouds, clusters_with_grasps, grasps_per_cluster):
        num_point_clouds = len(point_clouds)
        if len(point_clouds) == 0:
            raise NoPositiveGraspsException

        replace = num_samples > num_point_clouds
        point_cloud_indices = np.random.choice(range(num_point_clouds),
                                               size=num_samples,
                                               replace=replace).astype(np.int32)

        random_cluster_indices_with_grasps = []
        random_grasps_per_cluster = []
        for point_cloud_index in point_cloud_indices:
            num_cluters_with_grasp_for_current_point_cloud = len(clusters_with_grasps[point_cloud_index])
            random_cluster_index_with_grasp = np.random.randint(0, num_cluters_with_grasp_for_current_point_cloud)
            random_cluster_indices_with_grasps.append(
                clusters_with_grasps[point_cloud_index][random_cluster_index_with_grasp])

            num_grasps_in_random_cluster = len(grasps_per_cluster[point_cloud_index][random_cluster_index_with_grasp])
            random_grasp_index_for_random_query_point = np.random.randint(0, num_grasps_in_random_cluster)
            random_grasps_per_cluster.append(
                grasps_per_cluster[point_cloud_index][random_cluster_index_with_grasp][random_grasp_index_for_random_query_point])

        random_cluster_indices_with_grasps = np.asarray(random_cluster_indices_with_grasps)
        random_grasps_per_cluster = np.asarray(random_grasps_per_cluster)
        return point_cloud_indices, random_cluster_indices_with_grasps, random_grasps_per_cluster

    def make_dataset(self):
        files = glob.glob(self.opt.dataset_root_folder+"/*")
        return files

    def read_grasp_file(self, path):
        file_name = path
        if self.caching and file_name in self.cache:
            point_clouds, query_points_with_grasps, grasps_per_query_points, prerendered_camera_poses, mesh_file, mesh_scale = copy.deepcopy(
                self.cache[file_name])
            return point_clouds, query_points_with_grasps, grasps_per_query_points, prerendered_camera_poses, mesh_file, mesh_scale

        point_clouds, query_points_with_grasps, grasps_per_query_points, prerendered_camera_poses, mesh_file, mesh_scale = self.read_object_grasp_data(
            path)

        if self.caching:
            self.cache[file_name] = (point_clouds, query_points_with_grasps, grasps_per_query_points,
                                     prerendered_camera_poses, mesh_file, mesh_scale)
            return copy.deepcopy(self.cache[file_name])

        return point_clouds, query_points_with_grasps, grasps_per_query_points, prerendered_camera_poses, mesh_file, mesh_scale



    def read_object_grasp_data(self,
                               file_path
                               ):
        """
        Reads the grasps from the json path and loads the mesh and all the
        grasps.
        """
        grasps, pre_rendered_point_clouds, camera_poses_for_prerendered_point_clouds, all_query_points_per_point_cloud, grasp_indices_for_every_query_point_on_each_rendered_point_cloud, mesh_file, mesh_scale = self.load_data_from_file(
            file_path)

        clusters_with_grasps_for_each_point_cloud, grasps_for_each_cluster_per_point_cloud, point_clouds_to_keep = self.get_query_points_with_grasps_and_all_grasps(
            all_query_points_per_point_cloud, grasp_indices_for_every_query_point_on_each_rendered_point_cloud, grasps, camera_poses_for_prerendered_point_clouds)
        pre_rendered_point_clouds = pre_rendered_point_clouds[point_clouds_to_keep]
        return pre_rendered_point_clouds, clusters_with_grasps_for_each_point_cloud, grasps_for_each_cluster_per_point_cloud,  camera_poses_for_prerendered_point_clouds, mesh_file, mesh_scale

    def load_data_from_file(self, file_path):
        if file_path.endswith("h5"):
            return self.load_from_h5py_file(file_path)
        elif file_path.endswith("pickle"):
            return self.load_from_pickle_file(file_path)
        else:
            raise ValueError("Cannot read file with extension ", file_path.split(".")[-1])

    def load_from_h5py_file(self, h5py_file):
        h5_dict = utils.read_h5_file(h5py_file)
        grasps = h5_dict['grasps/transformations'][()]
        pre_rendered_point_clouds = h5_dict['rendering/point_clouds'][()]
        camera_poses_for_prerendered_point_clouds = h5_dict['rendering/camera_poses'][()]
        all_query_points_per_point_cloud = h5_dict["query_points/points_with_grasps_on_each_rendered_point_cloud"][()]
        grasp_indices_for_every_query_point_on_each_rendered_point_cloud = h5_dict[
            "query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud"][()]
        mesh_file = np.asarray(h5_dict["mesh/file"])
        mesh_scale = np.asarray(h5_dict["mesh/scale"])

        return grasps, pre_rendered_point_clouds, camera_poses_for_prerendered_point_clouds, all_query_points_per_point_cloud, grasp_indices_for_every_query_point_on_each_rendered_point_cloud, mesh_file, mesh_scale 

    def load_from_pickle_file(self, file):
        with open(file, "rb") as f:
            data = pickle.load(f)
            grasps = data['grasps/transformations']
            pre_rendered_point_clouds = np.asarray(data['rendering/point_clouds'])
            camera_poses_for_prerendered_point_clouds = np.asarray(data['rendering/camera_poses'])
            all_query_points_per_point_cloud = np.asarray(data["query_points/points_with_grasps_on_each_rendered_point_cloud"])
            grasp_indices_for_every_query_point_on_each_rendered_point_cloud = np.asarray(data[
                "query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud"])
            mesh_file = np.asarray(data["mesh/file"])
            mesh_scale = np.asarray(data["mesh/scale"])

        return grasps, pre_rendered_point_clouds, camera_poses_for_prerendered_point_clouds, all_query_points_per_point_cloud, grasp_indices_for_every_query_point_on_each_rendered_point_cloud, mesh_file, mesh_scale

    def get_query_points_with_grasps_and_all_grasps(self, all_clusters, grasp_indices_for_every_cluster_on_each_rendered_point_cloud,  grasps, camera_poses_for_prerendered_point_clouds):

        all_clusters_per_point_cloud = []
        all_grasps_per_cluster_per_point_cloud = []
        point_clouds_to_keep = []
        for point_cloud_index in range(all_clusters.shape[0]):
            all_clusters_with_grasps_per_point_cloud = all_clusters[point_cloud_index]
            if len(all_clusters_with_grasps_per_point_cloud) == 0:
                continue
            point_clouds_to_keep.append(point_cloud_index)
            all_clusters_per_point_cloud.append(all_clusters_with_grasps_per_point_cloud)

            all_grasps_per_cluster = []
            grasp_indices_for_all_clusters = grasp_indices_for_every_cluster_on_each_rendered_point_cloud[point_cloud_index]

            camera_pose_for_current_point_cloud_index = camera_poses_for_prerendered_point_clouds[point_cloud_index]
            for grasp_indices_per_cluster in grasp_indices_for_all_clusters:
                transformed_grasps = np.matmul(camera_pose_for_current_point_cloud_index,
                                               grasps[grasp_indices_per_cluster, :, :])
                all_grasps_per_cluster.append(transformed_grasps)
            all_grasps_per_cluster_per_point_cloud.append(all_grasps_per_cluster)

        return all_clusters_per_point_cloud, all_grasps_per_cluster_per_point_cloud, point_clouds_to_keep
