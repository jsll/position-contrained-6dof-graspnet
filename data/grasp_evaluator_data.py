import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
import random
import copy
import pickle
from utils.sample import Object


class GraspEvaluatorData(BaseDataset):
    def __init__(self, opt, ratio_positive=0.3, ratio_hardnegative=0.4, collision_hard_neg_min_translation=(-0.03, -0.03, -0.03),
                 collision_hard_neg_max_translation=(0.03, 0.03, 0.03),
                 collision_hard_neg_min_rotation=(-0.6, -0.2, -0.6),
                 collision_hard_neg_max_rotation=(+0.6, +0.2, +0.6),
                 collision_hard_neg_num_perturbations=10):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.collision_hard_neg_queue = {}
        # self.get_mean_std()
        self.ratio_positive = self.set_ratios(ratio_positive)
        self.ratio_hardnegative = self.set_ratios(ratio_hardnegative)

        self.collision_hard_neg_min_translation = collision_hard_neg_min_translation
        self.collision_hard_neg_max_translation = collision_hard_neg_max_translation
        self.collision_hard_neg_min_rotation = collision_hard_neg_min_rotation
        self.collision_hard_neg_max_rotation = collision_hard_neg_max_rotation
        self.collision_hard_neg_num_perturbations = collision_hard_neg_num_perturbations
        for i in range(3):
            assert (collision_hard_neg_min_rotation[i] <=
                    collision_hard_neg_max_rotation[i])
            assert (collision_hard_neg_min_translation[i] <=
                    collision_hard_neg_max_translation[i])



    def set_ratios(self, ratio):
        if int(self.opt.num_grasps_per_object * ratio) == 0:
            return 1 / self.opt.num_grasps_per_object
        return ratio

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            data = self.get_uniform_evaluator_data(path)
        except NoPositiveGraspsException:
            if self.opt.skip_error:
                return None
            else:
                return self.__getitem__(np.random.randint(0, self.size))

        gt_control_points = utils.transform_control_points_numpy(
            data[1], self.opt.num_grasps_per_object, mode='rt')

        meta = {}
        meta['pc'] = data[0][:, :, :3]
        meta['grasp_rt'] = gt_control_points[:, :, :3]
        meta['labels'] = data[2]
        return meta

    def __len__(self):
        return self.size

    def get_uniform_evaluator_data(self, path):
        pos_grasps, neg_grasps, obj_mesh, point_clouds, camera_poses_for_prerendered_point_clouds = self.read_grasp_file(
            path)

        output_pcs = []
        output_grasps = []
        output_labels = []

        num_positive = int(self.opt.batch_size * self.ratio_positive)
        num_hard_negative = int(self.opt.batch_size * self.ratio_hardnegative)
        num_flex_negative = self.opt.batch_size - num_positive - num_hard_negative
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps
                                                      )
        negative_clusters = self.sample_grasp_indexes(num_flex_negative,
                                                      neg_grasps)
        hard_neg_candidates = []
        # Fill in Positive Examples.

        for clusters, grasps in zip(
                [positive_clusters, negative_clusters], [pos_grasps, neg_grasps]):
            for cluster in clusters:
                selected_grasp = grasps[cluster[0]][cluster[1]]
                hard_neg_candidates += utils.perturb_grasp(
                    selected_grasp,
                    self.collision_hard_neg_num_perturbations,
                    self.collision_hard_neg_min_translation,
                    self.collision_hard_neg_max_translation,
                    self.collision_hard_neg_min_rotation,
                    self.collision_hard_neg_max_rotation,
                )

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self.collision_hard_neg_queue or len(
                self.collision_hard_neg_queue[path]) < num_hard_negative:
            if path not in self.collision_hard_neg_queue:
                self.collision_hard_neg_queue[path] = []
            # hard negatives are perturbations of correct grasps.
            collisions, heuristic_qualities = utils.evaluate_grasps(
                hard_neg_candidates, obj_mesh)

            hard_neg_mask = collisions | (heuristic_qualities < 0.001)
            hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
            np.random.shuffle(hard_neg_indexes)
            for index in hard_neg_indexes:
                self.collision_hard_neg_queue[path].append(
                    hard_neg_candidates[index])
            random.shuffle(self.collision_hard_neg_queue[path])

        # Adding positive grasps
        for positive_cluster in positive_clusters:
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_labels.append(1)

        # Adding hard neg
        for i in range(num_hard_negative):
            grasp = self.collision_hard_neg_queue[path][i]
            output_grasps.append(grasp)
            output_labels.append(0)

        self.collision_hard_neg_queue[path] = self.collision_hard_neg_queue[
            path][num_hard_negative:]

        # Adding flex neg
        if len(negative_clusters) != num_flex_negative:
            raise ValueError(
                'negative clusters should have the same length as num_flex_negative {} != {}'
                .format(len(negative_clusters), num_flex_negative))

        for negative_cluster in negative_clusters:
            selected_grasp = neg_grasps[negative_cluster[0]][
                negative_cluster[1]]
            output_grasps.append(selected_grasp)
            output_labels.append(0)

        point_cloud_indices = self.sample_point_clouds(self.opt.num_grasps_per_object, point_clouds)
        output_grasps = np.matmul(camera_poses_for_prerendered_point_clouds[point_cloud_indices], output_grasps)

        output_pcs = point_clouds[point_cloud_indices]  # np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)

        return output_pcs, output_grasps, output_labels

    def sample_point_clouds(self, num_samples, point_clouds):
        num_point_clouds = len(point_clouds)
        if num_point_clouds == 0:
            raise NoPositiveGraspsException

        replace_point_cloud_indices = num_samples > num_point_clouds
        point_cloud_indices = np.random.choice(range(num_point_clouds),
                                               size=num_samples,
                                               replace=replace_point_cloud_indices).astype(np.int32)

        return point_cloud_indices

    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache:
            pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds

        pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds = self.read_object_grasp_data(
            path,
            return_all_grasps=return_all_grasps)

        if self.caching:
            self.cache[file_name] = (pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds)
            return copy.deepcopy(self.cache[file_name])

        return pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds



    def read_object_grasp_data(self,
                               data_path,
                               return_all_grasps=True):
        """
        Reads the grasps from the json path and loads the mesh and all the 
        grasps.
        """
        num_clusters = 32

        if num_clusters <= 0:
            raise NoPositiveGraspsException

        json_dict = pickle.load(open(data_path, "rb"))

        object_model = Object(json_dict['mesh/file'])
        object_model.rescale(json_dict['mesh/scale'])
        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)

        object_model.vertices -= object_mean

        grasps = np.asarray(json_dict['grasps/transformations'])
        point_clouds = np.asarray(json_dict['rendering/point_clouds'])
        camera_poses_for_prerendered_point_clouds = np.asarray(json_dict['rendering/camera_poses'])
        flex_qualities = np.asarray(json_dict['grasps/successes'])

        successful_mask = (flex_qualities == 1)

        positive_grasp_indexes = np.where(successful_mask)[0]
        negative_grasp_indexes = np.where(~successful_mask)[0]

        positive_grasps = grasps[positive_grasp_indexes, :, :]
        negative_grasps = grasps[negative_grasp_indexes, :, :]

        def cluster_grasps(grasps):
            cluster_indexes = np.asarray(
                utils.farthest_points(grasps, num_clusters,
                                      utils.distance_by_translation_grasp))
            output_grasps = []

            for i in range(num_clusters):
                indexes = np.where(cluster_indexes == i)[0]
                output_grasps.append(grasps[indexes, :, :])

            output_grasps = np.asarray(output_grasps)

            return output_grasps

        if not return_all_grasps:
            positive_grasps = cluster_grasps(
                positive_grasps)
            negative_grasps = cluster_grasps(
                negative_grasps)
        return positive_grasps, negative_grasps, object_model, point_clouds, camera_poses_for_prerendered_point_clouds

    def sample_grasp_indexes(self, n, grasps):
        """
          Stratified sampling of the grasps.
        """
        nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
        num_clusters = len(nonzero_rows)
        replace = n > num_clusters
        if num_clusters == 0:
            raise NoPositiveGraspsException

        grasp_rows = np.random.choice(range(num_clusters),
                                      size=n,
                                      replace=replace).astype(np.int32)
        grasp_rows = [nonzero_rows[i] for i in grasp_rows]
        grasp_cols = []
        for grasp_row in grasp_rows:
            if len(grasps[grasp_rows]) == 0:
                raise ValueError('grasps cannot be empty')

            grasp_cols.append(np.random.randint(len(grasps[grasp_row])))

        grasp_cols = np.asarray(grasp_cols, dtype=np.int32)
        return np.vstack((grasp_rows, grasp_cols)).T

