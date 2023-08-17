import torch.utils.data
from data.base_dataset import collate_fn

def CreateDataset(opt):
    """loads dataset class"""

    from data.constrained_grasp_sampling_data import ConstrainedGraspSamplingData
    dataset = ConstrainedGraspSamplingData(opt)

    return dataset


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        self.create_dataset()

    def create_dataset(self):
        self.dataset = CreateDataset(self.opt)

    def split_dataset(self, split_size_percentage=[0.8, 0.15, 0.05]):
        dataset_size = len(self.dataset)
        number_of_training_samples = round(split_size_percentage[0]*dataset_size)
        number_of_test_samples = round(split_size_percentage[1]*dataset_size)
        number_of_validation_samples = dataset_size - number_of_training_samples - number_of_test_samples

        return torch.utils.data.random_split(
            self.dataset, [number_of_training_samples, number_of_test_samples, number_of_validation_samples])

    def create_dataloader(self, data_loader, shuffle_batches):
        self.dataloader = torch.utils.data.DataLoader(
            data_loader,
            batch_size=self.opt.num_objects_per_batch,
            shuffle=shuffle_batches,
            num_workers=int(self.opt.num_threads),
            collate_fn=collate_fn)
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataloader:
            yield data
