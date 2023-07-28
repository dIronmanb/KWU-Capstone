import importlib
import torch.utils.data
# from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    pass

def get_option_setter(dataset_name):
    pass

def create_dataset(opt):

    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset






class PointsDatasetDataLaoder()