import importlib
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name,model):
    if model.startswith('GEC'):
        dataset_filename = "data." + dataset_name + "pose_dataset"
    else:
        dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name,model):
    dataset_class = find_dataset_using_name(dataset_name,model)
    return dataset_class.modify_commandline_options
