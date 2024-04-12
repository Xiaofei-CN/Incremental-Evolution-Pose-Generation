import torch.utils.data
from data.base_data_loader import BaseDataLoader
import data

def CreateDataset(opt):
    '''
    dataset = None
    if opt.dataset_mode == 'fashion':
        from data.fashion_dataset import FashionDataset
        dataset = FashionDataset()
    else:
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    '''
    dataset = data.find_dataset_using_name(opt.dataset_mode,opt.model)()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.instance = CreateDataset(opt)
        shuffle = opt.phase == 'train'
        self.sampler = data_sampler(self.instance, shuffle=shuffle, distributed=opt.distributed)
        self.dataloader = torch.utils.data.DataLoader(
                    self.instance,
                    batch_size=opt.batchSize,
                    sampler=self.sampler,
                    num_workers=0,#int(opt.nThreads),
                    drop_last=shuffle
                )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.instance), self.opt.max_dataset_size)
