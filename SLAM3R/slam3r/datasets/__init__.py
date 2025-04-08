# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
import torch
import numpy as np

from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa: F401
from .replica_seq import Replica
from .scannetpp_seq import ScanNetpp_Seq_Full2
from .project_aria_seq import Aria_Seq
from .co3d_seq import Co3d_Seq
from .base.base_stereo_view_dataset import EasyDataset

def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from slam3r.utils.croco_misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader

class MultiDataLoader:
    def __init__(self, dataloaders:list, return_id=False):
        self.dataloaders = dataloaders
        self.len_dataloaders = [len(loader) for loader in dataloaders]
        self.total_length = sum(self.len_dataloaders)
        self.epoch = None
        self.return_id = return_id  
    
    def __len__(self):
        return self.total_length 
    
    def set_epoch(self, epoch):
        for data_loader in self.dataloaders:
            if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
                data_loader.dataset.set_epoch(epoch)
            if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
                data_loader.sampler.set_epoch(epoch)
        self.epoch = epoch
        
    def __iter__(self):
        loader_idx = []
        for idx, length in enumerate(self.len_dataloaders):
            loader_idx += [idx]*length
        loader_idx = np.array(loader_idx)
        assert loader_idx.shape[0] == self.total_length
        #是否需要一个统一的seed让每个进程遍历dataloaders的顺序相同？
        if self.epoch is None:
            assert len(self.dataloaders) == 1
        else:
            seed = 777 + self.epoch
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(loader_idx)
        batch_count = 0
        
        iters = [iter(loader) for loader in self.dataloaders] # iterator for each dataloader
        while True:
            idx = loader_idx[batch_count]
            try:
                batch = next(iters[idx])
            except StopIteration: # this won't happen in distribute mode if drop_last is False
                iters[idx] = iter(self.dataloaders[idx])
                batch = next(iters[idx])
            if self.return_id:
                batch = (batch, idx)
            yield batch    
            batch_count += 1
            if batch_count == self.total_length: 
                # batch_count -= self.total_length
                break
            

def get_multi_data_loader(dataset, batch_size, return_id=False, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from slam3r.utils.croco_misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    if isinstance(dataset, EasyDataset):
        datasets = [dataset]
    else:
        datasets = dataset
    print(datasets)
    assert isinstance(datasets,(tuple, list))
    
    world_size = get_world_size()
    rank = get_rank()
    dataloaders = []
    for dataset in datasets:
        try:
            sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                        rank=rank, drop_last=drop_last)
        except (AttributeError, NotImplementedError):
            # not avail for this dataset
            if torch.distributed.is_initialized():
                sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
                )
            elif shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )
        dataloaders.append(data_loader)

    multi_dataloader = MultiDataLoader(dataloaders, return_id=return_id)
    return multi_dataloader
