#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def init_process(rank, size, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12254'

    dist.init_process_group(backend, rank=rank, world_size=size)

    if rank == 0:
        print('SYS: ' +
              f'distributed environment initialized with {size} processes')

    # Disabling randomness
    torch.manual_seed(0)
    np.random.seed(0)


def clean_up(self):
    """kill all multiprocessing groups
    """
    dist.destroy_process_group()


def training_distributed(world_size):
    """start distributed training
    """

    # starts multiprocessing
    mp.spawn(run_training, args=(world_size,),
             nprocs=world_size, join=True)


def run_training(rank, world_size):

    # initialize multiprocessing environment
    init_process(rank, world_size)

    # smaple model: resnet50
    model = models.resnet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # fake data set for test: size=1000
    dataset = datasets.FakeData(
        size=1000,
        transform=transforms.ToTensor())

    # sampler
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        num_workers=0,
        pin_memory=False,
        sampler=sampler
    )

    device = torch.device('cpu')

    model.to(device)
    model = DDP(model, device_ids=None)

    dist.barrier()

    if rank == 0:
        tic = time.time()

    # Not in this example but if you have a number of epochs, calling
    # set_epoch function for the data sampler will enhance the model
    # accuracy e.g) sampler.set_epoch(epoch)

    for i, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

    dist.barrier()

    if rank == 0:
        print(f'Done in {time.time() - tic} [s]')

    clean_up()


if __name__ == "__main__":

    try:
        # to get number of core available when you submit the job
        # in the cluster using bsub -n command
        # this (probably) works with the cluster which is using lsf environment
        world_size = int(os.environ['LSB_DJOB_NUMPROC'])
    except KeyError:
        # otherwise, use maximum number of cpu available
        world_size = os.cpu_count()

    training_distributed(world_size)
