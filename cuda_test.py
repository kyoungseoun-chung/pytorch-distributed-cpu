#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


model = models.resnet50()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

dataset = datasets.FakeData(
    size=1000,
    transform=transforms.ToTensor())
loader = DataLoader(
    dataset,
    num_workers=1,
    pin_memory=True
)

device = torch.device('cuda')

model.to(device)

tic = time.time()
for i, (data, target) in enumerate(loader):
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    if i % 100 == 0:
        print(f'{i+1} / 1000')

print(f'Done in {time.time() - tic} [s]')
