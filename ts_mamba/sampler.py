import torch
import itertools
import random

class InfiniteRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        while True:
            yield from torch.randperm(len(self.dataset)).tolist()

    def __len__(self):
        return 2**63  # effectively infinite
