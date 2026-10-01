# Modified from DistributedSampler to avoid OOM
# Each process loads a specific subset of data
# - Difference: 
#     One process in DistributedSampler can access different subset per iteration (epoch)
#       (e.g. 1 3 5 for 1st iter, 2 4 5 for 2nd iter)
#     One process in FWIDistributedSampler always loads a specific subset every iteration
#       (e.g. 1 3 5 for 1st iter, 5 1 3 for 2nd iter if shuffle=True)
# - Limitation:
#     When data is not evenly divisible:
#       drop_last=False: first few elements will be used for padding 
#       drop_last=True: last few elements will be dropped
#     Not sure if randomness should be added

import math
from typing import TypeVar, Iterator

import torch
from torch.utils.data import DistributedSampler

T_co = TypeVar('T_co', covariant=True)

class FWIDistributedSampler(DistributedSampler[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        indices = list(range(len(self.dataset)))
        if not self.drop_last:
            # total_size = num_samples * num_replicas
            padding_size = self.total_size - len(self.dataset)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else: # num_replicas > len（dataset)
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        start_idx = self.rank * self.num_samples
        indices = indices[start_idx:(start_idx + self.num_samples)]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = [indices[i] for i in torch.randperm(self.num_samples, generator=g)]
        assert len(indices) == self.num_samples
        # print('Replica: ', self.rank, 'Len Idx: ', len(indices), 'Min Idx: ', min(indices), 'Max Idx: ', max(indices))
        return iter(indices)
