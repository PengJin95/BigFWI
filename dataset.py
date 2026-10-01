import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import transforms as T

class FWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''
    def __init__(self, anno, preload=True, sample_ratio=1, file_size=1000,
                    transform_data=None, transform_label=None):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        if preload: 
            self.data_list, self.label_list = [], []
            for batch in self.batches: 
                data, label = self.load_every(batch)
                self.data_list.append(data)
                if label is not None:
                    self.label_list.append(label)

    # Load from one line
    def load_every(self, batch):
        print('Loading ', batch[:-1])
        batch = batch.split('\t')
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')
        if len(batch) > 1:
            label_path = batch[1][:-1]    
            label = np.load(label_path)
            label = label.astype('float32')
        else:
            label = None
        return data, label
        
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx] if len(self.label_list) != 0 else None
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        return data, label if label is not None else np.array([])
        
    def __len__(self):
        return len(self.batches) * self.file_size

# Load data during the first epoch
# Add support for dataset embedding
class FWIDataset2(Dataset):
    def __init__(self, anno, sample_ratio=1, file_size=500, use_cache=True, use_dataset_id=False,
                    transform_data=None, transform_label=None):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        self.use_cache = use_cache
        self.cached_batches = {}
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        self.use_dataset_id = use_dataset_id
        self.dataset_list = ['FlatVel_A', 'FlatVel_B', 'CurveVel_A', 'CurveVel_B', 
            'FlatFault_A', 'FlatFault_B', 'CurveFault_A', 'CurveFault_B', 'Style_A', 'Style_B']

    # Load from one line
    def load_every(self, batch):
        print('Loading ', batch[:-1])
        batch = batch.split('\t')
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')
        if len(batch) > 1:
            label_path = batch[1][:-1]    
            label = np.load(label_path)
            label = label.astype('float32')
        else:
            label = None
        if self.use_dataset_id:
            for i, d in enumerate(self.dataset_list):
                if d in batch[0]: # check if dataset name in path
                    return data, label, i
        return data, label, -1
        
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.use_cache and batch_idx in self.cached_batches.keys():
            data, label, dataset_id = self.cached_batches[batch_idx]
        else:
            data, label, dataset_id = self.load_every(self.batches[batch_idx])
            if self.use_cache:
                self.cached_batches[batch_idx] = (data, label, dataset_id)
        data = data[sample_idx]
        label = label[sample_idx] if label is not None else None
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        if label is None:
            label = np.array([])
        return data, label, dataset_id
        
    def __len__(self):
        return len(self.batches) * self.file_size

if __name__ == '__main__':
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(-30, k=1), T.log_transform(60, k=1))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(1500, 4500)
    ])
    dataset = FWIDataset(f'relevant_files/debug.txt', 
                         transform_data=transform_data, 
                         transform_label=transform_label, 
                         file_size=50)
    data, label = dataset[0]
    print(data.shape)
    print(label.shape)

