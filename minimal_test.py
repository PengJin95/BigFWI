# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

# %%
# Make sure you first download data1.npy and model1.npy from the shared Google Drive folder of OpenFWI Style-A and put them in the current directory.
with open('sa_example.txt', 'w') as f:
    f.write('data1.npy\tmodel1.npy\n')

# %%
import json
import transforms as T
from torchvision.transforms import Compose
from dataset import FWIDataset
from torch.utils.data import SequentialSampler, default_collate

with open('dataset_config.json') as f:
    ctx = json.load(f)['open']
log_data_min = T.log_transform(ctx['data_min'])
log_data_max = T.log_transform(ctx['data_max'])
transform_valid_data = Compose([
    T.LogTransform(k=1),
    T.MinMaxNormalize(log_data_min, log_data_max),
])

transform_valid_label = Compose([
    T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
])
dataset_valid = FWIDataset(
    'sa_example.txt',
    file_size=ctx['file_size'],
    transform_data=transform_valid_data,
    transform_label=transform_valid_label
)
print(len(dataset_valid))
valid_sampler = SequentialSampler(dataset_valid)
dataloader_valid = torch.utils.data.DataLoader(
    dataset_valid, batch_size=8,
    sampler=valid_sampler, num_workers=0,
    pin_memory=False, collate_fn=default_collate)

# %%
import network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_model = 'BigFWI_B'
if test_model == 'BigFWI_B':
    model = network.model_dict['FCN4_Deep_2']().to(device)
    ckpt_b = torch.load('../BigFWI_B.pth', map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt_b['model'], strict=False)
elif test_model == 'BigFWI_L':
    model = network.model_dict['FCN4_Deep_2L']().to(device)
    ckpt_l = torch.load('../BigFWI_L.pth', map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt_l['model'], strict=False)
elif test_model == 'BigFWI_XL':
    model = network.model_dict['FCN4_Deep_2XL']().to(device)
    ckpt_xl = torch.load('../BigFWI_XL.pth', map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt_xl['model'], strict=False)
else:
    raise ValueError(f'Unknown test model: {test_model}')

print('Missing keys:', missing)
print('Unexpected keys:', unexpected)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The number of trainable parameters: {num_params:,}')
# 24.4M for FCN4_Deep_2
# 28.9M for FCN4_Deep_2
# 109.7M for FCN4_Deep_2XL

# %%
data, label = next(iter(dataloader_valid))

# %%
with torch.inference_mode():
    data = data.type(torch.FloatTensor).to(device)
    label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
    pred = model(data)
    label_pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False)

# %%
from matplotlib.colors import ListedColormap
rainbow_cmap = ListedColormap(np.load('rainbow256.npy'))

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
im = ax[0].imshow(label_np[0, 0], cmap=rainbow_cmap, vmin=ctx['label_min'], vmax=ctx['label_max'])
ax[1].imshow(label_pred_np[0, 0], cmap=rainbow_cmap, vmin=ctx['label_min'], vmax=ctx['label_max'])
fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
