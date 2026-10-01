import os
import sys
import time
import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision.transforms import Compose
import numpy as np

import utils
import network
from vis import *
from dataset import FWIDataset
import transforms as T
import pytorch_ssim


def evaluate(model, criterions, dataloader, device, k, ctx, lines,
                vis_path, vis_batch, vis_sample, missing, std, save_path):
    model.eval()
    
    label_list, label_pred_list= [], [] # store denormalized predcition & gt in numpy 
    label_tensor, label_pred_tensor = [], [] # store normalized prediction & gt in tensor
    if missing or std:
            data_list, data_noise_list = [], [] # store original data and noisy/muted data

    with torch.no_grad():
        batch_idx = 0
        for line, (data, label) in zip(lines, dataloader):
            data = data.type(torch.FloatTensor).to(device, non_blocking=True)
            label = label.type(torch.FloatTensor).to(device, non_blocking=True)
            
            label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
            label_list.append(label_np)
            label_tensor.append(label)
            
            if missing or std:
                # Add gaussian noise
                data_noise = torch.clip(data + (std ** 0.5) * torch.randn(data.shape).to(device, non_blocking=True), min=-1, max=1)

                # Mute some traces
                mute_idx = np.random.choice(data.shape[3], size=missing, replace=False) 
                data_noise[:, :, :, mute_idx] = data[0, 0, 0, 0]
                
                data_np = T.tonumpy_denormalize(data, ctx['data_min'], ctx['data_max'], k=k)
                data_noise_np = T.tonumpy_denormalize(data_noise,  ctx['data_min'], ctx['data_max'], k=k)
                data_list.append(data_np)
                data_noise_list.append(data_noise_np)
                pred = model(data_noise)
            else:
                pred = model(data)

            label_pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False)
            label_pred_list.append(label_pred_np)
            label_pred_tensor.append(pred)

            if save_path is not None:
                vel_path = line.split('\t')[-1][:-1]
                vel_name = os.path.basename(vel_path)
                np.save(f'{save_path}/{vel_name}', label_pred_np)

            # Visualization
            if vis_path and batch_idx < vis_batch: #== 10:
                for i in range(vis_sample):#[0]:
                    plot_velocity(label_pred_np[i, 0], label_np[i, 0], f'{vis_path}/V_{batch_idx}_{i}.png', vmin=ctx['label_min'], vmax=ctx['label_max'])
                    plot_velocity(label_pred_np[i, 0], label_np[i, 0] - label_pred_np[i, 0], f'{vis_path}/DF_{batch_idx}_{i}.png', vmin=-500, vmax=500)
                if missing or std:
                    for ch in [2]: # range(data.shape[1]): 
                        plot_seismic(data_np[i, ch], data_noise_np[i, ch], f'{vis_path}/S_{batch_idx}_{i}_{ch}.png', 
                            vmin=ctx['data_min'] * 0.01, vmax=ctx['data_max'] * 0.01)
            batch_idx += 1

    label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    print(f'MAE: {l1(label_t, pred_t)}')
    print(f'MSE: {l2(label_t, pred_t)}')
    print(f'RMSE: {torch.sqrt(l2(label_t, pred_t))}')
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    print(f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}') # (-1, 1) to (0, 1)



def main(args):

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    utils.mkdir(args.output_path)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

    print("Loading data")
    print("Loading validation data")
    log_data_min = T.log_transform(ctx['data_min'], k=args.k)
    log_data_max = T.log_transform(ctx['data_max'], k=args.k)
    transform_valid_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    transform_valid_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])
    dataset_valid = FWIDataset(
        args.val_anno,
        sample_ratio=args.sample_temporal,
        file_size=ctx['file_size'],
        transform_data=transform_valid_data,
        transform_label=transform_valid_label
    )

    with open(args.val_anno, 'r') as f:
        lines = f.readlines()

    # print("Creating data loaders")
    valid_sampler = SequentialSampler(dataset_valid)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    # print("Creating model")
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
     
    model = network.model_dict[args.model](upsample_mode=args.up_mode, 
        sample_spatial=args.sample_spatial, sample_temporal=args.sample_temporal, norm=args.norm).to(device)

    criterions = {
        'MAE': lambda x, y: np.mean(np.abs(x - y)),
        'MSE': lambda x, y: np.mean((x - y) ** 2)
    }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model_hr' in checkpoint.keys():
            model_checkpoint = checkpoint['model_hr']
        else: 
            model_checkpoint = checkpoint['model']
        model.load_state_dict(network.replace_legacy(model_checkpoint))
        # print('Loaded model checkpoint at Epoch {} / Step {}.'.format(checkpoint['epoch'], checkpoint['step']))
    
    if args.vis:
        # Create folder to store visualization results
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.output_path, vis_folder)
        utils.mkdir(vis_path)
    else:
        vis_path = None
    
    print("Start testing")
    start_time = time.time()
    evaluate(model, criterions, dataloader_valid, device, args.k, ctx, lines,
                vis_path, args.vis_batch, args.vis_sample, args.missing, args.std, args.save_path)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='open', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=str, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='relevant_files', help='annotation files location')
    parser.add_argument('-v', '--val-anno', default='open_val_shuffled.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='models', help='path to parent folder to save checkpoints')
    parser.add_argument('-sp', '--save-path', default=None, help='path to store dataset')
    parser.add_argument('-n', '--save-name', default='fcn_open', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, help='inverse model name')
    parser.add_argument('-no', '--norm', default='bn', help='normalization layer type, support bn, in, ln (default: bn)')
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')

    # Test related
    parser.add_argument('-b', '--batch-size', default=50, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--vis', help='visualization option', action="store_true")
    parser.add_argument('--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('--vis-batch', help='number of batch to be visualized', default=0, type=int)
    parser.add_argument('--vis-sample', help='number of samples in a batch to be visualized', default=0, type=int)
    parser.add_argument('--missing', default=0, type=int, help='number of missing traces')
    parser.add_argument('--std', default=0, type=float, help='standard deviation of gaussian noise')
    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.resume = os.path.join(args.output_path, args.resume)

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

"""
Example usage:
  python test.py -m FCN4_Deep_2 -s all_408k -r BigFWI_B.pth --vis --vis-suffix val --vis-batch 6 -vis-sample 10
"""