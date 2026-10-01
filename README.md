# An Empirical Study of Large-Scale Data-Driven Full Waveform Inversion (BigFWI)

<p align="center">
  <a href="https://arxiv.org/abs/2307.15388"><img src="https://img.shields.io/badge/Paper-arXiv:2307.15388-B31B1B.svg" height=25></a> &ensp;
  <a href="https://www.nature.com/articles/s41598-024-68573-7"><img src="https://img.shields.io/badge/paper-Scientific_Reports-blue.svg" height=25></a>
</p>

This is the official repository for paper "An empirical study of large-scale data-driven full waveform inversion".

***
## 📜 Abstract

This paper investigates the impact of big data on deep learning models to help solve the full waveform inversion (FWI) problem. While it is well known that big data can boost the performance of deep learning models in many tasks, its effectiveness has not been validated for FWI. To address this gap, we present an empirical study that investigates how deep learning models in FWI behave when trained on openfwi, a collection of large-scale, multi-structural, synthetic datasets published recently. In particular, we train and evaluate the FWI models on a combination of 10 2D subsets in openfwi that contain 470 K pairs of seismic data and velocity maps in total. Our experiments demonstrate that training on the combined dataset yields an average improvement of 13.03% in MAE, 7.19% in MSE and 1.87% in SSIM compared to each split dataset, and an average improvement of 28.60%, 21.55% and 8.22% in the leave-one-out generalization test. We further demonstrate that model capacity needs to scale in accordance with data size for optimal improvement, where our largest model yields an average improvement of 20.06%, 13.39% and 0.72% compared to the smallest one.

## 🔧 Dependencies and Installation
- python >=3.10
- PyTorch >= 1.13.0

```bash
conda create -n bigfwi python=3.10
conda activate bigfwi
git clone https://github.com/PengJin95/BigFWI.git
cd BigFWI
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```


## 📂 Data Preparation
We use OpenFWI to train our BigFWI models. More information can be found at OpenFWI's [project](https://smileunc.github.io/projects/openfwi) page. Before training, you can merge all the paths in the split files into a single file (one for training and one for validation) and pre-shuffle the paths. The resulted files can be stored as `relevant_files/open_train_shuffled.txt` and `relevant_files/open_val_shuffled.txt`.

## 🚀 Training

Launch single-GPU training using the following command:
```bash
python train_open.py -s all_408k -m FCN4_Deep2 -lm 150 160
```

The training script also supports multi-node multi-GPU training. The example command used in a Slurm script can be:
```bash
srun python -u train_open.py -s all_408k -m FCN4_Deep2 -lm 150 160 \
  --sync-bn --dist-url tcp://$MASTER:$MASTERPORT --world-size $SLURM_NTASKS
```
The variables can be extracted from
```
MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MASTERPORT=6000
```

## 🧪 Testing
The minimal test script is provided in `minimal_test.py` where you can manually test certain model on a few batches.

The full test can be launched using the following command:
```bash
python test.py -m FCN4_Deep_2 -s all_408k -r BigFWI_B.pth --vis --vis-suffix val --vis-batch 6 -vis-sample 10
``` 

## 📸 Checkpoints

The checkpoints of BigFWI-B, BigFWI-M and BigFWI-XL are provided on [Google Drive](https://drive.google.com/drive/folders/1Wy7UTRgzKytP_Aoe0hJpdmXXRejetuzA?usp=sharing). 

## 📄 Citation

If you use BigFWI in your research, please cite:

    @article{jin2024empirical,
      title={An empirical study of large-scale data-driven full waveform inversion},
      author={Jin, Peng and Feng, Yinan and Feng, Shihang and Wang, Hanchen and Chen, Yinpeng and Consolvo, Benjamin and Liu, Zicheng and Lin, Youzuo},
      journal={Scientific reports},
      volume={14},
      number={1},
      pages={20034},
      year={2024},
      publisher={Nature Publishing Group UK London}
    }
