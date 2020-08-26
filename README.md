# [Efficient Crowd Counting via Structured Knowledge Transfer](https://arxiv.org/abs/2003.10120) (ACM MM 2020)
Crowd counting is an application-oriented task and its inference efficiency is crucial for real-world applications. However, most previous works relied on heavy backbone networks and required prohibitive run-time consumption, which would seriously restrict their deployment scopes and cause poor scalability. To liberate these crowd counting models, we propose a novel Structured Knowledge Transfer (SKT) framework, which fully exploits the structured knowledge of a well-trained teacher network to generate a lightweight but still highly effective student network. 

Extensive evaluations on three benchmarks well demonstrate the effectiveness of our SKT for extensive crowd counting models. In this project, the well-trained teacher networks and the distilled student networks have been released at [GoogleDrive](https://drive.google.com/drive/folders/17oxen8sNHtumcFL8hu9Z0Owuc6dWD8zV?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/10_SLXF_FID9huRbzMHFT4A) (extract code: srpl). If you use this code and the released models for your research, please cite our paper:
```
@inproceedings{liu2020efficient,
  title={Efficient Crowd Counting via Structured Knowledge Transfer},
  author={Liu, Lingbo and Chen, Jiaqi and Wu, Hefeng and Chen, Tianshui and Li, Guanbin and Lin, Liang},
  booktitle={ACM International Conference on Multimedia},
  year={2020}
}
```

## Datasets
ShanghaiTech: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)

UCF-QNRF: [Link](https://www.crcv.ucf.edu/data/ucf-qnrf/)

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 2.7

PyTorch: 0.4.0

## Preprocessing

1. Generation the ground-truth density maps for training
```
# ShanghaiTech
python preprocess/ShanghaiTech_GT_generation.py

# UCF-QNRF
python preprocess/UCF_GT_generation.py --mode train
python preprocess/UCF_GT_generation.py --mode test
```

2. Make data path files and edit this file to change the path to your original datasets.
```
python preprocess/make_json.py
```


## Training
Edit this file for distillation training
```
bash SKT_distill.sh
```

## Testing
Edit this file for testing models
```
bash test.sh
```

## Models
The well-trained teacher networks and the distilled student networks are released at have been released at [GoogleDrive](https://drive.google.com/drive/folders/17oxen8sNHtumcFL8hu9Z0Owuc6dWD8zV?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/10_SLXF_FID9huRbzMHFT4A) (extract code: srpl
). In particular, only using around 6% of the parameters and computation cost of original models, our distilled VGG-based models obtain at least 6.5× speed-up on an Nvidia 1080 GPU and even achieve state-of-the-art performance.

#### Shanghaitech A (576×864)
| Method | MAE | RMSE | #Param (M) | FLOPs (G) | GPU (ms) | CPU (s) | Comment | 
| --- | --- |  --- | --- |--- | --- | --- | --- |
| CSRNet | 68.43 | 105.99 | 16.26 | 205.88 | 66.58 | 7.85  | teacher model, trained with [CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch) |
| 1/4-CSRNet + SKT | 71.55 | 114.40 | 1.02 | 13.09 | 8.88 | 0.87 | -- |
| BL | 61.46 | 103.17 | 21.50 | 205.32 | 47.89 |  8.84 | teacher model |
| 1/4-BL + SKT | 62.73 | 102.33 | 1.35 | 13.06 | 7.40 | 0.88 | -- |

#### UCF-QNRF (2032×2912)
| Method | MAE | RMSE | #Param (M) | FLOPs (G) | GPU (ms) | CPU (s) | Comment | 
| --- | --- |  --- | --- |--- | --- | --- | --- |
| CSRNet | 145.54 | 233.32 | 16.26 | 2447.91 | 823.84 | 119.67 | teacher model, trained with [CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch) |
| 1/4-CSRNet + SKT | 144.36 | 234.64 | 1.02 | 155.69 | 106.08 | 9.71 | -- |
| BL | 87.70 | 158.09 | 21.50 | 2441.23 | 595.72 | 130.76 | teacher model |
| 1/4-BL + SKT | 96.24 | 156.82 | 1.35 | 155.30 | 90.96 | 9.78 | The released model is much better. |
