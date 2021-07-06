# Blind restoration of solar images via the Channel Sharing Spatio-Temporal Network

## Introduction

Due to the existence of atmospheric turbulence, the image is seriously degraded when the ground-based telescope observes
the object. The adaptive optics (AO) system can achieve partial correction but not reach the diffraction limit. In order to further
improve the imaging quality, post-processing for AO closed-loop image is still necessary. Methods based on deep learning have
been proposed to AO image reconstruction, but the most of them assumed that the point spread function is spatially invariant. We construct clear solar images by using a sophisticated spatially variant end-to-end blind restoration network in this paper.

## Pretrained Models

You could find the pretrained model in `./ckpt/best-ckpt.pth.tar`. 

## Prerequisites

- `Linux(tested on Titan XP + Ubuntu16.04 + cuda9.0 +pytorch1.0.1 + python2.7)`
- `Linux (tested on 2060/2080Ti + Ubuntu18.04 + cuda10.0 + pytorch1.0.0 + python2.7)`
- `Linux (tested on 2080Ti + Ubuntu18.04 + cuda10.2 + pytorch1.4.0 + python2.7)`
- `Linux (tested on 1050 + Ubuntu16.04 + cuda9.0 + pytorch1.1.0 + python3.7)`
- `Linux (tested on 1080i + Ubuntu16.04 + cuda9.0 + pytorch1.0.0 + python2.7)`

We didn't try any other environment,  if your environment does not match the ones listed above,  you may need to modify the file  

`./models/FAC/kernelconv2d/KernelConv2D_cuda.cpp` and `./models/FAC/kernelconv2d/setup.py`.

#### Installation

```
pip install -r requirements.txt
bash install.sh
```

## Get Started

Use the following command to train the neural network:

```
python runner.py 
        --phase 'train'\
        --data [dataset path]\
        --out [output path]
```

Use the following command to test the neural network:

```
python runner.py \
        --phase 'test'\
        --weights './ckpt/best-ckpt.pth.tar'\
        --data [dataset path]\
        --out [output path]
```
Use the following command to resume training the neural network:

```
python runner.py 
        --phase 'resume'\
        --weights './ckpt/best-ckpt.pth.tar'\
        --data [dataset path]\
        --out [output path]
```
You can also use the following simple command, with changing the settings in config.py

```bash
python runner.py
```

## Citation
If you find CSSTN useful in your research, please consider citing:

```
Blind restoration of solar images via the Channel Sharing Spatio-Temporal Network
Shuai Wang, Qingqing Chen, Chunyuan He, Chi Zhang, Libo Zhong, Hua Bao, Lanqiang Zhang, and Changhui Rao
Astronomy & Astrophysics,
2021, 
DOI:10.1051/0004-6361/202140376
```

## Contact

We are glad to hear if you have any suggestions and questions.

Please send email to wangshuai0601@uestc.edu.cn or qingqingchen618@gmail.com.