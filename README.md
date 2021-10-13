# An-Almost-Real-time-Video-Denoising-Model
改编于 FastDVD + 旷视'Practical Deep Raw Image Denoising'（旷视的网络下面都称为megvii block）

FastDVDnet：https://arxiv.org/abs/1907.01361

旷视：https://arxiv.org/abs/2010.06935

# Overview
这个源代码提供了一个 Pytorch 实现的视频去噪算法。

# Datasets and Video Examples

所有的东西都在 **fastdvdnet_zky** 这个文件夹里。

## Trainset
训练集：2017 DAVIS dataset，单张分辨率960x540，路径../fastdvdnet_zky/trainset/

## Testset
测试集：Set8，路径../fastdvdnet_zky/testset/ 。之前所有的测试都用的 hypersmooth，motorbike，rafting，snowboard这四组，因为fastdvd的论文里只有这四组的结果。

## Dependencies
环境：lpz_vdenoising

所有的 dependencies 都在 requirements.yml 里面。

## Run codes
所需代码：
1. dataloaders.py
2. dataset.py
3. fastdvdnet.py
4. models_mevgii3.py
5. **test_temp.py**  (测试代码运行，运行可用test.sh)
6. **train_faster135.py** (训练代码运行，运行可用train_faster.sh)
7. utils.py

所需权重：

路径../fastdvdnet_zky/logs_mevgii3/ (注：这里的命名是mevgii，不是megvii，此锅owner不背）

## Best Performance So Far

![0390e64a1406d7e42fef061e4e553a1](https://user-images.githubusercontent.com/65483602/137082485-6c89675e-75a5-4fb2-b9b7-1d1265afccfe.png)

蓝色背景数字 **22.17** 为连续处理25帧图片时，每帧图片去噪需要 22.17ms ；黄色背景数字 **30.49** 为PSNR。

**该模型结构为：**

<img width="483" alt="1634108988(1)" src="https://user-images.githubusercontent.com/65483602/137084339-1419271d-b62c-4634-9f66-457c025357d5.png">

## 测试集去噪结果

路径：../fastdvdnet_zky/denoised_results/
