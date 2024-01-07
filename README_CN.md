<p align="right">English | <a href="./README_CN.md">简体中文</a></p>

<div align="center">
  <img src="assets/Logo01.PNG" width="100%" higth="100%">
  <h3 align="center"><strong>Segment Any Events via Weighted Adaptation of Pivotal Tokens </strong></h3>
    <p align="center">
    <a>Zhiwen Chen</a><sup>1</sup>&nbsp;&nbsp;
    <a>Zhiyu Zhu</a><sup>2</sup>&nbsp;&nbsp;
    <a>Yifan Zhang</a><sup>2</sup>&nbsp;&nbsp;
    <a>Junhui Hou</a><sup>2</sup>&nbsp;&nbsp;
    <a> Guangming Shi</a><sup>1</sup>&nbsp;&nbsp;
    <a>Jinjian Wu</a><sup>1</sup>
    <br>
    <sup>1</sup>Xidian University&nbsp;&nbsp;&nbsp;
    <sup>2</sup>City University of Hong Kong&nbsp;&nbsp;&nbsp;
</div>

## 项目概览
这个项目是Segment Any Events via Weighted Adaptation of Pivotal Tokens [[`📕论文`](https://arxiv.org/abs/2312.16222)] 的官方代码。 本文深入探讨了将SAM分割模型迁移到事件域的挑战，其目标是在事件域内实现鲁棒和通用的目标分割。
<div align="center">
  <img src="assets/Framework.PNG" width="80%" higth="80%">
</div>


## 项目开始

### 安装依赖项
我们的代码需要 `python>=3.8`, `pytorch>=1.7` 和 `torchvision>=0.8`等依赖项. 请同时安装PyTorch和TorchVision依赖项。

Clone the repository locally:
```
pip install git+https://github.com/happychenpipi/EventSAM.git
```
Install the packages:

```
cd EventSAM
pip install -r requirements.txt
```
### 数据准备
在这项工作中，我们从当前可用的像素级对齐数据集中收集了一个大规模的RGB-Event数据集，用于以事件的分割 ([VisEvent](https://sites.google.com/view/viseventtrack/) 和 [COESOT](https://github.com/Event-AHU/COESOT)), 命名为 RGBE-SEG. 为了进一步探讨我们方法的零样本泛化性能, 我们在MV[MVSEC](https://daniilidis-group.github.io/mvsec/) 数据集上显示了更多的分割结果. 请下载这些数据集并把它们放在./data文件夹下.

Format of RGBE_SEG/MVSEC datasets:
```Shell
├── RGBE_SEG dataset
    ├── Training Subset (473 sequences)
        ├── dvSave-2021_09_01_06_59_10
            ├── rgb_image
            ├── event_image
        ├── ... 
    ├── Testing Subset (108 sequences)
        ├── dvSave-2021_07_30_11_04_12
            ├── rgb_image
            ├── event_image
        ├── ... 
```

## 训练
首先下载相应的SAM预训练权重 (e.g. ViT-B SAM model) [SAM](https://github.com/facebookresearch/segment-anything/tree/main). 然后，我们运行RGB-Event知识蒸馏模型:

```
python ./event_encoder/train.py
```

## 评估
预测事件表征的分割掩码:
```
python ./evaluate/predict_mask.py
```

计算分割掩码的性能指标:
```
python ./evaluate/calculate_metric.py
```

## 可视化
<div align="center">
  <img src="assets/Visual.PNG" width="100%" higth="100%">
</div>

## EventSAM与LLM整合
为了进一步验证我们的EventSAM强大的零样本目标识别能力.我们将其与整合到视觉语言对象分割框架中 [LISA](https://github.com/dvlab-research/LISA). 通过这种方式，我们可以进一步解锁SAM中丰富语义知识，用于事件数据的交互式通用目标分割。这里是一些可视化实例：
<div align="center">
    <img src="assets/01.gif"  width="50%" height="50%" /><img src="assets/02.gif" width="50%" height="50%"/>
    <img src="assets/03.gif" width="50%" height="50%" /><img src="assets/04.gif"  width="50%" height="50%"/>
    <img src="assets/05.gif" width="50%" height="50%" /><img src="assets/06.gif" width="50%" height="50%">
</div>

## 致谢
Thanks to [VisEvent](https://sites.google.com/view/viseventtrack/), [COESOT](https://github.com/Event-AHU/COESOT), [MVSEC](https://daniilidis-group.github.io/mvsec/) datasets, [SAM](https://github.com/facebookresearch/segment-anything/tree/main) and [LISA](https://github.com/dvlab-research/LISA).

## 联系
Feedbacks and comments are welcome! Feel free to contact us via [zhiwen.chen@stu.xidian.edu.cn](zhiwen.chen@stu.xidian.edu.cn). 

## 引用EventSAM
If you use EventSAM in your research, please use the following BibTeX entry.

```
@article{chen2023segment,
  title={Segment Any Events via Weighted Adaptation of Pivotal Tokens},
  author={Chen, Zhiwen and Zhu, Zhiyu and Zhang, Yifan and Hou, Junhui and Shi, Guangming and Wu, Jinjian},
  journal={arXiv preprint arXiv:2312.16222},
  year={2023}
}
```
