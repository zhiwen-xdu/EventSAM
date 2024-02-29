<p align="right">English | <a href="./README_CN.md">简体中文</a></p>

<div align="center">
  <img src="assets/Logo02.PNG" width="100%" higth="100%">
  <h3 align="center"><strong>Segment Any Events via Weighted Adaptation of Pivotal Tokens [CVPR 2024] </strong></h3>
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

## About
Official Code for Segment Any Events via Weighted Adaptation of Pivotal Tokens [[`📕Paper`](https://arxiv.org/abs/2312.16222)]. This paper delves into the nuanced challenge of tailoring the Segment Anything Models (SAMs) for integration with event data, with the overarching objective of attaining robust and universal object segmentation within the event-centric domain. 
<div align="center">
  <img src="assets/Framework.PNG" width="80%" higth="80%">
</div>


## Getting Started

### Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please to install both PyTorch and TorchVision dependencies. 

Clone the repository locally:
```
pip install git+https://github.com/happychenpipi/EventSAM.git
```
Install the packages:

```
cd EventSAM
pip install -r requirements.txt
```
### Data Preparation
In this work, we collected a large-scale RGB-Event dataset for event-centric segmentation, from current available pixel-level aligned datasets ([VisEvent](https://sites.google.com/view/viseventtrack/) and [COESOT](https://github.com/Event-AHU/COESOT)), namely RGBE-SEG. To explore the zero-shot performance of our method, we showed more segmentation results on [MVSEC](https://daniilidis-group.github.io/mvsec/). Please download these datasets and put in ./data.

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

## Training
First download a pre-trained model checkpoint (e.g. ViT-B SAM model) [SAM](https://github.com/facebookresearch/segment-anything/tree/main). Then the model can be used as teacher for rgb-event knowledge distillation:

```
python ./event_encoder/train.py
```

## Evaluation
Predict the segment masks of event images:
```
python ./evaluate/predict_mask.py
```

Calculate metrics of predicted masks:
```
python ./evaluate/calculate_metric.py
```

## Visualization
<div align="center">
  <img src="assets/Visual.PNG" width="100%" higth="100%">
</div>

## EventSAM&LLM
To further validate the strong zero-shot object recognition ability of our event-adapt SAM. We integrate it with a visionlanguage object segmentation framework [LISA](https://github.com/dvlab-research/LISA). Through this, we could further unlock the rich semantic inherent in SAM, for interactive universal object segmentation with Event data. There are some visualizations.
<div align="center">
    <img src="assets/01.gif"  width="50%" height="50%" /><img src="assets/02.gif" width="50%" height="50%"/>
    <img src="assets/03.gif" width="50%" height="50%" /><img src="assets/04.gif"  width="50%" height="50%"/>
    <img src="assets/05.gif" width="50%" height="50%" /><img src="assets/06.gif" width="50%" height="50%">
</div>

## Acknowledgments
Thanks to [VisEvent](https://sites.google.com/view/viseventtrack/), [COESOT](https://github.com/Event-AHU/COESOT), [MVSEC](https://daniilidis-group.github.io/mvsec/) datasets, [SAM](https://github.com/facebookresearch/segment-anything/tree/main) and [LISA](https://github.com/dvlab-research/LISA).

## Contact
Feedbacks and comments are welcome! Feel free to contact us via [zhiwen.chen@stu.xidian.edu.cn](zhiwen.chen@stu.xidian.edu.cn) and [zhiyuzhu2-c@my.cityu.edu.hk](zhiyuzhu2-c@my.cityu.edu.hk). 

## Citing EventSAM
If you use EventSAM in your research, please use the following BibTeX entry.

```
@article{chen2023segment,
  title={Segment Any Events via Weighted Adaptation of Pivotal Tokens},
  author={Chen, Zhiwen and Zhu, Zhiyu and Zhang, Yifan and Hou, Junhui and Shi, Guangming and Wu, Jinjian},
  journal={arXiv preprint arXiv:2312.16222},
  year={2023}
}
```

