# EventSAM
Official Code for Segment Any Events via Weighted Adaptation of Pivotal Tokens [[`📕Paper`](https://arxiv.org/submit/5314117/view)] 

<div align="center">
  <img src="assets/Framework.PNG" width="80%" higth="80%">
</div>

This paper delves into the nuanced challenge of tailoring the Segment Anything Models (SAMs) for integration with event data, with the overarching objective of attaining robust and universal object segmentation within the event-centric domain. 

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
In this work, we collected a large-scale RGB-Event dataset for event-centric segmentation, from current available pixel-level aligned datasets ([VisEvent](https://sites.google.com/view/viseventtrack/) and [COESOT](https://github.com/Event-AHU/COESOT)), namely RGBE-SEG. Please download this dataset and put in ./data.


## Training

## Evaluation

## Acknowledgments

## Contact
Feedbacks and comments are welcome! Feel free to contact us via [zhiwen.chen@stu.xidian.edu.cn](zhiwen.chen@stu.xidian.edu.cn). 
