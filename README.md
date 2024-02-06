# SYMBOL: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning
[![ArXiv](https://img.shields.io/badge/arXiv-2402.02355-b31b1b.svg)](https://arxiv.org/abs/2402.02355)

This is the python implementation of our paper "[Symbol: Generating Flexible Black-Box Optimizers Through Symbolic Equation Learning](https://arxiv.org/abs/2402.02355)", which is accepted as a poster paper in ICLR 2024.

![Animation](fig/animation.gif)

## Installations
```bash
git clone https://github.com/GMC-DRL/SymBol.git
cd SymBol
```

## Requirements
* Platform preferences: Linux (for better parallelism).
  Windows is also compatible, but the running speed will be slower due to the limit of the currently using parallel strategy.

The dependencies of this project are listed in `requirements.txt`. You can install them using the following command.
```bash
pip install -r requirements.txt
```

## Quick Start

* Train
  
To trigger the training process:
```
python run.py --train
```
For more adjustable configurations, please refer to `options.py`.
* Test
```
python run.py --test --load_path _path_to_checkpoint
```

## Citing
```
@inproceedings{symbol,
author={Chen, Jiacheng and Ma, Zeyuan and Guo, Hongshu and Ma, Yining and Zhang, Jie and Gong, Yue-Jiao},
title={SYMBOL: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning},
booktitle = {The Twelfth International Conference on Learning Representations},
year={2024},
}
```

## TODO
1. Instructions of how to construct a self-defined teacher optimizer
2. Future direction
