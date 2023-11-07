# Steering-Angle-Estimation-via-Liquid-Time-Constant-RNNs
Steering Angle Estimation via Liquid Time Constant RNNs

Relevant papers used for this project:

[Liquid Time-Constant Networks](https://arxiv.org/pdf/2006.04439.pdf)

[Neural Circuit Policies Enabling Auditable Autonomy](https://publik.tuwien.ac.at/files/publik_292280.pdf)

This project uses source code from various files from https://github.com/mlech26l/ncps, licensed under the Apache 2.0 license. The original license is included in the file LICENSE.


## Run
### Setup
`#TODO: pip install -r requirements.txt`
### Data Preprocessing
`python3 process.py`
### Training
`python3 train.py --batch_size 32 --lr 0.0005 --epochs 2 --N 10 --units 16 --log_freq 100`
