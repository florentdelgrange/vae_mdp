#!/bin/sh

python train.py --flagfile inputs/Acrobot-v1 --seed 11111
python train.py --flagfile inputs/Acrobot-v1 --seed 22222
python train.py --flagfile inputs/Acrobot-v1 --seed 33333
python train.py --flagfile inputs/Acrobot-v1 --seed 44444
python train.py --flagfile inputs/Acrobot-v1 --seed 55555
# Run using a uniform replay buffer to plot the histogram latent space comparison
python train.py --flagfile inputs/Acrobot-v1 --seed 11111 --prioritized_experience_replay=False
