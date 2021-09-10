#!/bin/sh

python train.py --flagfile inputs/MountainCar-v0 --seed 20210517
python train.py --flagfile inputs/MountainCar-v0 --seed 11111111
python train.py --flagfile inputs/MountainCar-v0 --seed 22222222
python train.py --flagfile inputs/MountainCar-v0 --seed 33333333
python train.py --flagfile inputs/MountainCar-v0 --seed 44444444
# Run using a uniform replay buffer to plot the histogram latent space comparison
python train.py --flagfile inputs/MountainCar-v0 --seed 11111111 --prioritized_experience_replay=False
