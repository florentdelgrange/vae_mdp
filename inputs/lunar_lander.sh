#!/bin/sh

python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 20421
python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 11111
python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 22222
python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 33333
python train.py --flagfile inputs/LunarLanderContinuous-v2 --seed 44444
# Run using a uniform replay buffer to plot the histogram latent space comparison
python train.py --flagfile inputs/Pendulum-v0 --seed 20421 --prioritized_experience_replay=False
