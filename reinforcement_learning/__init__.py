import tensorflow as tf
import math


def lunar_lander_labels(s):
    """
    from LunarLander heuristic
    """

    labels = []
    angle_targ = s[..., 0] * 0.5 + s[..., 2] * 1.0    # angle should point towards center
    labels.append(tf.logical_or(angle_targ > 0.4,   # more than 0.4 radians (22 degrees) is bad
                                angle_targ < -0.4))
    angle_targ = tf.map_fn(lambda x: tf.cond(x > 0.4, lambda: 0.4, lambda: x), angle_targ)
    angle_targ = tf.map_fn(lambda x: tf.cond(x < -.4, lambda: -.4, lambda: x), angle_targ)
    hover_targ = 0.55 * tf.abs(s[..., 0])           # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[..., 4]) * 0.5 - (s[..., 5])*1.0
    hover_todo = (hover_targ - s[..., 1]) * 0.5 - (s[..., 3])*0.5

    # legs contact
    labels.append(tf.logical_or(tf.cast(s[..., 6], dtype=tf.bool), tf.cast(s[..., 7], dtype=tf.bool)))
    no_legs_contact = (1. - (s[..., 6] + s[..., 7]) / tf.maximum(s[..., 6] + s[..., 7], 1.))
    angle_todo = angle_todo * no_legs_contact
    # override to reduce fall speed, that's all we need after contact
    hover_todo = hover_todo * no_legs_contact
    hover_todo = tf.map_fn(lambda x: tf.cond(x == 0., lambda: -.5 * s[..., 3], lambda:  x), hover_todo)

    labels.append(tf.logical_and(hover_todo > tf.abs(angle_todo), hover_todo > 0.05))
    labels.append(angle_todo < -0.05)
    labels.append(angle_todo > 0.05)
    labels.append(tf.logical_and(s[..., 2] == 0.,  # horizontal speed is 0
                                 s[..., 3] == 0.))  # vertical speed is 0

    return labels


labeling_functions = {
    'HumanoidBulletEnv-v0':
        lambda observation: tf.stack([
            # falling down -- observation[0] is the head position, 0.8 is the initial position
            observation[..., 0] + 0.8 <= 0.78,
            # has stuck joints
            # tf.expand_dims(tf.math.count_nonzero(tf.abs(observation[..., 8: 42][0::2]) > 0.99) > 0, axis=-1),
            # feet contact
            tf.cast(observation[..., -2], tf.bool), tf.cast(observation[..., -1], tf.bool),
            # move forward
            observation[..., 3] > 0.,
        ], axis=-1),
    'BipedalWalker-v2':
        lambda observation: tf.stack([
            tf.math.abs(observation[..., 0]) > math.pi / 3.,  # hull angle too high/low (unsafe flag)
            tf.cast(observation[..., 8], tf.bool),  # contact of the left leg with the ground
            tf.cast(observation[..., 13], tf.bool)  # contact of the right leg with the ground
        ], axis=-1),
    'Pendulum-v0':  # safe labels
        lambda observation: tf.stack([
            # cos(θ) >= cos(π / 3 rad) = cos(2 π - π / 3 rad) = cos(60°) = cos(-60°)
            observation[..., 0] >= tf.math.cos(math.pi / 3),
            # cos(θ) >= cos(π / 4 rad) = cos(2 π - π / 4 rad) = cos(45°) = cos(-45°)
            # observation[..., 0] >= tf.math.cos(math.pi / 4),
            # cos(θ) >= cos(π / 6 rad) = cos(2 π - π / 6 rad) = cos(30°) = cos(-30°)
            # observation[..., 0] >= tf.math.cos(math.pi / 6),
            # cos(θ) >= cos(π / 9 rad) = cos(2 π - π / 9 rad) = cos(20°) = cos(-20°)
            # observation[..., 0] >= tf.math.cos(math.pi / 9),
            # push direction
            observation[..., 2] >= 0,
            # cos(θ) >= 0, i.e., the pendulum is at the top of the screen
            observation[..., 0] >= 0.,
            # sin(θ) >= 0, i.e., the pendulum is at the left side of the screen
            observation[..., 1] >= 0.,
            # first quadrant -- up right
            # tf.logical_and(observation[..., 0] >= 0., observation[..., 1] >= 0.),
            # second quadrant -- down left
            # tf.logical_and(observation[..., 0] < 0., observation[..., 1] >= 0.),
            # third quadrant -- down right
            # tf.logical_and(observation[..., 0] < 0., observation[..., 1] < 0.),
            # fourth quadrant -- up right
            # tf.logical_and(observation[..., 0] >= 0., observation[..., 1] < 0.),
        ], axis=-1),
    'CartPole-v0':  # safe labels
        lambda observation: tf.stack([
            tf.abs(observation[..., 0]) < 1.5,  # cart position is less than 1.5
            tf.abs(observation[..., 2]) < 0.15,  # pole angle is inferior to 9 degrees
        ], axis=-1),
    #  'LunarLander-v2':
    #      lambda observation: tf.stack([
    #          tf.abs(observation[..., 0]) <= 0.15,  # land along the lunar pad x-position
    #          tf.abs(observation[..., 1]) <= 0.02,  # land along the lunar pad y-position
    #          # tf.abs(observation[..., 0]) >= 0.8,  # close to the edge of the frame
    #          # close to the lunar pad
    #          tf.math.logical_and(tf.abs(observation[..., 1]) <= 0.3, tf.abs(observation[..., 0]) <= 0.3),
    #          tf.logical_and(observation[..., 2] == 0.,  # horizontal speed is 0
    #                         observation[..., 3] == 0.),  # vertical speed is 0
    #          # tf.abs(observation[..., 2] + observation[..., 3]) <= 1e-4,  # speed is almost 0
    #          # observation[..., 3] <= -0.5,  # fast vertical (landing) speed
    #          # tf.abs(observation[..., 4]) <= math.pi / 3,  # lander angle is safe
    #          # tf.abs(observation[..., 4]) <= math.pi / 6,  # weak lander angle
    #          # observation[..., 5] == 0.,  # angular velocity is 0
    #          tf.logical_and(tf.cast(observation[..., 6], dtype=tf.bool),
    #                         tf.cast(observation[..., 7], dtype=tf.bool))  # ground contact
    #      ], axis=-1),
    #  'LunarLander-v2': lambda observation: tf.stack([
    #      tf.abs(observation[..., 0]) <= 0.15,  # land along the lunar pad x-position
    #      tf.abs(observation[..., 0]) >= 0.8,  # close to the edge of the frame
    #      # close to the lunar pad
    #      tf.math.logical_and(tf.abs(observation[..., 1]) <= 0.3, tf.abs(observation[..., 0]) <= 0.3),
    #      tf.abs(observation[..., 1]) <= 0.02,  # land along the lunar pad y-position
    #      observation[..., 2] == 0.,  # horizontal speed is 0
    #      observation[..., 3] == 0.,  # vertical speed is 0
    #      observation[..., 3] <= -0.5,  # fast vertical (landing) speed
    #      tf.abs(observation[..., 4]) <= math.pi / 3,  # lander angle is safe
    #      tf.abs(observation[..., 4]) <= math.pi / 6,  # weak lander angle
    #      observation[..., 5] == 0.,  # angular velocity is 0
    #      tf.cast(observation[..., 6], dtype=tf.bool),  # left leg ground contact
    #      tf.cast(observation[..., 7], dtype=tf.bool)  # right leg ground contact
    #  ], axis=-1),
    'LunarLander-v2': lambda observation: tf.stack([lunar_lander_labels(observation)], axis=-1),
    'MountainCar-v0': lambda observation: tf.stack([
        observation[..., 0] >= 0.5,  # has reached the goal
        observation[..., 0] >= -.5,  # right-hand side -- positive slope
        observation[..., 1] >= 0.,  # is going forward
    ], axis=-1),
    'Acrobot-v1': lambda observation: tf.stack([
        (-1. * observation[..., 0] - observation[..., 2] * observation[..., 0] +
         observation[..., 3] * observation[..., 1] > 1.),  # objective
        observation[..., 0] >= 0.,  # cos of the first pendulum angle
        observation[..., 1] >= 0.,  # sin of the first pendulum angle
        observation[..., 2] >= 0.,  # cos of the second pendulum angle
        observation[..., 3] >= 0.,  # cos of the first pendulum angle
        observation[..., 4] >= 0.,  # angular velocity of the first pendulum
        observation[..., 5] >= 0.  # angular velocity of the second pendulum
    ], axis=-1)
}

reward_scaling = {
    'Pendulum-v0': 1. / (2 * (math.pi ** 2 + 0.1 * 8 ** 2 + 0.001 * 2 ** 2)),
    'CartPole-v0': 1. / 2,
    'LunarLander-v2': 1. / 400,
    'MountainCar-v0': 1. / 2,
    'Acrobot-v1': 1. / 2
}  # to scale the rewards in [-1./2, 1./2]

for d in [labeling_functions, reward_scaling]:
    d['LunarLanderContinuous-v2'] = d['LunarLander-v2']
    d['LunarLanderNoRewardShaping-v2'] = d['LunarLander-v2']
    d['LunarLanderRandomInit-v2'] = d['LunarLander-v2']
    d['LunarLanderContinuousRandomInit-v2'] = d['LunarLander-v2']
    d['LunarLanderContinuousRandomInitNoRewardShaping-v2'] = d['LunarLander-v2']
    d['LunarLanderRewardShapingAugmented-v2'] = d['LunarLander-v2']
    d['LunarLanderRandomInitRewardShapingAugmented-v2'] = d['LunarLander-v2']
    d['LunarLanderRandomInitNoRewardShaping-v2'] = d['LunarLander-v2']
    d['LunarLanderContinuousRewardShapingAugmented-v2'] = d['LunarLander-v2']
    d['LunarLanderContinuousRandomInitRewardShapingAugmented-v2'] = d['LunarLander-v2']
    d['MountainCarContinuous-v0'] = d['MountainCar-v0']
    d['PendulumRandomInit-v0'] = d['Pendulum-v0']
    d['AcrobotRandomInit-v1'] = d['Acrobot-v1']
