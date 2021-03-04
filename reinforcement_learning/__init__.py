import tensorflow as tf
import math

labeling_functions = {
    'HumanoidBulletEnv-v0':
        lambda observation: observation[..., 0] + 0.8 <= 0.78,  # falling down
    # np.count_nonzero(np.abs(observation[:, :, 8: 42][0::2]) > 0.99) > 0  # has stuck joints
    'BipedalWalker-v2':
        lambda observation: tf.math.abs(observation[..., 0]) > math.pi / 3.,  # hull angle too high/low
    'Pendulum-v0':
        lambda observation: tf.stack([
            tf.logical_and(observation[..., 0] > 0., tf.abs(observation[..., 1]) < tf.math.sin(math.pi / 2)),
            # easy: |theta| < 90
            tf.logical_and(observation[..., 0] > 0., tf.abs(observation[..., 1]) < tf.math.sin(math.pi / 6)),
            # soft: |theta| < 30
            tf.logical_and(observation[..., 0] > 0., tf.abs(observation[..., 1]) < tf.math.sin(math.pi / 9)),
            # hard: |theta| < 20
        ], axis=-1)

}
