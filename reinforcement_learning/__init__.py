labeling_functions = {
    'HumanoidBulletEnv-v0':
        lambda states: states.take(0, axis=-1) + 0.8 <= 0.78,  # falling down
        # np.count_nonzero(np.abs(observation[:, :, 8: 42][0::2]) > 0.99) > 0  # has stuck joints
}
