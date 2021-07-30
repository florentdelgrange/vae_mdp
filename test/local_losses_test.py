import os
import sys

path = os.path.dirname(os.path.abspath("__file__"))
sys.path.insert(0, path + '/../')

import gc
import variational_action_discretizer
import variational_mdp
import tf_agents.specs
from tf_agents.environments import suite_gym, parallel_py_environment
from tf_agents.environments import tf_py_environment
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')  # allows testing during training
from reinforcement_learning import labeling_functions
import tensorflow_probability as tfp

tfd = tfp.distributions

if __name__ == '__main__':
    tf_agents.system.multiprocessing.enable_interactive_mode()
    load_vae = {
        'MountainCar-v0': lambda: variational_mdp.load(
            '../saves/MountainCar-v0/models/vae_LS12_MC1_ER10.0-decay=1e-05-min=0_KLA0.0-growth=5e-05_TD0.67-0.50_1e-06'
            '-2e-06_seed=20210517_PER-priority_exponent=0.99-WIS_exponent=0.4-WIS_growth_rate=7.5e'
            '-05_bucket_based_priorities_params=full_vae_optimization-relaxed_state_encoding-latent_policy/base',
            step=780000,
            discrete_action=True),

        'CartPole-v0': lambda: variational_mdp.load(
            '../saves/CartPole-v0/models/vae_LS12_MC1_ER10.0-decay=0.00025-min=0_KLA0.0-growth=5e-06_TD0.67-0.50_1e-06'
            '-2e-06_seed=60421_params=relaxed_state_encoding-latent_policy/base',
            step=260000, discrete_action=True),

        'LunarLander-v2': lambda: variational_mdp.load(
            '../saves/LunarLander-v2/models/vae_LS20_MC1_ER10.0-decay=1e-05-min=0_KLA0.0-growth=5e-05_TD0.67-0.50_1e'
            '-06-2e-06_seed=20210510_PER-priority_exponent=0.99-WIS_exponent=0.4-WIS_growth_rate=7.5e'
            '-05loss_based_priorities_params=full_vae_optimization-relaxed_state_encoding-latent_policy/base',
            step=720000,
            discrete_action=True),
        'Pendulum-v0': lambda: variational_action_discretizer.load(
            '../saves/Pendulum-v0/models/vae_LS14_MC1_ER10.0-decay=7.5e-05-min=0_KLA0.0-growth=5e-05_TD0.67-0.50_1e'
            '-06-2e'
            '-06_seed=20210521/policy/action_discretizer/LA3_MC1_ER10.0-decay=7.5e-05-min=0_KLA0.0-growth=5e-05_TD0'
            '.50-0.33_1e-06-2e-06_PER-priority_exponent=0.99-WIS_exponent=0.4-WIS_growth_rate=1e'
            '-05loss_based_priorities_params=one_output_per_action-full_vae_optimization-relaxed_state_encoding/base',
            step=310000)}
    num_parallel_env = 4
    num_steps = 30000

    for environment_name in ['MountainCar-v0']:
        labeling_function = labeling_functions[environment_name]
        py_env = tf_agents.environments.parallel_py_environment.ParallelPyEnvironment(
            [lambda: suite_gym.load(environment_name)] * num_parallel_env)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        tf_env.reset()

        vae_mdp = load_vae[environment_name]()
        for estimate_probability_function in [True, False]:
            metrics = vae_mdp.estimate_local_losses_from_samples(
                environment=tf_env,
                steps=num_steps,
                labeling_function=labeling_function,
                estimate_transition_function_from_samples=estimate_probability_function,
                assert_estimated_transition_function_distribution=True)

            tf.print("{} environment".format(environment_name))
            tf.print("Empirical probability transition function estimation: {}".format(estimate_probability_function))
            tf.print("Latent space size:", 2 ** vae_mdp.latent_state_size)
            tf.print('Local reward loss:', metrics.local_reward_loss)
            tf.print('Local probability loss:', metrics.local_probability_loss)
            metrics.print_time_metrics()

        tf_env.close()
        del vae_mdp
        gc.collect()
