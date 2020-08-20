import os
from typing import Dict

import numpy as np
import tensorflow as tf
import tf_agents.environments
from absl import app
from absl import flags
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Lambda

import reinforcement_learning
import variational_action_discretizer
import variational_mdp
from util.io import dataset_generator

flags.DEFINE_string(
    "dataset_path",
    help="Path of the directory containing the dataset files in hdf5 format.",
    default='')
flags.DEFINE_integer("batch_size", default=128, help="Batch size.")
flags.DEFINE_integer(
    "mixture_components",
    default=1,
    help="Number of gaussian mixture components used to model the states reconstruction distribution.")
flags.DEFINE_bool(
    "full_covariance",
    default=False,
    help="If set, the states and rewards reconstruction distributions will use a full covariance matrix instead of"
         "a diagonal matrix."
)
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer("latent_size", default=17, help='Number of bits used for the discrete latent state space.')
flags.DEFINE_float(
    "encoder_temperature",
    default=0.99,
    help="Temperature of the binary concrete relaxation distribution over latent states of the encoder.")
flags.DEFINE_float(
    "prior_temperature",
    default=0.95,
    help="Temperature of the binary concrete relaxation prior distribution over latent states."
)
flags.DEFINE_float(
    "encoder_temperature_decay_rate",
    default=1e-6,
    help="Decay rate used to anneal the temperature of the encoder distribution over latent states."
)
flags.DEFINE_float(
    "prior_temperature_decay_rate",
    default=2e-6,
    help="Decay rate used to anneal the temperature of the prior distribution over latent states."
)
flags.DEFINE_float(
    "regularizer_scale_factor",
    default=0.,
    help="Cross-entropy regularizer scale factor."
)
flags.DEFINE_float(
    "regularizer_decay_rate",
    default=0.,
    help="Cross-entropy regularizer decay rate."
)
flags.DEFINE_float(
    "kl_annealing_scale_factor",
    default=1.,
    help='Scale factor of the KL terms of the ELBO.'
)
flags.DEFINE_float(
    "kl_annealing_growth_rate",
    default=0.,
    help='Annealing growth rate of the ELBO KL terms scale factor.'
)
flags.DEFINE_integer(
    "start_annealing_step",
    default=int(1e4),
    help="Step from which temperatures and scale factors start to be annealed."
)
flags.DEFINE_integer(
    "max_steps",
    default=int(1e6),
    help="Maximum number of training steps."
)
flags.DEFINE_string(
    "save_dir",
    default=".",
    help="Checkpoints and models save directory."
)
flags.DEFINE_bool(
    "display_progressbar",
    default=False,
    help="Display progressbar."
)
flags.DEFINE_bool(
    "action_discretizer",
    default=False,
    help="Discretize the action space via a VAE already trained. Require the flag --load_vae to be set."
)
flags.DEFINE_bool(
    "one_output_per_action",
    default=False,
    help="Set whether discrete action networks use one output per action or use the latent action as input."
)
flags.DEFINE_boolean(
    "do_not_eval",
    default=False,
    help="Set this flag to not perform an evaluation of the ELBO (using discrete latent variables) during training."
)
flags.DEFINE_bool(
    "full_vae_optimization",
    default=False,
    help='Set whether the ELBO is optimized over the whole VAE or if the optimization is only focused on the'
         'action discretizer part of the VAE.'
)
flags.DEFINE_bool(
    'relaxed_state_encoding',
    default=False,
    help='Use a relaxed encoding of states to optimize the action discretizer part of the VAE.'
)
flags.DEFINE_integer(
    "number_of_discrete_actions",
    default=16,
    help='Number of discrete actions per states to learn.'
)
flags.DEFINE_string(
    "load_vae",
    default='',
    help='Path of a VAE model already trained to load (saved via the tf.saved_model function).'
)
flags.DEFINE_multi_integer(
    "encoder_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the encoder.'
)
flags.DEFINE_multi_integer(
    "decoder_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the decoder.'
)
flags.DEFINE_multi_integer(
    "transition_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the transition network.'
)
flags.DEFINE_multi_integer(
    "reward_layers",
    default=[256, 256],
    help='Number of units to use for each layer of the reward network.'
)
flags.DEFINE_string(
    "policy_path",
    default='',
    help="Path of a policy in tf.saved_model format."
)
flags.DEFINE_string(
    "environment",
    default='',
    help="Name of the agent's environment."
)
flags.DEFINE_string(
    "env_suite",
    default='suite_gym',
    help='Name of the tf_agents environment suite.'
)
flags.DEFINE_integer(
    "parallel_env",
    default=1,
    help='Number of parallel environments to be used during training.'
)
FLAGS = flags.FLAGS


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()

    def check_missing_argument(name: str):
        if params[name] == '':
            raise RuntimeError('Missing argument: --{}'.format(name))

    if params['action_discretizer']:
        for param in ('load_vae', 'policy_path', 'environment'):
            check_missing_argument(param)
    else:
        for param in ('dataset_path',):
            check_missing_argument(param)

    dataset_path = params['dataset_path']

    batch_size = params['batch_size']
    mixture_components = params['mixture_components']
    latent_state_size = params['latent_size']  # depends on the number of bits reserved for labels

    if params['load_vae'][-1] == os.path.sep:
        params['load_vae'] = params['load_vae'][:-1]
    if params['policy_path'][-1] == os.path.sep:
        params['policy_path'] = params['policy_path'][:-1]

    vae_name = 'vae_LS{}_MC{}_CER{}_KLA{}_TD{:.2f}-{:.2f}_{}-{}'.format(
        latent_state_size,
        mixture_components,
        params['regularizer_scale_factor'],
        params['kl_annealing_scale_factor'],
        params['encoder_temperature'],
        params['prior_temperature'],
        params['encoder_temperature_decay_rate'],
        params['prior_temperature_decay_rate']) if not params['action_discretizer'] else \
        '{}_LA{}_CER{}_KLA{}_TD{:.2f}-{:.2f}_{}-{}_policy={}'.format(
            os.path.split(params['load_vae'])[-1],
            params['number_of_discrete_actions'],
            params['regularizer_scale_factor'],
            params['kl_annealing_scale_factor'],
            params['encoder_temperature'],
            params['prior_temperature'],
            params['encoder_temperature_decay_rate'],
            params['prior_temperature_decay_rate'],
            os.path.split(params['policy_path'])[-1])

    additional_parameters = {'one_output_per_action', 'full_vae_optimization', 'relaxed_state_encoding'}
    nb_additional_params = sum(map(lambda x: params[x], additional_parameters))
    if nb_additional_params > 0:
        vae_name += ('_params={}' + '-{}' * (nb_additional_params - 1)).format(
            *filter(lambda x: params[x], additional_parameters))

    cycle_length = batch_size // 2
    block_length = batch_size // cycle_length
    activation = getattr(tf.nn, params["activation"])

    def generate_dataset():
        return dataset_generator.create_dataset(hdf5_files_path=dataset_path,
                                                cycle_length=cycle_length,
                                                block_length=block_length)
    dataset_size = -1

    # Encoder body
    q = Sequential(name="encoder_network_body")
    for i, units in enumerate(params['encoder_layers']):
        q.add(Dense(units, activation=activation, name="encoder_{}".format(i)))

    # Transition network body
    p_t = Sequential(name="transition_network_body")
    for i, units in enumerate(params['transition_layers']):
        p_t.add(Dense(units, activation=activation, name='transition_{}'.format(i)))

    # Reward network body
    p_r = Sequential(name="reward_network_body")
    for i, units in enumerate(params['reward_layers']):
        p_r.add(Dense(units, activation=activation, name='reward_{}'.format(i)))

    # Decoder network body
    p_decode = Sequential(name="decoder_body")
    for i, units in enumerate(params['decoder_layers']):
        p_decode.add(Dense(units, activation=activation, name='decoder_{}'.format(i)))

    if params['load_vae'] == '':
        dummy_dataset = generate_dataset()
        dataset_size = dataset_generator.get_num_samples(dataset_path, batch_size=batch_size, drop_remainder=True)

        state_shape, action_shape, reward_shape, _, label_shape = \
            [tuple(spec.shape.as_list()[1:]) for spec in dummy_dataset.element_spec]

        del dummy_dataset

        vae_mdp_model = variational_mdp.VariationalMarkovDecisionProcess(
            state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape, label_shape=label_shape,
            encoder_network=q, transition_network=p_t, reward_network=p_r, decoder_network=p_decode,
            latent_state_size=latent_state_size, mixture_components=mixture_components,
            encoder_temperature=params['encoder_temperature'], prior_temperature=params['prior_temperature'],
            encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
            prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
            regularizer_scale_factor=params['regularizer_scale_factor'],
            regularizer_decay_rate=params['regularizer_decay_rate'],
            kl_scale_factor=params['kl_annealing_scale_factor'],
            kl_annealing_growth_rate=params['kl_annealing_growth_rate'],
            multivariate_normal_full_covariance=params['full_covariance'])
    else:
        vae_mdp_model = variational_mdp.load(params['load_vae'])

    if params['action_discretizer']:
        vae_mdp_model = variational_action_discretizer.VariationalActionDiscretizer(
            vae_mdp=vae_mdp_model,
            number_of_discrete_actions=params['number_of_discrete_actions'],
            action_encoder_network=q, transition_network=p_t, reward_network=p_r, action_decoder_network=p_decode,
            encoder_temperature=params['encoder_temperature'], prior_temperature=params['prior_temperature'],
            encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
            prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
            one_output_per_action=params['one_output_per_action'],
            relaxed_state_encoding=params['relaxed_state_encoding'],
            full_optimization=params['full_vae_optimization']
        )
        vae_mdp_model.kl_scale_factor = params['kl_annealing_scale_factor']
        vae_mdp_model.kl_growth_rate = params['kl_annealing_growth_rate']
        vae_mdp_model.regularizer_scale_factor = params['regularizer_scale_factor']
        vae_mdp_model.regularizer_decay_rate = params['regularizer_decay_rate']

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    step = tf.compat.v1.train.get_or_create_global_step()
    checkpoint_directory = os.path.join(params['save_dir'], "saves/{}/training_checkpoints".format(vae_name))
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vae_mdp_model, step=step)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_directory, max_to_keep=1)

    if params['action_discretizer']:
        policy = tf.compat.v2.saved_model.load(params['policy_path'])

        try:
            import importlib
            environment_suite = importlib.import_module('tf_agents.environments.' + params['env_suite'])
        except BaseException as err:
            serr = str(err)
            print("Error to load the module '" + params['env_suite'] + "': " + serr)

        environment_name = params['environment']
        variational_mdp.train_from_policy(vae_mdp_model, policy=policy, environment_suite=environment_suite,
                                          env_name=environment_name,
                                          labeling_function=reinforcement_learning.labeling_functions[environment_name],
                                          batch_size=batch_size, optimizer=optimizer, checkpoint=checkpoint,
                                          manager=manager, log_name=vae_name,
                                          start_annealing_step=params['start_annealing_step'],
                                          logs=True, annealing_period=1, max_steps=params['max_steps'],
                                          display_progressbar=params['display_progressbar'],
                                          save_directory=params['save_dir'],
                                          parallelization=params['parallel_env'] > 1,
                                          num_parallel_environments=params['parallel_env'],
                                          eval_steps=int(1e3) if not params['do_not_eval'] else 0)
    else:
        variational_mdp.train_from_dataset(vae_mdp_model, dataset_generator=generate_dataset,
                                           batch_size=batch_size, optimizer=optimizer, checkpoint=checkpoint,
                                           manager=manager, dataset_size=dataset_size, annealing_period=1,
                                           start_annealing_step=params['start_annealing_step'],
                                           log_name=vae_name, logs=True, max_steps=params['max_steps'],
                                           display_progressbar=params['display_progressbar'],
                                           save_directory=params['save_dir'],
                                           eval_ratio=int(1e3) if not params['do_not_eval'] else 0)

    return 0


if __name__ == '__main__':
    app.run(main)
