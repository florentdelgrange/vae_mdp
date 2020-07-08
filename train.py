import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

import variational_action_discretizer
import variational_mdp
from util.io import dataset_generator

flags.DEFINE_string(
    "dataset_path",
    help="Path of the directory containing the dataset files in hdf5 format.",
    default="dataset/reinforcement_learning")
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
    help="Step from which temperatures and scale factors are annealed."
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
FLAGS = flags.FLAGS


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()

    dataset_path = params['dataset_path']
    # dataset_path = '/home/florent/Documents/hpc-cluster/dataset/reinforcement_learning'
    # dataset_path = 'reinforcement_learning/dataset/reinforcement_learning'

    batch_size = params['batch_size']
    mixture_components = params['mixture_components']
    latent_state_size = params['latent_size']  # depends on the number of bits reserved for labels

    vae_name = 'vae_LS{}_MC{}_CER{}_KLA{}_TD{:.2f}-{:.2f}_{}-{}'.format(
        latent_state_size, mixture_components, params['regularizer_scale_factor'], params['kl_annealing_scale_factor'],
        params['encoder_temperature'], params['prior_temperature'],
        params['encoder_temperature_decay_rate'],
        params['prior_temperature_decay_rate']) \
        if not params['action_dicretizer'] else 'vae_LS{}_LA{}_MC{}_CER{}_KLA{}_TD{:.2f}-{:.2f}_{}-{}'.format(
        latent_state_size, params['number_of_discrete_actions'],
        mixture_components, params['regularizer_scale_factor'], params['kl_annealing_scale_factor'],
        params['encoder_temperature'], params['prior_temperature'],
        params['encoder_temperature_decay_rate'],
        params['prior_temperature_decay_rate'])

    cycle_length = batch_size // 2
    block_length = batch_size // cycle_length
    activation = getattr(tf.nn, params["activation"])

    def generate_dataset():
        return dataset_generator.create_dataset(hdf5_files_path=dataset_path,
                                                cycle_length=cycle_length,
                                                block_length=block_length)

    dummy_dataset = generate_dataset()
    print('Compute dataset size...')
    dataset_size = dataset_generator.get_num_samples(dataset_path, batch_size=batch_size, drop_remainder=True)
    print('{} samples.'.format(dataset_size))

    state_shape, action_shape, reward_shape, _, label_shape = \
        [tuple(spec.shape.as_list()[1:]) for spec in dummy_dataset.element_spec]

    del dummy_dataset

    # Encoder body
    encoder_input = \
        Input(shape=(np.prod(state_shape) * 2 + np.prod(action_shape) + np.prod(reward_shape)
                     if not params['action_discretizer'] else np.prod(latent_state_size) + np.prod(action_shape),),
              name='encoder_input')
    q = encoder_input
    for i, units in enumerate(params['encoder_layers']):
        q = Dense(units, activation=activation, name="encoder_{}".format(i))(q)
    q = Model(inputs=encoder_input, outputs=q, name="encoder_network_body")

    # Transition network body
    transition_input = Input(shape=(latent_state_size + np.prod(action_shape)
                                    if not params['action_discretizer'] else latent_state_size,),
                             name='transition_input')
    p_t = transition_input
    for i, units in enumerate(params['transition_layers']):
        p_t = Dense(units, activation=activation, name='transition_{}'.format(i))(p_t)
    p_t = Model(inputs=transition_input, outputs=p_t, name="transition_network_body")

    # Reward network body
    p_r_input = Input(shape=(latent_state_size * 2 + action_shape[-1]
                             if not params['action_discretizer'] else latent_state_size * 2,),
                      name="reward_input")
    p_r = p_r_input
    for i, units in enumerate(params['reward_layers']):
        p_r = Dense(units, activation=activation, name='reward_{}'.format(i))(p_r)
    p_r = Model(inputs=p_r_input, outputs=p_r, name="reward_network_body")

    # Decoder network body
    p_decoder_input = Input(shape=(latent_state_size
                                   if not params['action_discretizer'] else latent_state_size + np.prod(action_shape),),
                            name='decoder_input')
    p_decode = p_decoder_input
    for i, units in enumerate(params['decoder_layers']):
        p_decode = Dense(units, activation=activation, name='decoder_{}'.format(i))(p_decode)
    p_decode = Model(inputs=p_decoder_input, outputs=p_decode, name="decoder_body")

    if params['load_vae'] != '':
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
        # regularizer_scale_factor = 100., regularizer_decay_rate = 1.5e-4, )
        # kl_annealing_growth_rate=2e-5, kl_annealing_scale_factor=2e-5)
    else:
        vae_mdp_model = variational_mdp.load(params['load_vae'])

    if params['action_discretizer']:
        if params['load_vae'] == '':
            raise RuntimeError('Missing argument: --load_vae')
        vae_mdp_model = variational_action_discretizer.VariationalActionDiscretizer(
            vae_mdp=vae_mdp_model,
            number_of_discrete_actions=params['number_of_discrete_actions'],
            action_encoder_network=q, transition_network=p_t, reward_network=p_r, action_decoder_network=p_decode,
            encoder_temperature=params['encoder_temperature'], prior_temperature=params['prior_temperature'],
            encoder_temperature_decay_rate=params['encoder_temperature_decay_rate'],
            prior_temperature_decay_rate=params['prior_temperature_decay_rate'],
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    step = tf.compat.v1.train.get_or_create_global_step()
    checkpoint_directory = os.path.join(params['save_dir'], "saves/{}/training_checkpoints".format(vae_name))
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vae_mdp_model, step=step)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_directory, max_to_keep=1)

    variational_mdp.train(vae_mdp_model, dataset_generator=generate_dataset,
                          batch_size=batch_size, optimizer=optimizer, checkpoint=checkpoint, manager=manager,
                          dataset_size=dataset_size, annealing_period=1,
                          start_annealing_step=params['start_annealing_step'],
                          log_name=vae_name, logs=True, max_steps=params['max_steps'],
                          display_progressbar=params['display_progressbar'],
                          save_directory=params['save_dir'])


if __name__ == '__main__':
    app.run(main)
