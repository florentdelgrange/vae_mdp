import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from absl import flags
from absl import app

from util.io import dataset_generator
import variational_mdp

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
        params['encoder_temperature_decay_rate'], params['prior_temperature_decay_rate'])
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
        Input(shape=(np.prod(state_shape) * 2 + np.prod(action_shape) + np.prod(reward_shape),), name='encoder_input')
    q = Dense(256, activation=activation, name="encoder_0")(encoder_input)
    q = Dense(256, activation=activation, name="encoder_1")(q)
    q = Model(inputs=encoder_input, outputs=q, name="encoder_network_body")

    # Transition network body
    transition_input = Input(shape=(latent_state_size + action_shape[-1],), name='transition_input')
    p_t = Dense(256, activation=activation, name='transition_0')(transition_input)
    p_t = Dense(256, activation=activation, name='transition_1')(p_t)
    p_t = Model(inputs=transition_input, outputs=p_t, name="transition_network_body")

    # Reward network body
    p_r_input = Input(shape=(latent_state_size * 2 + action_shape[-1],), name="reward_input")
    p_r = Dense(256, activation=activation, name='reward_0')(p_r_input)
    p_r = Dense(256, activation=activation, name='reward_1')(p_r)
    p_r = Model(inputs=p_r_input, outputs=p_r, name="reward_network_body")

    # Decoder network body
    p_decoder_input = Input(shape=(latent_state_size,), name='decoder_input')
    p_decode = Dense(256, activation=activation, name='decoder_0')(p_decoder_input)
    p_decode = Dense(256, activation=activation, name='decoder_1')(p_decode)
    p_decode = Model(inputs=p_decoder_input, outputs=p_decode, name="decoder_body")

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
