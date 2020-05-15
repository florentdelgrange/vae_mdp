import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from util.io import dataset_generator
import variational_mdp

if __name__ == '__main__':
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    dataset_path = '/home/florent/Documents/hpc-cluster/dataset/reinforcement_learning'
    batch_size = 128
    num_of_gaussian_posteriors = 3
    latent_state_size = 16 + 1  # depends on the number of bits reserved for labels
    vae_name = 'DELETEvae_decay_temperature_latent{}'.format(latent_state_size)
    cycle_length = 8
    block_length = batch_size // cycle_length


    def dataset():
        return dataset_generator.create_dataset(hdf5_files_path=dataset_path,
                                                cycle_length=cycle_length,
                                                block_length=block_length)


    dummy_dataset = dataset()
    print('Computing dataset size...')
    dataset_size = dataset_generator.get_num_samples(dataset_path, batch_size=batch_size, drop_remainder=True)
    print('{} samples.'.format(dataset_size))

    state_shape, action_shape, reward_shape, _, label_shape = \
        [tuple(spec.shape.as_list()[1:]) for spec in dummy_dataset.element_spec]

    del dummy_dataset

    # Encoder body
    encoder_input = \
        Input(shape=(np.prod(state_shape) * 2 + np.prod(action_shape) + np.prod(reward_shape),), name='encoder_input')
    q = Dense(256, activation='relu', name="encoder_0")(encoder_input)
    q = Dense(256, activation='relu', name="encoder_1")(q)
    q = Model(inputs=encoder_input, outputs=q, name="encoder_network_body")

    # Transition network body
    transition_input = Input(shape=(latent_state_size + action_shape[-1],), name='transition_input')
    p_t = Dense(256, activation='relu', name='transition_0')(transition_input)
    p_t = Dense(256, activation='relu', name='transition_1')(p_t)
    p_t = Model(inputs=transition_input, outputs=p_t, name="transition_network_body")

    # Reward network body
    p_r_input = Input(shape=(latent_state_size * 2 + action_shape[-1],), name="reward_input")
    p_r = Dense(256, activation='relu', name='reward_0')(p_r_input)
    p_r = Dense(256, activation='relu', name='reward_1')(p_r)
    p_r = Model(inputs=p_r_input, outputs=p_r, name="reward_network_body")

    # Decoder network body
    p_decoder_input = Input(shape=(latent_state_size,), name='decoder_input')
    p_decode = Dense(256, activation='relu', name='decoder_0')(p_decoder_input)
    #  p_decode = tf.keras.layers.LeakyReLU(p_decode)
    p_decode = Dense(256, activation='relu', name='decoder_1')(p_decode)
    p_decode = Model(inputs=p_decoder_input, outputs=p_decode, name="decoder_body")

    vae_mdp_model = variational_mdp.VariationalMarkovDecisionProcess(
        state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape, label_shape=label_shape,
        encoder_network=q, transition_network=p_t, reward_network=p_r, decoder_network=p_decode,
        latent_state_size=latent_state_size, nb_gaussian_posteriors=num_of_gaussian_posteriors,
        temperature_1=(1-1e-6), temperature_2=(1-1e-6), temperature_1_decay_rate=1e-6, temperature_2_decay_rate=2e-6,
        debug=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    step = tf.compat.v1.train.get_or_create_global_step()
    checkpoint_directory = "saves/{}_{}gp/training_checkpoints".format(vae_name, num_of_gaussian_posteriors)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vae_mdp_model, step=step)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_directory, max_to_keep=1)

    variational_mdp.train(vae_mdp_model, dataset_generator=dataset,
                          batch_size=batch_size, optimizer=optimizer, checkpoint=None, manager=None,
                          dataset_size=dataset_size, decay_period=1, log_name=vae_name, logs=True)
