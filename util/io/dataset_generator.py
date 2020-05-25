import h5py
import os
import glob
import datetime
import random
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
import tf_agents.trajectories.time_step as ts
import time


def gather_rl_observations(
        iterator,
        labeling_function,
        dataset_path='dataset/reinforcement_learning',
        dataset_name='rl_exploration',
        scalar_rewards=True):
    """
    Writes the observations gathered through the training of an RL policy into an hdf5 dataset.
    Important: the next() call of the iterator function must yields a bash containing 3-steps of tf_agents Trajectories.
    The labeling function is defined over Trajectories observations.
    """
    data = iterator.next()[0]  # a tf_agents dataset typically returns a tuple (trajectories, information)
    states = data.observation[:, :2, :].numpy()
    actions = data.action[:, :2, :].numpy()
    rewards = data.reward[:, :2].numpy() if scalar_rewards else data.reward[:, :2, :].numpy()
    if scalar_rewards:
        rewards.reshape(list(rewards.shape) + [1])
    next_states = data.observation[:, 1:, :].numpy()
    next_labels = labeling_function(next_states)
    if next_labels.shape == states.shape[:-1]:
        next_labels.reshape(list(next_labels.shape) + [1])
    state_type = data.step_type[:, :2].numpy()
    next_state_type = data.next_step_type[:, :2].numpy()

    # remove transitions where the incident state is terminal and next state is initial
    # note: such transitions correspond to those where the reset() function has been called
    filtering = state_type[:, 0] != ts.StepType.LAST
    filtering &= state_type[:, 1] != ts.StepType.LAST
    filtering &= next_state_type[:, 0] != ts.StepType.FIRST
    filtering &= next_state_type[:, 1] != ts.StepType.FIRST

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with h5py.File(os.path.join(dataset_path, dataset_name + current_time + '.hdf5'), 'w') as h5f:
        h5f['state'] = states[filtering]
        h5f['action'] = actions[filtering]
        h5f['reward'] = rewards[filtering]
        h5f['next_state'] = next_states[filtering]
        h5f['next_state_label'] = next_labels[filtering]
        h5f['state_type'] = state_type[filtering]
        h5f['next_state_type'] = state_type[filtering]


class DatasetGenerator:

    def __init__(self, initial_dummy_state=None, initial_dummy_action=None):
        self.initial_dummy_state = initial_dummy_state
        self.initial_dummy_action = initial_dummy_action

    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            for (state, action, reward, next_state, label, state_type, next_state_type) in \
                    zip(hf['state'], hf['action'], hf['reward'], hf['next_state'],
                        hf['next_state_label'], hf['state_type'], hf['next_state_type']):

                if state.shape[:-1] == reward.shape:  # singleton shape
                    reward = reward.reshape(list(reward.shape) + [1])
                if state.shape[:-1] == label.shape:
                    label = label.reshape(list(label.shape) + [1])

                if state_type[0] == ts.StepType.FIRST:  # initial state handling
                    initial_state = self.initial_dummy_state if self.initial_dummy_state is not None \
                        else np.zeros(shape=state.shape[1:])
                    initial_action = self.initial_dummy_action if self.initial_dummy_action is not None \
                        else np.zeros(shape=action.shape[1:])
                    yield np.stack((initial_state, state[0])), \
                          np.stack((initial_action, action[0])), \
                          np.stack((np.zeros(shape=reward.shape[1:]), reward[0])), \
                          state, \
                          np.stack((label[0], label[0]))

                yield state, action, reward, next_state, label


def get_tensor_shape(h5file):
    with h5py.File(h5file, 'r') as hf:
        reward_shape = list(hf['reward'].shape[1:]) + \
                       ([1] if (tf.TensorShape(hf['state'].shape[:-1]) == hf['reward'].shape) else [])
        label_shape = list(hf['next_state_label'].shape[1:]) + \
                      ([1] if (tf.TensorShape(hf['state'].shape[:-1]) == hf['next_state_label'].shape) else [])
        return (tf.TensorShape(hf['state'].shape[1:]),
                tf.TensorShape(hf['action'].shape[1:]),
                tf.TensorShape(reward_shape),
                tf.TensorShape(hf['next_state'].shape[1:]),
                tf.TensorShape(label_shape))
        #  tf.TensorShape(hf['state_type'].shape[1:]),
        #  tf.TensorShape(hf['next_state_type'].shape[1:]))


def create_dataset(cycle_length=4,
                   block_length=32,
                   num_parallel_calls=tf.data.experimental.AUTOTUNE,
                   hdf5_files_path='dataset/reinforcement_learning',
                   regex='*.hdf5'):
    file_list: List[str] = glob.glob(os.path.join(hdf5_files_path, regex), recursive=True)
    random.shuffle(file_list)

    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    dataset = dataset.interleave(
        lambda filename: tf.data.Dataset.from_generator(
            DatasetGenerator(),
            (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),  # , tf.int8, tf.int8),
            get_tensor_shape(file_list[0]),  # all files are assumed to have same Tensor Shape
            args=(filename,)
        ),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=num_parallel_calls
    )
    return dataset


def get_num_samples(hdf5_files_path: str = 'dataset/reinforcement_learning', regex: str = '*.hdf5',
                    drop_remainder: bool = True, batch_size: int = 32):
    file_list: List[str] = glob.glob(os.path.join(hdf5_files_path, regex), recursive=True)
    size = 0
    for h5f_name in file_list:
        with h5py.File(h5f_name, 'r') as h5f:
            # we assume that all h5f datasets have the same length (= size of axis 0)
            h5f_size = h5f['state'].shape[0]
            size += (h5f_size // batch_size) * batch_size if drop_remainder else h5f_size
    return size


def merge_rl_observations_dataset(
        hdf5_files_path='dataset/reinforcement_learning',
        dataset_name='rl_exploration.hdf5',
        regex='*.hdf5',
        shuffle: bool = False):
    file_list: List[str] = glob.glob(os.path.join(hdf5_files_path, regex), recursive=True)

    print("File list:")
    print(file_list)

    length: int = 0
    h5f_indices: Dict[str, Tuple[int, int]] = {}
    shape: Dict[str, Tuple[int, ...]] = {}
    start: float = time.time()

    for h5f_name in file_list:
        with h5py.File(h5f_name, 'r') as h5f:
            # we assume that all h5f datasets have the same length (= size of axis 0)
            h5f_length = h5f['state'].shape[0]
            h5f_indices[h5f_name] = (length, length + h5f_length)
            length += h5f_length

    random_indices = np.arange(length) if shuffle else np.empty(0)
    np.random.shuffle(random_indices)

    with h5py.File(os.path.join(hdf5_files_path, dataset_name), 'w') as merged_h5f:
        for i, h5f_name in enumerate(file_list):
            with h5py.File(h5f_name, 'r') as h5f:
                first, last = h5f_indices[h5f_name]
                indices = np.sort(random_indices[first: last]) if shuffle else np.empty(0)
                for key in h5f:
                    if i == 0:  # dataset file initialization
                        shape[key] = (length,) + h5f[key].shape[1:]
                        merged_h5f.create_dataset(key, shape[key], dtype=h5f[key].dtype)
                    if shuffle:
                        merged_h5f[key][indices] = h5f[key]
                    else:
                        merged_h5f[key][first: last] = h5f[key]

        print("Dataset files merged into {}. Time: {:.3f} sec, size: {:.3f} GB).".format(
            dataset_name, time.time() - start,
                          os.path.getsize(os.path.join(hdf5_files_path, dataset_name)) / 2.0 ** 30))