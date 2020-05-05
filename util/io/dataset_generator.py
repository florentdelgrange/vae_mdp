import datetime

import h5py
import os
import tf_agents.trajectories.time_step as ts


def gather_rl_observations(iterator, labeling_function, dataset_path='dataset', dataset_name='rl_observations'):
    """
    Writes the observations gathered through the training of an RL policy into an hdf5 dataset.
    Important: the next() call of the iterator function must yields a bash containing 3-steps of tf_agents Trajectories.
    The labeling function is defined over Trajectories observations.
    """
    data = iterator.next()[0]  # a tf_agents dataset typically returns a tuple (trajectories, information)
    states = data.observation[:, :2, :].numpy()
    actions = data.action[:, :2, :].numpy()
    rewards = data.reward[:, :2].numpy()  # assuming rewards are scalar values
    next_states = data.observation[:, 1:, :].numpy()
    next_labels = labeling_function(next_states)
    # 0: initial state; 1: mid state; 2: terminal state
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
    with h5py.File(os.path.join(dataset_path, dataset_name + current_time), 'w') as h5f:
        h5f['state'] = states[filtering]
        h5f['action'] = actions[filtering]
        h5f['reward'] = rewards[filtering]
        h5f['next_state'] = next_states[filtering]
        h5f['next_state_label'] = next_labels[filtering]
        h5f['state_type'] = state_type[filtering]
        h5f['next_state_type'] = state_type[filtering]
