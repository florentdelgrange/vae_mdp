import os

import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents.policies
import tf_agents.trajectories.time_step as ts
import tf_agents.specs.tensor_spec
from tf_agents.policies import policy_saver
from tf_agents.trajectories.policy_step import PolicyStep

tfd = tfp.distributions

COLLECT_SPEC = 'collect_data_spec.pbtxt'


class SavedTFPolicy(tf_agents.policies.tf_policy.TFPolicy):

    def __init__(self, saved_policy_path, time_step_spec=None, action_spec=None):
        self.saved_policy = tf.compat.v2.saved_model.load(saved_policy_path)
        spec_path = [os.path.join(saved_policy_path, policy_saver.POLICY_SPECS_PBTXT),
                     os.path.join(saved_policy_path, COLLECT_SPEC)]
        policy_specs = None
        for path in spec_path:
            if os.path.exists(path):
                policy_specs = tf_agents.specs.tensor_spec.from_pbtxt_file(path)
                break

        if time_step_spec is None and policy_specs is not None:
            time_step_spec = ts.time_step_spec(policy_specs['collect_data_spec'].observation)
        if action_spec is None and policy_specs is not None:
            action_spec = policy_specs['collect_data_spec'].action

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            info_spec=policy_specs['collect_data_spec'].policy_info,
            policy_state_spec=policy_specs['policy_state_spec'])

    def _distribution(self, time_step, policy_state):
        step = self.saved_policy.action(time_step, policy_state)
        return PolicyStep(tfd.Deterministic(step.action), step.state, step.info)

    def _get_initial_state(self, batch_size):
        return self.saved_policy.get_initial_state(batch_size)

    def _variables(self):
        pass
