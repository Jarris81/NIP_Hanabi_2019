from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.agents.ppo import ppo_policy
from tf_agents.specs import distribution_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

tfd = tfp.distributions


class PPOPolicyCustom(ppo_policy.PPOPolicy):
  """An ActorPolicy that also returns policy_info needed for PPO training."""

  def apply_value_network(self, observations, step_types, policy_state):
    """Apply value network to time_step, potentially a sequence.
    If observation_normalizer is not None, applies observation normalization.
    Args:
      observations: A (possibly nested) observation tensor with outer_dims
        either (batch_size,) or (batch_size, time_index). If observations is a
        time series and network is RNN, will run RNN steps over time series.
      step_types: A (possibly nested) step_types tensor with same outer_dims as
        observations.
      policy_state: Initial policy state for value_network.
    Returns:
      The output of value_net, which is a tuple of:
        - value_preds with same outer_dims as time_step
        - policy_state at the end of the time series
    """
    # extract state from observations
    print('STEP_TYPES:', step_types)
    print('OBS:', observations)
    state = observations['state']
    if self._observation_normalizer:
      state = self._observation_normalizer.normalize(state)
    return self._value_network(state, step_types, policy_state)

  def _apply_actor_network(self, time_step, policy_state):
    # extract state from observations
    print('APPLY_ACTOR_NETWORK')
    print("TIME_STEP:", time_step)
    if self._observation_normalizer:
      state = time_step.observation['state']
      legal_moves_mask = time_step.observation['mask']
      state = self._observation_normalizer.normalize(
          state)
      obs = {'state': state, 'mask': legal_moves_mask}
      time_step = ts.TimeStep(time_step.step_type, time_step.reward,
                              time_step.discount, obs)

    return self._actor_network(
        time_step.observation, time_step.step_type, network_state=policy_state)