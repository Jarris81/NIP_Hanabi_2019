# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project inputs to a categorical distribution object."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks.categorical_projection_network import CategoricalProjectionNetwork 
from tf_agents.networks import utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec

#import gin.tf


class CategoricalProjectionNetworkCustom(network.DistributionNetwork):
  """Generates a tfp.distribution.Categorical by predicting logits."""

  def __init__(self,
               sample_spec,
               input_tensor_spec,
               logits_init_output_factor=0.1,
               name='CategoricalProjectionNetwork'):
    """Creates an instance of CategoricalProjectionNetwork.
    Args:
      sample_spec: An spec (either BoundedArraySpec or BoundedTensorSpec)
        detailing the shape and dtypes of samples pulled from the output
        distribution.
      logits_init_output_factor: Output factor for initializing kernel logits
        weights.
      name: A string representing name of the network.
    """
    unique_num_actions = np.unique(sample_spec.maximum - sample_spec.minimum +
                                   1)
    if len(unique_num_actions) > 1 or np.any(unique_num_actions <= 0):
      raise ValueError('Bounds on discrete actions must be the same for all '
                       'dimensions and have at least 1 action.')

    output_shape = sample_spec.shape.concatenate([unique_num_actions])
    output_spec = self._output_distribution_spec(output_shape, sample_spec)

    super(CategoricalProjectionNetworkCustom, self).__init__(
        # We don't need these, but base class requires them.
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    if not tensor_spec.is_bounded(sample_spec):
      raise ValueError(
          'sample_spec must be bounded. Got: %s.' % type(sample_spec))

    if not tensor_spec.is_discrete(sample_spec):
      raise ValueError('sample_spec must be discrete. Got: %s.' % sample_spec)

    if len(unique_num_actions) > 1:
      raise ValueError(
          'Projection Network requires num_actions to be equal '
          'across action dimentions. Implement a more general categorical '
          'projection if you need more flexibility.')

    self._sample_spec = sample_spec
    self._output_shape = output_shape

    self._projection_layer = tf.keras.layers.Dense(
        self._output_shape.num_elements(),
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=logits_init_output_factor),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='logits')

  def _output_distribution_spec(self, output_shape, sample_spec):
    input_param_spec = {
        'logits': tensor_spec.TensorSpec(shape=output_shape, dtype=tf.float32)
    }

    return distribution_spec.DistributionSpec(
        tfp.distributions.Categorical,
        input_param_spec,
        sample_spec=sample_spec,
        dtype=sample_spec.dtype)

  def call(self, inputs, outer_rank):
    print("CATEGORICAL_NET_CALL", inputs)
    # extract state and mask of legal moves from inputs
    state = inputs['state']
    legal_moves_mask = inputs['mask']
    # currently, legal moves have value 1 and illegal ones value 0, but we need legal moves to have
    # value 0 and illegal ones to have value -infinity
    #legal_moves_mask = np.where(legal_moves_mask == 0, np.NINF, 0)
    #np.reshape(legal_moves_mask, ()

    ### FROM SUPER CLASS ###
    # outer_rank is needed because the projection is not done on the raw
    # observations so getting the outer rank is hard as there is no spec to
    # compare to.
    batch_squash = utils.BatchSquash(outer_rank)
    state = batch_squash.flatten(state)
    state = tf.cast(state, tf.float32)

    logits = self._projection_layer(state)
    logits = tf.reshape(logits, [-1] + self._output_shape.as_list())
    logits = batch_squash.unflatten(logits)

    print('LOGITS SHAPE:', logits.shape)
    print('legal_moves_mask SHAPE:', inputs['mask'])
    # apply mask to remove illegal actions
    logits = tf.add(logits, legal_moves_mask)

    return self.output_spec.build_distribution(logits=logits)
