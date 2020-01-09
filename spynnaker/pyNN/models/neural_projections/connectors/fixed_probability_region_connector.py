# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math
import numpy
from spinn_utilities.overrides import overrides
from data_specification.enums.data_type import DataType
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spynnaker.pyNN.utilities import utility_calls
from .abstract_connector import AbstractConnector
from .abstract_generate_connector_on_machine import (
    AbstractGenerateConnectorOnMachine, ConnectorIDs)
from spinn_front_end_common.utilities.constants import BYTES_PER_WORD

CIRCLE, SQUARE = range(2)

class FixedProbabilityRegionConnector(AbstractGenerateConnectorOnMachine):
    """ For each pair of pre-post cells, the connection probability is constant.
    """

    __slots__ = [
        "__allow_self_connections",
        "_p_connect",
        "_pre_shape", 
        "_post_shape", 
        "_max_distance",
    ]

    def __init__(
            self, pre_shape, post_shape, max_distance, p_connect, 
            circle_or_square=CIRCLE, allow_self_connections=True, safe=True,
            callback=None, verbose=False, rng=None):
        """
        :param pre_shape: shape (rows, columns, cells per region) for the \
        pre-synaptic population.
        :type pre_shape: iterable
        
        :param post_shape: shape (rows, columns, cells per region) for the \
        post-synaptic population.
        :type post_shape: float
        
        :param max_distance: maximum distance between cell pairs for them \
        to be able to establish a connection
        :type max_distance: float

        :param p_connect: a number between zero and one. Each potential\
            connection is created with this probability.
        :type p_connect: float

        :param circle_or_square: an identifier to know what the maximum distance \
        boundary is. Square will take the distance along the axis while Circle does \
        the straight line between cells.
        :type circle_or_square: int

        :param allow_self_connections:
            if the connector is used to connect a Population to itself, this\
            flag determines whether a neuron is allowed to connect to itself,\
            or only to other neurons in the Population.
        :type allow_self_connections: bool
        :param `pyNN.Space` space:
            a Space object, needed if you wish to specify distance-dependent\
            weights or delays - not implemented
        """
        super(FixedProbabilityRegionConnector, self).__init__(
            safe, callback, verbose)
        self._p_connect = p_connect
        self._pre_shape = pre_shape
        self._post_shape = post_shape
        self._max_distance = max_distance
        self._max_dist2 = self._max_distance ** 2

        self._circle_or_square = circle_or_square
        
        self._max_pre = self._compute_max_pre_per_post()
        
        self.__allow_self_connections = allow_self_connections
        self._rng = rng
        if not 0 <= self._p_connect <= 1:
            raise ConfigurationException(
                "The probability must be between 0 and 1 (inclusive)")

    def _within_region(self, row_pre, col_pre, row_post, col_post):
        if self._circle_or_square == CIRCLE:
            d2 = self._max_dist2
            dn2 = (row_post - row_pre) ** 2 + (col_post - col_pre) ** 2
            return dn2 <= dn2
        else:
            within_rows = np.abs(row_post - row_pre) <= self._max_distance
            within_cols = np.abs(col_post - col_pre) <= self._max_distance
            return (within_rows and within_cols)
    
    def _neurons_in_region(self, scale=1.0):
        dist = scale * self._max_distance
        
        if self._circle_or_square == CIRCLE:
            #from http://mathworld.wolfram.com/GausssCircleProblem.html
            max_error = 2 * numpy.sqrt(2) * numpy.pi * dist
            n = numpy.pi * (dist ** 2)
        else:
            n = (2 * dist) ** 2
    
        return n
    
    def _compute_max_pre_per_post(self):
        scaling = self._pre_shape[0] / self._post_shape[0]
        scaling = 1.0 if scaling >= 1.0 else scaling
        n_pre = self._neurons_in_region(scaling)
        
        return int(np.ceil(n_pre * self._pre_shape[2]))

    
    def _compute_max_post_per_pre(self):
        scaling = self._post_shape[0] / self._pre_shape[0]
        scaling = 1.0 if scaling >= 1.0 else scaling
        n_post = self._neurons_in_region(scaling)
        
        return int(np.ceil(n_post * self._post_shape[2]))
        
        
    @overrides(AbstractConnector.get_delay_maximum)
    def get_delay_maximum(self, synapse_info):
        scaling = min(min(self._pre_shape[0] / self._post_shape[0],
                          self._post_shape[0] / self._pre_shape[0])
                      min(self._pre_shape[1] / self._post_shape[1],
                          self._post_shape[1] / self._pre_shape[1]))

        n_in_region = self._neurons_in_region()
        total = n_in_region * self._pre_shape[2] * self._post_shape[2] * scaling**2
        
        n_connections = utility_calls.get_probable_maximum_selected(
                            total, total, self._p_connect)

        return self._get_delay_maximum(synapse_info.delays, n_connections)

    @overrides(AbstractConnector.get_n_connections_from_pre_vertex_maximum)
    def get_n_connections_from_pre_vertex_maximum(
            self, post_vertex_slice, synapse_info, min_delay=None,
            max_delay=None):
        # pylint: disable=too-many-arguments
        n_connections = utility_calls.get_probable_maximum_selected(
            synapse_info.n_pre_neurons * synapse_info.n_post_neurons,
            post_vertex_slice.n_atoms, 
            self._p_connect, chance=1.0/10000.0)

        if min_delay is None or max_delay is None:
            return int(math.ceil(n_connections))

        return self._get_n_connections_from_pre_vertex_with_delay_maximum(
            synapse_info.delays,
            synapse_info.n_pre_neurons * synapse_info.n_post_neurons,
            n_connections, min_delay, max_delay)

    @overrides(AbstractConnector.get_n_connections_to_post_vertex_maximum)
    def get_n_connections_to_post_vertex_maximum(self, synapse_info):
        # pylint: disable=too-many-arguments
        n_connections = utility_calls.get_probable_maximum_selected(
            synapse_info.n_pre_neurons * synapse_info.n_post_neurons,
            synapse_info.n_pre_neurons, self._p_connect,
            chance=1.0/10000.0)
        return n_connections

    @overrides(AbstractConnector.get_weight_maximum)
    def get_weight_maximum(self, synapse_info):
        # pylint: disable=too-many-arguments
        n_connections = utility_calls.get_probable_maximum_selected(
            synapse_info.n_pre_neurons * synapse_info.n_post_neurons,
            synapse_info.n_pre_neurons * synapse_info.n_post_neurons,
            self._p_connect)
        return self._get_weight_maximum(synapse_info.weights, n_connections)

    @overrides(AbstractConnector.create_synaptic_block)
    def create_synaptic_block(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            synapse_type, synapse_info):
        # pylint: disable=too-many-arguments
        n_items = pre_vertex_slice.n_atoms * post_vertex_slice.n_atoms
        items = self._rng.next(n_items)

        # If self connections are not allowed, remove possibility the self
        # connections by setting them to a value of infinity
        if not self.__allow_self_connections:
            items[0:n_items:post_vertex_slice.n_atoms + 1] = numpy.inf

        present = items <= self._p_connect
        ids = numpy.where(present)[0]
        n_connections = numpy.sum(present)

        block = numpy.zeros(n_connections, dtype=self.NUMPY_SYNAPSES_DTYPE)
        block["source"] = (
            (ids // post_vertex_slice.n_atoms) + pre_vertex_slice.lo_atom)
        block["target"] = (
            (ids % post_vertex_slice.n_atoms) + post_vertex_slice.lo_atom)
        block["weight"] = self._generate_weights(
            n_connections, None, pre_vertex_slice, post_vertex_slice,
            synapse_info)
        block["delay"] = self._generate_delays(
            n_connections, None, pre_vertex_slice, post_vertex_slice,
            synapse_info)
        block["synapse_type"] = synapse_type
        return block

    def __repr__(self):
        return "FixedProbabilityConnector({})".format(self._p_connect)

    def _get_view_lo_hi(self, indexes):
        view_lo = indexes[0]
        view_hi = indexes[-1]
        return view_lo, view_hi

    @property
    @overrides(AbstractGenerateConnectorOnMachine.gen_connector_id)
    def gen_connector_id(self):
        return ConnectorIDs.FIXED_PROBABILITY_CONNECTOR.value

    @overrides(AbstractGenerateConnectorOnMachine.
               gen_connector_params)
    def gen_connector_params(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            synapse_type, synapse_info):
        params = []
        pre_view_lo = 0
        pre_view_hi = synapse_info.n_pre_neurons - 1
        if synapse_info.prepop_is_view:
            pre_view_lo, pre_view_hi = self._get_view_lo_hi(
                synapse_info.pre_population._indexes)

        params.extend([pre_view_lo, pre_view_hi])

        post_view_lo = 0
        post_view_hi = synapse_info.n_post_neurons - 1
        if synapse_info.postpop_is_view:
            post_view_lo, post_view_hi = self._get_view_lo_hi(
                synapse_info.post_population._indexes)

        params.extend([post_view_lo, post_view_hi])

        params.extend([self.__allow_self_connections])

        # If prob=1.0 has been specified, take care when scaling value to
        # ensure that it doesn't wrap round to zero as an unsigned long fract
        params.extend([DataType.U032.encode_as_int(
            DataType.U032.max if self._p_connect == 1.0 else self._p_connect)])

        params.extend(self._get_connector_seed(
            pre_vertex_slice, post_vertex_slice, self._rng))
        return numpy.array(params, dtype="uint32")

    @property
    @overrides(AbstractGenerateConnectorOnMachine.
               gen_connector_params_size_in_bytes)
    def gen_connector_params_size_in_bytes(self):
        # view + params + seeds
        return (4 + 2 + 4) * BYTES_PER_WORD
