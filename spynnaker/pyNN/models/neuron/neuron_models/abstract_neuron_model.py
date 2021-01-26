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

import numpy
from six import with_metaclass
from spinn_utilities.abstract_base import AbstractBase
from spinn_utilities.overrides import overrides
from spynnaker.pyNN.models.neuron.implementations import (
    AbstractStandardNeuronComponent)
from spinn_front_end_common.utilities.constants import BYTES_PER_WORD
from spynnaker.pyNN.utilities.struct import Struct


# with_metaclass due to https://github.com/benjaminp/six/issues/219
class AbstractNeuronModel(
        with_metaclass(AbstractBase, AbstractStandardNeuronComponent)):
    """ Represents a neuron model.
    """

    __slots__ = [
        "__global_struct", "requires_spike_mapping", "needs_dma_weights"
    ]

    def __init__(self, data_types, global_data_types=None,
                 requires_spike_mapping=False, needs_dma_weights=True):
        """
        :param list(~data_specification.enums.DataType) data_types:
            A list of data types in the neuron structure, in the order that
            they appear
        :param global_data_types:
            A list of data types in the neuron global structure, in the order
            that they appear
        :type global_data_types:
            list(~data_specification.enums.DataType) or None
        """
        super(AbstractNeuronModel, self).__init__(data_types)
        if global_data_types is None:
            global_data_types = []
        self.__global_struct = Struct(global_data_types)

        self.requires_spike_mapping = requires_spike_mapping
        self.needs_dma_weights = needs_dma_weights

    @property
    def global_struct(self):
        """ Get the global parameters structure

        :rtype: ~spynnaker.pyNN.utilities.struct.Struct
        """
        return self.__global_struct

    @property
    def local_only_compatible(self):
        return self.requires_spike_mapping and not self.needs_dma_weights

    @overrides(AbstractStandardNeuronComponent.get_dtcm_usage_in_bytes)
    def get_dtcm_usage_in_bytes(self, n_neurons):
        n = (1 if self.local_only_compatible else n_neurons)
        usage = super(AbstractNeuronModel, self).get_dtcm_usage_in_bytes(n)
        return usage + (self.__global_struct.get_size_in_whole_words() *
                        BYTES_PER_WORD)

    @overrides(AbstractStandardNeuronComponent.get_sdram_usage_in_bytes)
    def get_sdram_usage_in_bytes(self, n_neurons):
        n = (1 if self.local_only_compatible else n_neurons)
        usage = super(AbstractNeuronModel, self).get_sdram_usage_in_bytes(n)
        return usage + (self.__global_struct.get_size_in_whole_words() *
                        BYTES_PER_WORD)

    def get_global_values(self, ts):  # pylint: disable=unused-argument
        """ Get the global values to be written to the machine for this model

        :param float ts: The time to advance the model at each call
        :return: A list with the same length as self.global_struct.field_types
        :rtype: list(int or float) or ~numpy.ndarray
        """
        return numpy.zeros(0, dtype="uint32")

    @overrides(AbstractStandardNeuronComponent.get_data)
    def get_data(self, parameters, state_variables, vertex_slice, ts,
                 local_only_compatible=False):
        super_data = super(AbstractNeuronModel, self).get_data(
            parameters, state_variables, vertex_slice, ts,
            local_only_compatible)
        values = self.get_global_values(ts)
        global_data = self.__global_struct.get_data(values)
        return numpy.concatenate([global_data, super_data])

    @overrides(AbstractStandardNeuronComponent.read_data)
    def read_data(
            self, data, offset, vertex_slice, parameters, state_variables):

        # Assume that the global data doesn't change
        offset += (self.__global_struct.get_size_in_whole_words() *
                   BYTES_PER_WORD)
        return super(AbstractNeuronModel, self).read_data(
            data, offset, vertex_slice, parameters, state_variables)
