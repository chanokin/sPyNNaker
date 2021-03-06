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

from spinn_utilities.overrides import overrides
from data_specification.enums import DataType
from .abstract_synapse_type import AbstractSynapseType
from spynnaker.pyNN.models.neuron.implementations import (
    AbstractStandardNeuronComponent)
from spinn_front_end_common.utilities.constants import BYTES_PER_WORD

ISYN_EXC = "isyn_exc"

UNITS = {
    ISYN_EXC: "",
}


class SynapseTypeDeltaConv(AbstractSynapseType):
    """ This represents a synapse type with two delta synapses
    """
    __slots__ = [
        "__isyn_exc",
        ]

    def __init__(self, isyn_exc):
        """
        :param float isyn_exc: :math:`I^{syn}_e`

        """
        super(SynapseTypeDeltaConv, self).__init__([
            DataType.S1615,   # isyn_exc
            ])
        self.__isyn_exc = isyn_exc

        self.extend_state_variables = True
        self.needs_dma_weights = False
        self.requires_spike_mapping = True

    @overrides(AbstractSynapseType.get_n_cpu_cycles)
    def get_n_cpu_cycles(self, n_neurons):
        return 1 * n_neurons

    @overrides(AbstractSynapseType.add_parameters)
    def add_parameters(self, parameters):
        pass

    @overrides(AbstractSynapseType.add_state_variables)
    def add_state_variables(self, state_variables):
        state_variables[ISYN_EXC] = self.__isyn_exc

    @overrides(AbstractSynapseType.get_units)
    def get_units(self, variable):
        return UNITS[variable]

    @overrides(AbstractSynapseType.has_variable)
    def has_variable(self, variable):
        return variable in UNITS

    @overrides(AbstractSynapseType.get_values)
    def get_values(self, parameters, state_variables, vertex_slice, ts,
                   state_variables_indices=None):

        state_variables_indices.extend([0])
        # Add the rest of the data
        return [state_variables[ISYN_EXC]]

    @overrides(AbstractSynapseType.update_values)
    def update_values(self, values, parameters, state_variables):

        # Read the data
        (isyn_exc, ) = values

        state_variables[ISYN_EXC] = isyn_exc

    @overrides(AbstractSynapseType.get_n_synapse_types)
    def get_n_synapse_types(self):
        return 1

    @overrides(AbstractSynapseType.get_synapse_id_by_target)
    def get_synapse_id_by_target(self, target):
        if target == "excitatory":
            return 0
        return None

    @overrides(AbstractSynapseType.get_synapse_targets)
    def get_synapse_targets(self):
        return "excitatory"

    @property
    def isyn_exc(self):
        return self.__isyn_exc

    @isyn_exc.setter
    def isyn_exc(self, isyn_exc):
        self.__isyn_exc = isyn_exc

    @overrides(AbstractStandardNeuronComponent.get_sdram_usage_in_bytes)
    def get_sdram_usage_in_bytes(self, n_neurons):
        # current (isyn_X)
        num_state_variables = 1
        # none
        # num_shared_parameters = 0

        return (
            num_state_variables * n_neurons * self.get_n_synapse_types()
        ) * BYTES_PER_WORD
