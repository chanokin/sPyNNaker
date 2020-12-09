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
from .abstract_threshold_type import AbstractThresholdType
from spynnaker.pyNN.models.neuron.implementations import (
    AbstractStandardNeuronComponent)
from spinn_front_end_common.utilities.constants import BYTES_PER_WORD

V_THRESH = "v_thresh"

UNITS = {V_THRESH: "mV"}


class ThresholdTypeStaticConv(AbstractThresholdType):
    """ A threshold that is a static value.
    """
    __slots__ = ["__v_thresh"]

    def __init__(self, v_thresh):
        """
        :param float v_thresh: :math:`V_{thresh}`
        """
        super(ThresholdTypeStaticConv, self).__init__([
            DataType.S1615])  # v_thresh
        self.__v_thresh = v_thresh

        self.needs_dma_weights = False
        self.requires_spike_mapping = True
        self.extend_state_variables = True


    @overrides(AbstractThresholdType.get_n_cpu_cycles)
    def get_n_cpu_cycles(self, n_neurons):
        # Just a comparison, but 2 just in case!
        return 2 * n_neurons

    @overrides(AbstractThresholdType.add_parameters)
    def add_parameters(self, parameters):
        parameters[V_THRESH] = self.__v_thresh

    @overrides(AbstractThresholdType.add_state_variables)
    def add_state_variables(self, state_variables):
        pass

    @overrides(AbstractThresholdType.get_units)
    def get_units(self, variable):
        return UNITS[variable]

    @overrides(AbstractThresholdType.has_variable)
    def has_variable(self, variable):
        return variable in UNITS

    @overrides(AbstractThresholdType.get_values)
    def get_values(self, parameters, state_variables, vertex_slice, ts,
                   state_variables_indices=None):

        # Add the rest of the data
        return [parameters[V_THRESH]]

    @overrides(AbstractThresholdType.update_values)
    def update_values(self, values, parameters, state_variables):

        # Read the data
        (_v_thresh,) = values

    @property
    def v_thresh(self):
        """
        :math:`V_{thresh}`
        """
        return self.__v_thresh

    @v_thresh.setter
    def v_thresh(self, v_thresh):
        self.__v_thresh = v_thresh

    @overrides(AbstractStandardNeuronComponent.get_sdram_usage_in_bytes)
    def get_sdram_usage_in_bytes(self, n_neurons):
        # none
        # num_state_variables = 1
        # threshold
        num_shared_parameters = 1

        return num_shared_parameters * BYTES_PER_WORD

