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
from spinn_utilities.overrides import overrides
from data_specification.enums import DataType
from .abstract_neuron_model import AbstractNeuronModel
from spynnaker.pyNN.models.neuron.implementations import (
    AbstractStandardNeuronComponent)
from .neuron_model_leaky_integrate_and_fire import (
    NeuronModelLeakyIntegrateAndFire as LIF_base)

V = "v"
V_REST = "v_rest"
TAU_M = "tau_m"
CM = "cm"
I_OFFSET = "i_offset"
V_RESET = "v_reset"
TAU_REFRAC = "tau_refrac"
COUNT_REFRAC = "count_refrac"

UNITS = {
    V: 'mV',
    V_REST: 'mV',
    TAU_M: 'ms',
    CM: 'nF',
    I_OFFSET: 'nA',
    V_RESET: 'mV',
    TAU_REFRAC: 'ms'
}


class NeuronModelLeakyIntegrateAndFireConv(LIF_base):
    """ Classic leaky integrate and fire neuron model.
    """
    def __init__(
            self, v_init, v_rest, tau_m, cm, i_offset, v_reset, tau_refrac):
        r"""
        :param float v_init: :math:`V_{init}`
        :param float v_rest: :math:`V_{rest}`
        :param float tau_m: :math:`\tau_{m}`
        :param float cm: :math:`C_m`
        :param float i_offset: :math:`I_{offset}`
        :param float v_reset: :math:`V_{reset}`
        :param float tau_refrac: :math:`\tau_{refrac}`
        """
        super(NeuronModelLeakyIntegrateAndFireConv, self).__init__(
            v_init, v_rest, tau_m, cm, i_offset, v_reset, tau_refrac)  # tau_refrac
        self.requires_spike_mapping = True
        self.needs_dma_weights = False

