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

import logging
from spinn_utilities.overrides import overrides
from pacman.model.graphs.application import ApplicationEdge
from .projection_machine_edge import ProjectionMachineEdge
from spynnaker.pyNN.models.abstract_models import AbstractFilterableEdge

logger = logging.getLogger(__name__)
_DynamicsStructural = None


def _are_dynamics_structural(synapse_dynamics):
    global _DynamicsStructural
    if _DynamicsStructural is None:
        # Avoid import loop by postponing this import
        from spynnaker.pyNN.models.neuron.synapse_dynamics import (
            AbstractSynapseDynamicsStructural)
        _DynamicsStructural = AbstractSynapseDynamicsStructural
    return isinstance(synapse_dynamics, _DynamicsStructural)


class ProjectionApplicationEdge(ApplicationEdge, AbstractFilterableEdge):
    """ An edge which terminates on an :py:class:`AbstractPopulationVertex`.
    """
    __slots__ = [
        "__delay_edge",
        "__synapse_information",
        "__machine_edges_by_slices"]

    def __init__(
            self, pre_vertex, post_vertex, synapse_information, label=None):
        """
        :param AbstractPopulationVertex pre_vertex:
        :param AbstractPopulationVertex post_vertex:
        :param SynapseInformation synapse_information:
        :param str label:
        """
        super(ProjectionApplicationEdge, self).__init__(
            pre_vertex, post_vertex, label=label)

        # A list of all synapse information for all the projections that are
        # represented by this edge
        self.__synapse_information = [synapse_information]

        # The edge from the delay extension of the pre_vertex to the
        # post_vertex - this might be None if no long delays are present
        self.__delay_edge = None

        # Keep the machine edges by pre- and post-vertex
        self.__machine_edges_by_slices = dict()

    def add_synapse_information(self, synapse_information):
        """
        :param SynapseInformation synapse_information:
        """
        self.__synapse_information.append(synapse_information)

    @property
    def synapse_information(self):
        """
        :rtype: list(SynapseInformation)
        """
        return self.__synapse_information

    @property
    def delay_edge(self):
        """ Settable.

        :rtype: DelayedApplicationEdge or None
        """
        return self.__delay_edge

    @delay_edge.setter
    def delay_edge(self, delay_edge):
        self.__delay_edge = delay_edge

    @property
    def n_delay_stages(self):
        """
        :rtype: int
        """
        if self.__delay_edge is None:
            return 0
        return self.__delay_edge.pre_vertex.n_delay_stages

    @overrides(ApplicationEdge._create_machine_edge)
    def _create_machine_edge(
            self, pre_vertex, post_vertex, label):
        edge = ProjectionMachineEdge(
            self.__synapse_information, pre_vertex, post_vertex, self, label)
        self.__machine_edges_by_slices[
            pre_vertex.vertex_slice, post_vertex.vertex_slice] = edge
        if self.__delay_edge is not None:
            delayed = self.__delay_edge._get_machine_edge(
                pre_vertex, post_vertex)
            if delayed is not None:
                edge.delay_edge = delayed
        return edge

    def _get_machine_edge(self, pre_vertex, post_vertex):
        return self.__machine_edges_by_slices.get(
            (pre_vertex.vertex_slice, post_vertex.vertex_slice))

    @overrides(AbstractFilterableEdge.filter_edge)
    def filter_edge(self):
        return all(
            not _are_dynamics_structural(syn_info.synapse_dynamics)
            for syn_info in self.__synapse_information)
