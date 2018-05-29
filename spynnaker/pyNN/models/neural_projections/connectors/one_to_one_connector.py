import numpy
from spinn_utilities.overrides import overrides
from .abstract_connector import AbstractConnector

import logging
logger = logging.getLogger(__name__)


class OneToOneConnector(AbstractConnector):
    """
    Where the pre- and postsynaptic populations have the same size, connect\
    cell i in the presynaptic pynn_population.py to cell i in the\
    postsynaptic pynn_population.py for all i.
    """
    __slots__ = ["_random_number_class"]

    def __init__(
            self, random_number_class, safe=True, verbose=False):
        """
        """
        self._random_number_class = random_number_class
        super(OneToOneConnector, self).__init__(safe, verbose)

    @overrides(AbstractConnector.get_delay_maximum)
    def get_delay_maximum(self, delays):
        return self._get_delay_maximum(
            delays, max((self._n_pre_neurons, self._n_post_neurons)))

    @overrides(AbstractConnector.get_delay_variance)
    def get_delay_variance(
            self, delays, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        # pylint: disable=too-many-arguments
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        if max_lo_atom > min_hi_atom:
            return 0
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        return self._get_delay_variance(delays, [connection_slice])

    @overrides(AbstractConnector.get_n_connections_from_pre_vertex_maximum)
    def get_n_connections_from_pre_vertex_maximum(
            self, delays, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            min_delay=None, max_delay=None):
        # pylint: disable=too-many-arguments
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))

        if min_hi_atom < max_lo_atom:
            return 0
        if min_delay is None or max_delay is None:
            return 1
        if isinstance(delays, self._random_number_class):
            return 1
        elif numpy.isscalar(delays):
            if delays >= min_delay and delays <= max_delay:
                return 1
            return 0

        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        slice_min_delay = min(delays[connection_slice])
        slice_max_delay = max(delays[connection_slice])
        if slice_min_delay >= min_delay and slice_max_delay <= max_delay:
            return 1
        return 0

    @overrides(AbstractConnector.get_n_connections_to_post_vertex_maximum)
    def get_n_connections_to_post_vertex_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        # pylint: disable=too-many-arguments
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        if min_hi_atom < max_lo_atom:
            return 0
        return 1

    @overrides(AbstractConnector.get_weight_mean)
    def get_weight_mean(
            self, weights, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        # pylint: disable=too-many-arguments
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        n_connections = (min_hi_atom - max_lo_atom) + 1
        if n_connections <= 0:
            return 0
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        return self._get_weight_mean(weights, [connection_slice])

    @overrides(AbstractConnector.get_weight_maximum)
    def get_weight_maximum(
            self, weights, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        # pylint: disable=too-many-arguments
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        n_connections = (min_hi_atom - max_lo_atom) + 1
        if n_connections <= 0:
            return 0
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        return self._get_weight_maximum(
            weights, n_connections, [connection_slice])

    @overrides(AbstractConnector.get_weight_variance)
    def get_weight_variance(
            self, weights, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        # pylint: disable=too-many-arguments
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        if max_lo_atom > min_hi_atom:
            return 0
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        return self._get_weight_variance(weights, [connection_slice])

    @overrides(AbstractConnector.generate_on_machine)
    def generate_on_machine(self, weights, delays):
        return (
            not self._generate_lists_on_host(weights) and
            not self._generate_lists_on_host(delays))

    @overrides(AbstractConnector.create_synaptic_block)
    def create_synaptic_block(
            self, weights, delays, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            synapse_type):
        # pylint: disable=too-many-arguments
        max_lo_atom = max(
            (pre_vertex_slice.lo_atom, post_vertex_slice.lo_atom))
        min_hi_atom = min(
            (pre_vertex_slice.hi_atom, post_vertex_slice.hi_atom))
        n_connections = max((0, (min_hi_atom - max_lo_atom) + 1))
        if n_connections <= 0:
            return numpy.zeros(0, dtype=self.NUMPY_SYNAPSES_DTYPE)
        connection_slice = slice(max_lo_atom, min_hi_atom + 1)
        block = numpy.zeros(n_connections, dtype=self.NUMPY_SYNAPSES_DTYPE)
        block["source"] = numpy.arange(max_lo_atom, min_hi_atom + 1)
        block["target"] = numpy.arange(max_lo_atom, min_hi_atom + 1)
        block["weight"] = self._generate_weights(
            weights, n_connections, [connection_slice])
        block["delay"] = self._generate_delays(
            delays, n_connections, [connection_slice])
        block["synapse_type"] = synapse_type
        return block

    def __repr__(self):
        return "OneToOneConnector()"
