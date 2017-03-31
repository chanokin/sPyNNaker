"""
Synfirechain-like example
"""
import os
import unittest
import p7_integration_tests.scripts.synfire_run as synfire_run
import spynnaker.pyNN.utilities.utility_calls as utility_calls

n_neurons = 20
timestep = 1
max_delay = 14.40
delay = 1.7
neurons_per_core = n_neurons/2
runtime = 500
current_file_path = os.path.dirname(os.path.abspath(__file__))
current_file_path = os.path.join(current_file_path, "spikes.data")


class TestPrintSpikes(unittest.TestCase):
    """
    tests the printing of get spikes given a simulation
    """

    @unittest.skip("skipping test /0_1_time_steps/spikes/"
                   "test_synfire_0dot1_timestep_test_print_spikes.py")
    def test_print_spikes(self):
        results = synfire_run.do_run(n_neurons, timestep=timestep,
                                     max_delay=max_delay, delay=delay,
                                     neurons_per_core=neurons_per_core,
                                     runtimes=[runtime],
                                     spike_path=current_file_path)
        (v, gsyn, spikes) = results

        read_in_spikes = utility_calls.read_spikes_from_file(
            current_file_path, min_atom=0, max_atom=n_neurons,
            min_time=0, max_time=500)
        p.end()

        for spike_element, read_element in zip(spikes, read_in_spikes):
            self.assertEqual(round(spike_element[0], 1),
                             round(read_element[0], 1))
            self.assertEqual(round(spike_element[1], 1),
                             round(read_element[1], 1))


if __name__ == '__main__':
    unittest.main()
