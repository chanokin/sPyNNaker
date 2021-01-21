/*
 * Copyright (c) 2017-2019 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//! \file
//! \brief Leaky Integrate and Fire neuron implementation
#include "neuron_model_lif_conv_impl.h"

#include <debug.h>

global_neuron_params_t *global_params;

//! \brief simple Leaky I&F ODE
//! \param[in,out] neuron: The neuron to update
//! \param[in] V_prev: previous voltage
//! \param[in] input_this_timestep: The input to apply
static inline void lif_neuron_closed_form(
        neuron_t *neuron, REAL V_prev, input_t input_this_timestep) {
    REAL alpha = input_this_timestep * global_params->R_membrane + \
                 global_params->V_rest;

    // update membrane voltage
    neuron->V_membrane = alpha - (global_params->exp_TC * (alpha - V_prev));
}

void neuron_model_set_global_neuron_params(
        const global_neuron_params_t *params) {
    global_params = params;
}

state_t neuron_model_state_update(
		uint16_t num_excitatory_inputs, const input_t *exc_input,
		uint16_t num_inhibitory_inputs, const input_t *inh_input,
		input_t external_bias, neuron_t *restrict neuron) {

//	log_debug("Exc 1: %12.6k, Exc 2: %12.6k", exc_input[0], exc_input[1]);
//	log_debug("Inh 1: %12.6k, Inh 2: %12.6k", inh_input[0], inh_input[1]);

    // If outside of the refractory period
    if (neuron->refract_timer <= 0) {
		REAL total_exc = 0;
		REAL total_inh = 0;

		for (int i=0; i < num_excitatory_inputs; i++) {
			total_exc += exc_input[i];
//            log_info("Exc %d: %12.6k\ttotal: %k", i, exc_input[i], total_exc);
		}
		for (int i=0; i< num_inhibitory_inputs; i++) {
			total_inh += inh_input[i];
//            log_info("Inh %d: %12.6k", i, inh_input[i]);
		}
        // Get the input in nA
        input_t input_this_timestep =
                total_exc - total_inh + external_bias + global_params->I_offset;
//        log_info("total_input %k", input_this_timestep);
        lif_neuron_closed_form(
                neuron, neuron->V_membrane, input_this_timestep);
    } else {
        // countdown refractory timer
        neuron->refract_timer--;
    }
    return neuron->V_membrane;
}

void neuron_model_has_spiked(neuron_t *restrict neuron) {
//    log_info("Neuron has spiked!!!");
    // reset membrane voltage
    neuron->V_membrane = global_params->V_reset;

    // reset refractory timer
    neuron->refract_timer  = global_params->T_refract;
}

state_t neuron_model_get_membrane_voltage(const neuron_t *neuron) {
    return neuron->V_membrane;
}

void neuron_model_print_state_variables(
    UNUSED const neuron_t *neuron) {
//    log_info("V membrane    = %11.4k mv", neuron->V_membrane);
//    log_info("Refract time    = %u", neuron->refract_timer);
}

void neuron_model_print_parameters(
    UNUSED const neuron_t *neuron) {
//    log_info("V reset       = %11.4k mv", global_params->V_reset);
//    log_info("V rest        = %11.4k mv", global_params->V_rest);
//
//    log_info("I offset      = %11.4k nA", global_params->I_offset);
//    log_info("R membrane    = %11.4k Mohm", global_params->R_membrane);
//
//    log_info("exp(-ms/(RC)) = %11.4k [.]", global_params->exp_TC);
//
//    log_info("T refract     = %u timesteps", global_params->T_refract);
}
