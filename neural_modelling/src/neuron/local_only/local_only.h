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

#ifndef _LOCAL_ONLY_H_
#define _LOCAL_ONLY_H_

#define N_FROM_SHAPE(s) (s.width * s.height;)
#include <common/neuron-typedefs.h>
#include "local_only_typedefs.h"

bool local_only_initialise(address_t sdram_address);

bool local_only_is_compatible(void);

void local_only_process_spike(uint32_t key, uint32_t payload);

void local_only_coord_to_id(
    int32_t row, int32_t col, lc_shapes_t shapes, bool is_post,
    lc_neuron_id_t *output);

void local_only_id_to_coord(
    lc_neuron_id_t id, lc_shapes_t shapes, bool is_post,
    lc_coord_t *output);

void local_only_map_pre_to_post(lc_coord_t pre, lc_shapes_t shapes,
	lc_coord_t *output);

lc_dim_t local_only_get_ids_and_weights(
    lc_neuron_id_t pre_id, lc_shapes_t shapes, lc_weight_t* kernel,
    lc_neuron_id_t* post_ids, lc_weight_t* weights);

lc_dim_t num_possible_post(lc_shapes_t shapes);

#endif