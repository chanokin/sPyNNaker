#ifndef _LOCAL_ONLY_H_
#define _LOCAL_ONLY_H_

#define N_FROM_SHAPE(s) (s.width * s.height;)
#define LEN_SHAPE_DATA 5
#include <common/neuron-typedefs.h>
#include "local_only_typedefs.h"

bool local_only_initialise(address_t sdram_address);

bool local_only_is_compatible(void);

void local_only_process_spike(uint32_t key, uint32_t payload);

void local_only_coord_to_id(
    lc_coord_t coord, lc_shapes_t shapes, bool is_post,
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