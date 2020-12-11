#include "local_only.h"
#include "local_only_typedefs.h"
#include <common/neuron-typedefs.h>
#include <debug.h>


static uint32_t n_bytes = 0;

bool local_only_initialise(address_t sdram_address){
    log_debug("CONV init.");
    // total incoming local-only connections data size
    n_bytes = *((uint32_t*)sdram_address++);
    return true;
}

bool local_only_is_compatible(void){
    return false;
}

void local_only_process_spike(uint32_t key, uint32_t payload){
}

void local_only_coord_to_id(
    UNUSED lc_coord_t coord,
    UNUSED lc_shapes_t _shapes,
    UNUSED bool is_post,
    UNUSED lc_neuron_id_t *output){
}

void local_only_id_to_coord(
    UNUSED lc_neuron_id_t id,
    UNUSED lc_shapes_t shapes,
    UNUSED bool is_post,
    UNUSED lc_coord_t *output){
}

void local_only_map_pre_to_post(
    UNUSED lc_coord_t pre, UNUSED lc_shapes_t _shapes,
    UNUSED lc_coord_t *output){
}

lc_dim_t local_only_get_ids_and_weights(
    UNUSED lc_neuron_id_t pre_id, UNUSED lc_shapes_t _shapes,
    UNUSED lc_weight_t* kernel, UNUSED lc_neuron_id_t* post_ids,
    UNUSED lc_weight_t* currents){
    return 0;
}

lc_dim_t num_possible_post(UNUSED  lc_shapes_t _shapes){
    return 0;
}
