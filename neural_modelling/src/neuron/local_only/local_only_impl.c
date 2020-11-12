#include "local_only.h"
#include "local_only_typedefs.h"
#include <common/neuron-typedefs.h>

static lc_weight_t* conv_kernel;
static lc_shapes_t shapes;

bool local_only_initialise(address_t sdram_address);

bool local_only_is_compatible();

void local_only_process_spike();

lc_neuron_id_t local_only_coord_to_id(
    lc_coord_t row, lc_coord_t col, lc_shapes_t shapes, bool is_post);

lc_coord_t local_only_id_to_coord(
    lc_neuron_id_t id, lc_shapes_t shapes, bool is_post);

void local_only_get_ids_and_weights(
    lc_neuron_id_t pre_id, lc_shapes_t shapes, lc_weight_t* kernel,
    lc_neuron_id_t* post_ids, lc_weight_t* currents);
