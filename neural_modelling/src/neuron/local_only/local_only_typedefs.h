#ifndef _LOCAL_ONLY_STRUCTS_
#define _LOCAL_ONLY_STRUCTS_

#include <common/neuron-typedefs.h>

typedef int16_t lc_weight_t;
typedef int16_t lc_dim_t;
typedef uint32_t lc_neuron_id_t;

typedef struct {
    lc_dim_t row;
    lc_dim_t col;
} lc_coord_t;

typedef struct {
    lc_dim_t width;
    lc_dim_t height;
} lc_shape_t;

typedef struct {
    lc_neuron_id_t start;
    lc_shape_t pre;
    lc_shape_t post;
    lc_shape_t padding;
    lc_shape_t strides;
    lc_shape_t kernel;
} lc_shapes_t;

#endif