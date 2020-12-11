#include "local_only.h"
#include "local_only_typedefs.h"
#include <common/neuron-typedefs.h>
#include <debug.h>


static lc_weight_t* conv_kernel = NULL;
static lc_shapes_t shapes = {0};
static uint32_t n_bytes = 0;
static lc_weight_t* mapped_weights = NULL;
static lc_neuron_id_t* mapped_post_ids = NULL;
static uint32_t n_mapped = 0;

bool local_only_initialise(address_t sdram_address){
    log_info("+++++++++++++++++ CONV init ++++++++++++++++++++");
    // total incoming local-only connections data size
    n_bytes = *((uint32_t*)sdram_address++);
    log_info("num bytes %d", n_bytes);
    if(n_bytes == 0){
        return true;
    }

    // how many elements are in a single connector data
    uint32_t n_elem = *((uint32_t*)sdram_address);
//    log_info("Num elem %d", n_elem);

    // shapes are 16-bit uints, hopefully enough for future too?
    uint16_t *p = ((uint16_t*)(++sdram_address));
    // todo: can this be done with just a single memset?
    // todo: does it matter?
    shapes.pre.width = *((lc_dim_t*)(p++));
    shapes.pre.height = *((lc_dim_t*)(p++));
    shapes.post.width = *((lc_dim_t*)(p++));
    shapes.post.height = *((lc_dim_t*)(p++));
    shapes.padding.width = *((lc_dim_t*)(p++));
    shapes.padding.height = *((lc_dim_t*)(p++));
    shapes.strides.width = *((lc_dim_t*)(p++));
    shapes.strides.height = *((lc_dim_t*)(p++));
    shapes.kernel.width = *((lc_dim_t*)(p++));
    shapes.kernel.height = *((lc_dim_t*)(p));

//    log_info("shape pre %d, %d", shapes.pre.width, shapes.pre.height);
//    log_info("shape post %d, %d", shapes.post.width, shapes.post.height);
//    log_info("shape padding %d, %d", shapes.padding.width, shapes.padding.height);
//    log_info("shape strides %d, %d", shapes.strides.width, shapes.strides.height);
//    log_info("shape kernel %d, %d", shapes.kernel.width, shapes.kernel.height);


    // weight kernel data is also 16-bit
    lc_dim_t n_weights = shapes.kernel.width * shapes.kernel.height;
    conv_kernel = (lc_weight_t*)spin1_malloc(n_weights * 2);
    if(conv_kernel == NULL){
        log_error("Could not initialise convolution kernel weights");
        rt_error(RTE_SWERR);
    }

    mapped_weights = (lc_weight_t*)spin1_malloc(n_weights * 2);
    if(mapped_weights == NULL){
        log_error("Could not initialise weights buffer");
        rt_error(RTE_SWERR);
    }

    mapped_post_ids = (lc_neuron_id_t*)spin1_malloc(n_weights * 2);
    if(mapped_post_ids == NULL){
        log_error("Could not initialise post IDs buffer");
        rt_error(RTE_SWERR);
    }

    n_mapped = 0;

    p = (lc_weight_t *)(sdram_address + LEN_SHAPE_DATA);
    uint32_t *p32 = sdram_address + LEN_SHAPE_DATA;
    for(lc_dim_t r=0; r < shapes.kernel.height; r++){
        for(lc_dim_t c=0; c < shapes.kernel.width; c++){
            uint32_t idx = r * shapes.kernel.width + c;
            if((idx%2) == 0){
                conv_kernel[idx] = (lc_weight_t)(p32[idx/2] >> 16);
            } else {
                conv_kernel[idx] = (lc_weight_t)(p32[idx/2] & ((1 << 16) - 1));
            }


//            log_info("w(%d, %d) = 32 %u signed %d fixed-point %d.%u",
//            r, c,
//            p32[idx/2],
//            conv_kernel[idx],
//            conv_kernel[idx] >> 7,
//            conv_kernel[idx] & ((1 << 7) - 1));
        }
    }

//    *p32 = sdram_address;
//    for(uint32_t i = 0; i  < n_elem; i++){
//        log_info("%u", *p32++);
//    }
    return true;
}

bool local_only_is_compatible(void){
    return (n_bytes > 0);
}

void local_only_process_spike(lc_neuron_id_t key){
}

void local_only_coord_to_id(
    lc_coord_t coord, lc_shapes_t _shapes, bool is_post,
    lc_neuron_id_t *id){
    if (is_post){
        *id = coord.row * _shapes.post.width + coord.col;
    } else {
        *id = coord.row * _shapes.pre.width + coord.col;
    }
}

void local_only_id_to_coord(
    lc_neuron_id_t id, lc_shapes_t shapes, bool is_post,
    lc_coord_t *coord){

    if (is_post){
        coord->row = id / shapes.post.width;
        coord->col = id % shapes.post.width;
    } else {
        coord->row = id / shapes.pre.width;
        coord->col = id % shapes.pre.width;
    }
}

void local_only_map_pre_to_post(
	lc_coord_t pre, lc_shapes_t _shapes,
	lc_coord_t *post){
    post->row = (pre.row - _shapes.kernel.height - 1 + 2 * _shapes.padding.height);
    post->row /= _shapes.strides.height;
    post->row += 1;

    post->col = (pre.col - _shapes.kernel.width - 1 + 2 * _shapes.padding.width);
    post->col /= _shapes.strides.width;
    post->col += 1;
}

lc_dim_t local_only_get_ids_and_weights(
    lc_neuron_id_t pre_id, lc_shapes_t _shapes, lc_weight_t* kernel,
    lc_neuron_id_t* post_ids, lc_weight_t* weights){

    lc_dim_t n_out = 0;
    lc_coord_t pre = {0, 0};
    lc_coord_t post = {0, 0};
    lc_coord_t tmp = {0,0};
    lc_shape_t half_k = {_shapes.kernel.width/2, _shapes.kernel.height/2};

    local_only_id_to_coord(pre_id, _shapes, false, &pre);
    local_only_map_pre_to_post(pre, _shapes, &post);

    for(lc_dim_t r = -half_k.height; r <= half_k.height; r++){
        tmp.row = post.row + r;
        if((tmp.row < 0) || (tmp.row >= _shapes.post.height)){
            continue;
        }
        for(lc_dim_t c = -half_k.width; c <= half_k.width; c++){
            tmp.col = post.col + c;
            if((tmp.col < 0) || (tmp.row >= _shapes.post.width)){
                continue;
            }
            local_only_coord_to_id(tmp, _shapes, true,
                                   &post_ids[n_out]);
            weights[n_out] =
                kernel[(r + half_k.height) * _shapes.kernel.width +
                            (c + half_k.width)];
            n_out++;
        }
    }
    return n_out;
}

lc_dim_t num_possible_post(lc_shapes_t _shapes){
    return (lc_dim_t)(_shapes.kernel.width * _shapes.kernel.height);
}
