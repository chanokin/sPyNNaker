#include "local_only.h"
#include "local_only_typedefs.h"
#include <common/neuron-typedefs.h>
#include <debug.h>
#include "../population_table/population_table.h"

#define LEN_SHAPE_DATA 7

static uint32_t num_connectors = 0;
static uint32_t *jumps = NULL;
static lc_weight_t** conv_kernel = NULL;
static lc_shapes_t *shapes = NULL;
static uint32_t n_bytes = 0;
static lc_weight_t* mapped_weights = NULL;
static lc_neuron_id_t* mapped_post_ids = NULL;
static uint32_t n_mapped = 0;
extern address_t local_address_start = NULL;

bool local_only_initialise(address_t sdram_address){
    log_info("+++++++++++++++++ CONV init ++++++++++++++++++++");
    log_info("SDRAM address is 0x%08x", sdram_address);

    // total incoming local-only connections data size
    n_bytes = *((uint32_t*)sdram_address++);
//    log_info("num bytes %d", n_bytes);
    if(n_bytes == 0){
        return true;
    }

    num_connectors = *((uint32_t*)sdram_address++);
    if(num_connectors == 0 && n_bytes > 0){
        return false;
    }
    log_info("num connectors = %u", num_connectors);

    conv_kernel = (lc_weight_t**)spin1_malloc(num_connectors * sizeof(lc_weight_t*));
    if(conv_kernel == NULL){
        log_error("Can't allocate memory for convolution kernel pointers.");
        return false;
    }

    jumps = (uint32_t*)spin1_malloc(num_connectors * 4);
    if(jumps == NULL){
        log_error("Can't allocate memory for address jumps.");
        return false;
    }

    shapes = (lc_shapes_t*)spin1_malloc(n_bytes);
    if(shapes == NULL){
        log_error("Can't allocate memory for shape's information.");
        return false;
    }

    address_t _address = sdram_address;

    uint32_t remaining_bytes = n_bytes;
    uint32_t idx = 0;
    uint32_t max_n_weights = 0;
    uint32_t mem_size = 0;
    for(idx = 0; idx < num_connectors; idx++){
        _address = sdram_address + mem_size;
        // how many elements are in a single connector data
        uint32_t n_elem = *((uint32_t*)_address++);
        log_info("Num elem %d", n_elem);

        uint32_t start = *((uint32_t*)_address++);
        log_info("Slice start %d", start);
        shapes[idx].start = start;

        // shapes are 16-bit uints, hopefully enough for future too?
        uint16_t *p = ((uint16_t*)(_address));
        // todo: can this be done with just a single memset?
        // todo: does it matter?
        shapes[idx].pre.width = *((lc_dim_t*)(p++));
        shapes[idx].pre.height = *((lc_dim_t*)(p++));
        shapes[idx].post.width = *((lc_dim_t*)(p++));
        shapes[idx].post.height = *((lc_dim_t*)(p++));
        shapes[idx].padding.width = *((lc_dim_t*)(p++));
        shapes[idx].padding.height = *((lc_dim_t*)(p++));
        shapes[idx].strides.width = *((lc_dim_t*)(p++));
        shapes[idx].strides.height = *((lc_dim_t*)(p++));
        shapes[idx].kernel.width = *((lc_dim_t*)(p++));
        shapes[idx].kernel.height = *((lc_dim_t*)(p));

        log_info("shape pre %d, %d",
                 shapes[idx].pre.width, shapes[idx].pre.height);
        log_info("shape post %d, %d",
                 shapes[idx].post.width, shapes[idx].post.height);
        log_info("shape padding %d, %d",
                 shapes[idx].padding.width, shapes[idx].padding.height);
        log_info("shape strides %d, %d",
                 shapes[idx].strides.width, shapes[idx].strides.height);
        log_info("shape kernel %d, %d",
                 shapes[idx].kernel.width, shapes[idx].kernel.height);


        // weight kernel data is also 16-bit
        lc_dim_t n_weights = shapes[idx].kernel.width * shapes[idx].kernel.height;
        if(max_n_weights < n_weights){
            max_n_weights = n_weights;
        }

        log_info("n_elem = %u\tn_weights = %u\tmem_size = %u",
            n_elem, n_weights, mem_size);
        jumps[idx] = mem_size;
        mem_size += n_elem;

    }

    for(idx = 0; idx < num_connectors; idx++){
        lc_dim_t n_weights = shapes[idx].kernel.width * shapes[idx].kernel.height;


        conv_kernel[idx] = (lc_weight_t*)spin1_malloc(n_weights * 2);
        if(conv_kernel[idx] == NULL){
            log_error(
                "Could not initialise convolution kernel weights (size = %u)",
                n_weights);
            rt_error(RTE_SWERR);
        }

        uint32_t *p32 = sdram_address + jumps[idx] + LEN_SHAPE_DATA;
        for(lc_dim_t r=0; r < shapes[idx].kernel.height; r++){
            for(lc_dim_t c=0; c < shapes[idx].kernel.width; c++){
                uint32_t w_idx = r * shapes[idx].kernel.width + c;
                if((w_idx%2) == 0){
                    conv_kernel[idx][w_idx] = (lc_weight_t)(p32[w_idx/2] >> 16);
                } else {
                    conv_kernel[idx][w_idx] = (lc_weight_t)(p32[w_idx/2] & ((1 << 16) - 1));
                }

                log_info("w(%d, %d) = fixed-point %d.%u",
                r, c,
                conv_kernel[idx][w_idx] >> 7,
                conv_kernel[idx][w_idx] & ((1 << 7) - 1));
            }
        }

    }

    mapped_weights = (lc_weight_t*)spin1_malloc(max_n_weights * 2);
    if(mapped_weights == NULL){
        log_error("Could not initialise weights buffer");
        rt_error(RTE_SWERR);
    }

    mapped_post_ids = (lc_neuron_id_t*)spin1_malloc(max_n_weights * 2);
    if(mapped_post_ids == NULL){
        log_error("Could not initialise post IDs buffer");
        rt_error(RTE_SWERR);
    }


    return true;
}

bool local_only_is_compatible(void){
    return (n_bytes > 0);
}

void local_only_process_spike(uint32_t key, uint32_t payload){

    address_t row_address = 0;
    size_t n_bytes_to_transfer = 0;
    bool success = false;
    success = population_table_get_first_address(
        key, &row_address, &n_bytes_to_transfer);
    log_info("key %u\tpayload %d\taddress %u\tn_bytes %u\tsuccess = %u\n\n",
        key, payload, row_address, n_bytes_to_transfer, success);

    if(success){
        uint32_t local_spike_id;

    }
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
