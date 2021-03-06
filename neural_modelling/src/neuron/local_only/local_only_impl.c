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

#include "local_only.h"
#include "local_only_typedefs.h"
#include <common/neuron-typedefs.h>
#include <debug.h>
#include "../population_table/population_table.h"
#include "../neuron.h"

#define LEN_SHAPE_DATA 7

static uint32_t num_connectors = 0;
static uint32_t *jumps = NULL;
static lc_weight_t** conv_kernel = NULL;
static lc_shapes_t *shapes = NULL;
static uint32_t n_bytes = 0;
static lc_weight_t* mapped_weights = NULL;
static lc_neuron_id_t* mapped_post_ids = NULL;
//static uint32_t n_mapped = 0;

inline input_t to_s1615(lc_weight_t w){
    // conv weights are stored as s87 so we need to shift them
    // so we end up with s1615
    return (((input_t)w) >> 7);
}

bool local_only_initialise(address_t sdram_address){
    log_info("+++++++++++++++++ CONV init ++++++++++++++++++++");
//    log_info("SDRAM address is 0x%08x", sdram_address);

    // total incoming local-only connections data size
    n_bytes = *((uint32_t*)sdram_address++);
    log_info("num bytes %d", n_bytes);
    if (n_bytes == 0) {
        return true;
    }

    num_connectors = *((uint32_t*)sdram_address++);
    if (num_connectors == 0 && n_bytes > 0) {
        return false;
    }
    log_info("num connectors = %u", num_connectors);

    conv_kernel = (lc_weight_t**)spin1_malloc(num_connectors * sizeof(lc_weight_t*));
    if (conv_kernel == NULL) {
        log_error("Can't allocate memory for convolution kernel pointers.");
        return false;
    }

    jumps = (uint32_t*)spin1_malloc(num_connectors * 4);
    if (jumps == NULL) {
        log_error("Can't allocate memory for address jumps.");
        return false;
    }

    shapes = (lc_shapes_t*)spin1_malloc(n_bytes);
    if (shapes == NULL) {
        log_error("Can't allocate memory for shape's information.");
        return false;
    }

    address_t _address = sdram_address;

//    uint32_t remaining_bytes = n_bytes;
    uint32_t idx = 0;
    uint32_t max_n_weights = 0;
    uint32_t mem_size = 0;
    for (idx = 0; idx < num_connectors; idx++) {
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
        uint32_t n_weights = shapes[idx].kernel.width * shapes[idx].kernel.height;
        if (max_n_weights < n_weights) {
            max_n_weights = n_weights;
        }

//        log_info("n_elem = %u\tn_weights = %u\tmem_size = %u",
//            n_elem, n_weights, mem_size);
        jumps[idx] = mem_size;
        mem_size += n_elem;

    }

    for (idx = 0; idx < num_connectors; idx++) {
        lc_dim_t n_weights = shapes[idx].kernel.width * shapes[idx].kernel.height;


        conv_kernel[idx] = (lc_weight_t*)spin1_malloc(n_weights * 2);
        if (conv_kernel[idx] == NULL) {
            log_error(
                "Could not initialise convolution kernel weights (size = %u)",
                n_weights);
            return false;
//            rt_error(RTE_SWERR);
        }

        uint32_t *p32 = sdram_address + jumps[idx] + LEN_SHAPE_DATA;
        for (lc_dim_t r=0; r < shapes[idx].kernel.height; r++) {
            for (lc_dim_t c=0; c < shapes[idx].kernel.width; c++) {
                uint32_t w_idx = r * shapes[idx].kernel.width + c;
                if ((w_idx%2) == 0) {
                    conv_kernel[idx][w_idx] = (lc_weight_t)(p32[w_idx/2] >> 16);
                } else {
                    conv_kernel[idx][w_idx] = (lc_weight_t)(p32[w_idx/2] & 0xFFFF);
                }

                log_info("w(%d, %d) = fixed-point %k",
                r, c,
                to_s1615(conv_kernel[idx][w_idx]));
            }
        }

    }
	// 16-bit weights
    mapped_weights = (lc_weight_t*)spin1_malloc(
                                    max_n_weights * sizeof(lc_weight_t));
    if (mapped_weights == NULL) {
        log_error("Could not initialise weights buffer");
        return false;
//        rt_error(RTE_SWERR);
    }

    mapped_post_ids = (lc_neuron_id_t*)spin1_malloc(
                                    max_n_weights * sizeof(lc_neuron_id_t));
    if (mapped_post_ids == NULL) {
        log_error("Could not initialise post IDs buffer");
        return false;
//        rt_error(RTE_SWERR);
    }


    return true;
}

bool local_only_is_compatible(void){
    return (n_bytes > 0);
}

void local_only_process_spike(uint32_t key, UNUSED uint32_t payload){

    uint32_t conn_jump = 0;
    size_t pre_id_relative = 0;
    bool success = false;
    // see if spike key can be found in the population table,
    // get the number of incoming connector (conn_jump) and the
    // relative (to slice beginning) pre-synaptic neuron id
    success = population_table_get_first_address(
        key, &conn_jump, &pre_id_relative);

//    log_info("key %u\tpayload %d\tjump %u\tpre_rel %u\tsuccess = %u",
//        key, payload, conn_jump, pre_id_relative, success);

    // if the key was found in the pop table, then add current to
    // post-synaptic neuron
    if (success) {
        // compute the real pre-syn id (population id)
        lc_neuron_id_t pre_id = pre_id_relative + shapes[conn_jump].start;
//        log_info("real pre id %u\n", pre_id);
        // get the post-syn neuron ids (this core ids)
        // get the weights which will be added to each post-syn neuron
        // get how many post-syn neurons got hit by the current spike
        lc_dim_t n_out = local_only_get_ids_and_weights(
            pre_id, shapes[conn_jump], conv_kernel[conn_jump],
            mapped_post_ids, mapped_weights);

        // add the weight to each of the post neurons
        for (lc_dim_t i=0; i<n_out; i++) {
//            log_info("post %u, weight fixed-point s1615 %k",
//                mapped_post_ids[i],
//                to_s1615(mapped_weights[i]));
            neuron_add_inputs(
                0, // only one synapse type to save space
                mapped_post_ids[i],
                to_s1615(mapped_weights[i]));

        }
    }
}


void local_only_coord_to_id(
    int32_t row, int32_t col, lc_shapes_t _shapes, bool is_post,
    lc_neuron_id_t *id){
    if (is_post) {
//        log_info("id = %d", row * _shapes.post.width + col);
        *id = row * _shapes.post.width + col;
    } else {
        *id = row * _shapes.pre.width + col;
    }
}

void local_only_id_to_coord(
    lc_neuron_id_t id, lc_shapes_t shapes, bool is_post,
    lc_coord_t *coord){

    if (is_post) {
        coord->row = id / shapes.post.width;
        coord->col = id % shapes.post.width;
    } else {
        coord->row = id / shapes.pre.width;
        coord->col = id % shapes.pre.width;
    }
}

void local_only_map_pre_to_post(
    lc_coord_t pre, lc_shapes_t _shapes, lc_coord_t *post){
    int32_t _pre = pre.row;
    int32_t _post = 0;
    _post = (_pre - (_shapes.kernel.height>>1) + _shapes.padding.height) / _shapes.strides.height;
    _post += 1;
    post->row = _post;

    _pre = pre.col;
    _post = (_pre - (_shapes.kernel.width>>1) + _shapes.padding.width) / _shapes.strides.width;
    _post += 1;
    post->col = _post;

}

lc_dim_t local_only_get_ids_and_weights(
    lc_neuron_id_t pre_id, lc_shapes_t _shapes, lc_weight_t* kernel,
    lc_neuron_id_t* post_ids, lc_weight_t* weights){

    lc_dim_t n_out = 0;
    lc_coord_t pre = {0, 0};
    lc_coord_t post = {0, 0};
    int32_t tmp_row = 0;
    int32_t tmp_col = 0;
    int32_t half_kh = _shapes.kernel.height >> 1;
    int32_t half_kw = _shapes.kernel.width >> 1;
	log_debug("half k shape width %u, height %u", half_kw, half_kh);

    local_only_id_to_coord(pre_id, _shapes, false, &pre);
    log_debug("pre %u row %d, col %d", pre_id, pre.row, pre.col);

    local_only_map_pre_to_post(pre, _shapes, &post);
    log_debug("AS post row %d, col %d", post.row, post.col);

    for (int32_t r = -half_kh; r <= half_kh; r++) {
        tmp_row = post.row + r;
        log_debug("r %d : tmp row %d", r, tmp_row);
        if ((tmp_row < 0) || (tmp_row >= _shapes.post.height)) {
            log_debug("escape row %d %d", post.row, tmp_row);
            continue;
        }
        for (int32_t c = -half_kw; c <= half_kw; c++) {
            tmp_col = post.col + c;
            log_debug("c %d : tmp col %d", c, tmp_col);
            if ((tmp_col < 0) || (tmp_col >= _shapes.post.width)) {
                log_debug("escape col %d %d", post.col, tmp_col);
                continue;
            }
            log_debug("tmp_row %d, tmp_col %d", tmp_row, tmp_col);
            local_only_coord_to_id(tmp_row, tmp_col, _shapes, true,
                                   &post_ids[n_out]);

            weights[n_out] =
                kernel[(r + half_kh) * _shapes.kernel.width +
                            (c + half_kw)];
            log_debug("pre %u, post r %d, c %d, i %u, weight %k",
                pre_id, tmp_row, tmp_col,
                post_ids[n_out],
                to_s1615(weights[n_out]));
            n_out++;
        }
    }
    return n_out;
}

lc_dim_t num_possible_post(lc_shapes_t _shapes){
    return (lc_dim_t)(_shapes.kernel.width * _shapes.kernel.height);
}
