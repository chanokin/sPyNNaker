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
