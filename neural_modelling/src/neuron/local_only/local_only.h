#ifndef _LOCAL_ONLY_H_
#define _LOCAL_ONLY_H_

#include <common/neuron-typedefs.h>
//        klist = self.pack_kernel(
//            dtp.encode_as_numpy_int_array(self.kernel.flatten())
//        ).tolist()
//
//        shapes = [
//            shape2word(self.pre_shape[WIDTH], self.pre_shape[HEIGHT]),
//            shape2word(self.post_shape[WIDTH], self.post_shape[HEIGHT]),
//            shape2word(self.padding[WIDTH], self.padding[HEIGHT]),
//            shape2word(self.strides[WIDTH], self.strides[HEIGHT]),
//            shape2word(self.kernel_shape[WIDTH], self.kernel_shape[HEIGHT])]
//
//        ndata = len(klist) + len(shapes)
//        data = [ndata] + shapes + klist

typedef struct {
    uint16_t width;
    uint16_t height;
} _shape_t;

typedef struct {
    _shape_t pre;
    _shape_t post;
    _shape_t padding;
    _shape_t strides;
    _shape_t kernel;
} local_only_shapes_t;

#define N_FROM_SHAPE(s) (s.width * s.height;)



bool local_only_initialise(address_t sdram_address);

bool local_only_is_compatible();

void local_only_process_spike();

uint32_t local_only_decode_neuron_id(uint32_t row, uint32_t col,
    local_only_shapes_t shapes, bool is_post);

uint32_t local_only_decode_neuron_coordinates(uint32_t id,
    local_only_shapes_t shapes, bool is_post, uint32_t &row, uint32_t &col);


#endif