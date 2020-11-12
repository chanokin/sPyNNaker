#include "local_only.h"
#include <common/neuron-typedefs.h>


bool local_only_initialise(address_t sdram_address){
    return true;
}

bool local_only_is_compatible(){
    return false;
}

