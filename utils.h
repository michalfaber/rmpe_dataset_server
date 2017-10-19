#ifndef RMPE_DATASET_SERVER_UTILS_H
#define RMPE_DATASET_SERVER_UTILS_H
#include <boost/random.hpp>
#include "RNGen.h"

typedef boost::mt19937 rng_t;
inline rng_t* caffe_rng() {
  return static_cast<rng_t*>(RNGen::rng_stream().generator());
}

#endif //RMPE_DATASET_SERVER_UTILS_H
