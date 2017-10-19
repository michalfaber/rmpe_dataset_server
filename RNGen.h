#ifndef RMPE_DATASET_SERVER_RNGEN_H
#define RMPE_DATASET_SERVER_RNGEN_H

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;

class RNGen {
public:

  static RNGen& Get();

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
  public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
  private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  shared_ptr<RNG> random_generator_;

// Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }

};


#endif //RMPE_DATASET_SERVER_RNGEN_H
