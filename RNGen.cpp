#include "RNGen.h"
#include "utils.h"
#include <boost/thread.hpp>

// Make sure each thread can have different values.
static boost::thread_specific_ptr<RNGen> thread_instance_;

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

RNGen& RNGen::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new RNGen());
  }
  return *(thread_instance_.get());
}

class RNGen::RNG::Generator {
public:
  Generator() : rng_(new rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new rng_t(seed)) {}
  rng_t* rng() { return rng_.get(); }
private:
  shared_ptr<rng_t> rng_;
};

RNGen::RNG::RNG() : generator_(new Generator()) { }

RNGen::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

RNGen::RNG& RNGen::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* RNGen::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

