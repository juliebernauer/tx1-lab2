#ifndef CAFFE_UTIL_GPU_UTIL_H_
#define CAFFE_UTIL_GPU_UTIL_H_

#include <cuda_fp16.h>

namespace caffe {

template <typename Dtype>
inline __device__ Dtype caffe_gpu_atomic_add(const Dtype val, Dtype* address);

template <>
inline __device__
float caffe_gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
inline __device__
double caffe_gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      // NOLINT_NEXT_LINE(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template <>
inline __device__
float16 caffe_gpu_atomic_add(const float16 val, float16* address) {
// TODO check for FP16 implementation in future CUDA releases
   return float16(0);
#if 0
   union U {
      unsigned int i;
      __half2 h;
   };
   union Up {
      unsigned int * i;
      __half2 * h;
   };

   Up up;
   up.h = address;

   U old;
   old.h = *address;
   U assumed;

   do {
      assumed.i = old.i;
      U temp;
      temp.h = __hadd2(val, assumed.h);
      old.i = atomicCAS( up.i, assumed.i, temp.i);
   } while (assumed.i != old.i );
   return old.h;
# endif
}

}  // namespace caffe

#endif  // CAFFE_UTIL_GPU_UTIL_H_
