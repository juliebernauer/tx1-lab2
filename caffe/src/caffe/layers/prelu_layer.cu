#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype, typename Mtype>
__global__ void PReLUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : Dtype(in[index] * slope_data[c]);
  }
}

// CUDA kernel for bottom backward
template <typename Dtype, typename Mtype>
__global__ void PReLUBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* slope_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = Get<Dtype>( Get<Mtype>(in_diff[index]) * ((Get<Mtype>(in_data[index]) > 0)
        + (Get<Mtype>(in_data[index]) <= 0) * Get<Mtype>(slope_data[c])) );
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype, typename Mtype>
__global__ void PReLUParamBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = Get<Dtype>( Get<Mtype>(in_diff[index]) * Get<Mtype>(in_data[index]) * (Get<Mtype>(in_data[index]) <= 0) );
  }
}

template <typename Dtype, typename Mtype>
void PReLULayer<Dtype,Mtype>::Forward_gpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy<Dtype,Mtype>(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  PReLUForward<Dtype,Mtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, slope_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, typename Mtype>
void PReLULayer<Dtype,Mtype>::Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype,Mtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    // slope_diff is set as 0, then accumulated over batches
    caffe_gpu_set<Dtype,Mtype>(this->blobs_[0]->count(), Mtype(0), slope_diff);
    int cdim = channels * dim;
    Mtype dsum(0.);
    for (int n = 0; n < bottom[0]->num(); ++n) {
      // compute element-wise diff
      // NOLINT_NEXT_LINE(whitespace/operators)
      PReLUParamBackward<Dtype,Mtype><<<CAFFE_GET_BLOCKS(cdim),
          CAFFE_CUDA_NUM_THREADS>>>(
          cdim, top_diff + top[0]->offset(n),
          bottom_data + bottom[0]->offset(n),
          backward_buff_.mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      if (channel_shared_) {
        Mtype d;
        caffe_gpu_dot<Dtype,Mtype>(channels * dim, backward_buff_.gpu_diff(),
            multiplier_.gpu_data(), &d);
        dsum += d;
      } else {
        caffe_gpu_gemv<Dtype,Mtype>(CblasNoTrans, channels, dim, Mtype(1.),
            backward_buff_.gpu_diff(), multiplier_.gpu_data(), Mtype(1.),
            slope_diff);
      }
    }
    if (channel_shared_) {
      caffe_gpu_add_scalar<Dtype,Mtype>(this->blobs_[0]->count(), Get<Mtype>(dsum), slope_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    PReLUBackward<Dtype,Mtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PReLULayer);


}  // namespace caffe
