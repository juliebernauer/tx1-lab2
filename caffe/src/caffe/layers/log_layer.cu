#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void LogLayer<Dtype,Mtype>::Forward_gpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (input_scale_ == Get<Dtype>(1) && input_shift_ == Get<Dtype>(0)) {
    caffe_gpu_log<Dtype>(count, bottom_data, top_data);
  } else {
    caffe_copy<Dtype,Mtype>(count, bottom_data, top_data);
    if (input_scale_ != Get<Dtype>(1)) {
      caffe_gpu_scal<Dtype,Mtype>(count, Get<Mtype>(input_scale_), top_data);
    }
    if (input_shift_ != Get<Dtype>(0)) {
      caffe_gpu_add_scalar<Dtype,Mtype>(count, Get<Mtype>(input_shift_), top_data);
    }
    caffe_gpu_log<Dtype>(count, top_data, top_data);
  }
  if (base_scale_ != Get<Dtype>(1)) {
    caffe_gpu_scal<Dtype,Mtype>(count, Get<Mtype>(base_scale_), top_data);
  }
}

template <typename Dtype, typename Mtype>
void LogLayer<Dtype,Mtype>::Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy<Dtype,Mtype>(count, bottom_data, bottom_diff);
    if (input_scale_ != Get<Dtype>(1)) {
      caffe_gpu_scal<Dtype,Mtype>(count, Get<Mtype>(input_scale_), bottom_diff);
    }
    if (input_shift_ != Get<Dtype>(0)) {
      caffe_gpu_add_scalar<Dtype,Mtype>(count, Get<Mtype>(input_shift_), bottom_diff);
    }
    caffe_gpu_powx<Dtype,Mtype>(count, bottom_diff, Get<Mtype>(-1), bottom_diff);
    if (backward_num_scale_ != Get<Dtype>(1)) {
      caffe_gpu_scal<Dtype,Mtype>(count, Get<Mtype>(backward_num_scale_), bottom_diff);
    }
    caffe_gpu_mul<Dtype,Mtype>(count, top_diff, bottom_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(LogLayer);

}  // namespace caffe
