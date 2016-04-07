#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void ExpLayer<Dtype,Mtype>::Forward_gpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (inner_scale_ == Mtype(1)) {
    caffe_gpu_exp<Dtype,Mtype>(count, bottom_data, top_data);
  } else {
    caffe_gpu_scale<Dtype,Mtype>(count, inner_scale_, bottom_data, top_data);
    caffe_gpu_exp<Dtype,Mtype>(count, top_data, top_data);
  }
  if (outer_scale_ != Mtype(1)) {
    caffe_gpu_scal<Dtype,Mtype>(count, outer_scale_, top_data);
  }
}

template <typename Dtype, typename Mtype>
void ExpLayer<Dtype,Mtype>::Backward_gpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_mul<Dtype,Mtype>(count, top_data, top_diff, bottom_diff);
  if (inner_scale_ != Mtype(1)) {
    caffe_gpu_scal<Dtype,Mtype>(count, inner_scale_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpLayer);


}  // namespace caffe
