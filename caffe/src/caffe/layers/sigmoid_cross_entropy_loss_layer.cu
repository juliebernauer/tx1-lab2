#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void SigmoidCrossEntropyLossLayer<Dtype,Mtype>::Backward_gpu(
    const vector<Blob<Dtype,Mtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype,Mtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy<Dtype,Mtype>(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy<Dtype,Mtype>(count, Mtype(-1), target, bottom_diff);
    // Scale down gradient
    const Mtype loss_weight = Get<Mtype>(top[0]->cpu_diff()[0]);
    caffe_gpu_scal<Dtype,Mtype>(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);
INSTANTIATE_LAYER_GPU_BACKWARD_FF(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
