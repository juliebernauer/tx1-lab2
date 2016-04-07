#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void SigmoidCrossEntropyLossLayer<Dtype,Mtype>::LayerSetUp(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  LossLayer<Dtype,Mtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype, typename Mtype>
void SigmoidCrossEntropyLossLayer<Dtype,Mtype>::Reshape(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  LossLayer<Dtype,Mtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype, typename Mtype>
void SigmoidCrossEntropyLossLayer<Dtype,Mtype>::Forward_cpu(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Mtype loss(0.f);
  for (int i = 0; i < count; ++i) {
    Mtype input_val = Get<Mtype>(input_data[i]);
    loss -= input_val * (Get<Mtype>(target[i]) - (input_val >= 0)) -
        log(1 + exp(input_val - 2 * input_val * (input_val >= 0)));
  }
  top[0]->mutable_cpu_data()[0] = Get<Dtype>(loss / num);
}

template <typename Dtype, typename Mtype>
void SigmoidCrossEntropyLossLayer<Dtype,Mtype>::Backward_cpu(
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
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub<Dtype,Mtype>(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Mtype loss_weight = Get<Mtype>(top[0]->cpu_diff()[0]);
    caffe_scal<Dtype,Mtype>(count, Get<Mtype>(loss_weight / num), bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
