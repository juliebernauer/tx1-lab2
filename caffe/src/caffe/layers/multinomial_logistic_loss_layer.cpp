#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void MultinomialLogisticLossLayer<Dtype,Mtype>::Reshape(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  LossLayer<Dtype,Mtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype, typename Mtype>
void MultinomialLogisticLossLayer<Dtype,Mtype>::Forward_cpu(
    const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Mtype loss(0.f);
  for (int i = 0; i < num; ++i) {
    int label = Get<int>(bottom_label[i]);
    Mtype prob = std::max(
        Get<Mtype>(bottom_data[i * dim + label]), Mtype(kLOG_THRESHOLD));
    loss -= log(prob);
  }
  top[0]->mutable_cpu_data()[0] = Get<Dtype>(loss / num);
}

template <typename Dtype, typename Mtype>
void MultinomialLogisticLossLayer<Dtype,Mtype>::Backward_cpu(
    const vector<Blob<Dtype,Mtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype,Mtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Get<Dtype>(0), bottom_diff);
    const Mtype scale = - Get<Mtype>(top[0]->cpu_diff()[0]) / num;
    for (int i = 0; i < num; ++i) {
      int label = Get<int>(bottom_label[i]);
      Mtype prob = std::max(
          Get<Mtype>(bottom_data[i * dim + label]), Mtype(kLOG_THRESHOLD));
      bottom_diff[i * dim + label] = Get<Dtype>(scale / prob);
    }
  }
}

INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(MultinomialLogisticLoss);

}  // namespace caffe
