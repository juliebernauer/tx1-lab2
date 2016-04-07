#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void ContrastiveLossLayer<Dtype,Mtype>::LayerSetUp(
  const vector<Blob<Dtype,Mtype>*>& bottom, const vector<Blob<Dtype,Mtype>*>& top) {
  LossLayer<Dtype,Mtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Get<Dtype>(1);
}

template <typename Dtype, typename Mtype>
void ContrastiveLossLayer<Dtype,Mtype>::Forward_cpu(
    const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub<Dtype,Mtype>(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  Mtype margin(this->layer_param_.contrastive_loss_param().margin());
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Mtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = Get<Dtype>(caffe_cpu_dot<Dtype,Mtype>(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels)));
    if (Get<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += Get<Mtype>(dist_sq_.cpu_data()[i]);
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max((float)(margin - dist_sq_.cpu_data()[i]), 0.F);
      } else {
        Mtype dist = std::max((float)(margin - sqrt(dist_sq_.cpu_data()[i])), 0.F);
        loss += dist*dist;
      }
    }
  }
  loss = loss / static_cast<Mtype>(bottom[0]->num()) / Mtype(2);
  top[0]->mutable_cpu_data()[0] = Get<Dtype>(loss);
}

template <typename Dtype, typename Mtype>
void ContrastiveLossLayer<Dtype,Mtype>::Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype,Mtype>*>& bottom) {
  Mtype margin(this->layer_param_.contrastive_loss_param().margin());
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Mtype sign(i == 0 ? 1 : -1);
      const Mtype alpha = sign * Get<Mtype>(top[0]->cpu_diff()[0]) /
          static_cast<Mtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (Get<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby<Dtype,Mtype>(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Mtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          Mtype mdist(0.0);
          Mtype beta(0.0);
          if (legacy_version) {
            mdist = margin - Get<Mtype>(dist_sq_.cpu_data()[j]);
            beta = -alpha;
          } else {
            Mtype dist(sqrt(dist_sq_.cpu_data()[j]));
            mdist = margin - dist;
            beta = -alpha * mdist / (dist + Mtype(1e-4));
          }
          if (mdist > Mtype(0.0)) {
            caffe_cpu_axpby<Dtype,Mtype>(
                channels,
                beta,
                diff_.cpu_data() + (j*channels),
                Mtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Get<Dtype>(0), bout + (j*channels));
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(ContrastiveLossLayer);
REGISTER_LAYER_CLASS(ContrastiveLoss);

}  // namespace caffe
