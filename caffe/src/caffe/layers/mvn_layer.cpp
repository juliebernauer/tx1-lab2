#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void MVNLayer<Dtype,Mtype>::Reshape(const vector<Blob<Dtype,Mtype>*>& bottom,
      const vector<Blob<Dtype,Mtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  if ( this->layer_param_.mvn_param().across_channels() ) {
    sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                            bottom[0]->width());
  } else {
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Get<Dtype>(1), multiplier_data);
  eps_ = this->layer_param_.mvn_param().eps();
}

template <typename Dtype, typename Mtype>
void MVNLayer<Dtype,Mtype>::Forward_cpu(const vector<Blob<Dtype,Mtype>*>& bottom,
    const vector<Blob<Dtype,Mtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // put the squares of bottom into temp_
    caffe_powx<Dtype,Mtype>(bottom[0]->count(), bottom_data, Mtype(2),
        temp_.mutable_cpu_data());

    // computes variance using var(X) = E(X^2) - (EX)^2
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, num, dim, Mtype(1. / dim), bottom_data,
				sum_multiplier_.cpu_data(), Mtype(0.), mean_.mutable_cpu_data());  // EX
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, num, dim, Mtype(1. / dim), temp_.cpu_data(),
				sum_multiplier_.cpu_data(), Mtype(0.),
        variance_.mutable_cpu_data());  // E(X^2)
    caffe_powx<Dtype,Mtype>(mean_.count(), mean_.cpu_data(), Mtype(2),
        temp_.mutable_cpu_data());  // (EX)^2
    caffe_sub<Dtype,Mtype>(mean_.count(), variance_.cpu_data(), temp_.cpu_data(),
        variance_.mutable_cpu_data());  // variance

    // do mean and variance normalization
    // subtract mean
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Mtype(-1.f),
				mean_.cpu_data(), sum_multiplier_.cpu_data(), Mtype(0.f),
				temp_.mutable_cpu_data());

    caffe_add<Dtype,Mtype>(temp_.count(), bottom_data, temp_.cpu_data(), top_data);

    // normalize variance
    caffe_powx<Dtype,Mtype>(variance_.count(), variance_.cpu_data(), Mtype(0.5),
          variance_.mutable_cpu_data());

    caffe_add_scalar<Dtype,Mtype>(variance_.count(), eps_, variance_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Mtype(1.f),
				variance_.cpu_data(), sum_multiplier_.cpu_data(), Mtype(0.),
          temp_.mutable_cpu_data());

    caffe_div<Dtype,Mtype>(temp_.count(), top_data, temp_.cpu_data(), top_data);
  } else {
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, num, dim, Mtype(1. / dim), bottom_data,
				sum_multiplier_.cpu_data(), Mtype(0.), mean_.mutable_cpu_data());  // EX

    // subtract mean
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Mtype(-1.),
				mean_.cpu_data(), sum_multiplier_.cpu_data(), Mtype(0.),
            temp_.mutable_cpu_data());

    caffe_add<Dtype,Mtype>(temp_.count(), bottom_data, temp_.cpu_data(), top_data);
  }
}

template <typename Dtype, typename Mtype>
void MVNLayer<Dtype,Mtype>::Backward_cpu(const vector<Blob<Dtype,Mtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype,Mtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    caffe_mul<Dtype,Mtype>(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, num, dim, Mtype(1.), bottom_diff,
				sum_multiplier_.cpu_data(), Mtype(0.), mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Mtype(1.),
          mean_.cpu_data(), sum_multiplier_.cpu_data(), Mtype(0.f),
          bottom_diff);
    caffe_mul<Dtype,Mtype>(temp_.count(), top_data, bottom_diff, bottom_diff);

    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, num, dim, Mtype(1.), top_diff,
            sum_multiplier_.cpu_data(), Mtype(0.f), mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Mtype(1.),
            mean_.cpu_data(), sum_multiplier_.cpu_data(), Mtype(1.f),
            bottom_diff);

    caffe_cpu_axpby<Dtype,Mtype>(temp_.count(), Mtype(1), top_diff, Mtype(-1. / dim),
        bottom_diff);

    // put the squares of bottom into temp_
    caffe_powx<Dtype,Mtype>(temp_.count(), bottom_data, Mtype(2),
        temp_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Mtype(1.f),
        variance_.cpu_data(), sum_multiplier_.cpu_data(), Mtype(0.f),
        temp_.mutable_cpu_data());

    caffe_div<Dtype,Mtype>(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
  } else {
    caffe_cpu_gemv<Dtype,Mtype>(CblasNoTrans, num, dim, Mtype(1. / dim), top_diff,
      sum_multiplier_.cpu_data(), Mtype(0.f), mean_.mutable_cpu_data());
    caffe_cpu_gemm<Dtype,Mtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, Mtype(-1.f),
      mean_.cpu_data(), sum_multiplier_.cpu_data(), Mtype(0.f),
      temp_.mutable_cpu_data());
    caffe_add<Dtype,Mtype>(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(MVNLayer);
#endif

INSTANTIATE_CLASS(MVNLayer);
REGISTER_LAYER_CLASS(MVN);

}  // namespace caffe
