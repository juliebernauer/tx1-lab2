#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MaxPoolingDropoutTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  MaxPoolingDropoutTest()
      : blob_bottom_(new Blob<Dtype,Mtype>()),
        blob_top_(new Blob<Dtype,Mtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1703);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    ConstantFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MaxPoolingDropoutTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MaxPoolingDropoutTest, TestDtypesAndDevices);

TYPED_TEST(MaxPoolingDropoutTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  PoolingLayer<Dtype,Mtype> max_layer(layer_param);
  max_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  DropoutLayer<Dtype,Mtype> dropout_layer(layer_param);
  dropout_layer.SetUp(this->blob_top_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}


TYPED_TEST(MaxPoolingDropoutTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  PoolingLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  Mtype sum = 0.;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    sum += Get<Mtype>(top_data[i]);
  }
  EXPECT_EQ(sum, this->blob_top_->count());
  // Dropout in-place
  DropoutLayer<Dtype,Mtype> dropout_layer(layer_param);
  dropout_layer.SetUp(this->blob_top_vec_, this->blob_top_vec_);
  dropout_layer.Forward(this->blob_top_vec_, this->blob_top_vec_);
  sum = 0.;
  Mtype scale = 1. / (1. - layer_param.dropout_param().dropout_ratio());
  top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    sum += Get<Mtype>(top_data[i]);
  }
  EXPECT_GE(sum, 0);
  EXPECT_LE(sum, this->blob_top_->count()*scale);
}

TYPED_TEST(MaxPoolingDropoutTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  layer_param.set_phase(TRAIN);
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);
  PoolingLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = Get<Dtype>(1.);
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  const Dtype* bottom_diff = this->blob_bottom_->cpu_diff();
  Mtype sum = 0.;
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    sum += Get<Mtype>(bottom_diff[i]);
  }
  EXPECT_EQ(sum, this->blob_top_->count());
  // Dropout in-place
  DropoutLayer<Dtype,Mtype> dropout_layer(layer_param);
  dropout_layer.SetUp(this->blob_top_vec_, this->blob_top_vec_);
  dropout_layer.Forward(this->blob_top_vec_, this->blob_top_vec_);
  dropout_layer.Backward(this->blob_top_vec_, propagate_down,
                         this->blob_top_vec_);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  Mtype sum_with_dropout = 0.;
  bottom_diff = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    sum_with_dropout += Get<Mtype>(bottom_diff[i]);
  }
  EXPECT_GE(sum_with_dropout, sum);
}

}  // namespace caffe
