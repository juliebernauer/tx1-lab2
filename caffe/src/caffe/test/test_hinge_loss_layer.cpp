#include <cmath>
#include <cstdlib>
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
class HingeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  HingeLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype,Mtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype,Mtype>(10, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype,Mtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = Get<Dtype>(caffe_rng_rand() % 5);
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~HingeLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype,Mtype>* const blob_bottom_data_;
  Blob<Dtype,Mtype>* const blob_bottom_label_;
  Blob<Dtype,Mtype>* const blob_top_loss_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HingeLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(HingeLossLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  HingeLossLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(3e-2), Get<Dtype>(2e-3), 1701, Get<Dtype>(1.), Get<Dtype>(0.01));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(HingeLossLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  // Set norm to L2
  HingeLossParameter* hinge_loss_param = layer_param.mutable_hinge_loss_param();
  hinge_loss_param->set_norm(HingeLossParameter_Norm_L2);
  HingeLossLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-2), 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
