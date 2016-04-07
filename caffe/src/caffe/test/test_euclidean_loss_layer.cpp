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
class EuclideanLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  EuclideanLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype,Mtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype,Mtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype,Mtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~EuclideanLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    EuclideanLossLayer<Dtype,Mtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Mtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Mtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    EuclideanLossLayer<Dtype,Mtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Mtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Mtype kErrorMargin = 1e-5;
    EXPECT_NEAR(Get<Mtype>(loss_weight_1) * kLossWeight,
        Get<Mtype>(loss_weight_2), tol<Dtype>(kErrorMargin));
    // Make sure the loss is non-trivial.
    const Mtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(Get<Mtype>(loss_weight_1)), kNonTrivialAbsThresh);
  }

  Blob<Dtype,Mtype>* const blob_bottom_data_;
  Blob<Dtype,Mtype>* const blob_bottom_label_;
  Blob<Dtype,Mtype>* const blob_top_loss_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(EuclideanLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(EuclideanLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  const Mtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  EuclideanLossLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-2), 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
