#include <cmath>
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
class SoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  SoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        Mtype sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += Get<Mtype>(this->blob_top_->data_at(i, j, k, l));
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        Mtype scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l)));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(Get<Mtype>(this->blob_top_->data_at(i, j, k, l)) + tol<Dtype>(1e-4),
              exp(Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(Get<Mtype>(this->blob_top_->data_at(i, j, k, l)) - tol<Dtype>(1e-4),
              exp(Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(SoftmaxLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#ifdef USE_CUDNN
template <typename TypeParam>
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<TypeParam> {
 public:
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  CuDNNSoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNSoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNSoftmaxLayerTest, TestDtypes);

TYPED_TEST(CuDNNSoftmaxLayerTest, TestForwardCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        Mtype sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += Get<Mtype>(this->blob_top_->data_at(i, j, k, l));
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        Mtype scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l)));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(Get<Mtype>(this->blob_top_->data_at(i, j, k, l)) + tol<Dtype>(1e-4),
              exp(Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(Get<Mtype>(this->blob_top_->data_at(i, j, k, l)) - tol<Dtype>(1e-4),
              exp(Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(CuDNNSoftmaxLayerTest, TestGradientCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#endif

}  // namespace caffe
