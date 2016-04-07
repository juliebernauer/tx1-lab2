#include <cmath>
#include <cstring>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MVNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  MVNLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MVNLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MVNLayerTest, TestDtypesAndDevices);

TYPED_TEST(MVNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  MVNLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      Mtype sum = 0, var = 0;
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Mtype data = Get<Mtype>(this->blob_top_->data_at(i, j, k, l));
          sum += data;
          var += data * data;
        }
      }
      sum /= height * width;
      var /= height * width;

      const Mtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }
}

TYPED_TEST(MVNLayerTest, TestForwardMeanOnly) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param{normalize_variance: false}", &layer_param));
  MVNLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      Mtype sum = 0, var = 0;
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Mtype data = Get<Mtype>(this->blob_top_->data_at(i, j, k, l));
          sum += data;
          var += data * data;
        }
      }
      sum /= height * width;

      const Mtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
    }
  }
}

TYPED_TEST(MVNLayerTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param{across_channels: true}", &layer_param));
  MVNLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    Mtype sum = 0, var = 0;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Mtype data = Get<Mtype>(this->blob_top_->data_at(i, j, k, l));
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * channels;
    var /= height * width * channels;

    const Mtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(MVNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  MVNLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(6e-2), Get<Dtype>(1e-3));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MVNLayerTest, TestGradientMeanOnly) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param{normalize_variance: false}", &layer_param));
  MVNLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(5e-2), Get<Dtype>(1e-3));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MVNLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param{across_channels: true}", &layer_param));
  MVNLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(6e-2), Get<Dtype>(1e-3));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
