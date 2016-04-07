#include <algorithm>
#include <cstring>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NeuronLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  NeuronLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NeuronLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;

  void TestDropoutForward(const float dropout_ratio) {
    LayerParameter layer_param;
    // Fill in the given dropout_ratio, unless it's 0.5, in which case we don't
    // set it explicitly to test that 0.5 is the default.
    if (dropout_ratio != 0.5) {
      layer_param.mutable_dropout_param()->set_dropout_ratio(dropout_ratio);
    }
    DropoutLayer<Dtype,Mtype> layer(layer_param);
    layer_param.set_phase(TRAIN);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    Mtype scale = 1. / (1. - layer_param.dropout_param().dropout_ratio());
    const int count = this->blob_bottom_->count();
    // Initialize num_kept to count the number of inputs NOT dropped out.
    int num_kept = 0;
    for (int i = 0; i < count; ++i) {
      if (Get<Mtype>(top_data[i]) != 0) {
        ++num_kept;
        EXPECT_EQ(Get<Mtype>(top_data[i]), Get<Mtype>(bottom_data[i]) * scale);
      }
    }
    const Mtype std_error = sqrt(dropout_ratio * (1 - dropout_ratio) / count);
    // Fail if the number dropped was more than 1.96 * std_error away from the
    // expected number -- requires 95% confidence that the dropout layer is not
    // obeying the given dropout_ratio for test failure.
    const Mtype empirical_dropout_ratio = 1 - num_kept / Mtype(count);
    EXPECT_NEAR(empirical_dropout_ratio, dropout_ratio, 1.96 * std_error);
  }

  void TestExpForward(const float base, const float scale, const float shift) {
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base);
    layer_param.mutable_exp_param()->set_scale(scale);
    layer_param.mutable_exp_param()->set_shift(shift);
    ExpLayer<Dtype,Mtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    const float kDelta = 2e-4;
    const Dtype* bottom_data = blob_bottom_->cpu_data();
    const Dtype* top_data = blob_top_->cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      const float bottom_val = bottom_data[i];
      const float top_val = top_data[i];
      if (base == -1) {
        EXPECT_NEAR(top_val, exp(shift + scale * bottom_val), tol<Dtype>(kDelta));
      } else {
        EXPECT_NEAR(top_val, pow(base, shift + scale * bottom_val),
            sizeof(Dtype) == 2 ? tol<Dtype>(kDelta) * top_val : kDelta);
      }
    }
  }

  void TestExpGradient(const float base, const float scale, const float shift) {
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base);
    layer_param.mutable_exp_param()->set_scale(scale);
    layer_param.mutable_exp_param()->set_shift(shift);
    ExpLayer<Dtype,Mtype> layer(layer_param);
    GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
    checker.CheckGradientEltwise(&layer, blob_bottom_vec_, blob_top_vec_);
  }

  void TestPReLU(PReLULayer<Dtype,Mtype> *layer) {
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype* slope_data = layer->blobs()[0]->cpu_data();
    int hw = this->blob_bottom_->height() * this->blob_bottom_->width();
    int channels = this->blob_bottom_->channels();
    bool channel_shared = layer->layer_param().prelu_param().channel_shared();
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      int c = channel_shared ? 0 : (i / hw) % channels;
      EXPECT_NEAR(Get<Mtype>(top_data[i]),
          std::max(Get<Mtype>(bottom_data[i]), (Mtype)(0))
          + Get<Mtype>(slope_data[c]) * std::min(Get<Mtype>(bottom_data[i]), (Mtype)(0)),
          tol<Dtype>(1.e-6));
    }
  }

  void LogBottomInit() {
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
    caffe_exp<Dtype,Mtype>(this->blob_bottom_->count(), bottom_data, bottom_data);
  }

  void TestLogForward(const float base, const float scale, const float shift) {
    LogBottomInit();
    LayerParameter layer_param;
    layer_param.mutable_log_param()->set_base(base);
    layer_param.mutable_log_param()->set_scale(scale);
    layer_param.mutable_log_param()->set_shift(shift);
    LogLayer<Dtype,Mtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    const Mtype kDelta = Get<Mtype>(2e-4);
    const Dtype* bottom_data = blob_bottom_->cpu_data();
    const Dtype* top_data = blob_top_->cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      const Dtype bottom_val = bottom_data[i];
      const Dtype top_val = top_data[i];
      if (base == -1) {
        EXPECT_NEAR(Get<Mtype>(top_val), log(shift + scale * Get<Mtype>(bottom_val)),
                tol<Dtype>(kDelta));
      } else {
        EXPECT_NEAR(Get<Mtype>(top_val), log(shift + scale * Get<Mtype>(bottom_val)) / log(base),
                tol<Dtype>(kDelta));
      }
    }
  }

  void TestLogGradient(const float base, const float scale, const float shift) {
    LogBottomInit();
    LayerParameter layer_param;
    layer_param.mutable_log_param()->set_base(base);
    layer_param.mutable_log_param()->set_scale(scale);
    layer_param.mutable_log_param()->set_shift(shift);
    LogLayer<Dtype,Mtype> layer(layer_param);
    GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-2));
    checker.CheckGradientEltwise(&layer, blob_bottom_vec_, blob_top_vec_);
  }
};

TYPED_TEST_CASE(NeuronLayerTest, TestDtypesAndDevices);

TYPED_TEST(NeuronLayerTest, TestAbsVal) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  AbsValLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data    = this->blob_top_->cpu_data();
  const int count = this->blob_bottom_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(Get<Mtype>(top_data[i]), fabs(Get<Mtype>(bottom_data[i])));
  }
}

TYPED_TEST(NeuronLayerTest, TestAbsGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  AbsValLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestReLU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  ReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(Get<Mtype>(top_data[i]), 0.);
    EXPECT_TRUE(Get<Mtype>(top_data[i]) == 0 || Get<Mtype>(top_data[i]) == Get<Mtype>(bottom_data[i]));
  }
}

TYPED_TEST(NeuronLayerTest, TestReLUGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  ReLULayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestReLUWithNegativeSlope) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  ReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (Get<Mtype>(top_data[i]) >= 0) {
      EXPECT_FLOAT_EQ(Get<Mtype>(top_data[i]), Get<Mtype>(bottom_data[i]));
    } else {
      EXPECT_NEAR(Get<Mtype>(top_data[i]), Get<Mtype>(bottom_data[i]) * 0.01,
          tol<Dtype>(1.e-6));
    }
  }
}

TYPED_TEST(NeuronLayerTest, TestReLUGradientWithNegativeSlope) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  ReLULayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestSigmoid) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  SigmoidLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(Get<Mtype>(top_data[i]), 1. / (1 + exp(-Get<Mtype>(bottom_data[i]))),
        tol<Dtype>(1.e-6));
    // check that we squashed the value between 0 and 1
    EXPECT_GE(Get<Mtype>(top_data[i]), 0.);
    EXPECT_LE(Get<Mtype>(top_data[i]), 1.);
  }
}

TYPED_TEST(NeuronLayerTest, TestSigmoidGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  SigmoidLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestTanH) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  TanHLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test exact values
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      for (int k = 0; k < this->blob_bottom_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom_->width(); ++l) {
          EXPECT_GE(Get<Mtype>(this->blob_top_->data_at(i, j, k, l)) + tol<Dtype>(1e-4),
             (exp(2*Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) - 1) /
             (exp(2*Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) + 1));
          EXPECT_LE(Get<Mtype>(this->blob_top_->data_at(i, j, k, l)) - tol<Dtype>(1e-4),
             (exp(2*Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) - 1) /
             (exp(2*Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) + 1));
        }
      }
    }
  }
}

TYPED_TEST(NeuronLayerTest, TestTanHGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  TanHLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestExpLayer) {
  typedef typename TypeParam::Mtype Mtype;
  // Test default base of "-1" -- should actually set base := e.
  const Mtype kBase = -1;
  const Mtype kScale = 1;
  const Mtype kShift = 0;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradient) {
  typedef typename TypeParam::Mtype Mtype;
  // Test default base of "-1" -- should actually set base := e.
  const Mtype kBase = -1;
  const Mtype kScale = 1;
  const Mtype kShift = 0;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerBase2) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kBase = 2;
  const Mtype kScale = 1;
  const Mtype kShift = 0;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientBase2) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kBase = 2;
  const Mtype kScale = 1;
  const Mtype kShift = 0;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerBase2Shift1) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kBase = 2;
  const Mtype kScale = 1;
  const Mtype kShift = 1;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientBase2Shift1) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kBase = 2;
  const Mtype kScale = 1;
  const Mtype kShift = 1;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerBase2Scale3) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kBase = 2;
  const Mtype kScale = 3;
  const Mtype kShift = 0;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientBase2Scale3) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kBase = 2;
  const Mtype kScale = 3;
  const Mtype kShift = 0;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerBase2Shift1Scale3) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kBase = 2;
  const Mtype kScale = 3;
  const Mtype kShift = 1;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientBase2Shift1Scale3) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kBase = 2;
  const Mtype kScale = 3;
  const Mtype kShift = 1;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayer) {
  // Test default base of "-1" -- should actually set base := e.
  const float kBase = -1;
  const float kScale = 1;
  const float kShift = 0;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradient) {
  // Test default base of "-1" -- should actually set base := e.
  const float kBase = -1;
  const float kScale = 1;
  const float kShift = 0;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayerBase2) {
  const float kBase = 2;
  const float kScale = 1;
  const float kShift = 0;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradientBase2) {
  const float kBase = 2;
  const float kScale = 1;
  const float kShift = 0;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayerBase2Shift1) {
  const float kBase = 2;
  const float kScale = 1;
  const float kShift = 1;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradientBase2Shift1) {
  const float kBase = 2;
  const float kScale = 1;
  const float kShift = 1;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayerBase2Scale3) {
  const float kBase = 2;
  const float kScale = 3;
  const float kShift = 0;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradientBase2Scale3) {
  const float kBase = 2;
  const float kScale = 3;
  const float kShift = 0;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayerBase2Shift1Scale3) {
  const float kBase = 2;
  const float kScale = 3;
  const float kShift = 1;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradientBase2Shift1Scale3) {
  const float kBase = 2;
  const float kScale = 3;
  const float kShift = 1;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestDropoutHalf) {
  const float kDropoutRatio = 0.5;
  this->TestDropoutForward(kDropoutRatio);
}

TYPED_TEST(NeuronLayerTest, TestDropoutThreeQuarters) {
  const float kDropoutRatio = 0.75;
  this->TestDropoutForward(kDropoutRatio);
}

TYPED_TEST(NeuronLayerTest, TestDropoutTestPhase) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  DropoutLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (Get<Mtype>(top_data[i]) != 0) {
      EXPECT_EQ(Get<Mtype>(top_data[i]), Get<Mtype>(bottom_data[i]));
    }
  }
}

TYPED_TEST(NeuronLayerTest, TestDropoutGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  layer_param.set_phase(TRAIN);
  DropoutLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestDropoutGradientTest) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  DropoutLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestBNLL) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  BNLLLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(Get<Mtype>(top_data[i]), 0.);
    EXPECT_GE(Get<Mtype>(top_data[i]), Get<Mtype>(bottom_data[i]));
  }
}

TYPED_TEST(NeuronLayerTest, TestBNLLGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  BNLLLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestPReLUParam) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  PReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* slopes = layer.blobs()[0]->cpu_data();
  int count = layer.blobs()[0]->count();
  for (int i = 0; i < count; ++i, ++slopes) {
    EXPECT_EQ(Get<Mtype>(*slopes), 0.25);
  }
}

TYPED_TEST(NeuronLayerTest, TestPReLUForward) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  PReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype,Mtype> filler(filler_param);
  filler.Fill(layer.blobs()[0].get());
  this->TestPReLU(&layer);
}

TYPED_TEST(NeuronLayerTest, TestPReLUForwardChannelShared) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_channel_shared(true);
  PReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->TestPReLU(&layer);
}

TYPED_TEST(NeuronLayerTest, TestPReLUGradient) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  PReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype,Mtype> filler(filler_param);
  filler.Fill(layer.blobs()[0].get());
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestPReLUGradientChannelShared) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_channel_shared(true);
  PReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestPReLUConsistencyReLU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter prelu_layer_param;
  LayerParameter relu_layer_param;
  relu_layer_param.mutable_relu_param()->set_negative_slope(0.25);
  PReLULayer<Dtype,Mtype> prelu(prelu_layer_param);
  ReLULayer<Dtype,Mtype> relu(relu_layer_param);
  // Set up blobs
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_2;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_2;
  shared_ptr<Blob<Dtype,Mtype> > blob_bottom_2(new Blob<Dtype,Mtype>());
  shared_ptr<Blob<Dtype,Mtype> > blob_top_2(new Blob<Dtype,Mtype>());
  blob_bottom_vec_2.push_back(blob_bottom_2.get());
  blob_top_vec_2.push_back(blob_top_2.get());
  blob_bottom_2->CopyFrom(*this->blob_bottom_, false, true);
  // SetUp layers
  prelu.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  relu.SetUp(blob_bottom_vec_2, blob_top_vec_2);
  // Check forward
  prelu.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  relu.Forward(this->blob_bottom_vec_, blob_top_vec_2);
  for (int s = 0; s < blob_top_2->count(); ++s) {
    EXPECT_EQ(Get<Mtype>(this->blob_top_->cpu_data()[s]), Get<Mtype>(blob_top_2->cpu_data()[s]));
  }
  // Check backward
  shared_ptr<Blob<Dtype,Mtype> > tmp_blob(new Blob<Dtype,Mtype>());
  tmp_blob->ReshapeLike(*blob_top_2.get());
  FillerParameter filler_param;
  GaussianFiller<Dtype,Mtype> filler(filler_param);
  filler.Fill(tmp_blob.get());
  caffe_copy<Dtype,Mtype>(blob_top_2->count(), tmp_blob->cpu_data(),
      this->blob_top_->mutable_cpu_diff());
  caffe_copy<Dtype,Mtype>(blob_top_2->count(), tmp_blob->cpu_data(),
      blob_top_2->mutable_cpu_diff());
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  prelu.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  relu.Backward(blob_top_vec_2, propagate_down, blob_bottom_vec_2);
  for (int s = 0; s < blob_bottom_2->count(); ++s) {
    EXPECT_EQ(Get<Mtype>(this->blob_bottom_->cpu_diff()[s]), Get<Mtype>(blob_bottom_2->cpu_diff()[s]));
  }
}

TYPED_TEST(NeuronLayerTest, TestPReLUInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  // Set layer parameters
  LayerParameter ip_layer_param;
  LayerParameter prelu_layer_param;
  InnerProductParameter *ip_param =
      ip_layer_param.mutable_inner_product_param();
  ip_param->mutable_weight_filler()->set_type("gaussian");
  ip_param->set_num_output(3);
  InnerProductLayer<Dtype,Mtype> ip(ip_layer_param);
  PReLULayer<Dtype,Mtype> prelu(prelu_layer_param);
  InnerProductLayer<Dtype,Mtype> ip2(ip_layer_param);
  PReLULayer<Dtype,Mtype> prelu2(prelu_layer_param);
  // Set up blobs
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_2;
  vector<Blob<Dtype,Mtype>*> blob_middle_vec_2;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_2;
  shared_ptr<Blob<Dtype,Mtype> > blob_bottom_2(new Blob<Dtype,Mtype>());
  shared_ptr<Blob<Dtype,Mtype> > blob_middle_2(new Blob<Dtype,Mtype>());
  shared_ptr<Blob<Dtype,Mtype> > blob_top_2(new Blob<Dtype,Mtype>());
  blob_bottom_vec_2.push_back(blob_bottom_2.get());
  blob_middle_vec_2.push_back(blob_middle_2.get());
  blob_top_vec_2.push_back(blob_top_2.get());
  blob_bottom_2->CopyFrom(*this->blob_bottom_, false, true);
  // SetUp layers
  ip.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  prelu.SetUp(this->blob_top_vec_, this->blob_top_vec_);
  ip2.SetUp(blob_bottom_vec_2, blob_middle_vec_2);
  prelu2.SetUp(blob_middle_vec_2, blob_top_vec_2);
  caffe_copy<Dtype,Mtype>(ip2.blobs()[0]->count(), ip.blobs()[0]->cpu_data(),
      ip2.blobs()[0]->mutable_cpu_data());
  // Forward in-place
  ip.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  prelu.Forward(this->blob_top_vec_, this->blob_top_vec_);
  // Forward non-in-place
  ip2.Forward(blob_bottom_vec_2, blob_middle_vec_2);
  prelu2.Forward(blob_middle_vec_2, blob_top_vec_2);
  // Check numbers
  for (int s = 0; s < blob_top_2->count(); ++s) {
    EXPECT_EQ(Get<Mtype>(this->blob_top_->cpu_data()[s]), Get<Mtype>(blob_top_2->cpu_data()[s]));
  }
  // Fill top diff with random numbers
  shared_ptr<Blob<Dtype,Mtype> > tmp_blob(new Blob<Dtype,Mtype>());
  tmp_blob->ReshapeLike(*blob_top_2.get());
  FillerParameter filler_param;
  GaussianFiller<Dtype,Mtype> filler(filler_param);
  filler.Fill(tmp_blob.get());
  caffe_copy<Dtype,Mtype>(blob_top_2->count(), tmp_blob->cpu_data(),
      this->blob_top_->mutable_cpu_diff());
  caffe_copy<Dtype,Mtype>(blob_top_2->count(), tmp_blob->cpu_data(),
      blob_top_2->mutable_cpu_diff());
  // Backward in-place
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  prelu.Backward(this->blob_top_vec_, propagate_down, this->blob_top_vec_);
  ip.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  // Backward non-in-place
  prelu2.Backward(blob_top_vec_2, propagate_down, blob_middle_vec_2);
  ip2.Backward(blob_middle_vec_2, propagate_down, blob_bottom_vec_2);
  // Check numbers
  for (int s = 0; s < blob_bottom_2->count(); ++s) {
    EXPECT_EQ(Get<Mtype>(this->blob_bottom_->cpu_diff()[s]), Get<Mtype>(blob_bottom_2->cpu_diff()[s]));
  }
  for (int s = 0; s < ip.blobs()[0]->count(); ++s) {
    EXPECT_EQ(Get<Mtype>(ip.blobs()[0]->cpu_diff()[s]), Get<Mtype>(ip2.blobs()[0]->cpu_diff()[s]));
  }
  for (int s = 0; s < ip.blobs()[1]->count(); ++s) {
    EXPECT_EQ(Get<Mtype>(ip.blobs()[1]->cpu_diff()[s]), Get<Mtype>(ip2.blobs()[1]->cpu_diff()[s]));
  }
  for (int s = 0; s < prelu.blobs()[0]->count(); ++s) {
    EXPECT_EQ(Get<Mtype>(prelu.blobs()[0]->cpu_diff()[s]),
        Get<Mtype>(prelu2.blobs()[0]->cpu_diff()[s]));
  }
}

#ifdef USE_CUDNN
template <typename TypeParam>
class CuDNNNeuronLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  CuDNNNeuronLayerTest()
      : blob_bottom_(new Blob<Dtype,Mtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype,Mtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNNeuronLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
  vector<Blob<Dtype,Mtype>*> blob_bottom_vec_;
  vector<Blob<Dtype,Mtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNNeuronLayerTest, TestDtypes);

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CuDNNReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(Get<Mtype>(top_data[i]), 0.);
    EXPECT_TRUE(Get<Mtype>(top_data[i]) == 0 || Get<Mtype>(top_data[i]) == Get<Mtype>(bottom_data[i]));
  }
}

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUGradientCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CuDNNReLULayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUWithNegativeSlopeCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  CuDNNReLULayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (Get<Mtype>(top_data[i]) >= 0) {
      EXPECT_NEAR(Get<Mtype>(top_data[i]), Get<Mtype>(bottom_data[i]), tol<Dtype>(0.));
    } else {
      EXPECT_NEAR(Get<Mtype>(top_data[i]), Get<Mtype>(bottom_data[i]) * 0.01, tol<Dtype>(1.e-8));
    }
  }
}

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUGradientWithNegativeSlopeCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  CuDNNReLULayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestSigmoidCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CuDNNSigmoidLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(Get<Mtype>(top_data[i]), 1. / (1 + exp(-Get<Mtype>(bottom_data[i]))), tol<Dtype>(1.e-7));
    // check that we squashed the value between 0 and 1
    EXPECT_GE(Get<Mtype>(top_data[i]), 0.);
    EXPECT_LE(Get<Mtype>(top_data[i]), 1.);
  }
}

TYPED_TEST(CuDNNNeuronLayerTest, TestSigmoidGradientCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CuDNNSigmoidLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3), 1701, Get<Dtype>(0.), Get<Dtype>(0.01));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestTanHCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CuDNNTanHLayer<Dtype,Mtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test exact values
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      for (int k = 0; k < this->blob_bottom_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom_->width(); ++l) {
          EXPECT_GE(Get<Mtype>(this->blob_top_->data_at(i, j, k, l)) + tol<Dtype>(1e-4),
             (exp(2*Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) - 1) /
             (exp(2*Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) + 1));
          EXPECT_LE(Get<Mtype>(this->blob_top_->data_at(i, j, k, l)) - tol<Dtype>(1e-4),
             (exp(2*Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) - 1) /
             (exp(2*Get<Mtype>(this->blob_bottom_->data_at(i, j, k, l))) + 1));
        }
      }
    }
  }
}

TYPED_TEST(CuDNNNeuronLayerTest, TestTanHGradientCuDNN) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  LayerParameter layer_param;
  CuDNNTanHLayer<Dtype,Mtype> layer(layer_param);
  GradientChecker<Dtype,Mtype> checker(Get<Dtype>(1e-2), Get<Dtype>(1e-3));
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
#endif

}  // namespace caffe
