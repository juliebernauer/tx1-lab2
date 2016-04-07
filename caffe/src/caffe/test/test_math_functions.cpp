#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <climits>
#include <cmath>  // for std::fabs
#include <cstdlib>  // for rand_r

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MathFunctionsTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  MathFunctionsTest()
      : blob_bottom_(new Blob<Dtype,Mtype>()),
        blob_top_(new Blob<Dtype,Mtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(11, 17, 19, 23);
    this->blob_top_->Reshape(11, 17, 19, 23);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype,Mtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_top_);
  }

  virtual ~MathFunctionsTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  // http://en.wikipedia.org/wiki/Hamming_distance
  int ReferenceHammingDistance(const int n, const Dtype* x, const Dtype* y) {
    int dist = 0;
    uint64_t val;
    for (int i = 0; i < n; ++i) {
      if (sizeof(Dtype) == 8) {
        val = static_cast<uint64_t>(Get<Mtype>(x[i])) ^ static_cast<uint64_t>(Get<Mtype>(y[i]));
      } else if (sizeof(Dtype) == 4) {
        val = static_cast<uint32_t>(Get<Mtype>(x[i])) ^ static_cast<uint32_t>(Get<Mtype>(y[i]));
      } else if (sizeof(Dtype) == 2) {
        val = static_cast<uint16_t>(Get<Mtype>(x[i])) ^ static_cast<uint16_t>(Get<Mtype>(y[i]));
      } else {
        LOG(FATAL) << "Unrecognized Dtype size: " << sizeof(Dtype);
      }
      // Count the number of set bits
      while (val) {
        ++dist;
        val &= val - 1;
      }
    }
    return dist;
  }

  Blob<Dtype,Mtype>* const blob_bottom_;
  Blob<Dtype,Mtype>* const blob_top_;
};

template <typename TypeParam>
class CPUMathFunctionsTest
  : public MathFunctionsTest<CPUDevice<TypeParam> > {
};

TYPED_TEST_CASE(CPUMathFunctionsTest, TestDtypes);

TYPED_TEST(CPUMathFunctionsTest, TestNothing) {
  // The first test case of a test suite takes the longest time
  //   due to the set up overhead.
}

TYPED_TEST(CPUMathFunctionsTest, TestHammingDistance) {
  typedef typename TypeParam::Dtype Dtype;
  int n = this->blob_bottom_->count();
  const Dtype* x = this->blob_bottom_->cpu_data();
  const Dtype* y = this->blob_top_->cpu_data();
  int cpu_distance = caffe_cpu_hamming_distance(n, x, y);
  EXPECT_EQ(this->ReferenceHammingDistance(n, x, y),
            cpu_distance);
}

TYPED_TEST(CPUMathFunctionsTest, TestAsum) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  const Dtype* x = this->blob_bottom_->cpu_data();
  Mtype std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(Get<Mtype>(x[i]));
  }
  Mtype cpu_asum = caffe_cpu_asum<Dtype,Mtype>(n, x);
  EXPECT_LT((cpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(CPUMathFunctionsTest, TestSign) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  const Dtype* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sign<Dtype,Mtype>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const Dtype* signs = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(signs[i]), Get<Mtype>(Get<Mtype>(x[i]) > 0 ? 1 : (Get<Mtype>(x[i]) < 0 ? -1 : 0)));
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestSgnbit) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  const Dtype* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sgnbit<Dtype,Mtype>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const Dtype* signbits = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(signbits[i]), Get<Mtype>( Get<Mtype>(x[i]) < 0 ? 1 : 0) );
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestFabs) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  const Dtype* x = this->blob_bottom_->cpu_data();
  caffe_abs<Dtype>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const Dtype* abs_val = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(abs_val[i]), Get<Mtype>( Get<Mtype>(x[i]) > 0 ? x[i] : -x[i]) );
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestScale) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  Mtype alpha = Get<Mtype>(this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()]);
  caffe_cpu_scale<Dtype,Mtype>(n, alpha, this->blob_bottom_->cpu_data(),
                             this->blob_bottom_->mutable_cpu_diff());
  const Dtype* scaled = this->blob_bottom_->cpu_diff();
  const Dtype* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(scaled[i]), Get<Mtype>(x[i]) * alpha);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestCopy) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  const int n = this->blob_bottom_->count();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  Dtype* top_data = this->blob_top_->mutable_cpu_data();
  caffe_copy<Dtype,Mtype>(n, bottom_data, top_data);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(bottom_data[i]), Get<Mtype>(top_data[i]));
  }
}

#ifndef CPU_ONLY

template <typename TypeParam>
class GPUMathFunctionsTest : public MathFunctionsTest<GPUDevice<TypeParam> > {
};

TYPED_TEST_CASE(GPUMathFunctionsTest, TestDtypes);

// TODO: Fix caffe_gpu_hamming_distance and re-enable this test.
TYPED_TEST(GPUMathFunctionsTest, DISABLED_TestHammingDistance) {
  typedef typename TypeParam::Dtype Dtype;
  int n = this->blob_bottom_->count();
  const Dtype* x = this->blob_bottom_->cpu_data();
  const Dtype* y = this->blob_top_->cpu_data();
  int reference_distance = this->ReferenceHammingDistance(n, x, y);
  x = this->blob_bottom_->gpu_data();
  y = this->blob_top_->gpu_data();
  int computed_distance = caffe_gpu_hamming_distance(n, x, y);
  EXPECT_EQ(reference_distance, computed_distance);
}

TYPED_TEST(GPUMathFunctionsTest, TestAsum) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  const Dtype* x = this->blob_bottom_->cpu_data();
  Mtype std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(Get<Mtype>(x[i]));
  }
  Mtype gpu_asum;
  caffe_gpu_asum<Dtype,Mtype>(n, this->blob_bottom_->gpu_data(), &gpu_asum);
  EXPECT_LT((gpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(GPUMathFunctionsTest, TestSign) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  caffe_gpu_sign<Dtype,Mtype>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const Dtype* signs = this->blob_bottom_->cpu_diff();
  const Dtype* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(signs[i]), Get<Mtype>( Get<Mtype>(x[i]) > 0 ? 1 : (Get<Mtype>(x[i]) < 0 ? -1 : 0)) );
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestSgnbit) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  caffe_gpu_sgnbit<Dtype,Mtype>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const Dtype* signbits = this->blob_bottom_->cpu_diff();
  const Dtype* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(signbits[i]), Get<Mtype>(x[i]) < 0 ? 1 : 0);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestFabs) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  caffe_gpu_abs<Dtype,Mtype>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const Dtype* abs_val = this->blob_bottom_->cpu_diff();
  const Dtype* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(abs_val[i]), Get<Mtype>( Get<Mtype>(x[i]) > 0 ? Get<Mtype>(x[i]) : -Get<Mtype>(x[i])) );
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestScale) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  int n = this->blob_bottom_->count();
  Mtype alpha = Get<Mtype>(this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()]);
  caffe_gpu_scale<Dtype,Mtype>(n, alpha, this->blob_bottom_->gpu_data(),
                             this->blob_bottom_->mutable_gpu_diff());
  const Dtype* scaled = this->blob_bottom_->cpu_diff();
  const Dtype* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(scaled[i]), Get<Mtype>(x[i]) * alpha);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestCopy) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  const int n = this->blob_bottom_->count();
  const Dtype* bottom_data = this->blob_bottom_->gpu_data();
  Dtype* top_data = this->blob_top_->mutable_gpu_data();
  caffe_copy<Dtype,Mtype>(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(Get<Mtype>(bottom_data[i]), Get<Mtype>(top_data[i]));
  }
}

#endif


}  // namespace caffe
