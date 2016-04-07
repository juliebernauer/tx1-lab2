#include <cmath>
#include <cstring>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class RandomNumberGeneratorTest : public ::testing::Test {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
 protected:
  RandomNumberGeneratorTest()
     : mean_bound_multiplier_(3.8),  // ~99.99% confidence for test failure.
       sample_size_(10000),
       seed_(1701),
       data_(new SyncedMemory(sample_size_ * sizeof(Dtype))),
       data_2_(new SyncedMemory(sample_size_ * sizeof(Dtype))),
       int_data_(new SyncedMemory(sample_size_ * sizeof(int))),
       int_data_2_(new SyncedMemory(sample_size_ * sizeof(int))) {}

  virtual void SetUp() {
    Caffe::set_random_seed(this->seed_);
  }

  Mtype sample_mean(const Dtype* const seqs, const int sample_size) {
    Mtype sum = 0;
    for (int i = 0; i < sample_size; ++i) {
      sum += Get<Mtype>(seqs[i]);
    }
    return sum / sample_size;
  }

  Mtype sample_mean(const Dtype* const seqs) {
    return sample_mean(seqs, sample_size_);
  }

  Mtype sample_mean(const int* const seqs, const int sample_size) {
    Mtype sum = 0;
    for (int i = 0; i < sample_size; ++i) {
      sum += Get<Mtype>(seqs[i]);
    }
    return sum / sample_size;
  }

  Mtype sample_mean(const int* const seqs) {
    return sample_mean(seqs, sample_size_);
  }

  Mtype mean_bound(const Mtype std, const int sample_size) {
    return mean_bound_multiplier_ * std / sqrt(static_cast<Mtype>(sample_size));
  }

  Mtype mean_bound(const Mtype std) {
    return mean_bound(std, sample_size_);
  }

  void RngGaussianFill(const Mtype mu, const Mtype sigma, void* cpu_data) {
    Dtype* rng_data = static_cast<Dtype*>(cpu_data);
    caffe_rng_gaussian(sample_size_, mu, sigma, rng_data);
  }

  void RngGaussianChecks(const Mtype mu, const Mtype sigma,
                         const void* cpu_data, const Mtype sparse_p = 0) {
    const Dtype* rng_data = static_cast<const Dtype*>(cpu_data);
    const Mtype true_mean = mu;
    const Mtype true_std = sigma;
    // Check that sample mean roughly matches true mean.
    const Mtype bound = this->mean_bound(true_std);
    const Mtype sample_mean = this->sample_mean(
        static_cast<const Dtype*>(cpu_data));
    EXPECT_NEAR(sample_mean, true_mean, bound);
    // Check that roughly half the samples are above the true mean.
    int num_above_mean = 0;
    int num_below_mean = 0;
    int num_mean = 0;
    int num_nan = 0;
    for (int i = 0; i < sample_size_; ++i) {
      if (Get<Mtype>(rng_data[i]) > true_mean) {
        ++num_above_mean;
      } else if (Get<Mtype>(rng_data[i]) < true_mean) {
        ++num_below_mean;
      } else if (Get<Mtype>(rng_data[i]) == true_mean) {
        ++num_mean;
      } else {
        ++num_nan;
      }
    }
    EXPECT_EQ(0, num_nan);
    if (sparse_p == Mtype(0)) {
      if (sizeof(Dtype) == 2) {
        EXPECT_LE(num_mean, 1);
      } else {
        EXPECT_EQ(0, num_mean);
      }
    }
    const Mtype sample_p_above_mean =
        static_cast<Mtype>(num_above_mean) / sample_size_;
    const Mtype bernoulli_p = (1 - sparse_p) * 0.5;
    const Mtype bernoulli_std = sqrt(bernoulli_p * (1 - bernoulli_p));
    const Mtype bernoulli_bound = this->mean_bound(bernoulli_std);
    EXPECT_NEAR(bernoulli_p, sample_p_above_mean, bernoulli_bound);
  }

  void RngUniformFill(const Mtype lower, const Mtype upper, void* cpu_data) {
    CHECK_GE(upper, lower);
    Dtype* rng_data = static_cast<Dtype*>(cpu_data);
    caffe_rng_uniform<Dtype,Mtype>(sample_size_, lower, upper, rng_data);
  }

  void RngUniformChecks(const Mtype lower, const Mtype upper,
                        const void* cpu_data, const Mtype sparse_p = 0) {
    const Dtype* rng_data = static_cast<const Dtype*>(cpu_data);
    const Mtype true_mean = (Get<Mtype>(lower) + Get<Mtype>(upper)) / 2;
    const Mtype true_std = (Get<Mtype>(upper) - Get<Mtype>(lower)) / sqrt(12);
    // Check that sample mean roughly matches true mean.
    const Mtype bound = this->mean_bound(true_std);
    const Mtype sample_mean = this->sample_mean(rng_data);
    if (sizeof(Dtype) != 2) { //true_mean evaluates to 2147483648
      EXPECT_NEAR(sample_mean, true_mean, bound);
    }
    // Check that roughly half the samples are above the true mean, and none are
    // above upper or below lower.
    int num_above_mean = 0;
    int num_below_mean = 0;
    int num_mean = 0;
    int num_nan = 0;
    int num_above_upper = 0;
    int num_below_lower = 0;
    for (int i = 0; i < sample_size_; ++i) {
      if (Get<Mtype>(rng_data[i]) > true_mean) {
        ++num_above_mean;
      } else if (Get<Mtype>(rng_data[i]) < true_mean) {
        ++num_below_mean;
      } else if (Get<Mtype>(rng_data[i]) == true_mean) {
        ++num_mean;
      } else {
        ++num_nan;
      }
      if (Get<Mtype>(rng_data[i]) > upper) {
        ++num_above_upper;
      } else if (Get<Mtype>(rng_data[i]) < lower) {
        ++num_below_lower;
      }
    }
    EXPECT_EQ(0, num_nan);
    if (sizeof(Dtype) == 2) {
      EXPECT_LE(num_below_lower, 4);
    } else {
      EXPECT_EQ(0, num_below_lower);
    }
    if (sparse_p == Mtype(0)) {
      if (sizeof(Dtype) == 2) {
        EXPECT_LE(num_mean, 5);
      } else {
        EXPECT_EQ(0, num_above_upper);
        EXPECT_EQ(0, num_mean);
      }
    }
    const Mtype sample_p_above_mean =
        static_cast<Mtype>(num_above_mean) / sample_size_;
    const Mtype bernoulli_p = (1 - sparse_p) * 0.5;
    const Mtype bernoulli_std = sqrt(bernoulli_p * (1 - bernoulli_p));
    const Mtype bernoulli_bound = this->mean_bound(bernoulli_std);
    EXPECT_NEAR(bernoulli_p, sample_p_above_mean, tol<Dtype>(bernoulli_bound));
  }

  void RngBernoulliFill(const Mtype p, void* cpu_data) {
    int* rng_data = static_cast<int*>(cpu_data);
    caffe_rng_bernoulli<Dtype,Mtype>(sample_size_, p, rng_data);
  }

  void RngBernoulliChecks(const Mtype p, const void* cpu_data) {
    const int* rng_data = static_cast<const int*>(cpu_data);
    const Mtype true_mean = p;
    const Mtype true_std = sqrt(p * (1 - p));
    const Mtype bound = this->mean_bound(true_std);
    const Mtype sample_mean = this->sample_mean(rng_data);
    EXPECT_NEAR(sample_mean, true_mean, bound);
  }

#ifndef CPU_ONLY

  void RngGaussianFillGPU(const Mtype mu, const Mtype sigma, void* gpu_data) {
    Dtype* rng_data = static_cast<Dtype*>(gpu_data);
    caffe_gpu_rng_gaussian(sample_size_, mu, sigma, rng_data);
  }

  void RngUniformFillGPU(const Mtype lower, const Mtype upper, void* gpu_data) {
    CHECK_GE(upper, lower);
    Dtype* rng_data = static_cast<Dtype*>(gpu_data);
    caffe_gpu_rng_uniform<Dtype,Mtype>(sample_size_, lower, upper, rng_data);
  }

  // Fills with uniform integers in [0, UINT_MAX] using 2 argument form of
  // caffe_gpu_rng_uniform.
  void RngUniformIntFillGPU(void* gpu_data) {
    unsigned int* rng_data = static_cast<unsigned int*>(gpu_data);
    caffe_gpu_rng_uniform(sample_size_, rng_data);
  }

#endif

  int num_above_mean;
  int num_below_mean;

  Mtype mean_bound_multiplier_;

  size_t sample_size_;
  uint32_t seed_;

  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> data_2_;
  shared_ptr<SyncedMemory> int_data_;
  shared_ptr<SyncedMemory> int_data_2_;
};

TYPED_TEST_CASE(RandomNumberGeneratorTest, TestDtypes);

TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype mu = 0;
  const Mtype sigma = 1;
  void* gaussian_data = this->data_->mutable_cpu_data();
  this->RngGaussianFill(mu, sigma, gaussian_data);
  this->RngGaussianChecks(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian2) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype mu = -2;
  const Mtype sigma = 3;
  void* gaussian_data = this->data_->mutable_cpu_data();
  this->RngGaussianFill(mu, sigma, gaussian_data);
  this->RngGaussianChecks(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype lower = 0;
  const Mtype upper = 1;
  void* uniform_data = this->data_->mutable_cpu_data();
  this->RngUniformFill(lower, upper, uniform_data);
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform2) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype lower = -7.3;
  const Mtype upper = -2.3;
  void* uniform_data = this->data_->mutable_cpu_data();
  this->RngUniformFill(lower, upper, uniform_data);
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulli) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype p = 0.3;
  void* bernoulli_data = this->int_data_->mutable_cpu_data();
  this->RngBernoulliFill(p, bernoulli_data);
  this->RngBernoulliChecks(p, bernoulli_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulli2) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype p = 0.9;
  void* bernoulli_data = this->int_data_->mutable_cpu_data();
  this->RngBernoulliFill(p, bernoulli_data);
  this->RngBernoulliChecks(p, bernoulli_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianTimesGaussian) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  const Mtype mu = 0.;
  const Mtype sigma = 1.;

  // Sample from 0 mean Gaussian.
  Dtype* gaussian_data_1 =
      static_cast<Dtype*>(this->data_->mutable_cpu_data());
  this->RngGaussianFill(mu, sigma, gaussian_data_1);

  // Sample from 0 mean Gaussian again.
  Dtype* gaussian_data_2 =
      static_cast<Dtype*>(this->data_2_->mutable_cpu_data());
  this->RngGaussianFill(mu, sigma, gaussian_data_2);

  // Multiply Gaussians.
  for (int i = 0; i < this->sample_size_; ++i) {
    gaussian_data_1[i] = Get<Dtype>( Get<Mtype>(gaussian_data_1[i]) * Get<Mtype>(gaussian_data_2[i]));
  }

  // Check that result has mean 0.
  Mtype mu_product = pow(mu, 2);
  Mtype sigma_product = sqrt(pow(sigma, 2) / 2);
  this->RngGaussianChecks(mu_product, sigma_product, gaussian_data_1);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformTimesUniform) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  // Sample from Uniform on [-2, 2].
  const Mtype lower_1 = -2;
  const Mtype upper_1 = -lower_1;
  Dtype* uniform_data_1 =
      static_cast<Dtype*>(this->data_->mutable_cpu_data());
  this->RngUniformFill(lower_1, upper_1, uniform_data_1);

  // Sample from Uniform on [-3, 3].
  const Mtype lower_2 = -3;
  const Mtype upper_2 = -lower_2;
  Dtype* uniform_data_2 =
      static_cast<Dtype*>(this->data_2_->mutable_cpu_data());
  this->RngUniformFill(lower_2, upper_2, uniform_data_2);

  // Multiply Uniforms.
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data_1[i] = Get<Dtype>( Get<Mtype>(uniform_data_1[i]) * Get<Mtype>(uniform_data_2[i]));
  }

  // Check that result does not violate checked properties of Uniform on [-6, 6]
  // (though it is not actually uniformly distributed).
  const Mtype lower_prod = lower_1 * upper_2;
  const Mtype upper_prod = -lower_prod;
  this->RngUniformChecks(lower_prod, upper_prod, uniform_data_1);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianTimesBernoulli) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  // Sample from 0 mean Gaussian.
  const Mtype mu = 0;
  const Mtype sigma = 1;
  Dtype* gaussian_data =
      static_cast<Dtype*>(this->data_->mutable_cpu_data());
  this->RngGaussianFill(mu, sigma, gaussian_data);

  // Sample from Bernoulli with p = 0.3.
  const Mtype bernoulli_p = 0.3;
  int* bernoulli_data =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliFill(bernoulli_p, bernoulli_data);

  // Multiply Gaussian by Bernoulli.
  for (int i = 0; i < this->sample_size_; ++i) {
    gaussian_data[i] = Get<Dtype>( Get<Mtype>(gaussian_data[i]) * Get<Mtype>(bernoulli_data[i]));
  }

  // Check that result does not violate checked properties of sparsified
  // Gaussian (though it is not actually a Gaussian).
  this->RngGaussianChecks(mu, sigma, gaussian_data, 1 - bernoulli_p);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformTimesBernoulli) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  // Sample from Uniform on [-1, 1].
  const Mtype lower = -1;
  const Mtype upper = 1;
  Dtype* uniform_data =
      static_cast<Dtype*>(this->data_->mutable_cpu_data());
  this->RngUniformFill(lower, upper, uniform_data);

  // Sample from Bernoulli with p = 0.3.
  const Mtype bernoulli_p = 0.3;
  int* bernoulli_data =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliFill(bernoulli_p, bernoulli_data);

  // Multiply Uniform by Bernoulli.
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data[i] = Get<Dtype>( Get<Mtype>(uniform_data[i]) * Get<Mtype>(bernoulli_data[i]));
  }

  // Check that result does not violate checked properties of sparsified
  // Uniform on [-1, 1] (though it is not actually uniformly distributed).
  this->RngUniformChecks(lower, upper, uniform_data, 1 - bernoulli_p);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulliTimesBernoulli) {
  typedef typename TypeParam::Mtype Mtype;
  // Sample from Bernoulli with p = 0.5.
  const Mtype p_a = 0.5;
  int* bernoulli_data_a =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliFill(p_a, bernoulli_data_a);

  // Sample from Bernoulli with p = 0.3.
  const Mtype p_b = 0.3;
  int* bernoulli_data_b =
      static_cast<int*>(this->int_data_2_->mutable_cpu_data());
  this->RngBernoulliFill(p_b, bernoulli_data_b);

  // Multiply Bernoullis.
  for (int i = 0; i < this->sample_size_; ++i) {
    bernoulli_data_a[i] = Get<int>(Get<Mtype>(bernoulli_data_a[i]) * Get<Mtype>(bernoulli_data_b[i]));
  }
  int num_ones = 0;
  for (int i = 0; i < this->sample_size_; ++i) {
    if (Get<Mtype>(bernoulli_data_a[i]) != Mtype(0)) {
      EXPECT_EQ(Mtype(1), Get<Mtype>(bernoulli_data_a[i]));
      ++num_ones;
    }
  }

  // Check that resulting product has roughly p_a * p_b ones.
  const Mtype sample_p = this->sample_mean(bernoulli_data_a);
  const Mtype true_mean = p_a * p_b;
  const Mtype true_std = sqrt(true_mean * (1. - true_mean));
  const Mtype bound = this->mean_bound(true_std);
  EXPECT_NEAR(true_mean, sample_p, bound);
}

#ifndef CPU_ONLY

TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianGPU) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype mu = 0;
  const Mtype sigma = 1;
  void* gaussian_gpu_data = this->data_->mutable_gpu_data();
  this->RngGaussianFillGPU(mu, sigma, gaussian_gpu_data);
  const void* gaussian_data = this->data_->cpu_data();
  this->RngGaussianChecks(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian2GPU) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype mu = -2;
  const Mtype sigma = 3;
  void* gaussian_gpu_data = this->data_->mutable_gpu_data();
  this->RngGaussianFillGPU(mu, sigma, gaussian_gpu_data);
  const void* gaussian_data = this->data_->cpu_data();
  this->RngGaussianChecks(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformGPU) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype lower = 0;
  const Mtype upper = 1;
  void* uniform_gpu_data = this->data_->mutable_gpu_data();
  this->RngUniformFillGPU(lower, upper, uniform_gpu_data);
  const void* uniform_data = this->data_->cpu_data();
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform2GPU) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype lower = -7.3;
  const Mtype upper = -2.3;
  void* uniform_gpu_data = this->data_->mutable_gpu_data();
  this->RngUniformFillGPU(lower, upper, uniform_gpu_data);
  const void* uniform_data = this->data_->cpu_data();
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformIntGPU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  unsigned int* uniform_uint_gpu_data =
      static_cast<unsigned int*>(this->int_data_->mutable_gpu_data());
  this->RngUniformIntFillGPU(uniform_uint_gpu_data);
  const unsigned int* uniform_uint_data =
      static_cast<const unsigned int*>(this->int_data_->cpu_data());
  Dtype* uniform_data =
      static_cast<Dtype*>(this->data_->mutable_cpu_data());
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data[i] = Get<Dtype>(uniform_uint_data[i]);
  }
  const Mtype lower = 0;
  const Mtype upper = UINT_MAX;
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianTimesGaussianGPU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  const Mtype mu = 0;
  const Mtype sigma = 1;

  // Sample from 0 mean Gaussian.
  Dtype* gaussian_gpu_data_1 =
      static_cast<Dtype*>(this->data_->mutable_gpu_data());
  this->RngGaussianFillGPU(mu, sigma, gaussian_gpu_data_1);

  // Sample from 0 mean Gaussian again.
  Dtype* gaussian_gpu_data_2 =
      static_cast<Dtype*>(this->data_2_->mutable_gpu_data());
  this->RngGaussianFillGPU(mu, sigma, gaussian_gpu_data_2);

  // Multiply Gaussians.
  Dtype* gaussian_data_1 =
      static_cast<Dtype*>(this->data_->mutable_cpu_data());
  const Dtype* gaussian_data_2 =
      static_cast<const Dtype*>(this->data_2_->cpu_data());
  for (int i = 0; i < this->sample_size_; ++i) {
    gaussian_data_1[i] = Get<Dtype>( Get<Mtype>(gaussian_data_1[i]) * Get<Mtype>(gaussian_data_2[i]));
  }

  // Check that result does not violate checked properties of Gaussian
  // (though it is not actually a Gaussian).
  Mtype mu_product = pow(mu, 2);
  Mtype sigma_product = sqrt(pow(sigma, 2) / 2);
  this->RngGaussianChecks(mu_product, sigma_product, gaussian_data_1);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformTimesUniformGPU) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  // Sample from Uniform on [-2, 2].
  const Mtype lower_1 = -2;
  const Mtype upper_1 = -lower_1;
  Dtype* uniform_gpu_data_1 =
      static_cast<Dtype*>(this->data_->mutable_gpu_data());
  this->RngUniformFillGPU(lower_1, upper_1, uniform_gpu_data_1);

  // Sample from Uniform on [-3, 3].
  const Mtype lower_2 = -3;
  const Mtype upper_2 = -lower_2;
  Dtype* uniform_gpu_data_2 =
      static_cast<Dtype*>(this->data_2_->mutable_gpu_data());
  this->RngUniformFillGPU(lower_2, upper_2, uniform_gpu_data_2);

  // Multiply Uniforms.
  Dtype* uniform_data_1 =
      static_cast<Dtype*>(this->data_->mutable_cpu_data());
  const Dtype* uniform_data_2 =
      static_cast<const Dtype*>(this->data_2_->cpu_data());
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data_1[i] = Get<Dtype>( Get<Mtype>(uniform_data_1[i]) * Get<Mtype>(uniform_data_2[i]));
  }

  // Check that result does not violate properties of Uniform on [-7, -3].
  const Mtype lower_prod = lower_1 * upper_2;
  const Mtype upper_prod = -lower_prod;
  this->RngUniformChecks(lower_prod, upper_prod, uniform_data_1);
}

#endif

}  // namespace caffe
