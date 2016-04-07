#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template <typename TypeParam>
class GradientBasedSolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  GradientBasedSolverTest() :
      seed_(1701), num_(4), channels_(3), height_(10), width_(10),
      share_(false) {
        input_file_ = new string(
        CMAKE_SOURCE_DIR "caffe/test/test_data/solver_data_list.txt" CMAKE_EXT);
      }
  ~GradientBasedSolverTest() {
    delete input_file_;
  }

  string snapshot_prefix_;
  shared_ptr<SGDSolver<Dtype,Mtype> > solver_;
  shared_ptr<P2PSync<Dtype,Mtype> > sync_;
  int seed_;
  // Dimensions are determined by generate_sample_data.py
  // TODO this is brittle and the hdf5 file should be checked instead.
  int num_, channels_, height_, width_;
  bool share_;
  Dtype delta_;  // Stability constant for RMSProp, AdaGrad, AdaDelta and Adam

  // Test data: check out generate_sample_data.py in the same directory.
  string* input_file_;

  virtual SolverParameter_SolverType solver_type() = 0;
  virtual void InitSolver(const SolverParameter& param) = 0;

  virtual void InitSolverFromProtoString(const string& proto) {
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    // Set the solver_mode according to current Caffe::mode.
    switch (Caffe::mode()) {
      case Caffe::CPU:
        param.set_solver_mode(SolverParameter_SolverMode_CPU);
        break;
      case Caffe::GPU:
        param.set_solver_mode(SolverParameter_SolverMode_GPU);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
    }
    InitSolver(param);
    delta_ = Get<Dtype>((solver_type() == SolverParameter_SolverType_ADAGRAD) ?
         param.delta() : 0.F);
  }

  string RunLeastSquaresSolver(const Mtype learning_rate,
      const Mtype weight_decay, const Mtype momentum, const int num_iters,
      const int iter_size = 1, const int devices = 1,
      const bool snapshot = false, const char* from_snapshot = NULL) {
    ostringstream proto;
    int device_id = 0;
#ifndef CPU_ONLY
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaGetDevice(&device_id));
    }
#endif
    proto <<
       "snapshot_after_train: " << snapshot << " "
       "max_iter: " << num_iters << " "
       "base_lr: " << learning_rate << " "
       "lr_policy: 'fixed' "
       "iter_size: " << iter_size << " "
       "device_id: " << device_id << " "
       "net_param { "
       "  name: 'TestNetwork' "
       "  layer { "
       "    name: 'data' "
       "    type: 'HDF5Data' "
       "    hdf5_data_param { "
       "      source: '" << *(this->input_file_) << "' "
       "      batch_size: " << num_ / iter_size << " "
       "    } "
       "    top: 'data' "
       "    top: 'targets' "
       "  } ";
    if (share_) {
      proto <<
         "  layer { "
         "    name: 'slice' "
         "    type: 'Slice' "
         "    bottom: 'data' "
         "    top: 'data1' "
         "    top: 'data2' "
         "    slice_param { "
         "      axis: 0 "
         "    } "
         "  } ";
    }
    proto <<
       "  layer { "
       "    name: 'innerprod' "
       "    type: 'InnerProduct' "
       "    param { name: 'weights' } "
       "    param { name: 'bias' } "
       "    inner_product_param { "
       "      num_output: 1 "
       "      weight_filler { "
       "        type: 'gaussian' "
       "        std: 1.0 "
       "      } "
       "      bias_filler { "
       "        type: 'gaussian' "
       "        std: 1.0 "
       "      } "
       "    } "
       "    bottom: '" << string(share_ ? "data1": "data") << "' "
       "    top: '" << string(share_ ? "innerprod1": "innerprod") << "' "
       "  } ";
    if (share_) {
      proto <<
         "  layer { "
         "    name: 'innerprod2' "
         "    type: 'InnerProduct' "
         "    param { name: 'weights' } "
         "    param { name: 'bias' } "
         "    inner_product_param { "
         "      num_output: 1 "
         "      weight_filler { "
         "        type: 'gaussian' "
         "        std: 1.0 "
         "      } "
         "      bias_filler { "
         "        type: 'gaussian' "
         "        std: 1.0 "
         "      } "
         "    } "
         "    bottom: 'data2' "
         "    top: 'innerprod2' "
         "  } "
         "  layer { "
         "    name: 'concat' "
         "    type: 'Concat' "
         "    bottom: 'innerprod1' "
         "    bottom: 'innerprod2' "
         "    top: 'innerprod' "
         "    concat_param { "
         "      axis: 0 "
         "    } "
         "  } ";
    }
    proto <<
       "  layer { "
       "    name: 'loss' "
       "    type: 'EuclideanLoss' "
       "    bottom: 'innerprod' "
       "    bottom: 'targets' "
       "  } "
       "} ";
    if (weight_decay != 0) {
      proto << "weight_decay: " << weight_decay << " ";
    }
    if (momentum != 0) {
      proto << "momentum: " << momentum << " ";
    }
    MakeTempDir(&snapshot_prefix_);
    proto << "snapshot_prefix: '" << snapshot_prefix_ << "/' ";
    if (snapshot) {
      proto << "snapshot: " << num_iters << " ";
    }
    Caffe::set_random_seed(this->seed_);
    this->InitSolverFromProtoString(proto.str());
    if (from_snapshot != NULL) {
      this->solver_->Restore(from_snapshot);
      vector<Blob<Dtype,Mtype>*> empty_bottom_vec;
      for (int i = 0; i < this->solver_->iter(); ++i) {
        this->solver_->net()->Forward(empty_bottom_vec);
      }
    }
    if (devices == 1) {
      this->solver_->Solve();
    } else {
      LOG(INFO) << "Multi-GPU test on " << devices << " devices";
      vector<int> gpus;
      // put current device at the beginning
      int device_id = solver_->param().device_id();
      gpus.push_back(device_id);
      for (int i = 0; gpus.size() < devices; ++i) {
        if (i != device_id)
          gpus.push_back(i);
      }
      Caffe::set_solver_count(gpus.size());
      this->sync_.reset(new P2PSync<Dtype,Mtype>(
          this->solver_, NULL, this->solver_->param()));
      this->sync_->run(gpus);
      Caffe::set_solver_count(1);
    }
    if (snapshot) {
      ostringstream resume_file;
      resume_file << snapshot_prefix_ << "/_iter_" << num_iters
                  << ".solverstate";
      string resume_filename = resume_file.str();
      return resume_filename;
    }
    return string();
  }

  // Compute an update value given the current state of the train net,
  // using the analytical formula for the least squares gradient.
  // updated_params will store the updated weight and bias results,
  // using the blobs' diffs to hold the update values themselves.
  void ComputeLeastSquaresUpdate(const Mtype learning_rate,
      const Mtype weight_decay, const Mtype momentum, const int num_iters,
      vector<shared_ptr<Blob<Dtype,Mtype> > >* updated_params) {
    const int N = num_;
    const int D = channels_ * height_ * width_;

    // Run a forward pass, and manually compute the update values from the
    // result.
    Net<Dtype,Mtype>& net = *this->solver_->net();
    vector<Blob<Dtype,Mtype>*> empty_bottom_vec;
    net.Forward(empty_bottom_vec);
    ASSERT_TRUE(net.has_blob("data"));
    const Blob<Dtype,Mtype>& data = *net.blob_by_name("data");
    ASSERT_TRUE(net.has_blob("targets"));
    const Blob<Dtype,Mtype>& targets = *net.blob_by_name("targets");
    ASSERT_TRUE(net.has_layer("innerprod"));
    const vector<shared_ptr<Blob<Dtype,Mtype> > >& param_blobs =
        net.layer_by_name("innerprod")->blobs();
    const int num_param_blobs = 2;
    ASSERT_EQ(num_param_blobs, param_blobs.size());
    const Blob<Dtype,Mtype>& weights = *param_blobs[0];
    const Blob<Dtype,Mtype>& bias = *param_blobs[1];
    ASSERT_EQ(D * N, data.count());
    ASSERT_EQ(N, targets.count());
    ASSERT_EQ(D, weights.count());
    ASSERT_EQ(1, bias.count());

    updated_params->clear();
    updated_params->resize(num_param_blobs);
    for (int i = 0; i < num_param_blobs; ++i) {
      (*updated_params)[i].reset(new Blob<Dtype,Mtype>());
    }
    Blob<Dtype,Mtype>& updated_weights = *(*updated_params)[0];
    updated_weights.ReshapeLike(weights);
    Blob<Dtype,Mtype>& updated_bias = *(*updated_params)[1];
    updated_bias.ReshapeLike(bias);

    for (int i = 0; i <= D; ++i) {
      // Compute the derivative with respect to the ith weight (i.e., the ith
      // element of the gradient).
      Mtype grad = 0;
      for (int j = 0; j <= D; ++j) {
        // Compute element (i, j) of X^T * X.
        Mtype element = 0;
        for (int k = 0; k < N; ++k) {
          // (i, k) in X^T (== (k, i) in X) times (k, j) in X.
          const Mtype element_i = Get<Mtype>((i == D) ? Mtype(1.) : Get<Mtype>(data.cpu_data()[k * D + i]));
          const Mtype element_j = Get<Mtype>((j == D) ? Mtype(1.) : Get<Mtype>(data.cpu_data()[k * D + j]));
          element += element_i * element_j;
        }
        if (j == D) {
          grad += element * Get<Mtype>(bias.cpu_data()[0]);
        } else {
          grad += element * Get<Mtype>(weights.cpu_data()[j]);
        }
      }
      for (int k = 0; k < N; ++k) {
        const Mtype element_i = Get<Mtype>((i == D) ? Mtype(1.) : Get<Mtype>(data.cpu_data()[k * D + i]));
        grad -= element_i * Get<Mtype>(targets.cpu_data()[k]);
      }
      // Scale the gradient over the N samples.
      grad /= N;
      // Add the weight decay to the gradient.
      grad += weight_decay *
          Get<Mtype>(((i == D) ? Get<Mtype>(bias.cpu_data()[0]) : Get<Mtype>(weights.cpu_data()[i])));
      // Finally, compute update.
      const vector<shared_ptr<Blob<Dtype,Mtype> > >& history = solver_->history();
      if (solver_type() != SolverParameter_SolverType_ADADELTA
          && solver_type() != SolverParameter_SolverType_ADAM) {
        ASSERT_EQ(2, history.size());  // 1 blob for weights, 1 for bias
      } else {
        ASSERT_EQ(4, history.size());  // additional blobs for update history
      }
      Mtype update_value = learning_rate * grad;
      const Mtype history_value = (i == D) ?
    		  Get<Mtype>(history[1]->cpu_data()[0]) : Get<Mtype>(history[0]->cpu_data()[i]);
      const Mtype temp = momentum * history_value;
      switch (solver_type()) {
      case SolverParameter_SolverType_SGD:
        update_value += temp;
        break;
      case SolverParameter_SolverType_NESTEROV:
        update_value += temp;
        // step back then over-step
        update_value = (1 + momentum) * update_value - temp;
        break;
      case SolverParameter_SolverType_ADAGRAD:
        update_value /= std::sqrt(history_value + grad * grad) + Get<Mtype>(delta_);
        break;
      case SolverParameter_SolverType_RMSPROP: {
        const Mtype rms_decay = 0.95;
        update_value /= std::sqrt(rms_decay*history_value
            + grad * grad * (1 - rms_decay)) + Get<Mtype>(delta_);
        }
        break;
      case SolverParameter_SolverType_ADADELTA:
      {
        const Mtype update_history_value = (i == D) ?
        	Get<Mtype>(history[1 + num_param_blobs]->cpu_data()[0]) :
			Get<Mtype>(history[0 + num_param_blobs]->cpu_data()[i]);
        const Mtype weighted_gradient_average =
            momentum * history_value + (1 - momentum) * (grad * grad);
        update_value = grad * std::sqrt((update_history_value + Get<Mtype>(delta_)) /
            (weighted_gradient_average + Get<Mtype>(delta_))) * learning_rate;
        // not actually needed, just here for illustrative purposes
        // const Dtype weighted_update_average =
        //   momentum * update_history_value + (1 - momentum) * (update_value);
        break;
      }
      case SolverParameter_SolverType_ADAM: {
        const Mtype momentum2 = 0.999;
        const Mtype m = history_value;
        const Mtype v = (i == D) ?
        	Get<Mtype>(history[1 + num_param_blobs]->cpu_data()[0]) :
			Get<Mtype>(history[0 + num_param_blobs]->cpu_data()[i]);
        const Mtype val_m = (1 - momentum) * grad + momentum * m;
        const Mtype val_v = (1 - momentum2) * grad * grad + momentum2 * v;
        Mtype alpha_t = learning_rate *
            std::sqrt(Mtype(1) - pow(momentum2, num_iters)) /
            (Mtype(1.) - pow(momentum, num_iters));
        update_value = alpha_t * val_m / (std::sqrt(val_v) + Get<Mtype>(delta_));
        break;
      }
      default:
        LOG(FATAL) << "Unknown solver type: " << solver_type();
      }
      if (i == D) {
        updated_bias.mutable_cpu_diff()[0] = Get<Dtype>(update_value);
        updated_bias.mutable_cpu_data()[0] = bias.cpu_data()[0] - update_value;
      } else {
        updated_weights.mutable_cpu_diff()[i] = Get<Dtype>(update_value);
        updated_weights.mutable_cpu_data()[i] = weights.cpu_data()[i] - update_value;
      }
    }
  }

  void CheckLeastSquaresUpdate(
      const vector<shared_ptr<Blob<Dtype,Mtype> > >& updated_params) {
    const int D = channels_ * height_ * width_;

    const Blob<Dtype,Mtype>& updated_weights = *updated_params[0];
    const Blob<Dtype,Mtype>& updated_bias = *updated_params[1];

    Net<Dtype,Mtype>& net = *this->solver_->net();
    ASSERT_TRUE(net.has_layer("innerprod"));
    const vector<shared_ptr<Blob<Dtype,Mtype> > >& param_blobs =
        net.layer_by_name("innerprod")->blobs();
    ASSERT_EQ(2, param_blobs.size());
    const Blob<Dtype,Mtype>& solver_updated_weights = *param_blobs[0];
    ASSERT_EQ(D, solver_updated_weights.count());
    const double kPrecision = 5.e-1;
    const double kMinPrecision = 1e-7;
    for (int i = 0; i < D; ++i) {
      const Mtype expected_updated_weight = Get<Mtype>(updated_weights.cpu_data()[i]);
      const Mtype solver_updated_weight = Get<Mtype>(solver_updated_weights.cpu_data()[i]);
      const Mtype error_margin = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_updated_weight), fabs(solver_updated_weight)));
      EXPECT_NEAR(expected_updated_weight, solver_updated_weight, tol<Dtype>(error_margin));
    }
    const Blob<Dtype,Mtype>& solver_updated_bias_blob = *param_blobs[1];
    ASSERT_EQ(1, solver_updated_bias_blob.count());
    const Mtype expected_updated_bias = Get<Mtype>(updated_bias.cpu_data()[0]);
    const Mtype solver_updated_bias = Get<Mtype>(solver_updated_bias_blob.cpu_data()[0]);
    const Mtype error_margin = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_updated_bias), fabs(solver_updated_bias)));
    EXPECT_NEAR(expected_updated_bias, solver_updated_bias, tol<Dtype>(error_margin));

    // Check the solver's history -- should contain the previous update value.
    if (solver_type() == SolverParameter_SolverType_SGD) {
      const vector<shared_ptr<Blob<Dtype,Mtype> > >& history = solver_->history();
      ASSERT_EQ(2, history.size());
      for (int i = 0; i < D; ++i) {
        const Mtype expected_history = Get<Mtype>(updated_weights.cpu_diff()[i]);
        const Mtype solver_history = Get<Mtype>(history[0]->cpu_data()[i]);
        const Mtype error_margin_hist = std::max(kMinPrecision, kPrecision *
            std::min(fabs(expected_history), fabs(solver_history)));
        EXPECT_NEAR(expected_history, solver_history, tol<Dtype>(error_margin_hist));
      }
      const Mtype expected_history = Get<Mtype>(updated_bias.cpu_diff()[0]);
      const Mtype solver_history = Get<Mtype>(history[1]->cpu_data()[0]);
      const Mtype error_margin_hist = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_history), fabs(solver_history)));
      EXPECT_NEAR(expected_history, solver_history, error_margin_hist);
    }
  }

  void CheckAccumulation(const Mtype kLearningRate, const Mtype kWeightDecay,
      const Mtype kMomentum, const int kNumIters, const int kIterSize) {
    const double kPrecision = 1e-2;
    const double kMinPrecision = 1e-7;
    // Solve without accumulation and save parameters.
    this->RunLeastSquaresSolver(kLearningRate, kWeightDecay, kMomentum,
        kNumIters);
    // Save parameters for comparison.
    Net<Dtype,Mtype>& net = *this->solver_->net();
    const vector<shared_ptr<Blob<Dtype,Mtype> > >& param_blobs =
        net.layer_by_name("innerprod")->blobs();
    vector<shared_ptr<Blob<Dtype,Mtype> > > noaccum_params(param_blobs.size());
    for (int i = 0; i < param_blobs.size(); ++i) {
      noaccum_params[i].reset(new Blob<Dtype,Mtype>());
      noaccum_params[i]->CopyFrom(*param_blobs[i], false, true);
    }
    // Solve by equivalent accumulation of gradients over divided batches.
    this->RunLeastSquaresSolver(kLearningRate, kWeightDecay, kMomentum,
        kNumIters, kIterSize);
    Net<Dtype,Mtype>& net_accum = *this->solver_->net();
    const vector<shared_ptr<Blob<Dtype,Mtype> > >& accum_params =
        net_accum.layer_by_name("innerprod")->blobs();
    // Compare accumulated parameters against no accumulation standard.
    const int D = this->channels_ * this->height_ * this->width_;
    for (int i = 0; i < D; ++i) {
      const float expected_param = Get<float>(noaccum_params[0]->cpu_data()[i]);
      const float accum_param = Get<float>(accum_params[0]->cpu_data()[i]);
      const float error_margin = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_param), fabs(accum_param)));
      EXPECT_NEAR(expected_param, accum_param, tol<Dtype>(error_margin));
    }
    ASSERT_EQ(1, accum_params[1]->count());
    const float expected_bias = Get<float>(noaccum_params[1]->cpu_data()[0]);
    const float accum_bias = Get<float>(accum_params[1]->cpu_data()[0]);
    const float error_margin = std::max(kMinPrecision, kPrecision *
        std::min(fabs(expected_bias), fabs(accum_bias)));
    EXPECT_NEAR(expected_bias, accum_bias, error_margin);
  }

  // Test that the correct update is computed for a regularized least squares
  // problem:
  //
  //            E = (1/(2n)) || X w - y ||^2 + (lambda / 2) || w ||^2
  //   \nabla_w E = (1/n) (X^T X w - X^T y) + lambda * w
  //
  // X \in R^{n x (d+1)} (each example is a row, (d+1)th element is always 1)
  // w \in R^{(d+1) x 1} ((d+1)th element is the bias)
  // y \in R^{n x 1}
  // lambda is weight_decay
  //
  // TestLeastSquaresUpdate works "inductively", assuming that the solver
  // correctly updates the net K (= iter_to_check) times, then given the history
  // from the Kth update, we compute the (K+1)th update and check that it
  // matches the solver's (K+1)th update.
  void TestLeastSquaresUpdate(const Mtype learning_rate = 1.0,
      const Mtype weight_decay = 0.0, const Mtype momentum = 0.0,
      const int iter_to_check = 0) {
    const int kNum = num_;
    const int kIterSize = 1;
    // Test over all numbers of devices.
    int available_devices = 1;
#ifndef CPU_ONLY
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaGetDeviceCount(&available_devices));
    }
#endif
    for (int devices = 1; devices <= available_devices; ++devices) {
      // Configure batch size for single / multi device equivalence.
      // Constant data is needed for multi device as for accumulation.
      num_ = kNum * devices;

      // Initialize the solver and run K (= iter_to_check) solver iterations
      // (on single device).
      RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
                            iter_to_check, kIterSize, 1);

      // Compute the (K+1)th update using the analytic least squares gradient.
    vector<shared_ptr<Blob<Dtype,Mtype> > > updated_params;
      ComputeLeastSquaresUpdate(learning_rate, weight_decay, momentum,
          iter_to_check + 1, &updated_params);

      // Reinitialize the solver and run K+1 solver iterations.
      num_ = kNum;
      RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
          iter_to_check + 1, kIterSize, devices);

      // Check that the solver's solution matches ours.
      CheckLeastSquaresUpdate(updated_params);
    }
  }

  void TestSnapshot(const Mtype learning_rate = 1.0,
      const Mtype weight_decay = 0.0, const Mtype momentum = 0.0,
      const int num_iters = 1) {
    // Run the solver for num_iters * 2 iterations.
    const int total_num_iters = num_iters * 2;
    bool snapshot = false;
    const int kIterSize = 1;
    const int kDevices = 1;
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
        total_num_iters, kIterSize, kDevices, snapshot);

    // Save the resulting param values.
    vector<shared_ptr<Blob<Dtype,Mtype> > > param_copies;
    const vector<Blob<Dtype,Mtype>*>& orig_params =
        solver_->net()->learnable_params();
    param_copies.resize(orig_params.size());
    for (int i = 0; i < orig_params.size(); ++i) {
      param_copies[i].reset(new Blob<Dtype,Mtype>());
      const bool kReshape = true;
      for (int copy_diff = false; copy_diff <= true; ++copy_diff) {
        param_copies[i]->CopyFrom(*orig_params[i], copy_diff, kReshape);
      }
    }

    // Save the solver history
    vector<shared_ptr<Blob<Dtype,Mtype> > > history_copies;
    const vector<shared_ptr<Blob<Dtype,Mtype> > >& orig_history = solver_->history();
    history_copies.resize(orig_history.size());
    for (int i = 0; i < orig_history.size(); ++i) {
      history_copies[i].reset(new Blob<Dtype,Mtype>());
      const bool kReshape = true;
      for (int copy_diff = false; copy_diff <= true; ++copy_diff) {
        history_copies[i]->CopyFrom(*orig_history[i], copy_diff, kReshape);
      }
    }

    // Run the solver for num_iters iterations and snapshot.
    snapshot = true;
    string snapshot_name = RunLeastSquaresSolver(learning_rate, weight_decay,
        momentum, num_iters, kIterSize, kDevices, snapshot);

    // Reinitialize the solver and run for num_iters more iterations.
    snapshot = false;
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
        total_num_iters, kIterSize, kDevices,
        snapshot, snapshot_name.c_str());

    // Check that params now match.
    const vector<Blob<Dtype,Mtype>*>& params = solver_->net()->learnable_params();
    for (int i = 0; i < params.size(); ++i) {
      for (int j = 0; j < params[i]->count(); ++j) {
        EXPECT_EQ(param_copies[i]->cpu_data()[j], params[i]->cpu_data()[j])
            << "param " << i << " data differed at dim " << j;
        EXPECT_EQ(param_copies[i]->cpu_diff()[j], params[i]->cpu_diff()[j])
            << "param " << i << " diff differed at dim " << j;
      }
    }

    // Check that history now matches.
    const vector<shared_ptr<Blob<Dtype,Mtype> > >& history = solver_->history();
    for (int i = 0; i < history.size(); ++i) {
      for (int j = 0; j < history[i]->count(); ++j) {
        EXPECT_EQ(history_copies[i]->cpu_data()[j], history[i]->cpu_data()[j])
            << "history blob " << i << " data differed at dim " << j;
        EXPECT_EQ(history_copies[i]->cpu_diff()[j], history[i]->cpu_diff()[j])
            << "history blob " << i << " diff differed at dim " << j;
      }
    }
  }
};


template <typename TypeParam>
class SGDSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new SGDSolver<Dtype,Mtype>(param));
  }

  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_SGD;
  }
};

TYPED_TEST_CASE(SGDSolverTest, TestDtypesAndDevices);

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateLROneHundredth) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithWeightDecayMultiIter) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithMomentum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0;
  const Mtype kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0;
  const Mtype kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.5;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(SGDSolverTest, TestSnapshot) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}


template <typename TypeParam>
class AdaGradSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new AdaGradSolver<Dtype,Mtype>(param));
  }
  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_ADAGRAD;
  }
};

TYPED_TEST_CASE(AdaGradSolverTest, TestDtypesAndDevices);

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateLROneHundredth) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest,
      TestAdaGradLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaGradSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaGradSolverTest, TestSnapshot) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0);
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0);
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}


template <typename TypeParam>
class NesterovSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new NesterovSolver<Dtype,Mtype>(param));
  }
  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_NESTEROV;
  }
};

TYPED_TEST_CASE(NesterovSolverTest, TestDtypesAndDevices);

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateLROneHundredth) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(NesterovSolverTest,
           TestNesterovLeastSquaresUpdateWithWeightDecayMultiIter) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithMomentum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0;
  const Mtype kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0;
  const Mtype kMomentum = 0.5;
  const int kNumIters = sizeof(Dtype) == 2 ? 3 : 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest,
           TestNesterovLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(NesterovSolverTest, TestSnapshot) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template <typename TypeParam>
class AdaDeltaSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new AdaDeltaSolver<Dtype,Mtype>(param));
  }

  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_ADADELTA;
  }
};

TYPED_TEST_CASE(AdaDeltaSolverTest, TestDtypesAndDevices);

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdate) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.1;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.1;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.95;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithHalfMomentum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.1;
  const Mtype kWeightDecay = 0.0;
  const Mtype kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithMomentum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.1;
  const Mtype kWeightDecay = 0.0;
  const Mtype kMomentum = 0.95;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.1;
  const Mtype kWeightDecay = 0.0;
  const Mtype kMomentum = 0.95;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.1;
  const Mtype kWeightDecay = 0.1;
  const Mtype kMomentum = 0.95;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest,
           TestAdaDeltaLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.1;
  const Mtype kWeightDecay = 0.1;
  const Mtype kMomentum = 0.95;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.1);
  const Mtype kWeightDecay = Get<Mtype>(0.1);
  const Mtype kMomentum = Get<Mtype>(0.95);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.1);
  const Mtype kWeightDecay = Get<Mtype>(0.1);
  const Mtype kMomentum = Get<Mtype>(0.95);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaDeltaSolverTest, TestSnapshot) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.1);
  const Mtype kWeightDecay = Get<Mtype>(0.1);
  const Mtype kMomentum = Get<Mtype>(0.95);
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.1);
  const Mtype kWeightDecay = Get<Mtype>(0.1);
  const Mtype kMomentum = Get<Mtype>(0.95);
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template <typename TypeParam>
class AdamSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    SolverParameter new_param = param;
    const float momentum = 0.9F;
    new_param.set_momentum(momentum);
    const float momentum2 = 0.999F;
    new_param.set_momentum2(momentum2);
    this->solver_.reset(new AdamSolver<Dtype,Mtype>(new_param));
  }
  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_ADAM;
  }
};

TYPED_TEST_CASE(AdamSolverTest, TestDtypesAndDevices);

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdate) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0;
  const Mtype kMomentum = 0.9;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.9;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdamSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdamSolverTest, TestSnapshot) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.9);
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template <typename TypeParam>
class RMSPropSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  typedef typename TypeParam::Mtype Mtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    const float rms_decay = 0.95F;
    SolverParameter new_param = param;
    new_param.set_rms_decay(rms_decay);
    this->solver_.reset(new RMSPropSolver<Dtype,Mtype>(new_param));
  }
  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_RMSPROP;
  }
};

TYPED_TEST_CASE(RMSPropSolverTest, TestDtypesAndDevices);

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 1.0;
  const Mtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithRmsDecay) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.0;
  const Mtype kMomentum = 0.0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest,
      TestRMSPropLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = 0.01;
  const Mtype kWeightDecay = 0.5;
  const Mtype kMomentum = 0.0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.0);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(RMSPropSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0.0);
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(RMSPropSolverTest, TestSnapshot) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0);
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Mtype Mtype;
  const Mtype kLearningRate = Get<Mtype>(0.01);
  const Mtype kWeightDecay = Get<Mtype>(0.5);
  const Mtype kMomentum = Get<Mtype>(0);
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

}  // namespace caffe
