#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype, typename Mtype>
class Params {
 public:
  explicit Params(shared_ptr<Solver<Dtype,Mtype> > root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }

 protected:
  const size_t size_;           // Size of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient

DISABLE_COPY_AND_ASSIGN(Params);
};

// Params stored in GPU memory.
template<typename Dtype, typename Mtype>
class GPUParams : public Params<Dtype,Mtype> {
 public:
  GPUParams(shared_ptr<Solver<Dtype,Mtype> > root_solver, int device);
  virtual ~GPUParams();

  void configure(Solver<Dtype,Mtype>* solver) const;

 protected:
  using Params<Dtype,Mtype>::size_;
  using Params<Dtype,Mtype>::data_;
  using Params<Dtype,Mtype>::diff_;
 private:
  int buffer_device_;
};

class DevicePair {
 public:
  DevicePair(int parent, int device)
      : parent_(parent),
        device_(device) {
  }
  inline int parent() {
    return parent_;
  }
  inline int device() {
    return device_;
  }

  // Group GPUs in pairs, by proximity depending on machine's topology
  static void compute(const vector<int> devices, vector<DevicePair>* pairs);

 protected:
  int parent_;
  int device_;
};

// Synchronous data parallelism using map-reduce between local GPUs.
template<typename Dtype, typename Mtype>
class P2PSync : public GPUParams<Dtype,Mtype>, public Solver<Dtype,Mtype>::Callback,
    public InternalThread {
 public:
  explicit P2PSync(shared_ptr<Solver<Dtype,Mtype> > root_solver,
                   P2PSync<Dtype,Mtype>* parent, const SolverParameter& param);
  virtual ~P2PSync();

  inline const shared_ptr<Solver<Dtype,Mtype> >& solver() const {
    return solver_;
  }

  void run(const vector<int>& gpus);

 protected:
  void on_start();
  void on_gradients_ready();

  void InternalThreadEntry();

  P2PSync<Dtype,Mtype>* parent_;
  vector<P2PSync<Dtype,Mtype>*> children_;
  BlockingQueue<P2PSync<Dtype,Mtype>*> queue_;
  const int initial_iter_;
  Dtype* parent_grads_;
  shared_ptr<Solver<Dtype,Mtype> > solver_;

  using Params<Dtype,Mtype>::size_;
  using Params<Dtype,Mtype>::data_;
  using Params<Dtype,Mtype>::diff_;
};

}  // namespace caffe

#endif
