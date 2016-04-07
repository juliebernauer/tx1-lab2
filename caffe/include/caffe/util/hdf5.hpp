#ifndef CAFFE_UTIL_HDF5_H_
#define CAFFE_UTIL_HDF5_H_

#include <string>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"

namespace caffe {

std::vector<int> 
hdf5_load_nd_dataset_helper(hid_t file_id, 
			    const char* dataset_name, int min_dim, int max_dim);

template<class Data>
herr_t hdf5_load(hid_t file_id, const string& dataset_name, Data* data);
  
template<class Blob>
void hdf5_load_nd_dataset(hid_t file_id, const char* dataset_name,
			  int min_dim, int max_dim, Blob* blob) {
  blob->Reshape(hdf5_load_nd_dataset_helper(file_id, dataset_name, min_dim, max_dim));
  herr_t status = hdf5_load(file_id, dataset_name, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read dataset " << dataset_name;
}
  
template<class Data>
herr_t hdf5_save(hid_t file_id, const string& dataset_name, 
		 int num_axes, hsize_t *dims, const Data* data);
  
template<class Blob>
void hdf5_save_nd_dataset(hid_t file_id, const string& dataset_name, const Blob& blob,
			  bool write_diff = false) {
  int num_axes = blob.num_axes();
  hsize_t dims[num_axes];
  for (int i = 0; i < num_axes; ++i) {
    dims[i] = blob.shape(i);
  }
  herr_t status = hdf5_save(file_id, dataset_name, num_axes, dims, 
			    write_diff ? blob.cpu_diff() : blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make dataset " << dataset_name;
}

int hdf5_load_int(hid_t loc_id, const string& dataset_name);
void hdf5_save_int(hid_t loc_id, const string& dataset_name, int i);
string hdf5_load_string(hid_t loc_id, const string& dataset_name);
void hdf5_save_string(hid_t loc_id, const string& dataset_name, const string& s);

int hdf5_get_num_links(hid_t loc_id);
string hdf5_get_name_by_idx(hid_t loc_id, int idx);

}  // namespace caffe

#endif   // CAFFE_UTIL_HDF5_H_
