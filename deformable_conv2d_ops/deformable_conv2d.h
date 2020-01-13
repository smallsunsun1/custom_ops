//
// Created by 孙嘉禾 on 2019/12/31.
//

#ifndef TF_OPS_DEFORMABLE_CONV2D_H
#define TF_OPS_DEFORMABLE_CONV2D_H

#include <vector>
#include <iostream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
using TShape = std::vector<int>;

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

inline int ProdShape(const TShape &shape, int start, int end) {
    int res = 1;
    for (int i = start; i < end; ++i) {
        res *= shape[i];
    }
    return res;
}
inline int ProdShape(const TensorShape &shape, int start, int end) {
    int res = 1;
    for (int i = start; i < end; ++i) {
        res *= shape.dim_size(i);
    }
    return res;
}
template<typename Device, typename DType>
struct PureAddTo {
  void operator()(const Device &d, const int n, DType *result_data, const DType *right_data);
};
struct DeformableConv2DParameters {
  TShape dilations;
  TShape strides;
  Padding padding;
  int32_t num_groups;
  int32_t deformable_groups;
  int32_t im2col_step;
  bool no_bias;
  TensorFormat data_format;
};
struct DeformableConv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;
  int stride_rows;
  int stride_cols;
  int dilation_rows;
  int dilation_cols;
  int out_rows;
  int out_cols;
  int pad_rows;
  int pad_cols;
};
template<typename Device, typename T>
struct LaunchBatchMatMul;

template<typename Device, typename DType>
struct DeformableConv2DCol2ImCoord {
  void operator()(const Device &d, const DType *data_col, const DType *data_im, const DType *data_offset,
                  const DType *data_mask, const TShape &im_shape, const TShape &col_shape, const TShape &kernel_shape,
                  const TShape &pad, const TShape &stride, const TShape &dilation, const int32_t deformable_group,
                  DType *grad_offset, DType *grad_mask);
};
template<typename Device, typename DType>
struct SwapAxis {
  void operator()(const Device &d, DType *input_data, const TShape &origin_shape, const int axis_x, const int axis_y);
};
template<typename Device, typename DType>
struct DeformableConv2DCol2Im {
  void operator()(
      const Device &d,
      const DType *data_col, const DType *data_offset, const DType *data_mask,
      const TShape &im_shape, const TShape &col_shape, const TShape &kernel_shape,
      const TShape &pad, const TShape &stride,
      const TShape &dilation, const int32_t deformable_group,
      DType *grad_im
  );
};
template<typename Device, typename DType>
struct DeformableConv2DIm2Col {
  void operator()(
      const Device &d,
      const DType *data_im, const DType *data_offset, const DType *data_mask,
      const TShape &im_shape, const TShape &col_shape, const TShape &kernel_shape,
      const TShape &pad, const TShape &stride, const TShape &dilation,
      const int32_t deformable_group, DType *data_col
  );
};
template<typename Device, typename DType>
struct SetZeros {
  void operator()(const Device &d, int n, DType *result_data);
};
template<typename Device, typename DType>
struct SetOne {
  void operator()(const Device &d, int n, DType *result_data);
};
template<typename Device, typename DType>
struct SetNumAtIndex {
  void operator()(const Device &d, DType num, int index, DType *data);
};
#ifdef GOOGLE_CUDA
template <typename DType>
struct DeformableConv2DIm2Col<Eigen::GpuDevice, DType>{
    void operator()(
    const Eigen::GpuDevice& d,
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const int32_t deformable_group, DType* data_col
    );
};
template <typename DType>
struct DeformableConv2DCol2Im<Eigen::GpuDevice, DType>{
    void operator()(
    const Eigen::GpuDevice& d,
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_im
    );
};
template <typename DType>
struct DeformableConv2DCol2ImCoord<Eigen::GpuDevice, DType>{
    void operator()(
    const Eigen::GpuDevice& d, const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_offset, DType* grad_mask
    );
};
template <typename DType>
struct SetNumAtIndex<Eigen::GpuDevice, DType>{
    void operator()(const Eigen::GpuDevice& d, DType num, int index, DType* data);
};
template <typename DType>
struct SetZeros<Eigen::GpuDevice, DType>{
    void operator() (const Eigen::GpuDevice& d, int n, DType* result_data);
};
template <typename DType>
struct SetOne<Eigen::GpuDevice, DType>{
    void operator()(const Eigen::GpuDevice& d, int n, DType* result_data);
};
template <typename DType>
struct PureAddTo<Eigen::GpuDevice, DType>{
    void operator() (const Eigen::GpuDevice& d, const int n, DType* result_data, const DType* right_data);
};
#endif
template <typename DType>
struct DeformableConv2DIm2Col<Eigen::ThreadPoolDevice, DType>{
  void operator()(
      const Eigen::ThreadPoolDevice& d,
      const DType* data_im, const DType* data_offset, const DType* data_mask,
      const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
      const TShape& pad, const TShape& stride, const TShape& dilation,
      const int32_t deformable_group, DType* data_col
  );
};
template <typename DType>
struct DeformableConv2DCol2Im<Eigen::ThreadPoolDevice, DType>{
  void operator()(
      const Eigen::ThreadPoolDevice& d,
      const DType* data_col, const DType* data_offset, const DType* data_mask,
      const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
      const TShape& pad, const TShape& stride,
      const TShape& dilation, const int32_t deformable_group,
      DType* grad_im
  );
};
template <typename DType>
struct DeformableConv2DCol2ImCoord<Eigen::ThreadPoolDevice, DType>{
  void operator()(
      const Eigen::ThreadPoolDevice& d, const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
      const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
      const TShape& pad, const TShape& stride,
      const TShape& dilation, const int32_t deformable_group,
      DType* grad_offset, DType* grad_mask
  );
};
template <typename DType>
struct SetNumAtIndex<Eigen::ThreadPoolDevice, DType>{
  void operator()(const Eigen::ThreadPoolDevice& d, DType num, int index, DType* data);
};
template <typename DType>
struct SetZeros<Eigen::ThreadPoolDevice, DType>{
  void operator() (const Eigen::ThreadPoolDevice& d, int n, DType* result_data);
};
template <typename DType>
struct SetOne<Eigen::ThreadPoolDevice, DType>{
  void operator()(const Eigen::ThreadPoolDevice& d, int n, DType* result_data);
};
template <typename DType>
struct PureAddTo<Eigen::ThreadPoolDevice, DType>{
  void operator() (const Eigen::ThreadPoolDevice& d, const int n, DType* result_data, const DType* right_data);
};

template<typename T>
struct LaunchBatchMatMul<GPUDevice, T>{
  static void launch(OpKernelContext *context,
                     const TensorShape &in_x_shape,
                     const TensorShape &in_y_shape,
                     const T *in_x_ptr,
                     const T *in_y_ptr,
                     bool adj_x,
                     bool adj_y,
                     T *out);
};
template<typename T>
struct LaunchBatchMatMul<CPUDevice, T>{
  static void launch(OpKernelContext *context,
                     const TensorShape &in_x_shape,
                     const TensorShape &in_y_shape,
                     const T *in_x_ptr,
                     const T *in_y_ptr,
                     bool adj_x,
                     bool adj_y,
                     T *out);
};

}

#endif //TF_OPS_DEFORMABLE_CONV2D_H
