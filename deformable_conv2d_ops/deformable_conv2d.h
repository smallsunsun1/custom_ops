#ifndef KERNEL_DEFORMABLE_CONV_2D_H_
#define KERNEL_DEFORMABLE_CONV_2D_H_
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <vector>

namespace tensorflow{
typedef std::vector<int> TShape;

inline int ProdShape(const TShape &shape, int start, int end) {
    int res = 1;
    for(int i=start; i<end; i++) {
        res*=shape[i];
    }
    return res;
}    

inline int ProdShape(const TensorShape& shape, int start, int end){
    int res = 1;
    for(int i=start; i<end; i++) {
        res*=shape.dim_size(i);
    }
    return res;
}

template <typename Device, typename DType>
struct pureAddTo {
    void operator() (const Device& d, const int n, DType* result_data, const DType* right_data);
};

template <typename Device, typename Scalar>
struct LaunchBatchMatMul;

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

// Convolution dimensions inferred from parameters, input and filter tensors.
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

template <typename Device, typename DType>
struct DeformableConv2DCol2ImCoord{
    void operator()(
    const Device& d, const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_offset, DType* grad_mask
    );
};

template <typename Device, typename DType>
struct SwapAxis{
    void operator()(const Device& d, DType* input_data, const TShape& origin_shape, const int axis_x, const int axis_y);
};

template <typename Device, typename DType>
struct DeformableConv2DCol2Im{
    void operator()(
    const Device& d,
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_im
    );
};

template <typename Device, typename DType>
struct DeformableConv2DIm2Col{
    void operator()(
    const Device& d,
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const int32_t deformable_group, DType* data_col
    );
};

template <typename Device, typename DType>
struct setZero {
    void operator() (const Device& d, int n, DType* result_data);
};

template <typename Device, typename DType>
struct setOne {
    void operator()(const Device& d, int n, DType* result_data);
};

template <typename Device, typename DType>
struct setNumAtIndex{
    void operator()(const Device& d, DType num, int index, DType* data);
};


// 这里踩了一个深不见底的坑, c++很不熟悉的锅, 一开始我没有在h里部分特化,而是将部分特化放在了.cu.cc里,这样一来的话.cc里调用的时候一直都拿不到用
// GPUDEVICE特化的那个functor, 导致kernel一直没作用
// 此外, 还有一个坑是, 由于我只注册了GPU下的op,所以如果我想直接在cc里用Flat(Index index)这个重载来赋值的话是不行的, 因为包括.data()获得的指针的内容都是GPU
//下的地址, 即是显存的地址, 而cc是运行在cpu下的, 所以如果在cc里赋值的话, 会将值误赋到内存地址上, 会直接报段错误(Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)),
// 因为内存压根就没有分配空间, 所有allocate的空间都在显存上

#if GOOGLE_CUDA == 1
template <typename DType>
struct SwapAxis<Eigen::GpuDevice, DType>{
    void operator()(const Eigen::GpuDevice& d, DType* input_data, const TShape& origin_shape, const int axis_x, const int axis_y);
};
#endif

#if GOOGLE_CUDA == 1
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
#endif

#if GOOGLE_CUDA == 1
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
#endif

#if GOOGLE_CUDA == 1
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
#endif

#if GOOGLE_CUDA == 1
template <typename DType>
struct setNumAtIndex<Eigen::GpuDevice, DType>{
    void operator()(const Eigen::GpuDevice& d, DType num, int index, DType* data);
};
#endif

#if GOOGLE_CUDA == 1
template <typename DType>
struct setZero<Eigen::GpuDevice, DType>{
    void operator() (const Eigen::GpuDevice& d, int n, DType* result_data);
};
#endif

#if GOOGLE_CUDA == 1
template <typename DType>
struct setOne<Eigen::GpuDevice, DType>{
    void operator()(const Eigen::GpuDevice& d, int n, DType* result_data);
};
#endif

#if GOOGLE_CUDA == 1
template <typename DType>
struct pureAddTo<Eigen::GpuDevice, DType>{
    void operator() (const Eigen::GpuDevice& d, const int n, DType* result_data, const DType* right_data);
};
#endif


}
#endif