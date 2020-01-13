//
// Created by 孙嘉禾 on 2019/12/31.
//

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

// This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
//#define __CUDACC__

// These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
// we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
// a terrible idea :D
//#include <__clang_cuda_builtin_vars.h>
//#include <__clang_cuda_intrinsics.h>
//#include <__clang_cuda_math_forward_declares.h>
//#include <__clang_cuda_complex_builtins.h>
//#include <__clang_cuda_cmath.h>
#endif // __JETBRAINS_IDE__

#include "deformable_conv2d.h"
#include <cstdlib>
#include "tensorflow/core/util/gpu_kernel_helper.h"

#ifdef GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x, bool adj_y) {
    return Eigen::IndexPair<Eigen::DenseIndex>(adj_x ? 0 : 1, adj_y ? 1 : 0);
}

template<typename DType>
__host__ __device__ DType DmcnIm2colBilinear(const DType *bottom_data,
                                             const int data_width,
                                             const int height,
                                             const int width,
                                             DType h,
                                             DType w) {
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    DType lh = h - h_low;
    DType lw = w - w_low;
    DType hh = 1 - lh, hw = 1 - lw;

    DType v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low * data_width + w_low];
    DType v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low * data_width + w_high];
    DType v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high * data_width + w_low];
    DType v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high * data_width + w_high];

    DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}
template<typename DType>
__host__ __device__ DType DmcnGetGradientWeight(DType argmax_h,
                                                DType argmax_w,
                                                const int h,
                                                const int w,
                                                const int height,
                                                const int width) {
    /*
     * offset h, offset w, (h, w) coordinate
     */
    if (argmax_h <= -1 || argmax_w <= -1 || argmax_h >= height || argmax_w >= width) {
        return 0;
    }
    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;
    DType weight = 0;
    if (h == argmax_h_low && w == argmax_w_low)
        weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    if (h == argmax_h_low && w == argmax_w_high)
        weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    if (h == argmax_h_high && w == argmax_w_low)
        weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    if (h == argmax_h_high && w == argmax_w_high)
        weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    return weight;
}
template<typename DType>
__host__ __device__ DType DmcnGetCoordinateWeight(DType argmax_h,
                                                  DType argmax_w,
                                                  const int height,
                                                  const int width,
                                                  const DType *im_data,
                                                  const int data_width,
                                                  const int bp_dir) {
    if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
        //empty
        return 0;
    }

    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;

    DType weight = 0;

    if (bp_dir == 0) {
        if (argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
        if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
            weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
        if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
            weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
        if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
            weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
    } else if (bp_dir == 1) {
        if (argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
        if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
            weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
        if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
            weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
        if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
            weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
    }

    return weight;
}

#ifdef GOOGLE_CUDA
template<typename DType>
__global__ void SwapAxisKernel(const int n, const int cuda_mem_size, const int min_unit_size,
                               DType *input_data, const int dim_num, const int axis_x_dims, const int axis_y_dims,
                               const int axis_x, const int axis_y) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        DType *device_data = new DType[cuda_mem_size];
        DType *input_data_ptr = input_data + index * cuda_mem_size;
        for (int j = 0; j < axis_y_dims; j++) {
            for (int i = 0; i < axis_x_dims; i++) {
                DType *temp_ptr = input_data_ptr + (i * axis_x_dims + j) * min_unit_size;
                DType *device_data_temp_ptr = device_data + (j * axis_y_dims + i) * min_unit_size;
                for (int k = 0; k < min_unit_size; k++) {
                    *(device_data_temp_ptr + k) = *(temp_ptr + k);
                }
            }
        }
        for (int i = 0; i < cuda_mem_size; i++) {
            *(input_data_ptr + i) = *(device_data + i);
        }
        delete[]device_data;
    }
}
#endif

template<typename DType>
void SwapAxisKernel(const CPUDevice &d, const int n, const int cuda_mem_size, const int min_unit_size,
                    DType *input_data, const int dim_num, const int axis_x_dims, const int axis_y_dims,
                    const int axis_x, const int axis_y) {
    d.parallelFor(n,
                  Eigen::TensorOpCost(cuda_mem_size, cuda_mem_size, cuda_mem_size * axis_y_dims * axis_x_dims),
                  [min_unit_size, input_data, dim_num, axis_x_dims, axis_y_dims,
                      axis_x, axis_y, cuda_mem_size](int64 start, int64 end) {
                    for (int64 index = start; index < end; index++) {
                        DType *device_data = new DType[cuda_mem_size];
                        DType *input_data_ptr = input_data + index * cuda_mem_size;
                        for (int j = 0; j < axis_y_dims; j++) {
                            for (int i = 0; i < axis_x_dims; i++) {
                                DType *temp_ptr = input_data_ptr + (i * axis_x_dims + j) * min_unit_size;
                                DType *device_data_temp_ptr = device_data + (j * axis_y_dims + i) * min_unit_size;
                                for (int k = 0; k < min_unit_size; k++) {
                                    *(device_data_temp_ptr + k) = *(temp_ptr + k);
                                }
                            }
                        }
                        for (int idx = 0; idx < cuda_mem_size; idx++) {
                            *(input_data_ptr + idx) = *(device_data + idx);
                        }
                        delete[] device_data;
                    }
                  });
}
#ifdef GOOGLE_CUDA
template<typename DType>
__global__ void DeformableConv2DIm2ColKernel(const int n,
                                             const DType *data_im,
                                             const DType *data_offset,
                                             const DType *data_mask,
                                             const int height,
                                             const int width,
                                             const int kernel_h,
                                             const int kernel_w,
                                             const int pad_h,
                                             const int pad_w,
                                             const int stride_h,
                                             const int stride_w,
                                             const int dilation_h,
                                             const int dilation_w,
                                             const int channel_per_deformable_group,
                                             const int batch_size,
                                             const int num_channels,
                                             const int deformable_group,
                                             const int height_col,
                                             const int width_col,
                                             DType *data_col) {
    /*
     * channel_per_deformable_group // 输入图通道数除以deformable_group的数量,
     * //这里的batch_size代表的是im2col_step_, 一般就设为1了
     */
    CUDA_1D_KERNEL_LOOP(index, n) {
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int b_col = (index / width_col / height_col) % batch_size;
        const int c_im = (index / width_col / height_col) / batch_size;
        const int c_col = c_im * kernel_h * kernel_w;
        // compute deformable group index
        const int deformable_group_index = c_im / channel_per_deformable_group;
        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        DType *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
        const DType *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
        const DType *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
            kernel_h * kernel_w * height_col * width_col;
        const DType *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
                const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
                const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
                const DType offset_h = data_offset_ptr[data_offset_h_ptr];
                const DType offset_w = data_offset_ptr[data_offset_w_ptr];
                const DType mask = data_mask_ptr[data_mask_hw_ptr];
                DType val = static_cast<DType>(0);
                const DType h_im = h_in + i * dilation_h + offset_h;
                const DType w_im = w_in + j * dilation_w + offset_w;
                if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
                    val = DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);
                }
                *data_col_ptr = val * mask;
                data_col_ptr += batch_size * height_col * width_col;
            }
        }
    }
}
#endif
template<typename DType>
void DeformableConv2DIm2ColCPUKernel(const CPUDevice &d,
                                     const int n,
                                     const DType *data_im,
                                     const DType *data_offset,
                                     const DType *data_mask,

                                     const int height,
                                     const int width,
                                     const int kernel_h,
                                     const int kernel_w,
                                     const int pad_h,
                                     const int pad_w,
                                     const int stride_h,
                                     const int stride_w,
                                     const int dilation_h,
                                     const int dilation_w,

                                     const int channel_per_deformable_group, // 输入图通道数除以deformable_group的数量,
                                     const int batch_size,
                                     const int num_channels,
                                     const int deformable_group, //这里的batch_size代表的是im2col_step_, 一般就设为1了
                                     const int height_col,
                                     const int width_col,
                                     DType *data_col) {
    auto f = [n, data_im, data_offset, data_mask, height, width, kernel_h, kernel_w,
        pad_h, pad_w, stride_w, stride_h, dilation_w, dilation_h, channel_per_deformable_group,
        batch_size, num_channels, deformable_group, height_col, width_col, data_col](int64 start, int64 end) {
      for (int64 index = start; index < end; index++) {
          const int w_col = index % width_col;
          const int h_col = (index / width_col) % height_col;
          const int b_col = (index / width_col / height_col) % batch_size;
          const int c_im = (index / width_col / height_col) / batch_size;
          const int c_col = c_im * kernel_h * kernel_w;

          // compute deformable group index
          const int deformable_group_index = c_im / channel_per_deformable_group;

          const int h_in = h_col * stride_h - pad_h;
          const int w_in = w_col * stride_w - pad_w;

          DType *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
          const DType *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
          const DType *data_offset_ptr = data_offset
              + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col
                  * width_col; //

          const DType *data_mask_ptr = data_mask
              + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col; //
          for (int i = 0; i < kernel_h; ++i) {
              for (int j = 0; j < kernel_w; ++j) {
                  const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
                  const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
                  const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
                  const DType offset_h = data_offset_ptr[data_offset_h_ptr];
                  const DType offset_w = data_offset_ptr[data_offset_w_ptr];
                  const DType mask = data_mask_ptr[data_mask_hw_ptr];
                  DType val = static_cast<DType>(0);
                  const DType h_im = h_in + i * dilation_h + offset_h;
                  const DType w_im = w_in + j * dilation_w + offset_w;
                  if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
                      val = DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);
                  }
                  *data_col_ptr = val * mask;
                  data_col_ptr += batch_size * height_col * width_col;
              }
          }
      }
    };
    d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
#ifdef GOOGLE_CUDA
template<typename DType>
__global__ void DeformableConv2DCol2ImKernel(
    const int n,
    const DType *data_col, const DType *data_offset, const DType *data_mask,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int deformable_group,
    const int height_col, const int width_col,
    DType *grad_im) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        const int j = (index / width_col / height_col / batch_size) % kernel_w;
        const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
        const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
        // compute the start and end of the output
        const int deformable_group_index = c / channel_per_deformable_group;

        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int b = (index / width_col / height_col) % batch_size;
        int w_in = w_out * stride_w - pad_w;
        int h_in = h_out * stride_h - pad_h;

        const DType *data_offset_ptr = data_offset
            + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        const DType *data_mask_ptr =
            data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        const DType mask = data_mask_ptr[data_mask_hw_ptr];
        const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
        const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

        const DType cur_top_grad = data_col[index] * mask;
        const int cur_h = (int) cur_inv_h_data;
        const int cur_w = (int) cur_inv_w_data;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                if (cur_h + dy >= 0 && cur_h + dy < height &&
                    cur_w + dx >= 0 && cur_w + dx < width &&
                    abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
                    abs(cur_inv_w_data - (cur_w + dx)) < 1
                    ) {
                    int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
                    DType weight =
                        DmcnGetGradientWeight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
                    CudaAtomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
                }
            }
        }
    }
}
#endif

template<typename T>
void MutexAdd(T *address, T val) {
    static mutex mu;
    std::lock_guard<mutex> lock(mu);
    (*address) += val;
}

template<typename DType>
void DeformableConv2DCol2ImCPUKernel(const CPUDevice &d, const int n,
                                     const DType *data_col, const DType *data_offset, const DType *data_mask,
                                     const int channels, const int height, const int width,
                                     const int kernel_h, const int kernel_w,
                                     const int pad_h, const int pad_w,
                                     const int stride_h, const int stride_w,
                                     const int dilation_h, const int dilation_w,
                                     const int channel_per_deformable_group,
                                     const int batch_size, const int deformable_group,
                                     const int height_col, const int width_col,
                                     DType *grad_im) {
    auto f = [n, data_col, data_offset, data_mask, channels, height, width, kernel_h,
        kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        channel_per_deformable_group, batch_size, deformable_group, height_col, width_col, grad_im](int64 start,
                                                                                                    int64 end) {
      for (int64 index = start; index < end; ++index) {
          const int j = (index / width_col / height_col / batch_size) % kernel_w;
          const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
          const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
          // compute the start and end of the output
          const int deformable_group_index = c / channel_per_deformable_group;
          int w_out = index % width_col;
          int h_out = (index / width_col) % height_col;
          int b = (index / width_col / height_col) % batch_size;
          int w_in = w_out * stride_w - pad_w;
          int h_in = h_out * stride_h - pad_h;

          const DType *data_offset_ptr = data_offset
              + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
          const DType *data_mask_ptr = data_mask
              + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
          const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
          const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
          const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
          const DType offset_h = data_offset_ptr[data_offset_h_ptr];
          const DType offset_w = data_offset_ptr[data_offset_w_ptr];
          const DType mask = data_mask_ptr[data_mask_hw_ptr];
          const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
          const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

          const DType cur_top_grad = data_col[index] * mask;
          const int cur_h = (int) cur_inv_h_data;
          const int cur_w = (int) cur_inv_w_data;
          for (int dy = -2; dy <= 2; dy++) {
              for (int dx = -2; dx <= 2; dx++) {
                  if (cur_h + dy >= 0 && cur_h + dy < height &&
                      cur_w + dx >= 0 && cur_w + dx < width &&
                      abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
                      abs(cur_inv_w_data - (cur_w + dx)) < 1
                      ) {
                      int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
                      DType weight =
                          DmcnGetGradientWeight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
                      MutexAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
//                      *(grad_im + cur_bottom_grad_pos) += weight * cur_top_grad;
                  }
              }
          }
      }
    };
    d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
#ifdef GOOGLE_CUDA
template<typename DType>
__global__ void DeformableConv2DCol2ImCoordGPUKernel(
    const int n,
    const DType *data_col, const DType *data_im,
    const DType *data_offset, const DType *data_mask,
    const int channels, const int height, const int width, // 输入的C, H, W
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col,
    DType *grad_offset, DType *grad_mask) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        DType val = 0, mval = 0;
        int w = index % width_col;
        int h = (index / width_col) % height_col;
        int c = (index / width_col / height_col) % offset_channels;
        int b = (index / width_col / height_col) / offset_channels;
        // compute the start and end of the output

        const int deformable_group_index = c / (2 * kernel_h * kernel_w);
        const int col_step = kernel_h * kernel_w;
        int cnt = 0;
        const DType *data_col_ptr =
            data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
        const DType *data_im_ptr = data_im
            + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w
                * height * width;
        const DType *data_offset_ptr = data_offset
            + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        const DType *data_mask_ptr =
            data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

        const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

        for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
            const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
            const int bp_dir = offset_c % 2;

            int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
            int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
            int w_out = col_pos % width_col;
            int h_out = (col_pos / width_col) % height_col;
            int w_in = w_out * stride_w - pad_w;
            int h_in = h_out * stride_h - pad_h;
            const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
            const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
            const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
            const DType offset_h = data_offset_ptr[data_offset_h_ptr];
            const DType offset_w = data_offset_ptr[data_offset_w_ptr];
            const DType mask = data_mask_ptr[data_mask_hw_ptr];
            DType inv_h = h_in + i * dilation_h + offset_h;
            DType inv_w = w_in + j * dilation_w + offset_w;
            if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
                inv_h = inv_w = -2;
            } else {
                mval += data_col_ptr[col_pos]
                    * DmcnIm2colBilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
            }
            const DType weight = DmcnGetCoordinateWeight(
                inv_h, inv_w,
                height, width, data_im_ptr + cnt * height * width, width, bp_dir);
            val += weight * data_col_ptr[col_pos] * mask;
            cnt += 1;
        }

        grad_offset[index] = val;
        // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
        if (offset_c % 2 == 0) {
            grad_mask[
                (((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col
                    + h) * width_col + w] = mval;
            // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
        }
    }
}
#endif
template<typename DType>
void DeformableConv2DCol2ImCoordCPUKernel(
    const CPUDevice &d,
    const int n,
    const DType *data_col, const DType *data_im,
    const DType *data_offset, const DType *data_mask,
    const int channels, const int height, const int width, // 输入的C, H, W
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col,
    DType *grad_offset, DType *grad_mask) {
    auto f = [n, data_col, data_im, data_offset, data_mask, channels, height, width,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        channel_per_deformable_group, batch_size, offset_channels, deformable_group,
        height_col, width_col, grad_offset, grad_mask](int64 start, int64 end) {
      for (int64 index = start; index < end; index++) {
          DType val = 0, mval = 0;
          int w = index % width_col;
          int h = (index / width_col) % height_col;
          int c = (index / width_col / height_col) % offset_channels;
          int b = (index / width_col / height_col) / offset_channels;
          // compute the start and end of the output

          const int deformable_group_index = c / (2 * kernel_h * kernel_w);
          const int col_step = kernel_h * kernel_w;
          int cnt = 0;
          const DType *data_col_ptr =
              data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
          const DType *data_im_ptr = data_im
              + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w
                  * height * width;
          const DType *data_offset_ptr = data_offset
              + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
          const DType *data_mask_ptr = data_mask
              + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

          const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

          for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
              const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
              const int bp_dir = offset_c % 2;

              int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
              int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
              int w_out = col_pos % width_col;
              int h_out = (col_pos / width_col) % height_col;
              int w_in = w_out * stride_w - pad_w;
              int h_in = h_out * stride_h - pad_h;
              const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
              const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
              const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
              const DType offset_h = data_offset_ptr[data_offset_h_ptr];
              const DType offset_w = data_offset_ptr[data_offset_w_ptr];
              const DType mask = data_mask_ptr[data_mask_hw_ptr];
              DType inv_h = h_in + i * dilation_h + offset_h;
              DType inv_w = w_in + j * dilation_w + offset_w;
              if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
                  inv_h = inv_w = -2;
              } else {
                  mval += data_col_ptr[col_pos]
                      * DmcnIm2colBilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
              }
              const DType weight = DmcnGetCoordinateWeight(
                  inv_h, inv_w,
                  height, width, data_im_ptr + cnt * height * width, width, bp_dir);
              val += weight * data_col_ptr[col_pos] * mask;
              cnt += 1;
          }

          grad_offset[index] = val;
          // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
          if (offset_c % 2 == 0) {
              grad_mask[
                  (((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col
                      + h) * width_col + w] = mval;
              // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
          }
      }
    };
    d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
#ifdef GOOGLE_CUDA
template<typename DType>
__global__ void PureAddToKernel(const int n, DType *result_data, const DType *right_data) {

    CUDA_1D_KERNEL_LOOP(index, n) {
        CudaAtomicAdd(result_data + index, right_data[index]);
    }

}
template<typename DType>
__global__ void SetZeroKernel(const int n, DType *result_data) {

    CUDA_1D_KERNEL_LOOP(index, n) {
        *(result_data + index) = DType(0);
    }

}
template<typename DType>
__global__ void SetOneKernel(const int n, DType *result_data) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        *(result_data + index) = DType(1);
    }
}
#endif
template<typename DType>
__global__ void SetNumAtIndexKernel(DType num, int index, DType *data) {
    *(data + index) = num;
}
template<typename DType>
void PureAddToKernel(const CPUDevice &d, const int n, DType *result_data, const DType *right_data) {
    auto f = [n, result_data, right_data](int64 start, int64 end) {
      for (int64 index = start; index < end; index++) {
          *(result_data + index) += (right_data[index]);
      }
    };
    d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template<typename DType>
void SetZeroKernel(const CPUDevice &d, const int n, DType *result_data) {
    auto f = [n, result_data](int64 start, int64 end) {
      for (int64 index = start; index < end; ++index) {
          *(result_data + index) = DType(0);
      }
    };
    d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template<typename DType>
void SetOneKernel(const CPUDevice &d, const int n, DType *result_data) {
    auto f = [n, result_data](int64 start, int64 end) {
      for (int64 index = start; index < end; ++index) {
          *(result_data + index) = DType(1);
      }
    };
    d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
#ifdef GOOGLE_CUDA
template<typename DType>
void DeformableConv2DCol2ImCoord<GPUDevice, DType>::operator()(const Eigen::GpuDevice &d,
                                                               const DType *data_col,
                                                               const DType *data_im,
                                                               const DType *data_offset,
                                                               const DType *data_mask,
                                                               const TShape &im_shape,
                                                               const TShape &col_shape,
                                                               const TShape &kernel_shape,
                                                               const TShape &pad,
                                                               const TShape &stride,
                                                               const TShape &dilation,
                                                               const int32_t deformable_group,
                                                               DType *grad_offset,
                                                               DType *grad_mask) {
    int num_spatial_axes = kernel_shape.size();
    int num_kernels =
        col_shape[1] * col_shape[2] * col_shape[3] * 2 * kernel_shape[0] * kernel_shape[1] * deformable_group;
    int channel_per_deformable_group = col_shape[0] / deformable_group;
    // num_axes should be smaller than block size
    CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
    CHECK_LT(num_spatial_axes, config.thread_per_block);
    switch (num_spatial_axes) {
        case 2:
            // To avoid involving atomic operations, we will launch one kernel per
            // bottom dimension, and then in the kernel add up the top dimensions.
            // NOLINT_NEXT_LINE(whitespace/operators)

            DeformableConv2DCol2ImCoordGPUKernel<DType> <<< config.block_count, config.thread_per_block,
                0, d.stream() >>> (
                num_kernels, data_col, data_im, data_offset, data_mask, im_shape[1], im_shape[2], im_shape[3],
                    kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
                    dilation[0], dilation[1], channel_per_deformable_group,
                    col_shape[1], 2 * kernel_shape[0] * kernel_shape[1]
                    * deformable_group, deformable_group, col_shape[2], col_shape[3],
                    grad_offset, grad_mask);
            // MSHADOW_CUDA_POST_KERNEL_CHECK(DeformableConv2DCol2ImCoordGPUKernel);
            break;
        default:LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                           << num_spatial_axes << " spatial axes";
    }
}
template <typename DType>
void DeformableConv2DCol2Im<GPUDevice, DType>::operator()(
    const GPUDevice& d,
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_im)
{
    int  num_spatial_axes = kernel_shape.size();
    int  im_size = ProdShape(im_shape, 1, im_shape.size());
    int  channel_per_deformable_group = im_shape[1] / deformable_group;
    int  num_kernels = ProdShape(col_shape, 0, col_shape.size());
    // num_axes should be smaller than block size
    CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
    CHECK_LT(num_spatial_axes, config.thread_per_block);
    //   using namespace mxnet_op;
    switch (num_spatial_axes) {
        case 2:
            // To avoid involving atomic operations, we will launch one kernel per
            // bottom dimension, and then in the kernel add up the top dimensions.
            // NOLINT_NEXT_LINE(whitespace/operators)
            DeformableConv2DCol2ImKernel<DType><<<config.block_count, config.thread_per_block,
                0, d.stream()>>>(
                num_kernels, data_col, data_offset, data_mask, im_shape[1], im_shape[2], im_shape[3],
                    kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
                    dilation[0], dilation[1], channel_per_deformable_group,
                    col_shape[1], deformable_group, col_shape[2], col_shape[3], grad_im);
            // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_col2im_gpu_kernel);
            break;
        default:
            LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                       << num_spatial_axes << " spatial axes";
    }
}

template <typename DType>
void DeformableConv2DIm2Col<GPUDevice, DType>::operator()(
    const GPUDevice& d,
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const int32_t deformable_group, DType* data_col)
{
    int  num_spatial_axes = kernel_shape.size();
    int  channel_per_deformable_group = im_shape[1] / deformable_group; // imshape[1] = 输入图的通道数
    int  num_kernels = im_shape[1] * ProdShape(col_shape, 1, col_shape.size()); // K * N / k.Size(), k = filter, col_shape = [K, im2col_step_, H, W]
    CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
    CHECK_LT(num_spatial_axes, config.thread_per_block);
    switch (num_spatial_axes) {
        case 2:
            DeformableConv2DIm2ColKernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
                <<<config.block_count, config.thread_per_block, // 注意这里申请的block的个数是num_kernel个,
                0, d.stream()>>>(
                //CUDA对device(GPU )的内存管理主要通过cudaMalloc()、cudaFree()、cudaMemcpy() 进行管理。另外，从上述代码我们可以看到，
                //add() 函数的调用比较奇怪相对于C语言来说，需要用add<<<M，N>>> 这种形式表明这是一个从host(CPU)代码调用device的代码，
                //并且括号中的数值表明，M个block，每个block有 N个线程, 所以这个函数总共有M*N个线程。
                num_kernels,
                    data_im,
                    data_offset,
                    data_mask,
                    im_shape[2], im_shape[3],
                    kernel_shape[0], kernel_shape[1],
                    pad[0], pad[1],
                    stride[0], stride[1],
                    dilation[0], dilation[1],
                    channel_per_deformable_group,
                    col_shape[1], im_shape[1],
                    deformable_group,
                    col_shape[2], col_shape[3],
                    data_col);
            // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_im2col_gpu_kernel);
            break;
        default:
            LOG(FATAL) << "im2col_nd_gpu does not support computation with "
                       << num_spatial_axes << " spatial axes";
    }
}


template <typename DType>
void SetZeros<GPUDevice, DType>::operator()(const GPUDevice& d, int n, DType* result_data){
    CudaLaunchConfig config = GetCudaLaunchConfig(n ,d);
    SetZeroKernel<DType> <<< config.block_count, config.thread_per_block, 0, d.stream() >>>(n, result_data);
}

template <typename DType>
void PureAddTo<GPUDevice, DType>::operator()(const GPUDevice& d, const int n, DType* result_data, const DType* right_data){
    CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
    PureAddToKernel<DType> <<< config.block_count, config.thread_per_block, 0, d.stream() >>>(n, result_data, right_data);
}

template <typename DType>
void SetOne<GPUDevice, DType>::operator()(const GPUDevice& d, int n, DType* result_data){
    CudaLaunchConfig config = GetCudaLaunchConfig(n ,d);
    SetOneKernel<DType> <<< config.block_count, config.thread_per_block, 0, d.stream() >>>(n, result_data);
}

template <typename DType>
void SetNumAtIndex<GPUDevice, DType>::operator()(const GPUDevice& d, DType num, int index, DType* data){
    CudaLaunchConfig config = GetCudaLaunchConfig(1 ,d);
    SetNumAtIndexKernel<DType> <<<config.block_count, config.thread_per_block, 0, d.stream() >>>(num, index, data);
}


// 如果没有在这里实例化的话, 生成的.so会报类似于 undefined symbol: _ZN10tensorflow13setNumAtIndexIN5Eigen9GpuDeviceEfEclERKS2_fiPf的错误
// I guess the reason for instancing the functional structure below is that certifying single functor instance for every functor.
template struct DeformableConv2DIm2Col<GPUDevice, double>;
template struct DeformableConv2DCol2Im<GPUDevice, double>;
template struct DeformableConv2DCol2ImCoord<GPUDevice, double>;
template struct PureAddTo<GPUDevice, double>;
template struct SetOne<GPUDevice, double>;
template struct SetZeros<GPUDevice, double>;
template struct SwapAxis<GPUDevice, double>;
template struct SetNumAtIndex<GPUDevice, double>;

template struct DeformableConv2DIm2Col<GPUDevice, float>;
template struct DeformableConv2DCol2Im<GPUDevice, float>;
template struct DeformableConv2DCol2ImCoord<GPUDevice, float>;
template struct PureAddTo<GPUDevice, float>;
template struct SetOne<GPUDevice, float>;
template struct SetZeros<GPUDevice, float>;
template struct SwapAxis<GPUDevice, float>;
template struct SetNumAtIndex<GPUDevice, float>;
#endif

template<typename DType>
void DeformableConv2DCol2ImCoord<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                                               const DType *data_col,
                                                               const DType *data_im,
                                                               const DType *data_offset,
                                                               const DType *data_mask,
                                                               const TShape &im_shape,
                                                               const TShape &col_shape,
                                                               const TShape &kernel_shape,
                                                               const TShape &pad,
                                                               const TShape &stride,
                                                               const TShape &dilation,
                                                               const int32_t deformable_group,
                                                               DType *grad_offset,
                                                               DType *grad_mask) {
    int num_spatial_axes = kernel_shape.size();
    int num_kernels =
        col_shape[1] * col_shape[2] * col_shape[3] * 2 * kernel_shape[0] * kernel_shape[1] * deformable_group;
    int channel_per_deformable_group = col_shape[0] / deformable_group;
    switch (num_spatial_axes) {
        case 2:
            DeformableConv2DCol2ImCoordCPUKernel<DType>(d,
                                                        num_kernels,
                                                        data_col,
                                                        data_im,
                                                        data_offset,
                                                        data_mask,
                                                        im_shape[1],
                                                        im_shape[2],
                                                        im_shape[3],
                                                        kernel_shape[0],
                                                        kernel_shape[1],
                                                        pad[0],
                                                        pad[1],
                                                        stride[0],
                                                        stride[1],
                                                        dilation[0],
                                                        dilation[1],
                                                        channel_per_deformable_group,
                                                        col_shape[1],
                                                        2 * kernel_shape[0] * kernel_shape[1] * deformable_group,
                                                        deformable_group,
                                                        col_shape[2],
                                                        col_shape[3],
                                                        grad_offset,
                                                        grad_mask);
            break;
        default:
            LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                       << num_spatial_axes << "spatial axes";
    }
}

template<typename DType>
void DeformableConv2DCol2Im<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                                          const DType *data_col,
                                                          const DType *data_offset,
                                                          const DType *data_mask,
                                                          const TShape &im_shape,
                                                          const TShape &col_shape,
                                                          const TShape &kernel_shape,
                                                          const TShape &pad,
                                                          const TShape &stride,
                                                          const TShape &dilation,
                                                          const int32_t deformable_group,
                                                          DType *grad_im) {
    int num_spatial_axes = kernel_shape.size();
    int im_size = ProdShape(im_shape, 1, im_shape.size());
    int channel_per_deformable_group = im_shape[1] / deformable_group;
    int num_kernels = ProdShape(col_shape, 0, col_shape.size());
    // num_axes should be smaller than block size
    //   using namespace mxnet_op;
    switch (num_spatial_axes) {
        case 2:
            // To avoid involving atomic operations, we will launch one kernel per
            // bottom dimension, and then in the kernel add up the top dimensions.
            // NOLINT_NEXT_LINE(whitespace/operators)
            DeformableConv2DCol2ImCPUKernel<DType>(
                d, num_kernels, data_col, data_offset, data_mask, im_shape[1], im_shape[2], im_shape[3],
                kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
                dilation[0], dilation[1], channel_per_deformable_group,
                col_shape[1], deformable_group, col_shape[2], col_shape[3], grad_im);
            // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_col2im_gpu_kernel);
            break;
        default:
            LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                       << num_spatial_axes << " spatial axes";
    }
}

template<typename DType>
void DeformableConv2DIm2Col<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                                          const DType *data_im,
                                                          const DType *data_offset,
                                                          const DType *data_mask,
                                                          const TShape &im_shape,
                                                          const TShape &col_shape,
                                                          const TShape &kernel_shape,
                                                          const TShape &pad,
                                                          const TShape &stride,
                                                          const TShape &dilation,
                                                          const int32_t deformable_group,
                                                          DType *data_col) {
    int num_spatial_axes = kernel_shape.size();
    int channel_per_deformable_group = im_shape[1] / deformable_group; // imshape[1] = 输入图的通道数
    int num_kernels = im_shape[1] * ProdShape(col_shape,
                                              1,
                                              col_shape.size()); // K * N / k.Size(), k = filter, col_shape = [K, im2col_step_, H, W]
    switch (num_spatial_axes) {
        case 2:
            DeformableConv2DIm2ColCPUKernel<DType>(
                d,
                num_kernels,
                data_im,
                data_offset,
                data_mask,
                im_shape[2], im_shape[3],
                kernel_shape[0], kernel_shape[1],
                pad[0], pad[1],
                stride[0], stride[1],
                dilation[0], dilation[1],
                channel_per_deformable_group,
                col_shape[1], im_shape[1],
                deformable_group,
                col_shape[2], col_shape[3],
                data_col);
            // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_im2col_gpu_kernel);
            break;
        default:
            LOG(FATAL) << "im2col_nd_gpu does not support computation with "
                       << num_spatial_axes << " spatial axes";
    }
}

template<typename DType>
void SetZeros<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d, int n, DType *result_data) {
    SetZeroKernel(d, n, result_data);
}
template<typename DType>
void PureAddTo<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                             const int n,
                                             DType *result_data,
                                             const DType *right_data) {
    PureAddToKernel(d, n, result_data, right_data);
}
template<typename DType>
void SetOne<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d, int n, DType *result_data) {
    SetOneKernel(d, n, result_data);
}
template<typename DType>
void SetNumAtIndex<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d, DType num, int index, DType *data) {
    *(data + index) = num;
}

#ifdef GOOGLE_CUDA
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

class CublasScratchAllocator : public se::ScratchAllocator {
 public:
  using Stream = se::Stream;
  using DeviceMemoryBytes = se::DeviceMemory<uint8>;

  CublasScratchAllocator(OpKernelContext* context) : context_(context) {}

  int64 GetMemoryLimitInBytes() override { return -1; }

  se::port::StatusOr<DeviceMemoryBytes> AllocateBytes(int64 byte_size) override {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return se::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

  se::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
      Stream* stream, int64 byte_size) {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return se::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

 private:
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

template <typename Scalar>
void LaunchBatchMatMul<GPUDevice, Scalar>::launch(OpKernelContext* context, const TensorShape& in_x_shape, const TensorShape& in_y_shape, const Scalar* in_x_ptr,
                     const Scalar* in_y_ptr, bool adj_x, bool adj_y, Scalar* out) {
    constexpr se::blas::Transpose kTranspose =
        is_complex<Scalar>::value ? se::blas::Transpose::kConjugateTranspose
                                  : se::blas::Transpose::kTranspose;
    se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                   kTranspose};

    const uint64 m = in_x_shape.dim_size(adj_x ? 2 : 1);
    const uint64 k = in_x_shape.dim_size(adj_x ? 1 : 2);
    const uint64 n = in_y_shape.dim_size(adj_y ? 1 : 2);
    const uint64 batch_size = in_x_shape.dim_size(0);
    auto blas_transpose_a = trans[adj_x];
    auto blas_transpose_b = trans[adj_y];

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    typedef se::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_device_memory.reserve(batch_size);
    b_device_memory.reserve(batch_size);
    c_device_memory.reserve(batch_size);
    a_ptrs.reserve(batch_size);
    b_ptrs.reserve(batch_size);
    c_ptrs.reserve(batch_size);
    auto* a_base_ptr = in_x_ptr;
    auto* b_base_ptr = in_y_ptr;
    auto* c_base_ptr = out;
    for (int64 i = 0; i < batch_size; ++i) {
      a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    }

    typedef Scalar Coefficient;

    // Cublas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A', where ' stands for transpose (not adjoint).
    // TODO(yangzihao): Choose the best of the three strategies using autotune.
    if (batch_size == 1) {
      // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
      // overhead of the scratch allocator and the batch interface.
      if (n == 1 &&
          blas_transpose_b != se::blas::Transpose::kConjugateTranspose &&
          blas_transpose_a != se::blas::Transpose::kConjugateTranspose) {
        // This is a matrix*vector multiply so use GEMV to compute A * b.
        // Here we are multiplying in the natural order, so we have to flip
        // the transposition flag to compensate for the tensor being stored
        // row-major. Since GEMV doesn't provide a way to just conjugate an
        // argument, we have to defer those cases to GEMM below.
        auto gemv_trans_a = blas_transpose_a == se::blas::Transpose::kTranspose
                                ? se::blas::Transpose::kNoTranspose
                                : se::blas::Transpose::kTranspose;
        bool blas_launch_status =
            stream
                ->ThenBlasGemv(gemv_trans_a, adj_x ? m : k, adj_x ? k : m,
                               static_cast<Coefficient>(1.0), *(a_ptrs[0]),
                               adj_x ? m : k, *(b_ptrs[0]), 1,
                               static_cast<Coefficient>(0.0), c_ptrs[0], 1)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMV launch failed : a.shape=", in_x_shape.DebugString(),
              ", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
              ", k=", k));
        }
      } else {
        bool blas_launch_status =
            stream
                ->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k,
                               static_cast<Coefficient>(1.0), *(b_ptrs[0]),
                               adj_y ? k : n, *(a_ptrs[0]), adj_x ? m : k,
                               static_cast<Coefficient>(0.0), c_ptrs[0], n)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMM launch failed : a.shape=", in_x_shape.DebugString(),
              ", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
              ", k=", k));
        }
      }
    } else {
      CublasScratchAllocator scratch_allocator(context);
      bool blas_launch_status =
          stream
              ->ThenBlasGemmBatchedWithScratch(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
                  adj_x ? m : k, static_cast<Coefficient>(0.0), c_ptrs, n,
                  batch_size, &scratch_allocator)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMBatched launch failed : a.shape=",
            in_x_shape.DebugString(),
            ", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    }
}

#endif

template<typename T>
void LaunchBatchMatMul<CPUDevice, T>::launch(OpKernelContext *context,
                                             const TensorShape &in_x_shape,
                                             const TensorShape &in_y_shape,
                                             const T *in_x_ptr,
                                             const T *in_y_ptr,
                                             bool adj_x,
                                             bool adj_y,
                                             T *out) {
    const int64 m = in_x_shape.dim_size(adj_x ? 2 : 1);
    const int64 k = in_x_shape.dim_size(adj_x ? 1 : 2);
    const int64 n = in_y_shape.dim_size(adj_y ? 1 : 2);
    const uint64 batch_size = in_x_shape.dim_size(0);
    Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor>> t_in_x(in_x_ptr, in_x_shape.AsEigenDSizes<3, Eigen::DenseIndex>());
    Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor>> t_in_y(in_y_ptr, in_y_shape.AsEigenDSizes<3, Eigen::DenseIndex>());
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> t_out(out, batch_size, m, n);
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x, adj_y);
    auto &device = context->eigen_device<CPUDevice>();
    for (int i = 0; i < t_out.dimension(0); ++i) {
        t_out.template chip<0>(i).device(device) =
            (t_in_x.template chip<0>(i)).template contract(t_in_y.template chip<0>(i), contract_pairs);
    }
}

template
struct DeformableConv2DIm2Col<CPUDevice, double>;
template
struct DeformableConv2DCol2Im<CPUDevice, double>;
template
struct DeformableConv2DCol2ImCoord<CPUDevice, double>;
template
struct PureAddTo<CPUDevice, double>;
template
struct SetOne<CPUDevice, double>;
template
struct SetZeros<CPUDevice, double>;
template
struct SwapAxis<CPUDevice, double>;
template
struct SetNumAtIndex<CPUDevice, double>;

template
struct DeformableConv2DIm2Col<CPUDevice, float>;
template
struct DeformableConv2DCol2Im<CPUDevice, float>;
template
struct DeformableConv2DCol2ImCoord<CPUDevice, float>;
template
struct PureAddTo<CPUDevice, float>;
template
struct SetOne<CPUDevice, float>;
template
struct SetZeros<CPUDevice, float>;
template
struct SwapAxis<CPUDevice, float>;
template
struct SetNumAtIndex<CPUDevice, float>;

template
struct LaunchBatchMatMul<GPUDevice, float>;
template
struct LaunchBatchMatMul<GPUDevice, double>;
template
struct LaunchBatchMatMul<CPUDevice, float>;
template
struct LaunchBatchMatMul<CPUDevice, double>;

}
