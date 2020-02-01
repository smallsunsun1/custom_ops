这个库主要实现了可变形卷积V2和可变形池化的CPU和GPU版本

目前deformable_conv2d_ops_new目录下的实现包含了DeformablePSROIAlign和DeformableConv2D的实现，并对cpu和gpu代码的进行了分离，从而使用分离编译，NVCC编译gpu部分，gcc或者clang编译cpu部分，从而提高在cpu部分代码的performance，推荐使用该目录下的代码

目前是在tensorflow2.0-gpu上编译通过，在tensorflow1.14-gpu略做修改也可以编译通过，后续考虑添加一个支持tensorflow1.14的分支。

如果需要自己编译使用，修改CMakeLists中的${TF_INCLUDE_PATH}变量 和最后target_link_libraries(ops PUBLIC ${CUDA_LIBRARIES} /home/admin-seu/miniconda3/envs/py36/lib/python3.6/site-packages/tensorflow_core/libtensorflow_framework.so.2) 中最后手动给定的lib位置，这里给出的是我编译时用的lib位置，对于你自己使用的tensorflow库你需要修改为你目录下的地址, USE_CUDA默认设置为开启，cmake -DUSE_CUDA=OFF可以关闭对gpu部分的编译。

编译结束后使用deformable_conv2d.py进行测试

batch_conv:  使用depth_wise函数来模拟对和batch有关系的卷积操作

CARAFE: https://arxiv.org/pdf/1905.02188.pdf  CARAFE: Content-Aware ReAssembly of Features  根据这篇文章提出的算子实现，ICCV2019 oral论文

2020.1.4 update

add c++ operator deformable conv2d_v2, code is build is tensorflow2.0, No support for other tensorflow version

2020.1.13 update

add c++ operator deformable conv2d_v2(Both CPU and GPU), code is build is tensorflow2.0, No support for other tensorflow version
