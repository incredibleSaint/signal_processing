// This program implements a 1D convolution using CUDA
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <npp.h> // provided in CUDA SDK
#include "UtilNPP/ImagesCPU.h" // these image libraries are also in CUDA SDK
#include "UtilNPP/ImagesNPP.h"
#include "UtilNPP/ImageIO.h"

void test_nppiFilter()
{
    npp::ImageCPU_8u_C1 oHostSrc;
    npp::loadImage("lena.pgm", oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc); // malloc and memcpy to GPU
    NppiSize kernelSize = {3, 1}; // dimensions of convolution kernel (filter)
    NppiSize oSizeROI = {oHostSrc.width() - kernelSize.width + 1, oHostSrc.height() - kernelSize.height + 1};
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height); // allocate device image of appropriately reduced size
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    NppiPoint oAnchor = {2, 1}; // found that oAnchor = {2,1} or {3,1} works for kernel [-1 0 1]
    NppStatus eStatusNPP;

    Npp32s hostKernel[3] = {-1, 0, 1}; // convolving with this should do edge detection
    Npp32s* deviceKernel;
    size_t deviceKernelPitch;
    cudaMallocPitch((void**)&deviceKernel, &deviceKernelPitch, kernelSize.width*sizeof(Npp32s), kernelSize.height*sizeof(Npp32s));
    cudaMemcpy2D(deviceKernel, deviceKernelPitch, hostKernel,
                 sizeof(Npp32s)*kernelSize.width, // sPitch
                 sizeof(Npp32s)*kernelSize.width, // width
                 kernelSize.height, // height
                 cudaMemcpyHostToDevice);
    Npp32s divisor = 1; // no scaling

    eStatusNPP = nppiFilter_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                   oDeviceDst.data(), oDeviceDst.pitch(),
                                   oSizeROI, deviceKernel, kernelSize, oAnchor, divisor);

    std::cout << "NppiFilter error status " << eStatusNPP << std::endl; // prints 0 (no errors)
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch()); // memcpy to host
    saveImage("Lena_filter_1d.pgm", oHostDst);
}
using type = int16_t;
using type_kern = int32_t;
std::vector<type> cuda_conv(std::vector<type> src, std::vector<type_kern> kernel) {
    std::vector<type> dst(src.size());
    auto src_size_bytes = src.size() * sizeof(type);
    auto ker_size_bytes = kernel.size() * sizeof(type_kern);
    type *src_d, *dst_d;
    type_kern *kern_d;
    cudaMalloc(&src_d, src_size_bytes);
    cudaMemcpy(src_d, src.data(), src_size_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dst_d, src_size_bytes);
    cudaMalloc(&kern_d, ker_size_bytes);
    cudaMemcpy(kern_d, kernel.data(), ker_size_bytes, cudaMemcpyHostToDevice);

    NppiSize roi = {int(src.size()), 1};
    NppiSize ker_size = {int(kernel.size()), 1};
    NppiPoint anchor = {1, 0};
    const Npp16s divisor = 1;
    auto res = nppiFilter_16s_C1R(src_d, int(src_size_bytes),
                                  dst_d, int(src_size_bytes),
                                  roi, kern_d, ker_size,
                                  anchor, divisor);
    cudaMemcpy(dst.data(), dst_d, src_size_bytes, cudaMemcpyDeviceToHost);
    std::vector<type> src_from_gpu(src.size()), kern_from_gpu(kernel.size());
    cudaMemcpy(src_from_gpu.data(), src_d, src_size_bytes, cudaMemcpyDeviceToHost);

    cudaMemcpy(kern_from_gpu.data(), kern_d,ker_size_bytes, cudaMemcpyDeviceToHost);
    for (auto && el: dst) {
        std::cout << el << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl << res << std::endl;
}

void test_sig_conv() {

    std::vector<type> src{1, 3, 4, 5, 6, 9  };
    std::vector<type_kern> kernel{2, 4, 6, 8};
    cuda_conv(src, kernel);
//    std::vector<type> dst(src.size());
//    auto src_size_bytes = src.size() * sizeof(type);
//    auto ker_size_bytes = kernel.size() * sizeof(type_kern);
//    type *src_d, *dst_d;
//    type_kern *kern_d;
//    cudaMalloc(&src_d, src_size_bytes);
//    cudaMemcpy(src_d, src.data(), src_size_bytes, cudaMemcpyHostToDevice);
//    cudaMalloc(&dst_d, src_size_bytes);
//    cudaMalloc(&kern_d, ker_size_bytes);
//    cudaMemcpy(kern_d, kernel.data(), ker_size_bytes, cudaMemcpyHostToDevice);
//
//    NppiSize roi = {int(src.size()), 1};
//    NppiSize ker_size = {int(kernel.size()), 1};
//    NppiPoint anchor = {1, 0};
//    const Npp16s divisor = 1;
//    auto res = nppiFilter_16s_C1R(src_d, int(src_size_bytes),
//                                  dst_d, int(src_size_bytes),
//                                  roi, kern_d, ker_size,
//                                  anchor, divisor);
//    cudaMemcpy(dst.data(), dst_d, src_size_bytes, cudaMemcpyDeviceToHost);
//    std::vector<type> src_from_gpu(src.size()), kern_from_gpu(kernel.size());
//    cudaMemcpy(src_from_gpu.data(), src_d, src_size_bytes, cudaMemcpyDeviceToHost);
//
//    cudaMemcpy(kern_from_gpu.data(), kern_d,ker_size_bytes, cudaMemcpyDeviceToHost);
//    for (auto && el: dst) {
//        std::cout << el << " ";
//    }
//    std::cout << std::endl;
//    std::cout << std::endl << res << std::endl;

}

// 1-D convolution kernel
//  Arguments:
//      array   = padded array
//      mask    = convolution mask
//      result  = result array
//      n       = number of elements in array
//      m       = number of elements in the mask
__global__ void convolution_1d(int *array, int *mask, int *result, int n,
                               int m) {
    // Global thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate radius of the mask
    int r = m / 2;

    // Calculate the starting point for the element
    int start = tid - r;

    // Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for (int j = 0; j < m; j++) {
        // Ignore elements that hang off (0s don't contribute)
        if (((start + j) >= 0) && (start + j < n)) {
            // accumulate partial results
            temp += array[start + j] * mask[j];
        }
    }

    // Write-back the results
    result[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n, int m) {
    int radius = m / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < m; j++) {
            if ((start + j >= 0) && (start + j < n)) {
                temp += array[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}

int main() {
    test_sig_conv();
    return 0;
    test_nppiFilter();
    // Number of elements in result array
    int n = 10;

    // Size of the array in bytes
    int bytes_n = n * sizeof(int);

    // Number of elements in the convolution mask
    int m = 4;

    // Size of mask in bytes
    int bytes_m = m * sizeof(int);

    // Allocate the array (include edge elements)...
    std::vector<int> h_array(n);

    // ... and initialize it
    std::generate(begin(h_array), end(h_array), [](){ return rand() % 7; });

    // Allocate the mask and initialize it
    std::vector<int> h_mask(m);
    std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 4; });

    // Allocate space for the result
    std::vector<int> h_result(n);

    // Allocate space on the device
    int *d_array, *d_mask, *d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_array, h_array.data(), bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

    // Threads per TB
    int THREADS = 256;

    // Number of TBs
    int GRID = (n + THREADS - 1) / THREADS;

    // Call the kernel
    convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

    // Copy back the result
    cudaMemcpy(h_result.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(h_array.data(), h_mask.data(), h_result.data(), n, m);

    std::cout << std::endl << "First" << std::endl;
    for (auto && el : h_array) {
        std::cout << el << " ";
    }
    std::cout << std::endl << "Second" << std::endl;
    for (auto && el : h_mask) {
        std::cout << el << " ";
    }
    std::cout << std::endl << "Result" << std::endl;
    for (auto && el : h_result) {
        std::cout << el << " ";
    }
    std::cout << std::endl << "COMPLETED SUCCESSFULLY\n";
    std::cout << "Result length = " << h_result.size() << std::endl;
    // Free allocated memory on the device and host
    cudaFree(d_result);
    cudaFree(d_mask);
    cudaFree(d_array);

    return 0;
}
