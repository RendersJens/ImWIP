/*
:file:      warpAlgorithmsAffine.cu
:brief:     Affine warping algorithms
:author:    Jens Renders
*/

/*
This file is part of ImWIP.

ImWIP is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

ImWIP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of
the GNU General Public License along with ImWIP. If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>

#include <warpKernelsAffine.cu>
#include <warpAlgorithmsAffine.hu>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void affineWarp2D(
        const float* f,
        const float* A,
        const float* b,
        float* fWarped,
        int degree,
        int shape0,
        int shape1
    ){

    size_t size = shape0 * shape1 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_A, *d_b, *d_fWarped;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_A, 4 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_fWarped, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 4 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((shape0 + 15)/16, (shape1 + 15)/16);
    if(degree==1){
        affineLinearWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_A,
            d_b,
            d_fWarped,
            shape0,
            shape1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        float coeffs[] = {
            #include "cubic_2D_coefficients.inc"
        };
        float *d_coeffs;
        gpuErrchk(cudaMalloc(&d_coeffs, 16*16*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 16*16*sizeof(float), cudaMemcpyHostToDevice));
        affineCubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_A,
            d_b,
            d_fWarped,
            shape0,
            shape1,
            d_coeffs
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_coeffs);
    }else{
        throw "Only degree 1 and 3 are implemented";
    }

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(fWarped, d_fWarped, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_fWarped);
}


void adjointAffineWarp2D(
        const float* fWarped,
        const float* A,
        const float* b,
        float* f,
        int degree,
        int shape0,
        int shape1
    ){

    /*
    GPU implementation of 2D adjoint backward image warping along the DVF (u,v)
    with rectangular multivariate spline interpolation
    */


    size_t size = shape0 * shape1 * sizeof(float);

    // allocate vectors in device memory
    float *d_fWarped, *d_A, *d_b, *d_f;
    gpuErrchk(cudaMalloc(&d_fWarped, size));
    gpuErrchk(cudaMalloc(&d_A, 4 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_f, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 4 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((shape0 + 15)/16, (shape1 + 15)/16);
    if(degree==1){
        adjointAffineLinearWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_fWarped,
            d_A,
            d_b,
            d_f,
            shape0,
            shape1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        float coeffs[] = {
            #include "cubic_2D_coefficients.inc"
        };
        float *d_coeffs;
        gpuErrchk(cudaMalloc(&d_coeffs, 16*16*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 16*16*sizeof(float), cudaMemcpyHostToDevice));
        adjointAffineCubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_fWarped,
            d_A,
            d_b,
            d_f,
            shape0,
            shape1,
            d_coeffs
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_coeffs);
    }else{
        throw "Only degree 1 and 3 are implemented";
    }

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(f, d_f, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_fWarped);
}


void diffAffineWarp2D(
        const float* f,
        const float* A,
        const float* b,
        float* diffx,
        float* diffy,
        int shape0,
        int shape1
    ){

    size_t size = shape0 * shape1 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_A, *d_b, *d_diffx, *d_diffy;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_A, 4 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 2 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_diffx, size));
    gpuErrchk(cudaMalloc(&d_diffy, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 4 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffx, diffx, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffy, diffy, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((shape0 + 15)/16, (shape1 + 15)/16);
    float coeffsx[] = {
        #include "cubic_2D_coefficients_dx.inc"
    };
    float coeffsy[] = {
        #include "cubic_2D_coefficients_dy.inc"
    };
    float *d_coeffsx;
    float *d_coeffsy;
    gpuErrchk(cudaMalloc(&d_coeffsx, 16*16*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_coeffsy, 16*16*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffsx, coeffsx, 16*16*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeffsy, coeffsy, 16*16*sizeof(float), cudaMemcpyHostToDevice));
    affineCubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_A,
        d_b,
        d_diffx,
        shape0,
        shape1,
        d_coeffsx
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    affineCubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_A,
        d_b,
        d_diffy,
        shape0,
        shape1,
        d_coeffsy
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_coeffsx);
    cudaFree(d_coeffsy);

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(diffx, d_diffx, size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(diffy, d_diffy, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_diffx);
    cudaFree(d_diffy);
}


void affineWarp3D(
        const float* f,
        const float* A,
        const float* b,
        float* fWarped,
        int degree,
        int shape0,
        int shape1,
        int shape2
    ){

    size_t size = shape0 * shape1 * shape2 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_A, *d_b, *d_fWarped;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_A, 9 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_fWarped, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 3 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire image
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape2 + 7)/8, (shape1 + 7)/8, (shape0 + 7)/8); //faster order
    if(degree==1){
        affineLinearWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_A,
            d_b,
            d_fWarped,
            shape0,
            shape1,
            shape2
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        float coeffs[] = {
            #include "cubic_3D_coefficients.inc"
        };
        float *d_coeffs;
        gpuErrchk(cudaMalloc(&d_coeffs, 64*64*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 64*64*sizeof(float), cudaMemcpyHostToDevice));
        affineCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_A,
            d_b,
            d_fWarped,
            shape0,
            shape1,
            shape2,
            d_coeffs
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_coeffs);
    }else{
        throw "Only degree 1 and 3 are implemented";
    }

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(fWarped, d_fWarped, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_fWarped);
}


void adjointAffineWarp3D(
        const float* fWarped,
        const float* A,
        const float* b,
        float* f,
        int degree,
        int shape0,
        int shape1,
        int shape2
){
    /*
    GPU implementation of 3D adjoint backward image warping along the DVF (u,v,w)
    with rectangular multivariate spline interpolation
    */

    size_t size = shape0 * shape1 * shape2 * sizeof(float);

    // allocate vectors in device memory
    float *d_fWarped, *d_A, *d_b, *d_f;
    gpuErrchk(cudaMalloc(&d_fWarped, size));
    gpuErrchk(cudaMalloc(&d_A, 9 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_f, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 3 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape2 + 7)/8, (shape1 + 7)/8, (shape0 + 7)/8); //faster order

    if(degree==1){
        adjointAffineLinearWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
            d_fWarped,
            d_A,
            d_b,
            d_f,
            shape0,
            shape1,
            shape2
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        float coeffs[] = {
            #include "cubic_3D_coefficients.inc"
        };
        float *d_coeffs;
        gpuErrchk(cudaMalloc(&d_coeffs, 64*64*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 64*64*sizeof(float), cudaMemcpyHostToDevice));
        adjointAffineCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
            d_fWarped,
            d_A,
            d_b,
            d_f,
            shape0,
            shape1,
            shape2,
            d_coeffs
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_coeffs);
    }else{
        throw "Only degree 1 and 3 are implemented";
    }
    // copy the result back to the host
    gpuErrchk(cudaMemcpy(f, d_f, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_fWarped);
}


void diffAffineWarp3D(
        const float* f,
        const float* A,
        const float* b,
        float* diffx,
        float* diffy,
        float* diffz,
        int shape0,
        int shape1,
        int shape2
    ){

    size_t size = shape0 * shape1 * shape2 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_A, *d_b, *d_diffx, *d_diffy, *d_diffz;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_A, 9 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_diffx, size));
    gpuErrchk(cudaMalloc(&d_diffy, size));
    gpuErrchk(cudaMalloc(&d_diffz, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 3 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffx, diffx, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffy, diffy, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffz, diffz, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape2 + 7)/8, (shape1 + 7)/8, (shape0 + 7)/8); //faster order
    float coeffsx[] = {
        #include "cubic_3D_coefficients_dx.inc"
    };
    float coeffsy[] = {
        #include "cubic_3D_coefficients_dy.inc"
    };
    float coeffsz[] = {
        #include "cubic_3D_coefficients_dz.inc"
    };
    float *d_coeffsx;
    float *d_coeffsy;
    float *d_coeffsz;
    gpuErrchk(cudaMalloc(&d_coeffsx, 64*64*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_coeffsy, 64*64*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_coeffsz, 64*64*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffsx, coeffsx, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeffsy, coeffsy, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeffsz, coeffsz, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    affineCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_A,
        d_b,
        d_diffx,
        shape0,
        shape1,
        shape2,
        d_coeffsx
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    affineCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_A,
        d_b,
        d_diffy,
        shape0,
        shape1,
        shape2,
        d_coeffsy
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    affineCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_A,
        d_b,
        d_diffz,
        shape0,
        shape1,
        shape2,
        d_coeffsz
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_coeffsx);
    cudaFree(d_coeffsy);
    cudaFree(d_coeffsz);

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(diffx, d_diffx, size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(diffy, d_diffy, size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(diffz, d_diffz, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_diffx);
    cudaFree(d_diffy);
    cudaFree(d_diffz);
}


void diffAffineWarp3DMul(
        const float* f,
        const float* A,
        const float* b,
        float* diffx,
        float* diffy,
        float* diffz,
        int shape0,
        int shape1,
        int shape2,
        int shape3
    ){

    size_t size = shape0 * shape1 * shape2 * shape3 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_A, *d_b, *d_diffx, *d_diffy, *d_diffz;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_A, 9 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_diffx, size));
    gpuErrchk(cudaMalloc(&d_diffy, size));
    gpuErrchk(cudaMalloc(&d_diffz, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, 3 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffx, diffx, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffy, diffy, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffz, diffz, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape2 + 7)/8, (shape1 + 7)/8, (shape0 + 7)/8); //faster order
    float coeffsx[] = {
        #include "cubic_3D_coefficients_dx.inc"
    };
    float coeffsy[] = {
        #include "cubic_3D_coefficients_dy.inc"
    };
    float coeffsz[] = {
        #include "cubic_3D_coefficients_dz.inc"
    };
    float *d_coeffsx;
    float *d_coeffsy;
    float *d_coeffsz;
    gpuErrchk(cudaMalloc(&d_coeffsx, 64*64*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_coeffsy, 64*64*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_coeffsz, 64*64*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffsx, coeffsx, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeffsy, coeffsy, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeffsz, coeffsz, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    affineCubicWarp3DKernelMul<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_A,
        d_b,
        d_diffx,
        shape0,
        shape1,
        shape2,
        shape3,
        d_coeffsx
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    affineCubicWarp3DKernelMul<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_A,
        d_b,
        d_diffy,
        shape0,
        shape1,
        shape2,
        shape3,
        d_coeffsy
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    affineCubicWarp3DKernelMul<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_A,
        d_b,
        d_diffz,
        shape0,
        shape1,
        shape2,
        shape3,
        d_coeffsz
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_coeffsx);
    cudaFree(d_coeffsy);
    cudaFree(d_coeffsz);

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(diffx, d_diffx, size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(diffy, d_diffy, size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(diffz, d_diffz, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_diffx);
    cudaFree(d_diffy);
    cudaFree(d_diffz);
}