/*
:file:      warpAlgorithms.cu
:brief:     DVF based warping algorithms
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
*/
#include <stdio.h>

#include <warpKernels.cu>
#include <warpAlgorithms.hu>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void warp2D(
        const float* f,
        const float* u,
        const float* v,
        float* fWarped,
        int degree,
        int shape0,
        int shape1
    ){

    size_t size = shape0 * shape1 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_fWarped;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_fWarped, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((shape0 + 15)/16, (shape1 + 15)/16);
    if(degree==1){
        linearWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_u,
            d_v,
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
        cubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_u,
            d_v,
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
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_fWarped);
}


void adjointWarp2D(
        const float* fWarped,
        const float* u,
        const float* v,
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
    float *d_fWarped, *d_u, *d_v, *d_f;
    gpuErrchk(cudaMalloc(&d_fWarped, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_f, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((shape0 + 15)/16, (shape1 + 15)/16);
    if(degree==1){
        adjointLinearWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_fWarped,
            d_u,
            d_v,
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
        adjointCubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_fWarped,
            d_u,
            d_v,
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
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_fWarped);
}


void diffWarp2D(
        const float* f,
        const float* u,
        const float* v,
        float* diffx,
        float* diffy,
        int shape0,
        int shape1
    ){

    size_t size = shape0 * shape1 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_diffx, *d_diffy;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_diffx, size));
    gpuErrchk(cudaMalloc(&d_diffy, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
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
    cubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_diffx,
        shape0,
        shape1,
        d_coeffsx
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
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
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_diffx);
    cudaFree(d_diffy);
}


void jvpWarp2D(
        const float* f,
        const float* u,
        const float* v,
        const float* input,
        float* output,
        int degree,
        int shape0,
        int shape1
    ){

    size_t size = shape0 * shape1 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_input, *d_output;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_input, size));
    gpuErrchk(cudaMalloc(&d_output, 2*size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_output, output, 2*size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((shape0 + 15)/16, (shape1 + 15)/16);
    if(degree==1){
        jvpxLinearWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_u,
            d_v,
            d_input,
            d_output,
            shape0,
            shape1
        );
        jvpyLinearWarp2DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_u,
            d_v,
            d_input,
            d_output + shape0*shape1,
            shape0,
            shape1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }else if(degree==3){
        float coeffs_dx[] = {
            #include "cubic_2D_coefficients_dx.inc"
        };
        float coeffs_dy[] = {
            #include "cubic_2D_coefficients_dy.inc"
        };
        float *d_coeffs_dx;
        float *d_coeffs_dy;
        gpuErrchk(cudaMalloc(&d_coeffs_dx, 16*16*sizeof(float)));
        gpuErrchk(cudaMalloc(&d_coeffs_dy, 16*16*sizeof(float)));
        gpuErrchk(cudaMemcpy(d_coeffs_dx, coeffs_dx, 16*16*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_coeffs_dy, coeffs_dy, 16*16*sizeof(float), cudaMemcpyHostToDevice));
        jvpCubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
            d_u,
            d_v,
            d_input,
            d_output,
            shape0,
            shape1,
            d_coeffs_dx
        );
        jvpCubicWarp2DKernel<<<numBlocks, threadsPerBlock>>>(d_f,
            d_u,
            d_v,
            d_input,
            d_output + shape0*shape1,
            shape0,
            shape1,
            d_coeffs_dy
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(d_coeffs_dx);
        cudaFree(d_coeffs_dy);
    }else{
        throw "Only degree 1 and 3 are implemented";
    }

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(output, d_output, 2*size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_input);
    cudaFree(d_output);
}


void warp3D(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        float* fWarped,
        int degree,
        int shape0,
        int shape1,
        int shape2
    ){

    size_t size = shape0 * shape1 * shape2 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_w, *d_fWarped;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_w, size));
    gpuErrchk(cudaMalloc(&d_fWarped, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire image
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape0 + 7)/8, (shape1 + 7)/8, (shape2 + 7)/8);
    if(degree==1){
        linearWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_u,
            d_v,
            d_w,
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
        cubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
            d_f,
            d_u,
            d_v,
            d_w,
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
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_fWarped);
}


void adjointWarp3D(
        const float* fWarped,
        const float* u,
        const float* v,
        const float* w,
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
    float *d_fWarped, *d_u, *d_v, *d_w, *d_f;
    gpuErrchk(cudaMalloc(&d_fWarped, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_w, size));
    gpuErrchk(cudaMalloc(&d_f, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_fWarped, fWarped, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape0 + 7)/8, (shape1 + 7)/8, (shape2 + 7)/8);

    if(degree==1){
        adjointLinearWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
            d_fWarped,
            d_u,
            d_v,
            d_w,
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
        adjointCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
            d_fWarped,
            d_u,
            d_v,
            d_w,
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
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_fWarped);
}


void diffWarp3D(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        float* diffx,
        float* diffy,
        float* diffz,
        int shape0,
        int shape1,
        int shape2
    ){

    size_t size = shape0 * shape1 * shape2 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_w, *d_diffx, *d_diffy, *d_diffz;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_w, size));
    gpuErrchk(cudaMalloc(&d_diffx, size));
    gpuErrchk(cudaMalloc(&d_diffy, size));
    gpuErrchk(cudaMalloc(&d_diffz, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffx, diffx, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffy, diffy, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diffz, diffz, size, cudaMemcpyHostToDevice));

    // kernel invocation with 16*16 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape0 + 7)/8, (shape1 + 7)/8, (shape2 + 7)/8);
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
    cubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_w,
        d_diffx,
        shape0,
        shape1,
        shape2,
        d_coeffsx
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_w,
        d_diffy,
        shape0,
        shape1,
        shape2,
        d_coeffsy
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_w,
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
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_diffx);
    cudaFree(d_diffy);
    cudaFree(d_diffz);
}


void partialDiffWarp3D(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        int to,
        float* diff,
        int shape0,
        int shape1,
        int shape2
    ){

    size_t size = shape0 * shape1 * shape2 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_w, *d_diff;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_w, size));
    gpuErrchk(cudaMalloc(&d_diff, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_diff, diff, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape0 + 7)/8, (shape1 + 7)/8, (shape2 + 7)/8);
    float *coeffs;
    if(to == 0){
        float coeffsx[] = {
            #include "cubic_3D_coefficients_dx.inc"
        };
        coeffs = coeffsx;
    }else if(to == 1){
        float coeffsy[] = {
            #include "cubic_3D_coefficients_dy.inc"
        };
        coeffs = coeffsy;
    }else{
        float coeffsz[] = {
            #include "cubic_3D_coefficients_dz.inc"
        };
        coeffs = coeffsz;
    }
    float *d_coeffs;
    gpuErrchk(cudaMalloc(&d_coeffs, 64*64*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffs, coeffs, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    cubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_w,
        d_diff,
        shape0,
        shape1,
        shape2,
        d_coeffs
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_coeffs);

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(diff, d_diff, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_diff);
}


void jvpWarp3D(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        const float* input,
        float* output,
        int shape0,
        int shape1,
        int shape2
    ){

    size_t size = shape0 * shape1 * shape2 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_w, *d_input, *d_output;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_w, size));
    gpuErrchk(cudaMalloc(&d_input, size));
    gpuErrchk(cudaMalloc(&d_output, 3*size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_output, output, 3*size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape0 + 7)/8, (shape1 + 7)/8, (shape2 + 7)/8);

    float coeffs_dx[] = {
        #include "cubic_3D_coefficients_dx.inc"
    };
    float coeffs_dy[] = {
        #include "cubic_3D_coefficients_dy.inc"
    };
    float coeffs_dz[] = {
        #include "cubic_3D_coefficients_dz.inc"
    };
    float *d_coeffs_dx;
    float *d_coeffs_dy;
    float *d_coeffs_dz;
    gpuErrchk(cudaMalloc(&d_coeffs_dx, 64*64*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_coeffs_dy, 64*64*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_coeffs_dz, 64*64*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffs_dx, coeffs_dx, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeffs_dy, coeffs_dy, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_coeffs_dz, coeffs_dz, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    jvpCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_w,
        d_input,
        d_output,
        shape0,
        shape1,
        shape2,
        d_coeffs_dx
    );
    jvpCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_w,
        d_input,
        d_output + shape0*shape1*shape2,
        shape0,
        shape1,
        shape2,
        d_coeffs_dy
    );
    jvpCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_w,
        d_input,
        d_output + shape0*shape1*shape2*2,
        shape0,
        shape1,
        shape2,
        d_coeffs_dz
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_coeffs_dx);
    cudaFree(d_coeffs_dy);
    cudaFree(d_coeffs_dz);

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(output, d_output, 3*size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_input);
    cudaFree(d_output);
}


void jvpWarp3DY(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        const float* input,
        float* output,
        int shape0,
        int shape1,
        int shape2
    ){

    size_t size = shape0 * shape1 * shape2 * sizeof(float);

    // allocate vectors in device memory
    float *d_f, *d_u, *d_v, *d_w, *d_input, *d_output;
    gpuErrchk(cudaMalloc(&d_f, size));
    gpuErrchk(cudaMalloc(&d_u, size));
    gpuErrchk(cudaMalloc(&d_v, size));
    gpuErrchk(cudaMalloc(&d_w, size));
    gpuErrchk(cudaMalloc(&d_input, size));
    gpuErrchk(cudaMalloc(&d_output, size));

    // copy vectors from host memory to device memory
    gpuErrchk(cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_output, output, size, cudaMemcpyHostToDevice));

    // kernel invocation with 8*8*8 threads per block, and enough blocks
    // to cover the entire length of the vectors
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((shape0 + 7)/8, (shape1 + 7)/8, (shape2 + 7)/8);

    float coeffs_dy[] = {
        #include "cubic_3D_coefficients_dy.inc"
    };
    float *d_coeffs_dy;
    gpuErrchk(cudaMalloc(&d_coeffs_dy, 64*64*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_coeffs_dy, coeffs_dy, 64*64*sizeof(float), cudaMemcpyHostToDevice));
    jvpCubicWarp3DKernel<<<numBlocks, threadsPerBlock>>>(
        d_f,
        d_u,
        d_v,
        d_w,
        d_input,
        d_output,
        shape0,
        shape1,
        shape2,
        d_coeffs_dy
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    cudaFree(d_coeffs_dy);

    // copy the result back to the host
    gpuErrchk(cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost));

    // release the device memory
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_input);
    cudaFree(d_output);
}