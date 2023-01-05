/*
:file:      warpKernels.cu
:brief:     DVF based warping kernels
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

__global__ void linearWarp2DKernel(
        const float* f,
        const float* u,
        const float* v,
        float* fWarped,
        int shape0,
        int shape1
    ){
    
    /*
    Kernel of GPU implementation of 2D backward image warping along the DVF (u,v)
    with linear interpolation (rectangular multivariate spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = i+u[i*shape1 + j];
        float y = j+v[i*shape1 + j];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int Q[][2] = {{x1, y1},
                      {x1, y2},
                      {x2, y1},
                      {x2, y2}};

        // interpolation coefficients
        float coefficients[] = {(x2 - x)*(y2 - y),
                                (x2 - x)*(y - y1),
                                (x - x1)*(y2 - y),
                                (x - x1)*(y - y1)};

        for(int m = 0; m < 4; m++){
            if(0 <= Q[m][0] && Q[m][0] < shape0
            && 0 <= Q[m][1] && Q[m][1] < shape1){
                fWarped[i*shape1 + j] += coefficients[m] * f[Q[m][0]*shape1 + Q[m][1]];
            }
        }
    }
}


__global__ void adjointLinearWarp2DKernel(
        const float* fWarped,
        const float* u,
        const float* v,
        float* f,
        int shape0,
        int shape1
    ){

    /*
    Kernel of GPU implementation of 2D adjoint backward image warping along the
    DVF (u,v) with linear interpolation (rectangular multivariate spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = i+u[i*shape1 + j];
        float y = j+v[i*shape1 + j];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int Q[][2] = {{x1, y1},
                      {x1, y2},
                      {x2, y1},
                      {x2, y2}};

        // interpolation coefficients
        float coefficients[] = {(x2 - x)*(y2 - y),
                                (x2 - x)*(y - y1),
                                (x - x1)*(y2 - y),
                                (x - x1)*(y - y1)};

        for(int m = 0; m < 4; m++){
            if(0 <= Q[m][0] && Q[m][0] < shape0
            && 0 <= Q[m][1] && Q[m][1] < shape1){
                atomicAdd(&f[Q[m][0]*shape1 + Q[m][1]], coefficients[m] * fWarped[i*shape1 + j]);
            }
        }
    }
}


__global__ void jvpxLinearWarp2DKernel(
        const float* f,
        const float* u,
        const float* v,
        const float* input,
        float* output,
        int shape0,
        int shape1
    ){
    
    /*
    Kernel of GPU implementation of differentiated (to x)
    2D backward image warping along the DVF (u,v)
    with linear interpolation (rectangular multivariate spline)
    */

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = i+u[i*shape1 + j];
        float y = j+v[i*shape1 + j];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int Q[][2] = {{x1, y1},
                      {x1, y2},
                      {x2, y1},
                      {x2, y2}};

        // interpolation coefficients
        float coefficients[] = {-(y2 - y),
                                -(y - y1),
                                 (y2 - y),
                                 (y - y1)};

        for(int m = 0; m < 4; m++){
            if(0 <= Q[m][0] && Q[m][0] < shape0
            && 0 <= Q[m][1] && Q[m][1] < shape1){
                output[i*shape1 + j] += coefficients[m] * f[Q[m][0]*shape1 + Q[m][1]] * input[i*shape1 + j];
            }
        }
    }
}


__global__ void jvpyLinearWarp2DKernel(
        const float* f,
        const float* u,
        const float* v,
        const float* input,
        float* output,
        int shape0,
        int shape1
    ){
    
    /*
    Kernel of GPU implementation of differentiated (to y)
    2D backward image warping along the DVF (u,v)
    with linear interpolation (rectangular multivariate spline)
    */

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = i+u[i*shape1 + j];
        float y = j+v[i*shape1 + j];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int Q[][2] = {{x1, y1},
                      {x1, y2},
                      {x2, y1},
                      {x2, y2}};

        // interpolation coefficients
        float coefficients[] = {-(x2 - x),
                                (x2 - x),
                                -(x - x1),
                                (x - x1)};

        for(int m = 0; m < 4; m++){
            if(0 <= Q[m][0] && Q[m][0] < shape0
            && 0 <= Q[m][1] && Q[m][1] < shape0){
                output[i*shape1 + j] += coefficients[m] * f[Q[m][0]*shape1 + Q[m][1]] * input[i*shape1 + j];
            }
        }
    }
}


__global__ void linearWarp3DKernel(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        float* fWarped,
        int shape0,
        int shape1,
        int shape2
    ){
    
    /*
    Kernel of GPU implementation of 3D backward image warping along the DVF (u,v)
    with linear interpolation (rectangular multivariate spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = i+u[i*shape1*shape2 + j*shape2 + k];
        float y = j+v[i*shape1*shape2 + j*shape2 + k];
        float z = k+w[i*shape1*shape2 + j*shape2 + k];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int z2 = z1 + 1;
        int Q[][3] = {{x1, y1, z1},
                      {x2, y1, z1},
                      {x1, y2, z1},
                      {x2, y2, z1},
                      {x1, y1, z2},
                      {x2, y1, z2},
                      {x1, y2, z2},
                      {x2, y2, z2}};

        // interpolation coefficients
        float coefficients[] = {(x2 - x)*(y2 - y)*(z2 - z),
                                (x - x1)*(y2 - y)*(z2 - z),
                                (x2 - x)*(y - y1)*(z2 - z),
                                (x - x1)*(y - y1)*(z2 - z),
                                (x2 - x)*(y2 - y)*(z - z1),
                                (x - x1)*(y2 - y)*(z - z1),
                                (x2 - x)*(y - y1)*(z - z1),
                                (x - x1)*(y - y1)*(z - z1)};

        for(int m = 0; m < 8; m++){
            if(0 <= Q[m][0] && Q[m][0] < shape0
            && 0 <= Q[m][1] && Q[m][1] < shape1
            && 0 <= Q[m][2] && Q[m][2] < shape2){
                fWarped[i*shape1*shape2 + j*shape2 + k] += coefficients[m] * f[Q[m][0]*shape1*shape2 + Q[m][1]*shape2 + Q[m][2]];
            }
        }
    }
}


__global__ void adjointLinearWarp3DKernel(
        const float* fWarped,
        const float* u,
        const float* v,
        const float* w,
        float* f,
        int shape0,
        int shape1,
        int shape2
    ){
    
    /*
    Kernel of GPU implementation of 3D adjoint backward image warping along the
    DVF (u,v) with linear interpolation (rectangular multivariate spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = i+u[i*shape1*shape2 + j*shape2 + k];
        float y = j+v[i*shape1*shape2 + j*shape2 + k];
        float z = k+w[i*shape1*shape2 + j*shape2 + k];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        int z2 = z1 + 1;
        int Q[][3] = {{x1, y1, z1},
                      {x2, y1, z1},
                      {x1, y2, z1},
                      {x2, y2, z1},
                      {x1, y1, z2},
                      {x2, y1, z2},
                      {x1, y2, z2},
                      {x2, y2, z2}};

        // interpolation coefficients
        float coefficients[] = {(x2 - x)*(y2 - y)*(z2 - z),
                                (x - x1)*(y2 - y)*(z2 - z),
                                (x2 - x)*(y - y1)*(z2 - z),
                                (x - x1)*(y - y1)*(z2 - z),
                                (x2 - x)*(y2 - y)*(z - z1),
                                (x - x1)*(y2 - y)*(z - z1),
                                (x2 - x)*(y - y1)*(z - z1),
                                (x - x1)*(y - y1)*(z - z1)};

        for(int m = 0; m < 8; m++){
            if(0 <= Q[m][0] && Q[m][0] < shape0
            && 0 <= Q[m][1] && Q[m][1] < shape1
            && 0 <= Q[m][2] && Q[m][2] < shape2){
                atomicAdd(&f[Q[m][0]*shape1*shape2 + Q[m][1]*shape2 + Q[m][2]], coefficients[m] * fWarped[i*shape1*shape2 + j*shape2 + k]);
            }
        }
    }
}


__global__ void cubicWarp2DKernel(
        const float* f,
        const float* u,
        const float* v, 
        float* fWarped,
        int shape0,
        int shape1,
        const float* coeffs
    ){
    /*
    Kernel of GPU implementation of backward image warping along the DVF (u,v)
    with cubic interpolation (rectangular multivariate catmull-rom spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = i+u[i*shape1 + j];
        float y = j+v[i*shape1 + j];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        // xi = x1 - 1 + i

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float monomials[] = {1, xmx1, xmx1_2, xmx1_3, ymy1, xmx1*ymy1, xmx1_2*ymy1, xmx1_3*ymy1, ymy1_2, xmx1*ymy1_2, xmx1_2*ymy1_2, xmx1_3*ymy1_2, ymy1_3, xmx1*ymy1_3, xmx1_2*ymy1_3, xmx1_3*ymy1_3};
        
        int m = 0;
        for(int ii = 0; ii < 4; ii++){
            for(int jj = 0; jj < 4; jj++){
                int Q0 = x1 + ii - 1;
                int Q1 = y1 + jj - 1;
                if(0 <= Q0 && Q0 < shape0
                && 0 <= Q1 && Q1 < shape1){
                    float coefficient = 0;
                    for(int n = 0; n < 16; n++){
                        coefficient += coeffs[m*16 + n] * monomials[n];
                    }
                    fWarped[i*shape1 + j] += coefficient * f[Q0*shape1 + Q1];
                }
                m++;
            }
        }
    }
}


__global__ void adjointCubicWarp2DKernel(
        const float* fWarped,
        const float* u,
        const float* v, 
        float* f,
        int shape0,
        int shape1,
        const float* coeffs
    ){
    /*
    Kernel of GPU implementation of backward image warping along the DVF (u,v)
    with cubic interpolation (rectangular multivariate catmull-rom spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = i+u[i*shape1 + j];
        float y = j+v[i*shape1 + j];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        // xi = x1 - 1 + i

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float monomials[] = {1, xmx1, xmx1_2, xmx1_3, ymy1, xmx1*ymy1, xmx1_2*ymy1, xmx1_3*ymy1, ymy1_2, xmx1*ymy1_2, xmx1_2*ymy1_2, xmx1_3*ymy1_2, ymy1_3, xmx1*ymy1_3, xmx1_2*ymy1_3, xmx1_3*ymy1_3};

        int m = 0;
        for(int ii = 0; ii < 4; ii++){
            for(int jj = 0; jj < 4; jj++){
                int Q0 = x1 + ii - 1;
                int Q1 = y1 + jj - 1;
                if(0 <= Q0 && Q0 < shape0
                && 0 <= Q1 && Q1 < shape1){
                    float coefficient = 0;
                    for(int n = 0; n < 16; n++){
                        coefficient += coeffs[m*16 + n] * monomials[n];
                    }
                    atomicAdd(&f[Q0*shape1 + Q1], coefficient * fWarped[i*shape1 + j]);
                }
                m++;
            }
        }
    }
}


__global__ void jvpCubicWarp2DKernel(
        const float* f,
        const float* u,
        const float* v,
        const float* input, 
        float* output,
        int shape0,
        int shape1,
        const float* coeffs
    ){
    /*
    Kernel of GPU implementation of backward image warping along the DVF (u,v)
    with cubic interpolation (rectangular multivariate catmull-rom spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = i+u[i*shape1 + j];
        float y = j+v[i*shape1 + j];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        // xi = x1 - 1 + i

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float monomials[] = {1, xmx1, xmx1_2, xmx1_3, ymy1, xmx1*ymy1, xmx1_2*ymy1, xmx1_3*ymy1, ymy1_2, xmx1*ymy1_2, xmx1_2*ymy1_2, xmx1_3*ymy1_2, ymy1_3, xmx1*ymy1_3, xmx1_2*ymy1_3, xmx1_3*ymy1_3};

        int m = 0;
        for(int ii = 0; ii < 4; ii++){
            for(int jj = 0; jj < 4; jj++){
                int Q0 = x1 + ii - 1;
                int Q1 = y1 + jj - 1;
                if(0 <= Q0 && Q0 < shape0
                && 0 <= Q1 && Q1 < shape1){
                    float coefficient = 0;
                    for(int n = 0; n < 16; n++){
                        coefficient += coeffs[m*16 + n] * monomials[n];
                    }
                    output[i*shape1 + j] += coefficient * f[Q0*shape1 + Q1] * input[i*shape1 + j];
                }
                m++;
            }
        }
    }
}


__global__ void cubicWarp3DKernel(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        float* fWarped,
        int shape0,
        int shape1,
        int shape2,
        const float* coeffs
    ){
    
    /*
    Kernel of GPU implementation of 3D backward image warping along the DVF (u,v)
    with cubic interpolation (rectangular multivariate spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = i+u[i*shape1*shape2 + j*shape2 + k];
        float y = j+v[i*shape1*shape2 + j*shape2 + k];
        float z = k+w[i*shape1*shape2 + j*shape2 + k];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        // xi = fx + i - 1

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float zmz1 = z - z1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float zmz1_2 = zmz1 * zmz1;
        float zmz1_3 = zmz1 * zmz1_2;
        float monomials[] = {1, xmx1, xmx1_2, xmx1_3, ymy1, xmx1*ymy1, xmx1_2*ymy1, xmx1_3*ymy1, ymy1_2, xmx1*ymy1_2, xmx1_2*ymy1_2, xmx1_3*ymy1_2, ymy1_3, xmx1*ymy1_3, xmx1_2*ymy1_3, xmx1_3*ymy1_3, zmz1, xmx1*zmz1, xmx1_2*zmz1, xmx1_3*zmz1, ymy1*zmz1, xmx1*ymy1*zmz1, xmx1_2*ymy1*zmz1, xmx1_3*ymy1*zmz1, ymy1_2*zmz1, xmx1*ymy1_2*zmz1, xmx1_2*ymy1_2*zmz1, xmx1_3*ymy1_2*zmz1, ymy1_3*zmz1, xmx1*ymy1_3*zmz1, xmx1_2*ymy1_3*zmz1, xmx1_3*ymy1_3*zmz1, zmz1_2, xmx1*zmz1_2, xmx1_2*zmz1_2, xmx1_3*zmz1_2, ymy1*zmz1_2, xmx1*ymy1*zmz1_2, xmx1_2*ymy1*zmz1_2, xmx1_3*ymy1*zmz1_2, ymy1_2*zmz1_2, xmx1*ymy1_2*zmz1_2, xmx1_2*ymy1_2*zmz1_2, xmx1_3*ymy1_2*zmz1_2, ymy1_3*zmz1_2, xmx1*ymy1_3*zmz1_2, xmx1_2*ymy1_3*zmz1_2, xmx1_3*ymy1_3*zmz1_2, zmz1_3, xmx1*zmz1_3, xmx1_2*zmz1_3, xmx1_3*zmz1_3, ymy1*zmz1_3, xmx1*ymy1*zmz1_3, xmx1_2*ymy1*zmz1_3, xmx1_3*ymy1*zmz1_3, ymy1_2*zmz1_3, xmx1*ymy1_2*zmz1_3, xmx1_2*ymy1_2*zmz1_3, xmx1_3*ymy1_2*zmz1_3, ymy1_3*zmz1_3, xmx1*ymy1_3*zmz1_3, xmx1_2*ymy1_3*zmz1_3, xmx1_3*ymy1_3*zmz1_3};

        int m = 0;
        for(int ii = 0; ii < 4; ii++){
            for(int jj = 0; jj < 4; jj++){
                for(int kk = 0; kk < 4; kk++){
                    int Q0 = x1 + ii - 1;
                    int Q1 = y1 + jj - 1;
                    int Q2 = z1 + kk - 1;
                    if(0 <= Q0 && Q0 < shape0
                    && 0 <= Q1 && Q1 < shape1
                    && 0 <= Q2 && Q2 < shape2){
                        float coefficient = 0;
                        for(int n = 0; n < 64; n++){
                            coefficient += coeffs[m*64 + n] * monomials[n];
                        }
                        fWarped[i*shape1*shape2 + j*shape2 + k] += coefficient * f[Q0*shape1*shape2 + Q1*shape2 + Q2];
                    }
                    m++;
                }
            }
        }
    }
}


__global__ void adjointCubicWarp3DKernel(
        const float* fWarped,
        const float* u,
        const float* v,
        const float* w,
        float* f,
        int shape0,
        int shape1,
        int shape2,
        const float* coeffs
    ){
    
    /*
    Kernel of GPU implementation of adjoint 3D backward image warping along the
    DVF (u,v,w) with cubic interpolation (rectangular multivariate catmull-rom spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = i+u[i*shape1*shape2 + j*shape2 + k];
        float y = j+v[i*shape1*shape2 + j*shape2 + k];
        float z = k+w[i*shape1*shape2 + j*shape2 + k];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        // xi = fx + i - 1

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float zmz1 = z - z1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float zmz1_2 = zmz1 * zmz1;
        float zmz1_3 = zmz1 * zmz1_2;
        float monomials[] = {1, xmx1, xmx1_2, xmx1_3, ymy1, xmx1*ymy1, xmx1_2*ymy1, xmx1_3*ymy1, ymy1_2, xmx1*ymy1_2, xmx1_2*ymy1_2, xmx1_3*ymy1_2, ymy1_3, xmx1*ymy1_3, xmx1_2*ymy1_3, xmx1_3*ymy1_3, zmz1, xmx1*zmz1, xmx1_2*zmz1, xmx1_3*zmz1, ymy1*zmz1, xmx1*ymy1*zmz1, xmx1_2*ymy1*zmz1, xmx1_3*ymy1*zmz1, ymy1_2*zmz1, xmx1*ymy1_2*zmz1, xmx1_2*ymy1_2*zmz1, xmx1_3*ymy1_2*zmz1, ymy1_3*zmz1, xmx1*ymy1_3*zmz1, xmx1_2*ymy1_3*zmz1, xmx1_3*ymy1_3*zmz1, zmz1_2, xmx1*zmz1_2, xmx1_2*zmz1_2, xmx1_3*zmz1_2, ymy1*zmz1_2, xmx1*ymy1*zmz1_2, xmx1_2*ymy1*zmz1_2, xmx1_3*ymy1*zmz1_2, ymy1_2*zmz1_2, xmx1*ymy1_2*zmz1_2, xmx1_2*ymy1_2*zmz1_2, xmx1_3*ymy1_2*zmz1_2, ymy1_3*zmz1_2, xmx1*ymy1_3*zmz1_2, xmx1_2*ymy1_3*zmz1_2, xmx1_3*ymy1_3*zmz1_2, zmz1_3, xmx1*zmz1_3, xmx1_2*zmz1_3, xmx1_3*zmz1_3, ymy1*zmz1_3, xmx1*ymy1*zmz1_3, xmx1_2*ymy1*zmz1_3, xmx1_3*ymy1*zmz1_3, ymy1_2*zmz1_3, xmx1*ymy1_2*zmz1_3, xmx1_2*ymy1_2*zmz1_3, xmx1_3*ymy1_2*zmz1_3, ymy1_3*zmz1_3, xmx1*ymy1_3*zmz1_3, xmx1_2*ymy1_3*zmz1_3, xmx1_3*ymy1_3*zmz1_3};

        int m = 0;
        for(int ii = 0; ii < 4; ii++){
            for(int jj = 0; jj < 4; jj++){
                for(int kk = 0; kk < 4; kk++){
                    int Q0 = x1 + ii - 1;
                    int Q1 = y1 + jj - 1;
                    int Q2 = z1 + kk - 1;
                    if(0 <= Q0 && Q0 < shape0
                    && 0 <= Q1 && Q1 < shape1
                    && 0 <= Q2 && Q2 < shape2){
                        float coefficient = 0;
                        for(int n = 0; n < 64; n++){
                            coefficient += coeffs[m*64 + n] * monomials[n];
                        }
                        atomicAdd(&f[Q0*shape1*shape2 + Q1*shape2 + Q2], coefficient * fWarped[i*shape1*shape2 + j*shape2 + k]);
                    }
                    m++;
                }
            }
        }
    }
}


__global__ void jvpCubicWarp3DKernel(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        const float* input,
        float* output,
        int shape0,
        int shape1,
        int shape2,
        const float* coeffs
    ){
    
    /*
    Kernel of GPU implementation of 3D backward image warping along the DVF (u,v)
    with cubic interpolation (rectangular multivariate spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = i+u[i*shape1*shape2 + j*shape2 + k];
        float y = j+v[i*shape1*shape2 + j*shape2 + k];
        float z = k+w[i*shape1*shape2 + j*shape2 + k];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        // xi = fx + i - 1

        // interpolation coefficients
        float xmx1 = x - x1;
        float ymy1 = y - y1;
        float zmz1 = z - z1;
        float xmx1_2 = xmx1 * xmx1;
        float xmx1_3 = xmx1 * xmx1_2;
        float ymy1_2 = ymy1 * ymy1;
        float ymy1_3 = ymy1 * ymy1_2;
        float zmz1_2 = zmz1 * zmz1;
        float zmz1_3 = zmz1 * zmz1_2;
        float monomials[] = {1, xmx1, xmx1_2, xmx1_3, ymy1, xmx1*ymy1, xmx1_2*ymy1, xmx1_3*ymy1, ymy1_2, xmx1*ymy1_2, xmx1_2*ymy1_2, xmx1_3*ymy1_2, ymy1_3, xmx1*ymy1_3, xmx1_2*ymy1_3, xmx1_3*ymy1_3, zmz1, xmx1*zmz1, xmx1_2*zmz1, xmx1_3*zmz1, ymy1*zmz1, xmx1*ymy1*zmz1, xmx1_2*ymy1*zmz1, xmx1_3*ymy1*zmz1, ymy1_2*zmz1, xmx1*ymy1_2*zmz1, xmx1_2*ymy1_2*zmz1, xmx1_3*ymy1_2*zmz1, ymy1_3*zmz1, xmx1*ymy1_3*zmz1, xmx1_2*ymy1_3*zmz1, xmx1_3*ymy1_3*zmz1, zmz1_2, xmx1*zmz1_2, xmx1_2*zmz1_2, xmx1_3*zmz1_2, ymy1*zmz1_2, xmx1*ymy1*zmz1_2, xmx1_2*ymy1*zmz1_2, xmx1_3*ymy1*zmz1_2, ymy1_2*zmz1_2, xmx1*ymy1_2*zmz1_2, xmx1_2*ymy1_2*zmz1_2, xmx1_3*ymy1_2*zmz1_2, ymy1_3*zmz1_2, xmx1*ymy1_3*zmz1_2, xmx1_2*ymy1_3*zmz1_2, xmx1_3*ymy1_3*zmz1_2, zmz1_3, xmx1*zmz1_3, xmx1_2*zmz1_3, xmx1_3*zmz1_3, ymy1*zmz1_3, xmx1*ymy1*zmz1_3, xmx1_2*ymy1*zmz1_3, xmx1_3*ymy1*zmz1_3, ymy1_2*zmz1_3, xmx1*ymy1_2*zmz1_3, xmx1_2*ymy1_2*zmz1_3, xmx1_3*ymy1_2*zmz1_3, ymy1_3*zmz1_3, xmx1*ymy1_3*zmz1_3, xmx1_2*ymy1_3*zmz1_3, xmx1_3*ymy1_3*zmz1_3};

        int m = 0;
        for(int ii = 0; ii < 4; ii++){
            for(int jj = 0; jj < 4; jj++){
                for(int kk = 0; kk < 4; kk++){
                    int Q0 = x1 + ii - 1;
                    int Q1 = y1 + jj - 1;
                    int Q2 = z1 + kk - 1;
                    if(0 <= Q0 && Q0 < shape0
                    && 0 <= Q1 && Q1 < shape1
                    && 0 <= Q2 && Q2 < shape2){
                        float coefficient = 0;
                        for(int n = 0; n < 64; n++){
                            coefficient += coeffs[m*64 + n] * monomials[n];
                        }
                        output[i*shape1*shape2 + j*shape2 + k] += coefficient * f[Q0*shape1*shape2 + Q1*shape2 + Q2] +input[i*shape1*shape2 + j*shape2 + k];
                    }
                    m++;
                }
            }
        }
    }
}