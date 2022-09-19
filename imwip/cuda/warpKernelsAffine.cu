/*
:file:      warpKernelsAffine.cu
:brief:     Affine warping kernels
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
*/

__global__ void affineLinearWarp2DKernel(
        const float* f,
        const float* A,
        const float* b,
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
        float x = A[0]*i + A[1]*j + b[0];
        float y = A[2]*i + A[3]*j + b[1];

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


__global__ void adjointAffineLinearWarp2DKernel(
        const float* fWarped,
        const float* A,
        const float* b,
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
        float x = A[0]*i + A[1]*j + b[0];
        float y = A[2]*i + A[3]*j + b[1];

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


__global__ void affineLinearWarp3DKernel(
        const float* f,
        const float* A,
        const float* b,
        float* fWarped,
        int shape0,
        int shape1,
        int shape2
    ){
    
    /*
    Kernel of GPU implementation of 3D backward image warping along the DVF (u,v)
    with linear interpolation (rectangular multivariate spline)
    */

    // this order is faster then the other way around
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = A[0]*i + A[1]*j + A[2]*k + b[0];
        float y = A[3]*i + A[4]*j + A[5]*k + b[1];
        float z = A[6]*i + A[7]*j + A[8]*k + b[2];

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


__global__ void adjointAffineLinearWarp3DKernel(
        const float* fWarped,
        const float* A,
        const float* b,
        float* f,
        int shape0,
        int shape1,
        int shape2
    ){
    
    /*
    Kernel of GPU implementation of 3D adjoint backward image warping along the
    DVF (u,v) with linear interpolation (rectangular multivariate spline)
    */

    // this order is faster then the other way around
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = A[0]*i + A[1]*j + A[2]*k + b[0];
        float y = A[3]*i + A[4]*j + A[5]*k + b[1];
        float z = A[6]*i + A[7]*j + A[8]*k + b[2];

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


__global__ void affineCubicWarp2DKernel(
        const float* f,
        const float* A,
        const float* b,
        float* fWarped,
        int shape0,
        int shape1,
        const float* coeffs
    ){


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = A[0]*i + A[1]*j + b[0];
        float y = A[2]*i + A[3]*j + b[1];

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


__global__ void adjointAffineCubicWarp2DKernel(
        const float* fWarped,
        const float* A,
        const float* b,
        float* f,
        int shape0,
        int shape1,
        const float* coeffs
    ){

    /*
    Kernel of GPU implementation of adjoint backward image warping along the
    DVF (u,v) with cubic interpolation (rectangular multivariate catmull-rom spline)
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < shape0 && j < shape1){

        // position at which to iterpolate
        float x = A[0]*i + A[1]*j + b[0];
        float y = A[2]*i + A[3]*j + b[1];

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


__global__ void affineCubicWarp3DKernel(
        const float* f,
        const float* A,
        const float* b,
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

    // this order is faster then the other way around
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = A[0]*i + A[1]*j + A[2]*k + b[0];
        float y = A[3]*i + A[4]*j + A[5]*k + b[1];
        float z = A[6]*i + A[7]*j + A[8]*k + b[2];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        // xi = x1 - 1 + i

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


__global__ void adjointAffineCubicWarp3DKernel(
        const float* fWarped,
        const float* A,
        const float* b,
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

    // this order is faster then the other way around
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < shape0 && j < shape1 && k < shape2){

        // position at which to iterpolate
        float x = A[0]*i + A[1]*j + A[2]*k + b[0];
        float y = A[3]*i + A[4]*j + A[5]*k + b[1];
        float z = A[6]*i + A[7]*j + A[8]*k + b[2];

        // points from which to interpolate
        int x1 = floorf(x);
        int y1 = floorf(y);
        int z1 = floorf(z);
        // xi = x1 - 1 + i

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