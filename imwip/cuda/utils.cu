/*
:file:      utils.cu
:brief:     utils
:date:      1 AUG 2022
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
*/
#include <stdio.h>
#include <utils.hu>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void setDevice(int device){
    gpuErrchk(cudaSetDevice(device));
}