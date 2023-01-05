/*
:file:      utils.cu
:brief:     utils
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


int getDevice(){
    int* device;
    gpuErrchk(cudaGetDevice(device));
    return *device;
}


void setDevice(int device){
    gpuErrchk(cudaSetDevice(device));
}


int getDeviceCount(){
    int* count;
    gpuErrchk(cudaGetDeviceCount(count));
    return *count;
}

std::string getDeviceName(int device){
    cudaDeviceProp* prop;
    gpuErrchk(cudaGetDeviceProperties(prop, device));
    std::string name(prop->name);
    return name;
}