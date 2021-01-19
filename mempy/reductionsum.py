import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.gpuarray as gpuarray
import time

# Start timer from 0
def tic():
    global t
    t = time.time()

# Lap timer
def toc():
    global t
    print(str(time.time() - t) + str("\t"), end='')


    
N = 69
blockSize = 32
gridSize = N/blockSize
remaining = N
recursions = 0
while (remaining > 1):
    remaining /= blockSize
    recursions += 1



program = SourceModule("""
    __global__ void reduce0(
        int *g_idata, 
        int *g_odata
    ){ 
        extern __shared__ int sdata[];
        // each thread loads one element from global to shared mem
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
        sdata[tid] = g_idata[i];

        __syncthreads();

        // do reduction in shared mem
        for(unsigned int s=1; s < blockDim.x; s *= 2) {
             if (tid % (2*s) == 0) {
                sdata[tid] += sdata[tid + s]; 
            }
            __syncthreads(); 
        }
        
        // write result for this block to global mem
        if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
    }

    __global__ void reduce1(
        int *g_idata, 
        int *g_odata
    ){ 
        extern __shared__ int sdata[];
        // each thread loads one element from global to shared mem
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
        sdata[tid] = g_idata[i];

        __syncthreads();

        // do reduction in shared mem
        for (unsigned int s=1; s < blockDim.x; s *= 2) {
            int index = 2 * s * tid;
            if (index < blockDim.x) { 
                sdata[index] += sdata[index + s];
            }
            __syncthreads(); 
        }
        
        // write result for this block to global mem
        if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
    }

    __global__ void reduce2(
        int *g_idata, 
        int *g_odata
    ){ 
        extern __shared__ int sdata[];
        // each thread loads one element from global to shared mem
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
        sdata[tid] = g_idata[i];

        __syncthreads();

        // do reduction in shared mem
        for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s]; 
            }
            __syncthreads(); 
        }
        
        // write result for this block to global mem
        if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
    }

    __global__ void reduce3(
        int *g_idata, 
        int *g_odata
    ){ 
        extern __shared__ int sdata[];
        // each thread loads one element from global to shared mem
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
        sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];

        __syncthreads();

        // do reduction in shared mem
        for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s]; 
            }
            __syncthreads(); 
        }
        
        // write result for this block to global mem
        if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
    }

    


""")

#Copy variables from host to GPU
def hostToDevice(host_variable):
    device_variable = cuda.mem_alloc(host_variable.nbytes)
    cuda.memcpy_htod(device_variable, host_variable)
    return device_variable


def sum(sumType):

    print(sumType + str("\t"), end='')

    func = program.get_function(sumType)

    idata = np.zeros(N,dtype=np.int32) + 1
    g_idata = hostToDevice(idata)
    odata = np.zeros(N,dtype=np.int32)
    g_odata = hostToDevice(odata)
    func.prepare("PP")

    tic()
    for n in range(1000):
        for i in range(recursions):
            func.prepared_call(
                    (int(gridSize),1,1),
                    (int(blockSize),1,1), 
                    g_idata, 
                    g_odata,
                    shared_size=int(blockSize)
            )
            g_idata = g_odata
    toc()
    cuda.memcpy_dtoh(odata, g_odata)
    print(str(odata) + str("\t"), end='')

    func = program.get_function(sumType)
    idata = np.zeros(N,dtype=np.int32) + 1
    g_idata = hostToDevice(idata)
    odata = np.zeros(N,dtype=np.int32)
    g_odata = hostToDevice(odata)

    tic()
    for n in range(1000):
        for i in range(recursions):
            func(
                    g_idata, 
                    g_odata,
                    grid=(int(gridSize),1,1),
                    block=(int(blockSize),1,1), 
                    shared=int(blockSize)
            )
            g_idata = g_odata
    toc()
    cuda.memcpy_dtoh(odata, g_odata)
    print(str(odata))




sum("reduce0")
sum("reduce1")
sum("reduce2")
sum("reduce3")