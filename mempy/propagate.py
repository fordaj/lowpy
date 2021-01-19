import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import pycuda.gpuarray as gpuarray

# Start timer from 0
def tic():
    global t
    t = time.time()

# Lap timer
def toc():
    global t
    print(time.time() - t)


mod = SourceModule("""
    __global__ void propagate(
        const   int                     I,
        const   double * __restrict__   x,
        const   double * __restrict__   w,
        const   double * __restrict__   b,
                double * __restrict__   y,
                double * __restrict__   z
    ){
        int j = blockIdx.x;
        double sum = 0;
        for (int i = 0; i < I; i++) sum += x[i]*w[j*I+i];
        y[j] = sum + b[j];
        z[j] = 1/(1 + exp(-y[j]));
    }
    __global__ void mac(
        const   int                     I,
        const   double * __restrict__   x,
        const   double * __restrict__   w,
        const   double * __restrict__   b,
                double * __restrict__   y
    ){
        extern __shared__ int sdata[];
        // each thread loads one element from global to shared mem
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
        sdata[tid] = g_idata[i];

        __syncthreads();

        // do reduction in shared mem
        for(unsigned int s=1; s < 1024; s *= 2) {
             if (tid % (2*s) == 0) {
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



I_h = 784
I_d = np.int32(I_h)
J_h = 533
J_d = np.int32(I_d)
x_h = np.ones(I_h,dtype=np.float64)
x_d = hostToDevice(x_h)
w_h = np.ones((J_h,I_h),dtype=np.float64)
w_d = hostToDevice(w_h)
b_h = np.ones(J_h,dtype=np.float64)
b_d = hostToDevice(b_h)
y_h = np.zeros(J_h,dtype=np.float64)
y_d = hostToDevice(y_h)
z_h = np.zeros(J_h,dtype=np.float64)
z_d = hostToDevice(z_h)


grid = (J_h, 1, 1)
block = (1, 1, 1)

propagate = mod.get_function("propagate")

tic()
for i in range(60000):
    propagate(
        I_d,
        x_d, 
        w_d, 
        b_d,
        y_d,
        z_d,
        grid=grid, 
        block=block
    )
toc()



"""
tic()
for i in range(600000):
    func.prepared_call(grid, block, a_gpu, b_gpu)
toc()

tic()
for i in range(600000):
    func.prepared_call(grid, block, a_gpu, b_gpu)
toc()
tic()
for i in range(600000):
    twofer.prepared_call(grid, block, a_gpu)
    niner.prepared_call(grid, block, b_gpu)
toc()
"""
