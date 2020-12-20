import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time


def tic():
    global t
    t = time.time()

def toc():
    global t
    print(time.time() - t)



class dense:
    def __init__(self, number_of_inputs, number_of_outputs):
        self.I = number_of_inputs
        self.J = number_of_outputs
        self.x = np.ones(self.I,dtype=np.float32)
        self.x_d = cuda.mem_alloc(self.x.nbytes)
        cuda.memcpy_htod(self.x_d, self.x)
        self.w = np.ones((self.J,self.I),dtype=np.float32)
        self.w_d = cuda.mem_alloc(self.w.nbytes)
        cuda.memcpy_htod(self.w_d, self.w)
        self.b = np.ones(self.J,dtype=np.float32)
        self.b_d = cuda.mem_alloc(self.b.nbytes)
        cuda.memcpy_htod(self.b_d, self.b)
        self.y = np.zeros(self.J,dtype=np.float32)
        self.y_d = cuda.mem_alloc(self.b.nbytes)
        cuda.memcpy_htod(self.y_d, self.y)
        self.z = np.zeros(self.J,dtype=np.float32)
        self.z_d = cuda.mem_alloc(self.b.nbytes)
        cuda.memcpy_htod(self.z_d, self.z)
        self.program = SourceModule("""
        __global__ void propagate(
            const int I,
            const float *x,
            const float *w,
            const float *b,
            float *y,
            float *z
            ){
                int j = blockIdx.x;
                float sum = 0;
                for (int i = 0; i < I; i++) sum += x[i]*w[j*I+i];
                y[j] = sum + b[j];
                z[j] = 1/(1 + exp(-y[j]));
            }
        """)
    def propagate(self):
        propagate = self.program.get_function("propagate")
        propagate(np.int32(self.I), self.x_d, self.w_d, self.b_d, self.y_d, self.z_d, block=(1,1,1), grid=(self.J,1,1))


layer = dense(784,5)
tic()
for i in range(60000):
    layer.propagate()
    layer.propagate()
    layer.propagate()
    layer.propagate()
    layer.propagate()
    layer.propagate()
    layer.propagate()
    layer.propagate()
    layer.propagate()
toc()
cuda.memcpy_dtoh(layer.y, layer.y_d)
print(layer.y)
cuda.memcpy_dtoh(layer.z, layer.z_d)
print(layer.z)