import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
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
  __global__ void doublify(float *a, float *b)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
    b[idx] = 99;
  }
  """)
func = mod.get_function("doublify")




a = numpy.random.randn(4,4)
a = a.astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
b_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(b_gpu, a)

grid = (1, 1)
block = (4, 4, 1)
func.prepare("PP")

tic()
for i in range(600000):
    func.prepared_call(grid, block, a_gpu, b_gpu)
toc()

tic()
for i in range(600000):
    func(a_gpu, b_gpu, grid=grid, block=block)
toc()


a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)



print(a_doubled)
print(a)