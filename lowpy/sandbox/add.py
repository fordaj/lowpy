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

N = 500000

mod = SourceModule("""
    __global__ void add(float * __restrict__ a, float * __restrict__ b )
    {
        int idx = blockIdx.x;
        b[idx] = 2 * a[idx];
  }

  """)


func = mod.get_function("add")



a = numpy.random.randn(N)
a = a.astype(numpy.int32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
b_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(b_gpu, a)


grid = (N, 1, 1)
block = (1, 1, 1)
func.prepare("PP")
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

tic()
for i in range(100000):
    func(a_gpu, b_gpu, grid=grid, block=block)
toc()


a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)



print(a_doubled)
print(a)