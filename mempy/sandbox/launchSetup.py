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
  __global__ void doublify(float *a, float *b, int i)
  {
    int idx = blockIdx.x + blockIdx.y*4;
    a[idx] *= 2;
    b[idx] = 99;
  }
   __global__ void twofer(float *a)
  {
    int idx = blockIdx.x + blockIdx.y*4;
    a[idx] *= 2;
  }
   __global__ void niner(float *b)
  {
    int idx = blockIdx.x + blockIdx.y*4;
    b[idx] = 99;
  }

  """)


func = mod.get_function("doublify")
twofer = mod.get_function("twofer")
niner = mod.get_function("niner")



a = numpy.random.randn(4,4)
a = a.astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
b_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(b_gpu, a)


grid = (4, 4, 1)
block = (1, 1, 1)
func.prepare("PP")
twofer.prepare("P")
niner.prepare("P")
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
for i in range(600000):
    func(a_gpu, b_gpu, numpy.int32(i), grid=grid, block=block)
toc()

tic()
for i in range(600000):
    func(a_gpu, b_gpu, grid=grid, block=block)
toc()
tic()
for i in range(600000):
    twofer(a_gpu, grid=grid, block=block)
    niner(b_gpu, grid=grid, block=block)
toc()

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)



print(a_doubled)
print(a)