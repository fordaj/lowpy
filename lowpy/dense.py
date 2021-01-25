import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curand
import numpy as np
import math
import time
import pkg_resources

# Start timer from 0
def tic():
    global t
    t = time.time()

# Lap timer
def toc():
    global t
    print(time.time() - t)


class Dense:
    # GPU functors
    class functors:
        def __init__(self):
            self.propagate      = None
            self.backpropagate  = None
            self.argmax         = None

    # Constructor for model initialization
    def __init__(self, output_shape, input_shape=784, alpha=0.01, beta=0, weight_initialization="normal", sigma_i=0):
        # Layer dimensions are invariable across networks
        self.I                      = input_shape
        self.J                      = output_shape
        self.alpha                  = gpuarray.to_gpu(np.array(alpha,dtype=np.float64))
        self.beta                   = gpuarray.to_gpu(np.array(beta,dtype=np.float64))
        self.weight_initialization  = weight_initialization
        self.sigma_i                = gpuarray.to_gpu(np.array(sigma_i,dtype=np.float64))
        self.gpu                    = self.functors()

    # Layer constructor
    def build(self,number_of_networks,input_shape,output_shape):
        # Layer dimensions
        (self.L,self.I,self.J)  = (number_of_networks, input_shape, output_shape)
        # Input
        self.x          = gpuarray.zeros((self.L,self.I),dtype=np.float64)
        # Weights
        if (self.weight_initialization=="uniform"):
            self.w          = np.random.rand(self.L,self.J,self.I).astype(np.float64) * 2 - 1
            self.b          = np.random.rand(self.L,self.J).astype(np.float64) * 2 - 1
        else:
            self.w          = np.random.normal(0,math.sqrt(2/self.I),(self.L,self.J,self.I)).astype(np.float64)
            self.b          = np.random.normal(0,math.sqrt(2/self.I),(self.L,self.J)).astype(np.float64)
        for l in range(self.L):
            if (self.sigma_i[l].get() > 0):
                self.w[l]       = np.random.normal(self.w[l],self.sigma_i[l].get())
                self.b[l]       = np.random.normal(self.b[l],self.sigma_i[l].get())
        self.w          = gpuarray.to_gpu(self.w)
        self.b          = gpuarray.to_gpu(self.b)
        np.random.normal()
        # Momentum
        self.mw        = gpuarray.zeros((self.L,self.J,self.I),dtype=np.float64)
        self.mb        = gpuarray.zeros((self.L,self.J),dtype=np.float64)
        # Outputs
        self.y          = gpuarray.zeros((self.L,self.J),dtype=np.float64)
        self.z          = gpuarray.zeros((self.L,self.J),dtype=np.float64)
        # Gradients
        self.dedz       = gpuarray.zeros((self.L,self.J),dtype=np.float64)
        self.dzdy       = gpuarray.zeros((self.L,self.J),dtype=np.float64)
        # Next layer attributes
        self.n_J        = self.J
        self.n_w        = self.w
        self.n_z        = self.z
        self.n_dedz     = self.dedz
        self.n_dzdy     = self.dzdy
        # Cuda kernels
        self.program            = SourceModule(open(pkg_resources.resource_filename('lowpy', 'dense.cu')).read())
        self.gpu.propagate      = self.program.get_function("propagate")
        self.gpu.backpropagate  = self.program.get_function("backpropagate")
        self.gpu.argmax         = self.program.get_function("argmax")
        self.gpu.propagate.prepare("iiPPPPP")
        self.gpu.backpropagate.prepare("iiPPiPPPPPiPPPPPP")
        self.gpu.argmax.prepare("iPP")

    # Link attributes from next layer into current layer
    def linkNextLayer(self, nextLayer):
        self.n_J        = nextLayer.J
        self.n_w        = nextLayer.w
        self.n_z        = nextLayer.z
        self.n_dedz     = nextLayer.dedz
        self.n_dzdy     = nextLayer.dzdy

    # Set inputs of current layer equal to outputs of previous layer
    def linkPreviousLayer(self, previousLayer):
        self.x  = previousLayer.z

    # Propagate 
    def propagate(self,iteration=-1):
        self.gpu.propagate.prepared_call(
            (self.L,self.J,1),
            (1,1,1),
            np.int32(self.I), 
            np.int32(iteration),
            self.x.gpudata, 
            self.w.gpudata, 
            self.b.gpudata, 
            self.y.gpudata, 
            self.z.gpudata
        )

    # Backpropagate
    def backpropagate(self,iteration=-1,label=-1):
        self.gpu.backpropagate.prepared_call(
            (self.L,self.J,1),
            (1,1,1), 
            np.int32(iteration),
            np.int32(label),
            self.dedz.gpudata,
            self.z.gpudata,
            np.int32(self.n_J),
            self.n_w.gpudata,
            self.n_dedz.gpudata,
            self.n_dzdy.gpudata,
            self.dzdy.gpudata,
            self.alpha.gpudata,
            np.int32(self.I),
            self.b.gpudata,
            self.w.gpudata,
            self.x.gpudata,
            self.beta.gpudata,
            self.mb.gpudata,
            self.mw.gpudata
        )

    # Find winning neuron
    def argmax(self, label, hits):
        self.gpu.argmax.prepared_call(
            (self.L,self.J,1),
            (1,1,1),
            np.int32(label),
            self.z.gpudata,
            hits.gpudata
        )
        

