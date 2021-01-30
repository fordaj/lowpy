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
    def __init__(self, output_shape, input_shape=784, alpha=0.01, beta=0, initialization_type="uniform", initialization_parameter=2, sigma=0):
        # Layer dimensions are invariable across networks
        self.I                          = input_shape
        self.J                          = output_shape
        # Varied hyperparameters on a per-layer basis
        self.alpha                      = gpuarray.to_gpu(np.array(alpha,dtype=np.float64))
        self.beta                       = gpuarray.to_gpu(np.array(beta,dtype=np.float64))
        self.sigma                      = gpuarray.to_gpu(np.array(sigma,dtype=np.float64))
        self.initialization_type        = initialization_type
        self.initialization_parameter   = initialization_parameter
        self.gpu                        = self.functors()

    # Layer constructor
    def build(self,number_of_networks,input_shape,output_shape):
        # Layer dimensions
        (self.V,self.I,self.J)  = (number_of_networks, input_shape, output_shape)
        # Input
        self.x          = gpuarray.zeros((self.V,self.I),dtype=np.float64)
        # Weight Initializations
        self.w          = np.random.rand(self.V,self.J,self.I).astype(np.float64)
        self.b          = np.random.rand(self.V,self.J).astype(np.float64)
        for l in range(self.V):
            if (self.initialization_type == "uniform"):
                self.w[l]       = self.w[l] * self.initialization_parameter[l] - self.initialization_parameter[l]/2
                self.b[l]       = self.b[l] * self.initialization_parameter[l] - self.initialization_parameter[l]/2
            elif (self.initialization_type == "normal"):
                self.w          = np.random.normal(0,self.initialization_parameter[l],(self.J,self.I)).astype(np.float64)
                self.b          = np.random.normal(0,self.initialization_parameter[l],(self.J)).astype(np.float64)
            # Write variability
            if (self.sigma[l].get() > 0):
                self.w[l]       = np.random.normal(self.w[l],self.sigma[l].get())
                self.b[l]       = np.random.normal(self.b[l],self.sigma[l].get())
        self.w          = gpuarray.to_gpu(self.w)
        self.b          = gpuarray.to_gpu(self.b)
        np.random.normal()
        # Momentum
        self.mw        = gpuarray.zeros((self.V,self.J,self.I),dtype=np.float64)
        self.mb        = gpuarray.zeros((self.V,self.J),dtype=np.float64)
        # Outputs
        self.y          = gpuarray.zeros((self.V,self.J),dtype=np.float64)
        self.z          = gpuarray.zeros((self.V,self.J),dtype=np.float64)
        # Gradients
        self.dedz       = gpuarray.zeros((self.V,self.J),dtype=np.float64)
        self.dzdy       = gpuarray.zeros((self.V,self.J),dtype=np.float64)
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
            (self.V,self.J,1),
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
            (self.V,self.J,1),
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
            (self.V,self.J,1),
            (1,1,1),
            np.int32(label),
            self.z.gpudata,
            hits.gpudata
        )