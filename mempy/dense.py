import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import math



class Dense:
    # Constructor for model initialization
    def __init__(self, output_shape, input_shape=784, alpha=0.01, beta=0, sigma_i=0):
        self.I_h = input_shape
        self.J_h = output_shape
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.sigma_i = np.float64(sigma_i)

    # Layer constructor
    def build(self,input_shape,output_shape):
        # Layer dimensions
        self.I_h        = input_shape
        self.I_d        = np.int32(self.I_h)
        self.J_h        = output_shape
        self.J_d        = np.int32(self.J_h)
        # Input
        self.x_h        = np.ones(self.I_h,dtype=np.float64)
        self.x_d        = self.hostToDevice(self.x_h)
        # Weights
        #self.w_h        = np.random.rand(self.J_h,self.I_h).astype(np.float64) * 2 - 1
        #self.w_h        = np.ones((self.J_h,self.I_h),dtype=np.float64) * 0.01
        self.w_h        = np.random.normal(0,math.sqrt(2/784),(self.J_h,self.I_h)).astype(np.float64)
        if (self.sigma_i > 0):
            self.w_h    = np.random.normal(self.w_h,self.sigma_i)
        self.w_d        = self.hostToDevice(self.w_h)
        #self.b_h        = np.random.rand(self.J_h).astype(np.float64) * 2 - 1
        #self.b_h        = np.ones(self.J_h,dtype=np.float64) * 0.01
        self.b_h        = np.random.normal(0,math.sqrt(2/784),self.J_h).astype(np.float64)
        if (self.sigma_i > 0):
            self.b_h    = np.random.normal(self.b_h,self.sigma_i)
        self.b_d        = self.hostToDevice(self.b_h)
        # Momentum
        self.vtw_h      = np.zeros((self.J_h,self.I_h),dtype=np.float64)
        self.vtw_d      = self.hostToDevice(self.vtw_h)
        self.vtb_h      = np.zeros(self.J_h,dtype=np.float64)
        self.vtb_d      = self.hostToDevice(self.vtb_h)
        # Outputs
        self.y_h        = np.zeros(self.J_h,dtype=np.float64)
        self.y_d        = self.hostToDevice(self.y_h)
        self.z_h        = np.zeros(self.J_h,dtype=np.float64)
        self.z_d        = self.hostToDevice(self.z_h)
        # Gradients
        self.dedz_h     = np.zeros(self.J_h,dtype=np.float64)
        self.dedz_d     = self.hostToDevice(self.dedz_h)
        self.dzdy_h     = np.zeros(self.J_h,dtype=np.float64)
        self.dzdy_d     = self.hostToDevice(self.dzdy_h)
        # Next layer attributes
        self.n_J_d      = self.J_d
        self.n_w_d      = self.w_d
        self.n_z_d      = self.z_d
        self.n_dedz_d   = self.dedz_d
        self.n_dzdy_d   = self.dzdy_d
        self.hits_h     = np.zeros(self.J_h,dtype=np.float64)
        self.hits_d     = self.hostToDevice(self.hits_h)
        # Cuda programs
        self.program = SourceModule("""
        __global__ void propagate(
            const int I,
            const double *x,
            const double *w,
            const double *b,
            double *y,
            double *z
            ){
                int j = blockIdx.x;
                double sum = 0;
                for (int i = 0; i < I; i++) sum += x[i]*w[j*I+i];
                y[j] = sum + b[j];
                z[j] = 1/(1 + exp(-y[j]));
            }
        __global__ void backpropagate(
                const int label,
                double *dedz,
                const double *z,
                const int J_n,
                double *w_n,
                const double *dedz_n,
                const double *dzdy_n,
                double *dzdy,
                const double alpha,
                const int I,
                double *b,
                double *w,
                const double *x,
                const double beta,
                double *vtb,
                double *vtw
            ){
                int j   = blockIdx.x;
                int I_n = gridDim.x;
                if (label > -1){
                    if (j == label){
                        dedz[j] = z[j] - 1;
                    }else{
                        dedz[j] = z[j] - 0;
                    }
                }else{
                    double sum = 0;
                    for (int j_n = 0; j_n < J_n; j_n++) sum += w_n[j+j_n*I_n] * dedz_n[j_n] * dzdy_n[j_n];
                    dedz[j] = sum;
                }
                dzdy[j] = z[j] * (1 - z[j]);
                b[j]   -= (beta * vtb[j] + alpha * dedz[j] * dzdy[j]);
                vtb[j]  = beta * vtb[j] + alpha * dedz[j] * dzdy[j];
                for (int i = 0; i < I; i++){
                    w[j*I+i]   -= (beta * vtw[j*I+i] + alpha * dedz[j] * dzdy[j] * x[i]);
                    vtw[j*I+i]  = beta * vtw[j*I+i] + alpha * dedz[j] * dzdy[j] * x[i];
                }
            }
        __global__ void argmax(
                const int label,
                const double *z,
                double *hits
            ){
                int j = blockIdx.x;
                if (j == 0){
                    double maxVal = 0;
                    int maxIdx = 0;
                    for (int i = 0; i < 10; i++){
                        if (z[i] > maxVal){
                            maxIdx = i;
                            maxVal = z[i];
                        }
                    }
                    if (maxIdx == label){
                        hits[0] += 1;
                    }
                    hits[1] = maxIdx;
                    hits[2] = maxVal;
                }
            }
        """)

    # Link attributes from next layer into current layer
    def linkNextLayer(self, nextLayer):
        self.n_J_d      = nextLayer.J_d
        self.n_w_d      = nextLayer.w_d
        self.n_z_d      = nextLayer.z_d
        self.n_dedz_d   = nextLayer.dedz_d
        self.n_dzdy_d   = nextLayer.dzdy_d

    # Set inputs of current layer equal to outputs of previous layer
    def linkPreviousLayer(self, previousLayer):
        self.x_d    = previousLayer.z_d

    # Copy layer attributes from GPU back to host
    def deviceToHost(self):
        cuda.memcpy_dtoh(   self.x_h,       self.x_d    )
        cuda.memcpy_dtoh(   self.w_h,       self.w_d    )
        cuda.memcpy_dtoh(   self.b_h,       self.b_d    )
        cuda.memcpy_dtoh(   self.y_h,       self.y_d    )
        cuda.memcpy_dtoh(   self.z_h,       self.z_d    )
        cuda.memcpy_dtoh(   self.dedz_h,    self.dedz_d )
        cuda.memcpy_dtoh(   self.dzdy_h,    self.dzdy_d )
        cuda.memcpy_dtoh(   self.hits_h,    self.hits_d )
        cuda.memcpy_dtoh(   self.vtb_h,     self.vtb_d  )
        cuda.memcpy_dtoh(   self.vtw_h,     self.vtw_d  )

    #Copy variables from host to GPU
    def hostToDevice(self,host_variable):
        device_variable = cuda.mem_alloc(host_variable.nbytes)
        cuda.memcpy_htod(device_variable, host_variable)
        return device_variable

    # Reset the hit counter
    def resetHits(self):
        self.hits_h = np.zeros(self.J_h,dtype=np.float64)
        self.hits_d = self.hostToDevice(self.hits_h)

    # Propagate 
    def propagate(self):
        propagate = self.program.get_function("propagate")
        propagate(
                self.I_d, 
                self.x_d, 
                self.w_d, 
                self.b_d, 
                self.y_d, 
                self.z_d,
                block=(1,1,1), 
                grid=(self.J_h,1,1)
        )

    # Backpropagate
    def backpropagate(self, label=np.int32(-1)):
        backpropagate = self.program.get_function("backpropagate")
        backpropagate(
                label,
                self.dedz_d,
                self.z_d,
                self.n_J_d,
                self.n_w_d,
                self.n_dedz_d,
                self.n_dzdy_d,
                self.dzdy_d,
                self.alpha,
                self.I_d,
                self.b_d,
                self.w_d,
                self.x_d,
                self.beta,
                self.vtb_d,
                self.vtw_d,
                block=(1,1,1), 
                grid=(self.J_h,1,1)
        )

    # Find winning neuron
    def argmax(self, label, hits_d):
        argmax = self.program.get_function("argmax")
        argmax(
                label,
                self.z_d,
                hits_d,
                block=(1,1,1), 
                grid=(self.J_h,1,1)
        )


