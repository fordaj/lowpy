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

class Sequential:
    def __init__(self):
        self.layer = []
    def add(self,layer_type):
        self.layer.append(layer_type)
        numLayers = len(self.layer)
        if (numLayers == 1):
            self.layer[0].build(self.layer[0].I,self.layer[0].J)
        else:
            self.layer[numLayers-1].build(self.layer[numLayers-2].J,self.layer[numLayers-1].J)
            self.layer[numLayers-1].linkPreviousLayer(self.layer[numLayers-2])
            self.layer[numLayers-2].linkNextLayer(self.layer[numLayers-1])
    def deviceToHost(self):
        for l in self.layer:
            l.deviceToHost()
    def propagate(self):
        for l in self.layer:
            l.propagate()
    def backpropagate(self):
        numLayers = len(self.layer)
        self.layer[numLayers-1].backpropagate(np.float32(5))
        for l in range(numLayers-2,-1,-1):
            self.layer[l].backpropagate()


class Dense:
    def __init__(self, output_shape, input_shape=784, alpha=0.01):
        self.I = input_shape
        self.J = output_shape
        self.alpha = np.float32(alpha)
    def build(self,input_shape,output_shape):
        self.I = input_shape
        self.J = output_shape
        self.x = np.ones(self.I,dtype=np.float32)
        self.x_d = cuda.mem_alloc(self.x.nbytes)
        cuda.memcpy_htod(self.x_d, self.x)
        self.w = np.random.rand(self.J,self.I).astype(np.float32) * 2 - 1
        self.w_d = cuda.mem_alloc(self.w.nbytes)
        cuda.memcpy_htod(self.w_d, self.w)
        self.b = np.random.rand(self.J).astype(np.float32) * 2 - 1
        self.b_d = cuda.mem_alloc(self.b.nbytes)
        cuda.memcpy_htod(self.b_d, self.b)
        self.y = np.zeros(self.J,dtype=np.float32)
        self.y_d = cuda.mem_alloc(self.y.nbytes)
        cuda.memcpy_htod(self.y_d, self.y)
        self.z = np.zeros(self.J,dtype=np.float32)
        self.z_d = cuda.mem_alloc(self.z.nbytes)
        cuda.memcpy_htod(self.z_d, self.z)
        self.dedz = np.zeros(self.J,dtype=np.float32)
        self.dedz_d = cuda.mem_alloc(self.dedz.nbytes)
        cuda.memcpy_htod(self.dedz_d, self.dedz)
        self.dzdy = np.zeros(self.J,dtype=np.float32)
        self.dzdy_d = cuda.mem_alloc(self.dzdy.nbytes)
        cuda.memcpy_htod(self.dzdy_d, self.dzdy)
        self.J_n = self.J
        self.w_dn = self.w_d
        self.z_dn = self.z_d
        self.dedz_dn = self.dedz_d
        self.dzdy_dn = self.dzdy_d
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
        __global__ void backpropagate(
                float alpha,
                float *dedz,
                float *dzdy,
                const float *z,
                float *b,
                float label = -1,
                int J_n = -1,
                const float *dedz_n = NULL,
                const float *dzdy_n = NULL,
                const float *w_n = NULL
            ){
                int j = blockIdx.x;
                if (label > -1){
                    if (j == label){
                        dedz[j] = z[j] - 1;
                    }else{
                        dedz[j] = z[j] - 0;
                    }
                }else{
                    int J = blockDim.x;
                    float sum = 0;
                    for (unsigned long i = 0; i < J_n; i++) sum += w_n[i*J+j] * dedz_n[i] * dzdy_n[i];
                    dedz[j] = sum;
                }
                dzdy[j] = z[j] * (1 - z[j]);
                b[j] -= alpha * dedz[j] * dzdy[j];
            }
        """)
    def linkNextLayer(self, nextLayer):
        self.J_n = nextLayer.J
        self.w_dn = nextLayer.w
        self.z_dn = nextLayer.z_d
        self.dedz_dn = nextLayer.dedz_d
        self.dzdy_dn = nextLayer.dzdy_d
    def linkPreviousLayer(self, previousLayer):
        self.x = previousLayer.z
        self.x_d = previousLayer.z_d
    def deviceToHost(self):
        cuda.memcpy_dtoh(self.x, self.x_d)
        cuda.memcpy_dtoh(self.w, self.w_d)
        cuda.memcpy_dtoh(self.b, self.b_d)
        cuda.memcpy_dtoh(self.y, self.y_d)
        cuda.memcpy_dtoh(self.z, self.z_d)
        cuda.memcpy_dtoh(self.dedz, self.dedz_d)
        cuda.memcpy_dtoh(self.dzdy, self.dzdy_d)
    def propagate(self):
        propagate = self.program.get_function("propagate")
        propagate(np.int32(self.I), self.x_d, self.w_d, self.b_d, self.y_d, self.z_d, block=(1,1,1), grid=(self.J,1,1))
    def inference(self,label):
        i=20
    def backpropagate(self, label=np.float32(-1)):
        backpropagate = self.program.get_function("backpropagate")
        if label == np.float32(-1):
            backpropagate(self.alpha, self.dedz_d, self.dzdy_d, self.z_d, self.b_d, label, block=(1,1,1), grid=(self.J,1,1))
        else:
            backpropagate(self.alpha, self.dedz_d, self.dzdy_d, self.z_d, self.b_d, label, np.int32(self.J_n), self.dedz_dn, self.dzdy_dn, self.w_dn, block=(1,1,1), grid=(self.J,1,1))



# Create the model
# model = Sequential()
# model.add(Dense(350, input_shape=input_shape, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

model = Sequential()
model.add(Dense(533,input_shape=784))
model.add(Dense(533))
model.add(Dense(10))

tic()
model.deviceToHost()
model.propagate()
model.deviceToHost()
model.backpropagate()
model.deviceToHost()
toc()