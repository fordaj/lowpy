import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
from keras.datasets import mnist
import numpy as np
import math

# Start timer from 0
def tic():
    global t
    t = time.time()

# Lap timer
def toc():
    global t
    print(time.time() - t)

#Copy variables from host to GPU
def hostToDevice(host_variable):
    device_variable = cuda.mem_alloc(host_variable.nbytes)
    cuda.memcpy_htod(device_variable, host_variable)
    return device_variable



###############################
#----- MODEL ARCHITECTURE-----#
###############################

class Sequential:
    # Constructor for model initialization
    def __init__(self):
        self.layer = []
        self.numLayers = 0

    # Function for appending layer objects to the model
    def add(self,layer_type):
        self.layer.append(layer_type)
        numLayers = len(self.layer)
        if (numLayers == 1):
            self.layer[0].build(self.layer[0].I_h,self.layer[0].J_h)
        else:
            self.layer[numLayers-1].build(self.layer[numLayers-2].J_h,self.layer[numLayers-1].J_h)
            self.layer[numLayers-1].linkPreviousLayer(self.layer[numLayers-2])
            self.layer[numLayers-2].linkNextLayer(self.layer[numLayers-1])
        self.numLayers += 1
    
    # Copy all layer attributes from GPU to host
    def deviceToHost(self):
        for l in self.layer:
            l.deviceToHost()

    # Forward pass
    def propagate(self, input):
        self.layer[0].x_d = input
        for l in self.layer:
            l.propagate()
            
    # Backward pass
    def backpropagate(self,label):
        self.layer[self.numLayers-1].backpropagate(label)
        for l in range(self.numLayers-2,-1,-1):
            self.layer[l].backpropagate()

    # Decision
    def inference(self,label,hits_d):
        self.layer[self.numLayers-1].argmax(label,hits_d)

    # Track model data
    class metrics:
        def __init__(self):
            self.train = self.trialData()
            self.test = self.trialData()
            self.alpha = []
            self.beta = []
        class trialData:
            def __init__(self):
                self.iteration = []
                self.accuracy = []
                self.loss = []

    # Import dataset
    def importDataset(self,trainData,trainLabels,testData,testLabels):
        numTrain = len(trainData)
        self.trainData_d     = []
        self.trainLabels_d   = []
        self.testData_d      = []
        self.testLabels_d    = []
        for i in range(numTrain):
            self.trainData_d.append(hostToDevice(trainData[i]))
        self.trainLabels_d = np.int32(trainLabels)
        for i in range(len(testData)):
            self.testData_d.append(hostToDevice(testData[i]))
        self.testLabels_d = np.int32(testLabels)

    # Test model
    def validate(self):
        numTests = len(self.testData_d)
        testHits_h = np.zeros(3,dtype=np.float64)
        testHits_d = hostToDevice(testHits_h)
        self.layer[self.numLayers-1].hits_h = 0
        for i in range(numTests):
            self.propagate(self.testData_d[i])
            self.inference(self.testLabels_d[i], testHits_d)
        cuda.memcpy_dtoh(testHits_h, testHits_d)
        accuracy = testHits_h[0]/numTests
        loss = 1-accuracy
        self.history.test.accuracy.append(accuracy)
        self.history.test.loss.append(loss)
        print("Testing\t\tAccuracy: " + f'{accuracy*100:.3f}' + "%\tLoss: " + f'{loss:.5f}')
        if (accuracy <= self.peakAccuracy):
            self.numConverged += 1
        else:
            self.numConverged = 0
            self.peakAccuracy = accuracy

    # Train model
    def fit(self, trainData, trainLabels, validation_data, epochs, batch_size=-1):
        self.history = self.metrics()
        self.importDataset(trainData,trainLabels,testData,testLabels)
        numTrain = len(trainData)
        self.numConverged = 0
        self.peakAccuracy = 0
        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            self.validate()
            trainHits_h = np.zeros(3,dtype=np.float64)
            trainHits_d = hostToDevice(trainHits_h)
            self.layer[self.numLayers-1].resetHits()
            for i in range(numTrain):
                #self.deviceToHost()
                self.propagate(self.trainData_d[i])
                #self.deviceToHost()
                self.backpropagate(self.trainLabels_d[i])
                #self.deviceToHost()
                self.inference(self.trainLabels_d[i],trainHits_d)
            self.deviceToHost()
            cuda.memcpy_dtoh(trainHits_h, trainHits_d)
            accuracy = trainHits_h[0]/numTrain
            loss = 1 - accuracy
            self.history.train.accuracy.append(accuracy)
            self.history.train.loss.append(loss)
            print("Training\tAccuracy: " + f'{accuracy*100:.3f}' + "%\tLoss: " + f'{loss:.5f}')
            if (self.numConverged == 3):
                break
        return self.history



#################################
#-----FULLY CONNECTED LAYER-----#
#################################

class Dense:
    # Constructor for model initialization
    def __init__(self, output_shape, input_shape=784, alpha=0.01, beta=0):
        self.I_h = input_shape
        self.J_h = output_shape
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)

    # Layer constructor
    def build(self,input_shape,output_shape):
        # Layer dimensions
        self.I_h    = input_shape
        self.I_d    = np.int32(self.I_h)
        self.J_h    = output_shape
        self.J_d    = np.int32(self.J_h)
        # Input
        self.x_h        = np.ones(self.I_h,dtype=np.float64)
        self.x_d        = hostToDevice(self.x_h)
        # Weights
        #self.w_h        = np.random.rand(self.J_h,self.I_h).astype(np.float64) * 2 - 1
        #self.w_h        = np.ones((self.J_h,self.I_h),dtype=np.float64) * 0.01
        self.w_h        = np.random.normal(0,math.sqrt(2/784),(self.J_h,self.I_h)).astype(np.float64)
        self.w_d        = hostToDevice(self.w_h)
        #self.b_h        = np.random.rand(self.J_h).astype(np.float64) * 2 - 1
        #self.b_h        = np.ones(self.J_h,dtype=np.float64) * 0.01
        self.b_h        = np.random.normal(0,math.sqrt(2/784),self.J_h).astype(np.float64)
        self.b_d        = hostToDevice(self.b_h)
        # Momentum
        self.vtw_h      = np.zeros((self.J_h,self.I_h),dtype=np.float64)
        self.vtw_d      = hostToDevice(self.vtw_h)
        self.vtb_h      = np.zeros(self.J_h,dtype=np.float64)
        self.vtb_d      = hostToDevice(self.vtb_h)
        # Outputs
        self.y_h        = np.zeros(self.J_h,dtype=np.float64)
        self.y_d        = hostToDevice(self.y_h)
        self.z_h        = np.zeros(self.J_h,dtype=np.float64)
        self.z_d        = hostToDevice(self.z_h)
        # Gradients
        self.dedz_h     = np.zeros(self.J_h,dtype=np.float64)
        self.dedz_d     = hostToDevice(self.dedz_h)
        self.dzdy_h     = np.zeros(self.J_h,dtype=np.float64)
        self.dzdy_d     = hostToDevice(self.dzdy_h)
        # Next layer attributes
        self.n_J_d      = self.J_d
        self.n_w_d      = self.w_d
        self.n_z_d      = self.z_d
        self.n_dedz_d   = self.dedz_d
        self.n_dzdy_d   = self.dzdy_d
        self.hits_h     = np.zeros(self.J_h,dtype=np.float64)
        self.hits_d     = hostToDevice(self.hits_h)
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
        cuda.memcpy_dtoh(self.x_h, self.x_d)
        cuda.memcpy_dtoh(self.w_h, self.w_d)
        cuda.memcpy_dtoh(self.b_h, self.b_d)
        cuda.memcpy_dtoh(self.y_h, self.y_d)
        cuda.memcpy_dtoh(self.z_h, self.z_d)
        cuda.memcpy_dtoh(self.dedz_h, self.dedz_d)
        cuda.memcpy_dtoh(self.dzdy_h, self.dzdy_d)
        cuda.memcpy_dtoh(self.hits_h, self.hits_d)
        cuda.memcpy_dtoh(self.vtb_h, self.vtb_d)
        cuda.memcpy_dtoh(self.vtw_h, self.vtw_d)

    # Reset the hit counter
    def resetHits(self):
        self.hits_h = np.zeros(self.J_h,dtype=np.float64)
        self.hits_d = hostToDevice(self.hits_h)

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




#-----IMPORT DATASET-----#
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
trainData = np.float64(trainData.reshape(trainData.shape[0],trainData.shape[1]*trainData.shape[2])) # Reshape train data from [n,28,28] to [n,784]
trainLabels = np.int32(trainLabels)
testData = np.float64(testData.reshape(testData.shape[0],testData.shape[1]*testData.shape[2])) # Reshape test data from [n,28,28] to [n,784]
testLabels = np.int32(testLabels)

#-----NORMALIZE DATA-----#
trainData = np.true_divide(trainData, max(np.max(trainData), np.max(testData)))
testData = np.true_divide(testData, max(np.max(trainData), np.max(testData)))

#-----BUILD MODEL-----#
model = Sequential()
model.add(Dense(    10,     input_shape=784,    alpha=1,    beta=0  ))
#model.add(Dense(    533,                        alpha=0.025,    beta=0.1    ))
#model.add(Dense(    10,                         alpha=0.025,    beta=0.1    ))

#-----TRAIN MODEL-----#
history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels))
print(history)











# Create the model
# model = Sequential()
# model.add(Dense(350, input_shape=input_shape, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))