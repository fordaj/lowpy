import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# Start timer from 0
def tic():
    global t
    t = time.time()

# Lap timer
def toc():
    global t
    print(time.time() - t)

class Sequential:
    # Constructor for model initialization
    def __init__(self):
        self.layer = []
        self.numLayers = 0
        self.verbose = True

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

    #Copy variables from host to GPU
    def hostToDevice(self,host_variable):
        device_variable = cuda.mem_alloc(host_variable.nbytes)
        cuda.memcpy_htod(device_variable, host_variable)
        return device_variable

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

    # Model architecture
    def describe(self):
        print("---------------------------------------------------------------")
        print ("{:<7} {:<11} {:<15} {:<15} {:<12}".format("Type", "Neurons", "Learning Rate", "Momentum Rate","Variability"))
        for l in self.layer:
            print ("{:<7} {:<11} {:<15} {:<15} {:<12}".format("Dense", str(l.I_h)+"->"+str(l.J_h), f'{l.alpha:.5f}', f'{l.beta:.5f}', f'{l.sigma_i:.5f}'))
        print("---------------------------------------------------------------")

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
            self.trainData_d.append(self.hostToDevice(trainData[i]))
        self.trainLabels_d = np.int32(trainLabels)
        for i in range(len(testData)):
            self.testData_d.append(self.hostToDevice(testData[i]))
        self.testLabels_d = np.int32(testLabels)

    # Test model
    def validate(self):
        numTests = len(self.testData_d)
        testHits_h = np.zeros(3,dtype=np.float64)
        testHits_d = self.hostToDevice(testHits_h)
        self.layer[self.numLayers-1].hits_h = 0
        for i in range(numTests):
            self.propagate(self.testData_d[i])
            self.inference(self.testLabels_d[i], testHits_d)
        cuda.memcpy_dtoh(testHits_h, testHits_d)
        accuracy = testHits_h[0]/numTests
        loss = 1-accuracy
        self.history.test.accuracy.append(accuracy)
        self.history.test.loss.append(loss)
        if (self.verbose):
            print("Testing\t\tAccuracy: " + f'{accuracy*100:.3f}' + "%\tLoss: " + f'{loss:.5f}')
        if (accuracy <= self.peakAccuracy):
            self.numConverged += 1
        else:
            self.numConverged = 0
            self.peakAccuracy = accuracy

    # Train model
    def fit(self, trainData, trainLabels, validation_data, epochs, batch_size=-1, verbose=True):
        self.verbose = verbose
        self.history = self.metrics()
        testData = validation_data[0]
        testLabels = validation_data[1]
        self.importDataset(trainData,trainLabels,testData,testLabels)
        numTrain = 1000#len(trainData)
        self.numConverged = 0
        self.peakAccuracy = 0
        self.describe()
        tic()
        for epoch in range(epochs):
            if (self.verbose):
                print("Epoch " + str(epoch))
            self.validate()
            trainHits_h = np.zeros(3,dtype=np.float64)
            trainHits_d = self.hostToDevice(trainHits_h)
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
            if (verbose):
                print("Training\tAccuracy: " + f'{accuracy*100:.3f}' + "%\tLoss: " + f'{loss:.5f}')
            if (self.numConverged == 5):
                break
        toc()
        self.epoch = epoch
        return self.history
