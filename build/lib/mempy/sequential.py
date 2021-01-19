import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
from progress.bar import Bar



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
        self.layer      = []
        self.numLayers  = 0
        self.verbose    = True
        self.epoch      = 0
        self.i          = 0
        self.numTrain   = 0

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
            self.train      = self.trialData()
            self.test       = self.trialData()
            self.alpha      = []
            self.beta       = []
            self.sigma_i    = []
        class trialData:
            def __init__(self):
                self.iteration  = []
                self.accuracy   = []
                self.loss       = []

    # Import dataset
    def importDataset(self,trainData,trainLabels,testData,testLabels):
        numTrain = len(trainData)
        numTest = len(testData)
        self.trainData_d     = []
        self.trainLabels_d   = []
        self.testData_d      = []
        self.testLabels_d    = []
        print()
        bar = Bar('Importing dataset', max=int((numTrain+numTest)/1000), suffix='%(percent)d%%')
        for i in range(numTrain):
            self.trainData_d.append(self.hostToDevice(trainData[i]))
            if (i%1000 == 1):
                bar.next()
        self.trainLabels_d = np.int32(trainLabels)
        for i in range(numTest):
            self.testData_d.append(self.hostToDevice(testData[i]))
            if (i%1000 == 1):
                bar.next()
        bar.finish()
        print()
        self.testLabels_d = np.int32(testLabels)

    # Test model
    def validate(self):
        numTests = len(self.testData_d)
        testHits_h = np.zeros(3,dtype=np.float64)
        testHits_d = self.hostToDevice(testHits_h)
        self.layer[self.numLayers-1].resetHits()
        for i in range(numTests):
            self.propagate(self.testData_d[i])
            self.inference(self.testLabels_d[i], testHits_d)
        cuda.memcpy_dtoh(testHits_h, testHits_d)
        accuracy = testHits_h[0]/numTests
        loss = 1-accuracy
        self.history.test.accuracy.append(accuracy)
        self.history.test.loss.append(loss)
        self.history.test.iteration.append(self.epoch*self.numTrain+self.i)
        if (self.verbose):
            print ("{:<20} {:<20} {:<20}".format("    Testing", f'{accuracy*100:.2f}' + "%", f'{loss:.5f}'))
        if (accuracy <= self.peakAccuracy):
            self.numConverged += 1
        else:
            self.numConverged = 0
            self.peakAccuracy = accuracy

    # Train model
    def fit(self, trainData, trainLabels, validation_data, epochs, batch_size=-1, tests_per_epoch=1, convergenceTracking=-1, verbose=True):
        self.verbose = verbose
        self.history = self.metrics()
        for l in self.layer:
            self.history.alpha.append(l.alpha)
            self.history.beta.append(l.beta)
            self.history.sigma_i.append(l.sigma_i)
        testData = validation_data[0]
        testLabels = validation_data[1]
        self.importDataset(trainData,trainLabels,testData,testLabels)
        self.numTrain = len(trainData)
        self.numConverged = 0
        self.peakAccuracy = 0
        self.describe()
        print ("{:<20} {:<20} {:<20}".format("Epoch", "Accuracy", "Loss"))
        for self.epoch in range(epochs):
            if (self.verbose):
                print("  " + str(self.epoch))
            trainHits_h = np.zeros(3,dtype=np.float64)
            trainHits_d = self.hostToDevice(trainHits_h)
            self.layer[self.numLayers-1].resetHits()
            for self.i in range(self.numTrain):
                if (self.i%(int(self.numTrain/tests_per_epoch))==0):
                    self.validate()
                #self.deviceToHost()
                self.propagate(self.trainData_d[self.i])
                #self.deviceToHost()
                self.backpropagate(self.trainLabels_d[self.i])
                #self.deviceToHost()
                self.inference(self.trainLabels_d[self.i],trainHits_d)
            self.deviceToHost()
            cuda.memcpy_dtoh(trainHits_h, trainHits_d)
            accuracy = trainHits_h[0]/self.numTrain
            loss = 1 - accuracy
            self.history.train.accuracy.append(accuracy)
            self.history.train.loss.append(loss)
            self.history.train.iteration.append(self.epoch*self.numTrain+self.i)
            if (verbose):
                print ("{:<20} {:<20} {:<20}".format("    Training", f'{accuracy*100:.2f}'+"%", f'{loss:.5f}'))
            if (self.numConverged >= 5 and convergenceTracking > -1):
                break
        print("---------------------------------------------------------------")
        return self.history
