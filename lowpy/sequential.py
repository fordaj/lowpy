import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
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
        self.layer      = []
        self.numLayers  = 0
        self.verbose    = True
        self.epoch      = 0
        self.i          = 0
        self.numTrain   = 0

    # Function for appending layer objects to the model
    def add(self,newLayer):
        self.layer.append(newLayer)
        numLayers = len(self.layer)
        if (numLayers == 1):                                                # If this is the first layer added:
            self.layer[0].build(self.layer[0].I,self.layer[0].J)        # Run the build function with the user-specified input shape
        else:                                                               # Otherwise, run the build function with the previous layer's output length as the current input shape
            self.layer[numLayers-1].build(self.layer[numLayers-2].J,self.layer[numLayers-1].J)
            self.layer[numLayers-1].linkPreviousLayer(self.layer[numLayers-2]) # Link the previous layer to the current one
            self.layer[numLayers-2].linkNextLayer(self.layer[numLayers-1])     # Link the current layer to the previous one
        self.numLayers += 1

    # Forward pass
    def propagate(self,iteration):
        for l in range(self.numLayers):
            if (l==0):
                self.layer[l].propagate(iteration=iteration)
            else:
                self.layer[l].propagate()
            
    # Backward pass
    def backpropagate(self,iteration,label):
        if (self.numLayers==1):
            self.layer[self.numLayers-1].backpropagate(iteration=iteration,label=label)    
        else:
            self.layer[self.numLayers-1].backpropagate(label=label)
            for l in range(self.numLayers-2,-1,-1):
                if (l==0):
                    self.layer[l].backpropagate(iteration=iteration)
                else:
                    self.layer[l].backpropagate()

    # Decision
    def inference(self,label,hits_d):
        self.layer[self.numLayers-1].argmax(label,hits_d)

    # Model architecture
    def describe(self):
        print("---------------------------------------------------------------")
        print ("{:<7} {:<11} {:<15} {:<15} {:<12}".format("Type", "Neurons", "Learning Rate", "Momentum Rate","Variability"))
        for l in self.layer:
            print ("{:<7} {:<11} {:<15} {:<15} {:<12}".format("Dense", str(l.I)+"->"+str(l.J), f'{l.alpha:.5f}', f'{l.beta:.5f}', f'{l.sigma_i:.5f}'))
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
        """Send the dataset to the GPU.

        trainData : dtype=np.float64()
            Data to fit the network to. Currently supports 2D.
        trainLabels : dtype=np.float64()
            Labels corresponding to the inputs in trainData.
        trainData : dtype=np.float64()
            Data to validate the network with. Currently supports 2D.
        trainData : dtype=np.float64()
            Labels corresponding to the inputs in testData.
        """
        self.trainData      = gpuarray.to_gpu(trainData)
        self.trainLabels    = trainLabels
        self.testData       = gpuarray.to_gpu(testData)
        self.testLabels     = testLabels

    # Test model
    def validate(self):
        numTests = len(self.testData)
        testHits = gpuarray.zeros(3,dtype=np.float64)
        self.layer[self.numLayers-1].resetHits()
        for i in range(numTests):
            self.propagate(i)
            self.inference(self.testLabels[i], testHits)
        accuracy = testHits.get()[0]/numTests
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
        self.importDataset(trainData,trainLabels,validation_data[0],validation_data[1])
        self.layer[0].x = self.trainData
        self.numTrain = len(trainData)
        self.numConverged = 0
        self.peakAccuracy = 0
        self.describe()
        print ("{:<20} {:<20} {:<20}".format("Epoch", "Accuracy", "Loss"))
        for self.epoch in range(epochs):
            if (self.verbose):
                print("  " + str(self.epoch))
            trainHits = gpuarray.zeros(3,dtype=np.float64)
            self.layer[self.numLayers-1].resetHits()
            for self.i in range(self.numTrain):
                if (self.i%(int(self.numTrain/tests_per_epoch))==0):
                    self.layer[0].x = self.testData
                    self.validate()
                    self.layer[0].x = self.trainData
                self.propagate(self.i)
                self.backpropagate(self.i,self.trainLabels[self.i])
                self.inference(self.trainLabels[self.i],trainHits)
            accuracy = trainHits.get()[0]/self.numTrain
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
