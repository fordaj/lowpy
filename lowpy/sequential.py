import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import pandas as pd
from datetime import datetime
import os
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
    def __init__(self,number_of_networks=1):
        self.L          = number_of_networks
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
            self.layer[0].build(self.L,self.layer[0].I,self.layer[0].J)         # Run the build function with the user-specified input shape
        else:                                                               # Otherwise, run the build function with the previous layer's output length as the current input shape
            self.layer[numLayers-1].build(self.L,self.layer[numLayers-2].J,self.layer[numLayers-1].J)
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
    def inference(self,label,hits):
        self.layer[self.numLayers-1].argmax(label,hits)

    # Model architecture
    def describe(self):
        print("---------------------------------------------------------------")
        print("")
        neurons = str(self.layer[0].I)
        for l in range(len(self.layer)):
            neurons += "->" + str(self.layer[l].J)
        print(neurons + " with " + str(self.L) + " variants:")
        print("")
        for l in range(len(self.layer)):
            print("Layer " + str(l)+ " -------------------------------------------------------")
            print("        {:<10}".format("alpha"),end="")
            for c in range(self.L):
                print("{:<10}".format(self.layer[l].alpha.get()[c]),end="")
            print("")
            print("        {:<10}".format("beta"),end="")
            for c in range(self.L):
                print("{:<10}".format(self.layer[l].beta.get()[c]),end="")
            print("")
            print("        {:<10}".format("sigma_i"),end="")
            for c in range(self.L):
                print("{:<10}".format(self.layer[l].sigma_i.get()[c]),end="")
            print("")
        print("")
        print("---------------------------------------------------------------")

    # Track model data
    class metrics:
        def __init__(self):
            self.train      = self.trialData()
            self.test       = self.trialData()
            self.metricsDir    = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.trainDir = self.metricsDir + "/Train"
            self.testDir = self.metricsDir + "/Test"
            os.mkdir(self.metricsDir)
            os.mkdir(self.trainDir)
            os.mkdir(self.testDir)
        class trialData:
            def __init__(self):
                self.accuracy   = pd.DataFrame(columns=np.arange(11))
                self.loss       = pd.DataFrame(columns=np.arange(11))

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
        self.testHits = gpuarray.zeros(self.L,dtype=np.int32)
        for i in range(numTests):
            self.propagate(i)
            self.inference(self.testLabels[i], self.testHits)
        accuracy = self.testHits.get()/numTests
        loss = 1-accuracy
        self.history.test.accuracy = self.history.test.accuracy.append(pd.DataFrame([np.concatenate([[self.i+self.epoch*self.numTrain],accuracy])]))
        self.history.test.loss = self.history.test.loss.append(pd.DataFrame([np.concatenate([[self.i+self.epoch*self.numTrain],loss])]))
        self.history.test.accuracy.to_csv(self.history.testDir + "/Accuracy.csv")
        self.history.test.loss.to_csv(self.history.testDir + "/Loss.csv")
        if (self.verbose):
            print("        {:<10}".format("Test"),end="")
            for c in range(self.L):
                print("{:<10}".format(f'{accuracy[c]*100:.2f}'+"%"),end="")
            print("")
        # if (accuracy <= self.peakAccuracy):
        #     self.numConverged += 1
        # else:
        #     self.numConverged = 0
        #     self.peakAccuracy = accuracy

    # Train model
    def fit(self, trainData, trainLabels, validation_data, epochs, batch_size=-1, tests_per_epoch=1, convergenceTracking=-1, verbose=True):
        self.verbose = verbose
        self.importDataset(trainData,trainLabels,validation_data[0],validation_data[1])
        self.layer[0].x = self.trainData
        self.numTrain = len(trainData)
        self.numConverged = 0
        self.peakAccuracy = 0
        self.describe()
        self.history = self.metrics()
        for self.epoch in range(epochs):
            if (self.verbose):
                print("Epoch " + str(self.epoch))
            self.trainHits = gpuarray.zeros(self.L,dtype=np.int32)
            for self.i in range(self.numTrain):
                if (self.i%(int(self.numTrain/tests_per_epoch))==0):
                    self.layer[0].x = self.testData
                    self.validate()
                    self.layer[0].x = self.trainData
                self.propagate(self.i)
                self.backpropagate(self.i,self.trainLabels[self.i])
                self.inference(self.trainLabels[self.i],self.trainHits)
            accuracy = self.trainHits.get()/self.numTrain
            loss = 1 - accuracy
            if (verbose):
                print("        {:<10}".format("Train"),end="")
                for c in range(self.L):
                    print("{:<10}".format(f'{accuracy[c]*100:.2f}'+"%"),end="")
                print("")
            if (self.numConverged >= 5 and convergenceTracking > -1):
                break
        self.layer[0].x = self.testData
        self.validate()
        print("---------------------------------------------------------------")
        return self.metrics
