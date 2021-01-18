import numpy as np
import time
from keras.datasets import mnist
import numpy as np

from sequential import Sequential
from dense import Dense

# Start timer from 0
def tic():
    global t
    t = time.time()

# Lap timer
def toc():
    global t
    print(time.time() - t)


#-----IMPORT DATASET-----#
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
trainData   = np.float64(trainData.reshape( trainData.shape[0], trainData.shape[1]*trainData.shape[2]   )) # Reshape train data from [n,28,28] to [n,784]
trainLabels = np.int32(trainLabels)
testData    = np.float64(testData.reshape(  testData.shape[0],  testData.shape[1]*testData.shape[2]     )) # Reshape test data from [n,28,28] to [n,784]
testLabels  = np.int32(testLabels)
#-----NORMALIZE DATA-----#
trainData   = np.true_divide(trainData, max(np.max(trainData), np.max(testData)))
testData    = np.true_divide(testData,  max(np.max(trainData), np.max(testData)))
#-----BUILD MODEL-----#
model = Sequential()
model.add(  Dense(  10,    input_shape=784,    alpha=0.025,    beta=0,     sigma_i=0   ))
#model.add(  Dense(  10,                         alpha=0.025,    beta=0,     sigma_i=0   ))
#-----TRAIN MODEL-----#
history = model.fit(trainData, trainLabels, epochs=10, batch_size=60000, validation_data=(testData, testLabels), verbose=True)














"""
# 2LP Weight Initialization Sims
variability = 0.75
for i in range(5):
    #-----BUILD MODEL-----#
    model = Sequential()
    model.add(Dense(533, input_shape=784, alpha=0.025, beta=0, sigma_i=variability))
    model.add(Dense( 10,                  alpha=0.025, beta=0, sigma_i=variability))
    #-----TRAIN MODEL-----#
    history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels), verbose=False)
    print("Epochs Trained: " + str(model.epoch) + "\tPeak accuracy: " + str(model.peakAccuracy*100) + "%")
    variability /= 10

variability = 0.5
for i in range(5):
    #-----BUILD MODEL-----#
    model = Sequential()
    model.add(Dense(533, input_shape=784, alpha=0.025, beta=0, sigma_i=variability))
    model.add(Dense( 10,                  alpha=0.025, beta=0, sigma_i=variability))
    #-----TRAIN MODEL-----#
    history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels), verbose=False)
    print("Epochs Trained: " + str(model.epoch) + "\tPeak accuracy: " + str(model.peakAccuracy*100) + "%")
    variability /= 10

variability = 0.25
for i in range(5):
    #-----BUILD MODEL-----#
    model = Sequential()
    model.add(Dense(533, input_shape=784, alpha=0.025, beta=0, sigma_i=variability))
    model.add(Dense( 10,                  alpha=0.025, beta=0, sigma_i=variability))
    #-----TRAIN MODEL-----#
    history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels), verbose=False)
    print("Epochs Trained: " + str(model.epoch) + "\tPeak accuracy: " + str(model.peakAccuracy*100) + "%")
    variability /= 10

variability = 0.1
for i in range(5):
    #-----BUILD MODEL-----#
    model = Sequential()
    model.add(Dense(533, input_shape=784, alpha=0.025, beta=0, sigma_i=variability))
    model.add(Dense( 10,                  alpha=0.025, beta=0, sigma_i=variability))
    #-----TRAIN MODEL-----#
    history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels), verbose=False)
    print("Epochs Trained: " + str(model.epoch) + "\tPeak accuracy: " + str(model.peakAccuracy*100) + "%")
    variability /= 10

"""
"""
# 1LP Weight Initialization Sims
variability = 0.75
for i in range(5):
    #-----BUILD MODEL-----#
    model = Sequential()
    model.add(Dense(10, input_shape=784, alpha=0.05, beta=0, sigma_i=variability))
    #-----TRAIN MODEL-----#
    history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels), verbose=False)
    print("Epochs Trained: " + str(model.epoch) + "\tPeak accuracy: " + str(model.peakAccuracy*100) + "%")
    variability /= 10

variability = 0.5
for i in range(5):
    #-----BUILD MODEL-----#
    model = Sequential()
    model.add(Dense(10, input_shape=784, alpha=0.005, beta=0, sigma_i=variability))
    #-----TRAIN MODEL-----#
    history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels), verbose=False)
    print("Epochs Trained: " + str(model.epoch) + "\tPeak accuracy: " + str(model.peakAccuracy*100) + "%")
    variability /= 10

variability = 0.25
for i in range(5):
    #-----BUILD MODEL-----#
    model = Sequential()
    model.add(Dense(10, input_shape=784, alpha=0.005, beta=0, sigma_i=variability))
    #-----TRAIN MODEL-----#
    history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels), verbose=False)
    print("Epochs Trained: " + str(model.epoch) + "\tPeak accuracy: " + str(model.peakAccuracy*100) + "%")
    variability /= 10

variability = 0.1
for i in range(5):
    #-----BUILD MODEL-----#
    model = Sequential()
    model.add(Dense(10, input_shape=784, alpha=0.005, beta=0, sigma_i=variability))
    #-----TRAIN MODEL-----#
    history = model.fit(trainData, trainLabels, epochs=100, batch_size=60000, validation_data=(testData, testLabels), verbose=False)
    print("Epochs Trained: " + str(model.epoch) + "\tPeak accuracy: " + str(model.peakAccuracy*100) + "%")
    variability /= 10
"""

# Create the model
# model = Sequential()
# model.add(Dense(350, input_shape=input_shape, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))