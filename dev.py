import numpy as np
import time
from keras.datasets import mnist
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['backend'] = "WXAgg" #"Qt4Agg"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import lowpy as lp

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
history = []
#----GLOBAL PARAMETERS---#
input_shape             = 784
alpha                   = 0.05
beta                    = 0
sigma_i                 = 0
weight_initialization   = "normal"
#--SIMULATION PARAMETERS-#
epochs = 10
batch_size = 60000
tests_per_epoch = 5
validation_data = (testData,testLabels)
verbose = True



numSims = 6
divideBy5 = True
for s in range(numSims):
    model = lp.Sequential()
    # Build Model
    model.add(  
        lp.Dense(   
            533,    
            input_shape=input_shape,    
            alpha=alpha,    
            beta=beta,     
            weight_initialization=weight_initialization, 
            sigma_i=sigma_i   
        )
    )
    model.add(  
        lp.Dense(   
            10,      
            alpha=alpha,    
            beta=beta,     
            weight_initialization=weight_initialization, 
            sigma_i=sigma_i   
        )
    )
    # Simulate Model
    history.append(
        model.fit(
            trainData, 
            trainLabels, 
            epochs=epochs, 
            batch_size=batch_size, 
            tests_per_epoch=tests_per_epoch, 
            validation_data=(testData, testLabels), 
            verbose=True
        )
    )
    # Change parameter
    if(divideBy5):
        sigma_i /= 5
        divideBy5 = False
    else:
        sigma_i /= 2
        divideBy5 = True



plt.figure(figsize=(6, 4.5))
for h in history:
    plt.plot(h.test.iteration,h.test.accuracy, label=f'{h.sigma_i[0]:.3f}')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("2LP with Varied sigma_i")
plt.legend(loc='lower right')
plt.grid()
plt.show()
plt.savefig('2LPhjsadkf.png',dpi=1200)

print("done")






# Create the model
# model = Sequential()
# model.add(Dense(350, input_shape=input_shape, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))