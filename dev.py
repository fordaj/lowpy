import numpy as np
import time
from keras.datasets import mnist
import numpy as np
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
number_of_networks          = 11
input_shape                 = len(trainData[0])
alpha                       = np.ones(number_of_networks) * 0.2
beta                        = np.zeros(number_of_networks) #np.linspace(0,1,number_of_networks)
sigma                       = np.zeros(number_of_networks) #np.logspace(-1*(number_of_networks-1),0,number_of_networks)
sigma[0]                    = 0
initialization_type         = "normal"
initialization_parameter    = np.linspace(0.1,2,number_of_networks) # math.sqrt(2/input_shape)
#--SIMULATION PARAMETERS-#
epochs = 15
batch_size = 60000
tests_per_epoch = 4
validation_data = (testData,testLabels)
verbose = True
#------BUILD MODEL------#
model = lp.Sequential(number_of_networks=number_of_networks)
model.add(  
    lp.Dense(   
        533,    
        input_shape=input_shape,    
        alpha=alpha,    
        beta=beta,     
        initialization_type=initialization_type, 
        initialization_parameter=initialization_parameter,
        sigma=sigma   
    )
)
model.add(  
    lp.Dense(   
        10,      
        alpha=alpha,    
        beta=beta,     
        initialization_type=initialization_type, 
        initialization_parameter=initialization_parameter,
        sigma=sigma   
    )
)
# Simulate Model
#--------SIMULATE--------#
history = model.fit(
    trainData, 
    trainLabels, 
    epochs=epochs, 
    batch_size=batch_size, 
    tests_per_epoch=tests_per_epoch, 
    validation_data=(testData, testLabels), 
    verbose=True
)