import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pandas as pd

 # Track model data
class metrics:
    def __init__(self):
        self.train          = self.trialData()
        self.test           = self.trialData()
        self.trainDir       = "Train"
        self.testDir        = "Test"
        if not os.path.exists(os.path.join(os.getcwd(), self.trainDir)):
            os.mkdir(self.trainDir)
        else:
            if os.path.exists(os.path.join(os.getcwd(), self.trainDir, "Loss.csv")):
                os.remove(os.path.join(os.getcwd(), self.trainDir, "Loss.csv"))
            if os.path.exists(os.path.join(os.getcwd(), self.trainDir, "Accuracy.csv")):
                os.remove(os.path.join(os.getcwd(), self.trainDir, "Accuracy.csv"))
        if not os.path.exists(os.path.join(os.getcwd(), self.testDir)):
            os.mkdir(self.testDir)
        else:
            if os.path.exists(os.path.join(os.getcwd(), self.testDir, "Loss.csv")):
                os.remove(os.path.join(os.getcwd(), self.testDir, "Loss.csv"))
            if os.path.exists(os.path.join(os.getcwd(), self.testDir, "Accuracy.csv")):
                os.remove(os.path.join(os.getcwd(), self.testDir, "Accuracy.csv"))
        self.architecture   = pd.DataFrame()
    class trialData:
        def __init__(self):
            self.accuracy   = pd.DataFrame()
            self.loss       = pd.DataFrame()