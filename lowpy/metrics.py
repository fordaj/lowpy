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
        shouldRemove = 'u'
        if not os.path.exists(os.path.join(os.getcwd(), self.trainDir)):
            os.mkdir(self.trainDir)
        else:
            trainLossFile = os.path.join(os.getcwd(), self.trainDir, "Loss.csv")
            trainAccuracyFile = os.path.join(os.getcwd(), self.trainDir, "Accuracy.csv")
            if os.path.exists(trainLossFile):
                if (shouldRemove == 'u'):
                    shouldRemove = input("Are you sure you want to delete " + str(trainLossFile) + " ? (y/n/all): ")
                if (shouldRemove == 'y' or shouldRemove == 'all'):
                    os.remove(trainLossFile)
                    if (shouldRemove != 'all'):
                        shouldRemove = 'u'
            if os.path.exists(trainAccuracyFile):
                if (shouldRemove == 'u'):
                    shouldRemove = input("Are you sure you want to delete " + str(trainAccuracyFile) + "? (y/n/all): ")
                if (shouldRemove == 'y' or shouldRemove == 'all'):
                    os.remove(trainAccuracyFile)
                    if (shouldRemove != 'all'):
                        shouldRemove = 'u'
        if not os.path.exists(os.path.join(os.getcwd(), self.testDir)):
            os.mkdir(self.testDir)
        else:
            testLossFile = os.path.join(os.getcwd(), self.testDir, "Loss.csv")
            testAccuracyFile = os.path.join(os.getcwd(), self.testDir, "Accuracy.csv")
            if os.path.exists(testLossFile):
                if (shouldRemove == 'u'):
                    shouldRemove = input("Are you sure you want to delete " + str(testLossFile) + "? (y/n/all): ")
                if (shouldRemove == 'y' or shouldRemove == 'all'):
                    os.remove(testLossFile)
                    if (shouldRemove != 'all'):
                        shouldRemove = 'u'
            if os.path.exists(testAccuracyFile):
                if (shouldRemove == 'u'):
                    shouldRemove = input("Are you sure you want to delete " + str(testAccuracyFile) + "? (y/n/all): ")
                if (shouldRemove == 'y' or shouldRemove == 'all'):
                    os.remove(testAccuracyFile)
                    if (shouldRemove != 'all'):
                        shouldRemove = 'u'
        self.architecture   = pd.DataFrame()
    class trialData:
        def __init__(self):
            self.accuracy   = pd.DataFrame()
            self.loss       = pd.DataFrame()