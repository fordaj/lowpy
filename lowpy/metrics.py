import os
import pandas as pd
import datetime

class metrics:
    def __init__(self):
        self.directory_name    = "LowPy " + str(datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
        self.accuracy          = self.trialData(self.directory_name, "Accuracy")
        self.loss              = self.trialData(self.directory_name, "Loss")
        self.updates           = self.trialData(self.directory_name, "Updates")
        os.mkdir(self.directory_name)
        self.architecture   = pd.DataFrame()
    class trialData:
        def __init__(self, directory_name, file_name):
            self.file_name_full = os.path.join(os.getcwd(), directory_name, file_name + ".csv")
            self.data   = pd.DataFrame()
        def add_value(self, value, index, header):
            # try: # to access the column, and subsequently append new value
            #     self.data[header]
            #     self.data = self.data[header].dropna().append(pd.DataFrame({0:[value]}),ignore_index=True)
            # except: # when the column is uninitialized
            #     try: # and add a column to the existing dataframe
            #         self.data[header] = pd.DataFrame({header:[value]})
            #     except: # when the dataframe is empty
            #         self.data[header] = [value]
            # try:
            #     self.data[header]
            #     self.data = pd.concat([self.data,pd.DataFrame([[value]],columns=[header],index=[index])],axis=0)
            # except:
            #     self.data = pd.concat([self.data,pd.DataFrame([[value]],columns=[header],index=[index])],axis=1)
            self.data.at[index,header] = value
            self.data.to_csv(self.file_name_full)
    def export_weights(self, updates_directory_name, cell_updates):
        os.mkdir(self.directory_name + "/" + str(updates_directory_name))
        for i in range(len(cell_updates)):
            pd.DataFrame(cell_updates[i].numpy()).to_csv(self.directory_name + "/" + str(updates_directory_name) + "/Weights" + str(i) + ".csv")