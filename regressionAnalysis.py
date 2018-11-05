##Kexin Zhai - INFO3401
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class AnalysisData:
    def __init__(self, type1):
        self.type = type1
        self.dataset=[]
        self.variables = []

    def getData(self, filename):
        if(self.type == "csv"):        
            self.dataset = pd.read_csv(filename)
            self.variables = list(self.dataset.columns.values)
#            for row in reader:
#                self.dataset.append(row)
        else:
            self.dataset = open(filename).read()
        return self.dataset
    
    def get_dataVar(self, variable):
#        df = pd.DataFrame(self.dataset)
#        return df
        var_col = []
        var_col = self.dataset[variable].tolist()
        return var_col    
#    def get_index(self, variable):
#        index_v = self.variables.index(variable)
#        return index_v
                
        
#data = AnalysisData("csv")
#data1 = data.getData("candy-data.csv")
#targetY = data.get_dataVar("chocolate")
#print(targetY)
#data2 = data.variables
#print(data2)

class LinearAnalysis:
    def __init__(self, target_Y):
        self.bestX = [] #best X predictor for your data
        self.targetY =target_Y #index to the target dependent variable
        self.fit =0 #how well bestX predicts your target variable
    def runSimpleAnlysis(self, data):
        dataset_1 = data.getData("candy-data.csv")
        dep_value = data.get_dataVar(self.targetY)
        
        reDep_value = dataset_1.drop(self.targetY, axis = 1)
        indep_value = reDep_value.drop("competitorname", axis = 1)
        
        regr = LinearRegression()
        regr.fit(indep_value, dep_value)

        predict_value = regr.predict(indep_value)

        de = pd.DataFrame(list(zip(indep_value.columns, regr.predict(indep_value))), columns = ['variables', 'predict'])
        get_fit = de.loc[de['predict'].idxmax()]
        self.bestX.append(get_fit['variables'])
        self.fit = r2_score(dep_value, predict_value)
        return self.bestX
      
la = LinearAnalysis("sugarpercent")
get_variable = la.runSimpleAnlysis(AnalysisData("csv"))
print(get_variable)
print(la.fit)
#class LogisticAnalysis:
#    def __init__(self, input_y):
#        self.bestX = []
#        self.targetY = input_y
#        self.fit = []