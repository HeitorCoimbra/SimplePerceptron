import pandas as pd 
import random as rand
import numpy as np

# Actual implementation of the Perceptron algorithm

class Perceptron:
    def __init__(self, input_size):
        self.weights=np.zeros(input_size) # Fills the initial weights vector with zeroes
        self.input_size=input_size
    
    def trainWeights(self):
        while(True):
            count=0
            # A for loop goes through the training dataframe to make corrections to the hyperplane/weights vector if the
            # "current" prediction of any point is inaccurate
            for i in range (len(traindf.index)):
                x = traindf.iloc[i]
                vector = (x.drop(labels=['species']))
                vector['bias_coefficient']=1 # The bias coefficient's purpose is to simplify the operations and only
                # work with vectors
                y = x['species']
                if (np.dot(self.weights, vector)*y <= 0):
                    self.weights = self.weights + y*vector
                    count=count+1
            if (count==0):
                break
                    
    def predict(self, x):
        x=x.iloc[:self.input_size-1]
        x['bias_coefficient']=1
        return np.dot(self.weights, x)
    
# Converts each numerical data column to z-score to standardize the data.

def standardize_Dataframe(df):
    result = df.copy()
    
    for feature_name in df.columns:
        dtype=df[feature_name].dtypes

        # If the data type of the column is either integer or floating point,
        if dtype=='float64' or dtype=='int64':
            
            # subtract each element by it's column's mean and divide the result by the standard deviation.
            mean = df[feature_name].mean()
            std = df[feature_name].std()
            result[feature_name] = (df[feature_name] - mean) / std
            
    return result

# Data manipulation, subdivision and sampling
maindf = standardize_Dataframe(pd.read_csv('https://bit.ly/2ow0oJO').iloc[:100]).sample(frac=1).reset_index(drop=True)
maindf['species']= maindf['species'].apply(lambda x: 1 if x == 'setosa' else -1)
setosadf=maindf.iloc[:50]
versicolordf=maindf.iloc[50:]
traindf = setosadf.iloc[:40]
traindf = traindf.append(versicolordf.iloc[:40]).sample(frac=1).reset_index(drop=True)
testdf = setosadf.iloc[40:]
testdf = testdf.append(versicolordf.iloc[40:]).sample(frac=1).reset_index(drop=True)

# Construction and training of the Perceptron
model = Perceptron(5)
model.trainWeights()

# Sanity Check
bool=True
for i in range (len(testdf.index)):
    bool = (bool and (model.predict(testdf.iloc[i])*testdf.iloc[i].species > 0))
print (bool)
