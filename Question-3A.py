
from scipy.io import arff
import numpy as np
import pandas as pd
from pandas import DataFrame as df

def calculatePCC(xDF , yDF, ySumSqurd, yMean):
    xSumSqurd = np.sum(np.square(xDF))
    #ySumSqurd = np.sum(np.square(yDF))
    sumCoProduct = np.sum( xDF * yDF )
    xMean = np.mean(xDF)
    
    xPopSD = np.sqrt( (xSumSqurd / float(len(xDF))) - (xMean**2) )
    #print((ySumSqurd / len(yDF)) - (yMean**2))
    yPopSD = np.sqrt( (ySumSqurd / float(len(yDF))) - (yMean**2) )
    xyCov = ( (sumCoProduct / len(yDF)) - (xMean * yMean) )
    
    correlation = ( xyCov / (xPopSD * yPopSD) )
    #print(correlation)
    
    return correlation

data = arff.loadarff("veh-prime.arff")
trainDF = pd.DataFrame(data[0])

# Updating noncar as 0 and car as 1.
trainDF['CLASS'] = np.where(trainDF['CLASS'] == b'noncar', 0, 1)

#print(trainDF)

# Calculating Sum Squared Y (For class lebel)
ySumSqurd = np.sum(np.square(trainDF['CLASS']))
yMean = np.mean(trainDF['CLASS'])

#print(ySumSqurd)
#print(yMean)

pccList = []
abspccList = []
featureList = []
for counter in range(len(trainDF.columns) - 1):
    #print(trainDF.columns[counter])
    featureList.append(trainDF.columns[counter])
    
    pcc = calculatePCC(trainDF[trainDF.columns[counter]], trainDF['CLASS'], ySumSqurd, yMean)
    pccList.append( pcc )
    abspccList.append( np.abs(pcc) )
    
  
tmpDict = {'feature' : featureList , 'pcc' : pccList, '|pcc|' : abspccList}
#print(len(pccList))
#print(len(columnList))
pccDF = pd.DataFrame(tmpDict)
print(pccDF.sort_values(['|pcc|'] , ascending=0))
