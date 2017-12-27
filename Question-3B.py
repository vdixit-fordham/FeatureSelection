from __future__ import division
from scipy.io import arff
import numpy as np
import pandas as pd
from pandas import DataFrame as df

def normalizeDF(dataFrame):
    dfNormalized = dataFrame.copy()
    colList = list(dataFrame.columns)
    #print(cols)
    
    for col in range(len(colList)):
        colMean = dataFrame[colList[col]].mean()
        colStd = dataFrame[colList[col]].std()
        #print(col,'= ', colMean)
        #print(col,'= ', colStd)
        dfNormalized[colList[col]] = (dataFrame[colList[col]] - colMean)/colStd
    
    return dfNormalized

def calculatePCC(xDF , yDF, ySumSqurd, yMean):
    xSumSqurd = np.sum(np.square(xDF))
    #ySumSqurd = np.sum(np.square(yDF))
    sumCoProduct = np.sum( xDF * yDF )
    xMean = np.mean(xDF)
    
    xPopSD = np.sqrt( (xSumSqurd / float(len(xDF))) - (xMean**2) )
    yPopSD = np.sqrt( (ySumSqurd / float(len(yDF))) - (yMean**2) )
    xyCov = ( (sumCoProduct / float(len(yDF))) - (xMean * yMean) )
    
    correlation = ( xyCov / (xPopSD * yPopSD) )
    #print(correlation)
    
    return correlation

def getPredictedClassUsingKNN(trainDF , testRow, trainLableDF):
    kValue = 7
    #print(trainDF)
    #print(testRow)
    #print(trainLableDF)
    #This DF will have the distance sorted (ascending)
    distanceDF = calculateEculidDist(trainDF , testRow)
    #print(distanceDF)
    kRows = distanceDF.iloc[:kValue]
    #print("kRows ------ > ", kRows)
    #print("kRows ------ > ", kRows.index)
    #print(trainLableDF.iloc[kRows.index.tolist()]['CLASS'].values)

    tmp = trainLableDF.iloc[kRows.index.tolist()]['CLASS'].value_counts()
    #print(tmp)
    
    return tmp.idxmax()

def calculateEculidDist(trainDF , testRow):
    #np.sqrt(np.sum(np.square((trainArray - testArray))))
    tmp = (((trainDF.sub( testRow, axis=1))**2).sum(axis=1))**0.5
    tmp.sort_values(axis=0, ascending=True, inplace=True)
    #print(type(tmp))
    
    return tmp

data = arff.loadarff("veh-prime.arff")
trainDF = pd.DataFrame(data[0])
trainLableDF = trainDF[['CLASS']].copy()
trainDF.drop('CLASS' , axis=1, inplace=True)
# Updating noncar as 0 and car as 1.
trainLableDF['CLASS'] = np.where(trainLableDF['CLASS'] == b'noncar', 0, 1)

# Z score normalization
trainDFNormalized = normalizeDF(trainDF) 

#print(trainDF)

# Calculating Sum Squared Y (For class lebel)
ySumSqurd = np.sum(np.square(trainLableDF['CLASS']))
yMean = np.mean(trainLableDF['CLASS'])

#print(ySumSqurd)
#print(yMean)

pccList = []
abspccList = []
featureList = []
for counter in range(len(trainDF.columns)):
    #print(trainDF.columns[counter])
    featureList.append(trainDF.columns[counter])
    
    pcc = calculatePCC(trainDF[trainDF.columns[counter]], trainLableDF['CLASS'], ySumSqurd, yMean)
    pccList.append( pcc )
    abspccList.append( np.abs(pcc) )
    
 
#print(featureList)
tmpDict = {'feature' : featureList , 'pcc' : pccList, 'abspcc' : abspccList}
pccDF = pd.DataFrame(tmpDict)
pccDF.sort_values(['abspcc'] , ascending=0 , inplace=True)
#print(pccDF)

rankedFeatureList = pccDF['feature'].tolist()
#rankedFeatureList = ['f4']
#print(rankedFeatureList)

for counter in range(len(rankedFeatureList)):
    print("Selected Feature set -- ", rankedFeatureList[:counter+1])
    #tmptrainDF = trainDF[rankedFeatureList[:counter+1]]
    tmptrainDF = trainDFNormalized[rankedFeatureList[:counter+1]]
    #print(tmptrainDF)
    index = 0
    accuracyCount = 0
    for row in tmptrainDF.itertuples(index=False):
        #print("For row --- ", index)
        #if(index > 2):
            #break
        tmpDF = tmptrainDF.drop(index)
        predictedClass = getPredictedClassUsingKNN(tmpDF, row, trainLableDF) 
        #print("predictedClass---- ", predictedClass, " class ---- ", trainLableDF.iloc[index]['CLASS'])
        if(predictedClass == trainLableDF.iloc[index]['CLASS']):
            accuracyCount += 1        
        index += 1
     
    print("Accuracy Count       = ", accuracyCount)
    print("Accuracy Percentage  = ", round((accuracyCount / len(trainDFNormalized))*100, 2))
    print("\n")

#print(trainDF[['f0', 'f1']])
