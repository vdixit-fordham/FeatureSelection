from __future__ import division
from scipy.io import arff
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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

def getPredictedClassUsingKNN(trainDF , testRow, trainLableDF):
    kValue = 7
    #print(trainDF)
    #print(testRow)
    #print("trainLableDF =", len(trainLableDF))
    #This DF will have the distance sorted (ascending)
    distanceDF = calculateEculidDist(trainDF , testRow)
    #print(distanceDF)
    kRows = distanceDF.iloc[:kValue]
    #print("kRows ------ > ", kRows)
    #print("kRows ------ > ", kRows.index)
    #print(trainLableDF.iloc[kRows.index.tolist()]['CLASS'].values)

    #print("Index =", kRows.index.tolist())
    tmp = trainLableDF.iloc[kRows.index.tolist()]['CLASS'].value_counts()
    #print(tmp)
    #print(tmp.idxmax())
    return tmp.idxmax()

def calculateEculidDist(trainDF , testRow):
    #np.sqrt(np.sum(np.square((trainArray - testArray))))
    tmp = (((trainDF.sub( testRow, axis=1))**2).sum(axis=1))**0.5
    tmp.sort_values(axis=0, ascending=True, inplace=True)
    #print(type(tmp))
    return tmp


startTime = time.clock()

data = arff.loadarff("veh-prime.arff")
trainDF = pd.DataFrame(data[0])
trainLableDF = trainDF[['CLASS']].copy()
trainDF.drop('CLASS' , axis=1, inplace=True)
# Updating noncar as 0 and car as 1.
trainLableDF['CLASS'] = np.where(trainLableDF['CLASS'] == b'noncar', 0, 1)

print(trainLableDF)
# Z score normalization
trainDFNormalized = normalizeDF(trainDF) 

#print(trainDFNormalized)

featureList = trainDF.columns.tolist()
remainingFeatureList = trainDF.columns.tolist()
print(featureList)

selectedFeatureList = []
accuracyAttain = 0
#tmp = [1,5,8000,10,1000]
#print(max(tmp))
#print(tmp.index(max(tmp)))
iteration = 1
print("********* Starting feature selection using wrapper method (with empty set of feature) **********")
while (len(remainingFeatureList) > 0):  
    print("Iteration = ", iteration)
    iteration += 1
    tmpAccuracyList = []
    for counter in range(len(remainingFeatureList)):
        tmpFeatureList = selectedFeatureList + [remainingFeatureList[counter]]
        #print(tmpFeatureList)
        tmptrainDF = trainDFNormalized[tmpFeatureList]
        #print(tmptrainDF.columns)
        index = 0
        accuracyCount = 0
        predictedClassList = []
        for row in tmptrainDF.itertuples(index=False):
            tmpDF = tmptrainDF.drop(index)
            #tmpTrainLableDF = trainLableDF.drop(index)
            #print(len(tmpTrainLableDF))
            predictedClass = getPredictedClassUsingKNN(tmpDF, row, trainLableDF) 
            #print("predictedClass---- ", predictedClass, " class ---- ", trainLableDF.iloc[index]['CLASS'])
            #if(predictedClass == trainLableDF.iloc[index]['CLASS']):
                #accuracyCount += 1        
            #index += 1
            predictedClassList.append(predictedClass)
        
        predictedTestLabelDF = pd.DataFrame({"CLASS" : predictedClassList})
        #print(predictedTestLabelDF)
        differenceLabel = trainLableDF.sub(predictedTestLabelDF , axis=1)
        #print(differenceLabel)
        accuracyCount = len(differenceLabel[ differenceLabel['CLASS'] ==0 ])
        
        
        tmpAccuracyList.append(round(((accuracyCount/len(trainDFNormalized))*100),2))
    
    print("Features    = ",remainingFeatureList )
    print("Accuracies  = ",tmpAccuracyList )  
    
    maxAccuracy = max(tmpAccuracyList)
    maxAccuracyIndex = tmpAccuracyList.index(max(tmpAccuracyList))
    maxAccuracyFeature = remainingFeatureList[maxAccuracyIndex]
        
    print("Maximum Accuracy achieved is ",maxAccuracy, "%, with feature ",maxAccuracyFeature)
    if(maxAccuracy >= accuracyAttain):
        selectedFeatureList.append(maxAccuracyFeature)
        remainingFeatureList.remove(maxAccuracyFeature)
        accuracyAttain = maxAccuracy
        print("New Selected feature subset is ",selectedFeatureList)
    else:
        print("Accuracy is not increased from the previous feature set, Breaking the iteration")
        break
    

print("Final Selected Feature set is ,", selectedFeatureList)
print("Final Accuracy with above feature set is ", accuracyAttain)

print('Total Time taken is ', (time.clock() - startTime))
#featureList.remove('f4')

#print(featureList)
