# FeatureSelection
This project is to implement feature selection techniques using filter and wrapper method

For this project, we will be extending our KNN classifier to include automated feature selection. Feature selection is used to remove irrelevant or correlated features in order to improve classification performance. We will be performing feature selection on a variant of the UCI vehicle dataset in the file veh-prime.arff. We will be comparing 2 different feature selection methods: <br />
Filter method which doesn't make use of cross-validation performance and the Wrapper method which does. <br />
<br />
Fix the KNN parameter to be k = 7 for all runs of LOOCV in both task. <br />
<br />
Filter Method <br />
Make the class labels numeric (set \noncar"=0 and \car"=1) and calculate the Pearson Correlation Coeficient (PCC) of each feature with the numeric class label. The PCCvalue is commonly referred to as r. For a simple method to calculate the PCC that is both computationally eficient and numerically stable, see the pseudo code in the pearson.html file.<br />
(a) Fine the features from highest |r| (the absolute value of r) to lowest, along with their |r| values. <br />
(b) Select the features that have the highest m values of |r|, and run LOOCV on the dataset restricted to only those m features. Which value of m gives the highest LOOCV classification accuracy, and what is the value of this optimal accuracy?<br />
<br />

Wrapper Method<br />
Starting with the empty set of features, use a greedy approach to add the single feature that improves performance by the largest amount when added to the feature set. This is Sequential Forward Selection. Define performance as the LOOCV classification accuracy of the KNN classifier using only the features in the selection set (including the ?candidate? feature). Stop adding features only when there is no candidate that when added to the selection set increases the LOOCV accuracy. <br />
(a) Show the set of selected features at each step, as it grows from size zero to its final size (increasing in size by exactly one feature at each step). <br />
(b) What is the LOOCV accuracy over the final set of selected features?
