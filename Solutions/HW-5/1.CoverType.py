# coding: utf-8
# Name: Danyang Zhangn
# Email: daz040@eng.ucsd.edu
# PID: A53104006

# Result: 15 {'test': 0.08173837962296733, 'train': 0.060181149901666391}
# In[14]:
PYBOLT = True
# need to remove %, ! in .py file in order to run on PYBOLT
if PYBOLT:
    from pyspark import SparkContext
    sc = SparkContext()

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import string
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils


# ### Cover Type
# 
# Classify geographical locations according to their predicted tree cover:
# 
# * **URL:** http://archive.ics.uci.edu/ml/datasets/Covertype
# * **Abstract:** Forest CoverType dataset
# * **Data Set Description:** http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info

# In[15]:

#define a dictionary of cover types
CoverTypes = {1.0: 'Spruce/Fir',
              2.0: 'Lodgepole Pine',
              3.0: 'Ponderosa Pine',
              4.0: 'Cottonwood/Willow',
              5.0: 'Aspen',
              6.0: 'Douglas-fir',
              7.0: 'Krummholz' 
}



# Define the feature names
cols_txt="""
Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology,
Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways,
Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
Horizontal_Distance_To_Fire_Points, Wilderness_Area (4 binarycolumns), 
Soil_Type (40 binary columns), Cover_Type
"""

# In[19]:

# In[27]:

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
if not PYBOLT:
    path = 'covtype/covtype.data'
else:
    path = '/covtype/covtype.data'
    
inputRDD = sc.textFile(path)
inputRDD.first()


# In[21]:

# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.map(lambda line: [float(string.strip(x)) for x in line.split(',')])     .map(lambda p: LabeledPoint(p[-1], p[:-1]))
Data.first()
        


# In[22]:


# ### Making the problem binary
# 
# The implementation of BoostedGradientTrees in MLLib supports only binary problems. the `CovType` problem has
# 7 classes. To make the problem binary we choose the `Lodgepole Pine` (label = 2.0). We therefore transform the dataset to a new dataset where the label is `1.0` is the class is `Lodgepole Pine` and is `0.0` otherwise.
# 
# $\rightarrow$ 1-vs-all

# In[23]:

Label = 2.0
Data = inputRDD.map(lambda line: [float(x) for x in line.split(',')])       .map(lambda p: LabeledPoint(1.0 if p[-1] == Label else 0.0, p[:-1]))


# ### Reducing data size
# In order to see the effects of overfitting more clearly, we reduce the size of the data by a factor of 10

# In[24]:

# Data1, .1 data 
if not PYBOLT:
    Data1 = Data.sample(False, 0.1, seed=255).cache()
else:
    Data1 = Data.cache()
    
(trainingData, testData)=Data1.randomSplit([0.7, 0.3], seed=255)
trainingData.cache()
testData.cache()

if not PYBOLT:
    # TODO: expected output - Sizes: Data1=58022, trainingData=40674, testData=17348
    print 'Sizes: Data1=%d, trainingData=%d, testData=%d'%(Data1.count(), trainingData.cache().count(), testData.cache().count())


# In[25]:

counts = testData.map(lambda lp: (lp.label, 1)).reduceByKey(lambda x,y: x+y).collect()
counts.sort(key=lambda x:x[1],reverse=True)
counts


# ### Gradient Boosted Trees
# 
# * Following [this example](http://spark.apache.org/docs/latest/mllib-ensembles.html#gradient-boosted-trees-gbts) from the mllib documentation
# 
# * [pyspark.mllib.tree.GradientBoostedTrees documentation](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.tree.GradientBoostedTrees)
# 
# #### Main classes and methods
# 
# * `GradientBoostedTrees` is the class that implements the learning trainClassifier,
#    * It's main method is `trainClassifier(trainingData)` which takes as input a training set and generates an instance of `GradientBoostedTreesModel`
#    * The main parameter from train Classifier are:
#       * **data** – Training dataset: RDD of LabeledPoint. Labels should take values {0, 1}.
#       * categoricalFeaturesInfo – Map storing arity of categorical features. E.g., an entry (n -> k) indicates that feature n is categorical with k categories indexed from 0: {0, 1, ..., k-1}.
#       * **loss** – Loss function used for minimization during gradient boosting. Supported: {“logLoss” (default), “leastSquaresError”, “leastAbsoluteError”}.
#       * **numIterations** – Number of iterations of boosting. (default: 100)
#       * **learningRate** – Learning rate for shrinking the contribution of each estimator. The learning rate should be between in the interval (0, 1]. (default: 0.1)
#       * **maxDepth** – Maximum depth of the tree. E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (default: 3)
#       * **maxBins** – maximum number of bins used for splitting features (default: 32) DecisionTree requires maxBins >= max categories
#       
#       
# * `GradientBoostedTreesModel` represents the output of the boosting process: a linear combination of classification trees. The methods supported by this class are:
#    * `save(sc, path)` : save the tree to a given filename, sc is the Spark Context.
#    * `load(sc,path)` : The counterpart to save - load classifier from file.
#    * `predict(X)` : predict on a single datapoint (the `.features` field of a `LabeledPont`) or an RDD of datapoints.
#    * `toDebugString()` : print the classifier in a human readable format.

# In[26]:

from time import time
errors = {}

if not PYBOLT:
    depths = [1, 3, 6, 10, 15, 20]
else:
    depths = [15]

for depth in depths:
    start = time()
    model = GradientBoostedTrees.trainClassifier(trainingData, {}, maxDepth=depth, numIterations=10)
    # numIterations is the numTrees
    #print model.toDebugString()
    
    errors[depth] = {}
    dataSets = {'train': trainingData, 'test': testData}
    for name in dataSets.keys():  # Calculate errors on TRAIN and TEST sets
        data = dataSets[name]
        Predicted = model.predict(data.map(lambda x: x.features))  
        LabelsAndPredictions = data.map(lambda x: x.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v, p): v != p).count()/float(data.count())  # zip
        errors[depth][name] = Err
    if not PYBOLT:
        print depth, errors[depth], int(time()-start), 'seconds'
    else:
        print depth, errors[depth]