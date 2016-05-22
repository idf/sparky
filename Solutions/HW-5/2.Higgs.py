# coding: utf-8

# In[2]:

# 10 {'test': 0.26783881760716044, 'train': 0.20966912394994}
# Name: Danyang Zhangn
# Email: daz040@eng.ucsd.edu
# PID: A53104006
PYBOLT = True
# need to remove %, ! in .py file in order to run on PYBOLT
if PYBOLT:
    from pyspark import SparkContext
    sc = SparkContext()


from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils



# ### Higgs data set
# * **URL:** http://archive.ics.uci.edu/ml/datasets/HIGGS#  
# * **Abstract:** This is a classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.
# 
# **Data Set Information:**  
# The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.
# 
# 

# In[3]:

#define feature names
feature_text = ('lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, '
                'jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, '
                'jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb')
features=[a.strip() for a in feature_text.split(',')]
if not PYBOLT:
    print len(features), features

# In[6]:

if PYBOLT:
    path = '/HIGGS/HIGGS.csv'
else:
    path = 'higgs/HIGGS.csv'
    
inputRDD = sc.textFile(path)
# inputRDD.first()


# In[7]:

Data = inputRDD.map(lambda line: [float(x.strip()) for x in line.split(',')])     .map(lambda p: LabeledPoint(p[0], p[1:]))
# Data.first()


# ### Reducing data size
# In order to see the effects of overfitting more clearly, we reduce the size of the data by a factor of 100

# In[8]:

if not PYBOLT:
    Data1 = Data.sample(False, 1/100.0, seed=255).cache()
else:
    Data1 = Data.sample(False, 10/100.0, seed=255).cache()

(trainingData, testData)=Data1.randomSplit([0.7, 0.3], seed=255)
trainingData.cache()
testData.cache()

if not PYBOLT:
    print 'Sizes: Data1=%d, trainingData=%d, testData=%d'%(Data1.count(), trainingData.cache().count(), testData.cache().count())


# In[9]:

counts = testData.map(lambda lp: (lp.label, 1)).reduceByKey(lambda x,y: x+y).collect()
counts.sort(key=lambda x: x[1], reverse=True)
counts


# ### Gradient Boosted Trees

# In[ ]:

from time import time
errors = {}
if not PYBOLT:
    depths = [1, 3, 6, 10]
else:
    depths = [10]
    
for depth in depths:  # 15, 20
    start = time()
    model = GradientBoostedTrees.trainClassifier(trainingData, {}, maxDepth=depth, numIterations=30, learningRate=0.3)
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