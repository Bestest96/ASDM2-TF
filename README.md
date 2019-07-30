# ASDM2-TF
An implementation of ASDM2 optimizer in TensorFlow. 

## Description
This repository contains a TensorFlow implementation of ASDM2 algorithm. It is defined
as an Optimizer class in [asdm2.py](asdm2.py) file. There is also a demo
code that allows to launch learning on defined data sets using the chosen algorithm
(CM, AG, AdaDelta, AdaGrad, RMSProp, ADAM, ASDM2) with configurable parameters. 

## Instructions
### Cloning the repository
To clone this repository using Git:
```
git clone https://github.com/Bestest96/ASDM2-TF.git
```
### Using ASDM2 optimizer
To use ASDM2 optimizer in Python with TensorFlow, You will need to 
import ASDM2Optimizer class from [asdm2.py](asdm2.py):
```python
from asdm2 import ASDM2Optimizer # assuming asdm2.py is in working directory
```
Then, You will need to define optimizer:
```python
optimizer = ASDM2Optimizer().minimize(loss) # assuming loss op is defined earlier
```
and later pass it during training to run learning operations.
```python
with tf.Session() as sess:
    ...
    opt, loss = sess.run([optimizer, loss], ...)
    ...
```
### Launching demo
To use the demo, You need to have TensorFlow and Numpy installed. You can do it
by using pip or by creating Anaconda virtual environment and installing packages
through conda. 

Assuming Your working directory is in [demo](/demo) folder, You can launch demo
with command:
```
python main.py "[problemDescriptionList] [iterationCount] [reportLength] [minibatchLength] [runCount] [testRunDescriptionList] [ID]" [optArguments]
```
where primary arguments are:

problemDescriptionList: \[problemDescription1,problemDescription2,...\]	

problemDescription: \[problemName\]
	
iterationCount: - integer > 0 - number of teaching algorithm iterations

reportLength: - integer > 0 - number of samples after which reporting happens

minibatchLength: - integer > 0 - the size of a minibatch

runCount: - integer > 0 - number of runs of one experiment - results will be averaged

testRunDescriptionList:	\[testRunDescription1,testRunDescription2,...\]

testRunDescription: \[algorithmName\](parameterName1=value1,parameterName2=value2,...). (...) part is optional, not specified parameters will be set to default values

ID: file identifier to which logs will be saved (as ID.log file)

Primary arguments are mandatory and must be provided in quotation marks.

Problem description can be chosen from \[HandwrittenDigitsMnist, CreditCardUci\]

Algorithm name can be chosen from \[CM, NAG, Adagrad, Adadelta, RMSProp, Adam, ASDM2\]

All algorithms are provided by default params that can be changed
by providing them in parentheses separated by semicolon. 

Optional arguments:

--allow-growth - Tell TensorFlow to take graphics card memory as needed (if using a GPU)
		
--save-graph - Save computation graph that can be later displayed in TensorBoard
		
--alg-first - Iterate over problems algorithms first

#### Example launches
```
python main.py "CreditCardUci 10000000 10000 200 10 ASDM2 cc_asdm2"
python main.py "CreditCardUci 10000000 10000 100 1 ASDM2(use_ag=1) cc_asdm2_ag" # 1 will be converted to True
python main.py "HandwrittenDigitsMnist 100000000 10000 200 10 ASDM2(use_ag=1;use_grad_scaling=1) hd_asdm2_ags" --allow-growth
python main.py "HandwrittenDigitsMnist 10000000 10000 200 10 ASDM2(use_grad_scaling=1) hd_asdm2_s" --allow-growth
python main.py "HandwrittenDigitsMnist 10000000 10000 200 10 Adam(learning_rate=0.01) hd_adam_0_01" --allow-growth
```

#### Miscellaneous
For standard algorithms, their provided parameters are displayed (as they don't change).

For ASDM2, values of beta, lambda, gamma, 1.0 - exp(-nu), loss 
and bar_loss (for theta_bar vars) are displayed, averaged over reporting time. 


For demo purposes, ASDM2 has additional return values added to _finish() method, 
representing reported values described above. 


Datasets are taken from:

[CreditCardUci](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
[HandwrittenDigitsMnist](http://yann.lecun.com/exdb/mnist/)


Datasets were preprocessed and are available in [DataSets](/demo/DataSets) folder.


Handwritten digits are stored in zip archive and need to be extracted before usage.  


Implemented and tested using Tensorflow 1.12.
