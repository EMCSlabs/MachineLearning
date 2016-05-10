# This is HY_matlab_NN. 
                                                              Hyungwon Yang
                                                                 2016.02.02
                                                                   EMCS lab

Machine learning toolbox based on matlab.



Linux and MacOSX (This script is not tested on Window)
---

Matlab R2015a
(This script was not tested on the other versions.)



PREREQUISITE
---
1. Data preprocessing: Your data should be pre-processed before setting it 
 on the DNN. Please refer to preprocessing instrcution described as below.

 - Your data should contain 2 variables and be saved as 'inputData' and
  'targetData'. As the name implies, input data should be named 'inputData',
   and target or output data should be named 'targetData'.

 - Variable structure should be examples * features matrix. For examples,
   if your input data has 40 features and you have 1,000 of that kind of
   feature data. Data structure will be 1000 * 40 matrix.

 - For RBM stacking, your input data needs to be normalized scaled between
   0 to 1. (Choose nomalize option 'on' if you didn't normalize your data)
   Normalizing target data is not requisite, since the RBM training is 
   unsupervised.

 - In classification problem, target data should be cosisted of binary 
   information: 0 or 1. For example, if your target data is MNIST and one 
   of the target data is digit 6, then your target data will look like
   [0 0 0 0 0 1 0 0 0 0].



CONTENTS
---------------------------------------------------------------------------
Deep Neural Network (DNN) for classification & curve-fitting problems.

 - DNN
 DNN model to solve classification and curve-fitting problems. For beginners, 
 remain all the default parameters in order to test the given datasets and 
 try to understand the layer structures and its results.
 The users are highly recommended to adjust the parameters in order to 
 improve performance.

 - runDNN
 It contains a whole procedure of training & testing data by machine learning.
 Please follow the instrcution for better undestadning of its usage.

 - Netbuild
 This function builds a main structrue that contains training data and 
 parameters.

		
CONTACTS
---------------------------------------------------------------------------

Hosung Nam / hnam@korea.ac.kr

Hyungwon Yang / hyung8758@gmail.com


VERSION HISTORY
---------------------------------------------------------------------------
1.0. (2016.02.02)
 - 5 scripts were uploaded. Each script works independently. Therefore,
users should select a specific script for training data.
: ANN for classificiaton and curve-fitting,
DBN for classification and curve-fitting,
Netbuild function for building networks
Sample training datasets

2.0. (2016.03.05)
 - 4 types of ANN and DBN are combined. By adjusting parameters, users can
use ANN or DBN based on their deisre. 

2.1. (2016.03.06)
 - Batch size problem fixed.
 - Plotting added: error change, confusion, roc.
       (it is displayed nicely when the monitor is 13 inches.)

2.2. (2016.03.08)
 - Remained batch problem has been perfectly solved.

2.3. (2016.03.11)
 - Error occured when the input numbers are odd(ex, 987) in the testing 
       section. This bug was fixed.
    
2.4. (2016.03.19)
 - Wiehgt visualization during training process is now available. if plotOption
is 'on' in the training session, weights will be visualized. (it may 
occure training delay)
 - Large MNIST dataset is added. 
 - Dataset folder has been created. All datasets will be saved here.
 - demo_ANN script is added.

2.5. (2016.3.23)
 - Reorganize all the functions. Move functions into specific directories.
 - pathOrganizer is added.
 - Error rates from training and validation processes are visualized.
 - trainRatio parameter is added. Its ratio represents the number of 
datasets that will be used for training process. Rest of them will 
be allocated to validation process automatically.
ex. trainRatio = 80 means 80% of datasets for training and 20% of them for validation.
 - epochTrain mode is added. If epochTrain is off then validation error
 will be compared with training error while training the data and it 
 will stop the training process automatically when the error changes 
 between validation and training is not significant any longer.

2.6. (2016.4.02)
 - Validation system changed. 
 - CIFAR10 image datasets are added.
       
