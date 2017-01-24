# This is HY_python_NN.
                                                              Hyungwon Yang
                                                                 2015.02.02
                                                                   EMCS lab

Machine learning toolbox based on Python and Theano.



Linux and MacOSX (This script is not tested on Window)
---

~~python 2.7~~ (Python 2.7 was used to be supported but not anymore.)

Python 3.5

Theano



CONTENTS
---
ANN and DBN for classification & curve-fitting models.

- loadfile
  - It contains example files for running scripts.

- ANN_model
  - This script constructs ANN and DBN networks.
  - Feedforward neural network and stochastic gradient descent.

- SetValues
  - This class initiates all the default values including inputs, outputs, weight and biases

- errortools
  - Visual plotting and error calculation.
  - MSE, Pearson r, and correlation plot.

- ANNforClass
  - ANN model to solve classification problem

- ANNforCurvefit
  - ANN model to solve curve-fitting problem (regression)

- DBNforClass
  - DBN model to solve classification problem

- DBNforCurvefit
  - DBN model to solve curve-fitting problem (regression)

		
CONTACTS
---------------------------------------------------------------------------

Hosung Nam / hnam@korea.ac.kr

Hyungwon Yang / hyung8758@gmail.com

VERSION HISTORY
---------------------------------------------------------------------------
1.0 (2015.02.22)
 All 8 scripts were added. 
 - loadfile: It contains example files for running scripts.
 - ANN_model: This script constructs ANN and DBN networks. 
 Feedforward neural network and stochastic gradient descent.
 - SetValues: This class initiates all the default values including inputs, 
 outputs, weight and biases
 - errortools: Visual plotting and error calculation. 
 MSE, Pearson r, and correlation plot.
 - ANNforClass: ANN model to solve classification problem
 - ANNforCurvefit: ANN model to solve curve-fitting problem (regression)
 - DBNforClass: DBN model to solve classification problem
 - DBNforCurvefit: DBN model to solve curve-fitting problem (regression)

1.1 (2015.03.07)
 - In classification, prediction for accuracy bug fixed.

1.2 (2015.04.30)
 - ANN scripts that runnable in python2 is moved to 'VersionHistory' folder. (refer to for_python2.zip file)
 - Main scripts are re-written in order to run it in python3. From now on, ANN script is only supportable in python3.


