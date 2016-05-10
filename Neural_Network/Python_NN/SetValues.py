# -*- coding: utf-8 -*-
'''
For setting default values of ANN and DBN.

                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 10
                                                                                   EMCS Labs
'''

import numpy as np
import theano
import theano.tensor as T


# Setting the ANN default value.
class SetANN(object):

    def __init__(self,inputs,outputs,learningRate,
                 momentum,epochNum,hiddenUnits,W=None,b=None):
        self.inputs = inputs
        self.outputs = outputs
        self.lr = learningRate if learningRate is not None else 0.01
        self.momentum = momentum if momentum is not None else 0.9
        self.epochNum = epochNum
        self.hiddenUnits = hiddenUnits
        self.W = W
        self.b = b

        # check inputs and outputs
        assert len(self.inputs) and len(self.outputs) > 0

        #  initial weight and bias
        seed_one,seed_two = np.random.randint(100,999,2)
        self.genRan_w_ih = np.random.RandomState(seed_one)
        self.genRan_w_ho = np.random.RandomState(seed_two)
        self.addrange = 0.1

    def epochs(self):
        return self.epochNum

    def floatX(self,value):
        changed_value = np.asarray(value, dtype=theano.config.floatX)
        return changed_value

    # Generate weights.
    def genWeight(self):

        if self.W is None:
            sampleNum,unitNum = self.inputs.shape
            outputSize = self.outputs.shape[1]
            ihMatrix = theano.shared(self.floatX(self.genRan_w_ih.rand(unitNum,self.hiddenUnits) * self.addrange * 2 - self.addrange))
            hoMatrix = theano.shared(self.floatX(self.genRan_w_ho.rand(self.hiddenUnits,outputSize) * self.addrange * 2 - self.addrange))
            return ihMatrix, hoMatrix

    # Generate biases.
    def genBias(self):
        if self.b is None:
            outputSize = self.outputs.shape[1]
            hBiasMatrix = theano.shared(self.floatX(np.ones(self.hiddenUnits)))
            oBiasMatrix = theano.shared(self.floatX(np.ones(outputSize)))
            return hBiasMatrix, oBiasMatrix

    # Generate symbolic matrices.
    def genMatrices(self):
        self.inX = T.fmatrix('inputs')
        self.outY = T.fmatrix('outputs')
        return self.inX, self.outY


# Setting the DBN default value.
class SetDNN(object):

    def __init__(self,inputData,targetData,fineLearningRate,preLearningRate,
                 batchSize,momentum,preTrainEpoch,fineTrainEpoch,
                 hiddenUnits,W=None,b=None):
        self.inputs = inputData
        self.outputs = targetData
        self.finelr = fineLearningRate if fineLearningRate is not None else 0.01
        self.prelr = preLearningRate if preLearningRate is not None else 0.01
        self.batchSize = batchSize
        self.momentum = momentum
        self.pre_epoch = preTrainEpoch
        self.fine_epoch = fineTrainEpoch
        self.hiddenUnits = hiddenUnits
        self.W = W
        self.b = b

        # check inputs and outputs
        assert len(self.inputs) and len(self.outputs) > 0

        #  initial weight and bias
        self.addrange = 0.1

    def epochs(self):
        return self.pre_epoch, self.fine_epoch

    def params(self):
        return self.finelr, self.prelr, self.momentum, self.batchSize


    def floatX(self,value):
        changed_value = np.asarray(value, dtype=theano.config.floatX)
        return changed_value

    def genHiddenBox(self):

        inputNum = self.inputs.shape[1]
        outputNum = self.outputs.shape[1]
        hiddenBox = inputNum
        for iter in self.hiddenUnits:
            hiddenBox = np.append(hiddenBox,iter)

        hiddenBox= np.append(hiddenBox,outputNum)
        return hiddenBox


    def genWeight(self):

        hid = self.genHiddenBox()
        weightMatrix = []
        for layer in range(len(hid)-1):

            seed = np.random.randint(100,999)
            self.genRan = np.random.RandomState(seed)
            weight = self.genRan.rand(hid[layer],hid[layer+1]) * self.addrange * 2 - self.addrange
            weightMatrix.append(weight)

        return weightMatrix

    def genBias(self):

        hid = self.genHiddenBox()
        biasMatrix = []
        for layer in hid:

            if layer < 2:
                biasInit = 1
            else:
                biasInit = np.log10((1.0/layer) / (1.0-(1.0/layer)))

            bias = np.array([np.tile(biasInit,layer)])
            biasMatrix.append(bias)

        return biasMatrix

    # Generate symbolic matrices.
    def genMatrices(self):
        self.inX = T.matrix('inputs')
        self.outY = T.matrix('outputs')
        return self.inX, self.outY

    # generate shared variables.
    def sharing(self,weightMatrix,biasMatrix):

        weightBox = []
        biasBox = []
        for w in weightMatrix:
            weight = theano.shared(self.floatX(w))
            weightBox.append(weight)

        for b in biasMatrix:
            bias = theano.shared(self.floatX(b))
            biasBox.append(bias)

        return weightBox, biasBox

def shakeBatch(inputNumber,batchSize,option):

    # Get batchNumber
    if np.remainder(inputNumber,batchSize) is 0:
        batchNumber = (inputNumber/batchSize)
    else:
        batchNumber = (inputNumber/batchSize) + 1

    # Check option
    if option is 'train':
        batchBox = np.random.permutation(inputNumber)
    elif option is 'test':
        batchBox = range(inputNumber)
    else:
        raise Exception('Option variable of shakeBatch function is inapplicable. It should be train or test.')

    start = -1
    final = -1
    batchIndex = []
    for run in range(batchNumber):

        start = final + 1
        final = final + batchSize
        if final > inputNumber:
            final -= final - inputNumber
        batchIndex.append(batchBox[start:final+1])

    return batchIndex[0:-1]




