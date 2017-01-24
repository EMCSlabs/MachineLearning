# -*- coding: utf-8 -*-
'''
NetworkModel for constructing ANN and DBN models
Feedforward networks, gradient descent.


                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 07
                                                                                    EMCS Lab
'''
import numpy as np
import theano.tensor as T
import random
import errortools as et

from binarySigmoid import binarySigmoid
from SetValues import shakeBatch


# This is for DNN
class DNN_model(object):

    def __init__(self,inX,outY,weightMatrix,biasMatrix,fineLearningRate):
        self.inputs = inX
        self.targets = outY
        self.weightMatrix = weightMatrix
        self.biasMatrix = biasMatrix
        self.finelr = fineLearningRate if fineLearningRate is not None else 0.01

    def fnn(self,hiddenFunction,outputFunction):
        '''
        This feedforward network is not intended for batch_learning. Instead, it is on-lines based learning networks
        that take more learning time but update weights and biases frequently.
        '''

        # iteration
        hiddenNumber = len(self.weightMatrix)-1
        outputStorage = []
        outputActivation = []

        # First up
        outputStorage.append(T.dot(self.inputs,self.weightMatrix[0]) + self.biasMatrix[1])
        if hiddenFunction is 'sigmoid':
            outputActivation.append(T.nnet.sigmoid(outputStorage[0]))

            # Hidden up
            for iter in range(1,hiddenNumber):

                # Hiddenlayer activation.
                outputStorage.append(T.dot(outputActivation[iter-1],self.weightMatrix[iter]) + self.biasMatrix[iter+1])
                outputActivation.append(T.nnet.sigmoid(outputStorage[iter]))

        elif hiddenFunction is 'tanh':
            outputActivation.append(T.tanh(outputStorage[0]))

            # Hidden up
            for iter in range(1,hiddenNumber):

                # Hiddenlayer activation.
                outputStorage.append(T.dot(outputActivation[iter-1],self.weightMatrix[iter]) + self.biasMatrix[iter+1])
                outputActivation.append(T.tanh(outputStorage[iter]))

        if outputFunction is 'softmax':
            outputStorage.append(T.nnet.softmax(T.dot(outputActivation[-1],self.weightMatrix[-1]) + self.biasMatrix[-1]))
            outputActivation.append(T.argmax(outputStorage[-1],axis=1))

        elif outputFunction is 'linear':
            outputStorage.append(T.dot(outputActivation[-1],self.weightMatrix[-1]) + self.biasMatrix[-1])
            outputActivation.append(outputStorage[-1])

        elif outputFunction is 'sigmoid':
            outputStorage.append(T.nnet.sigmoid(T.dot(outputActivation[-1],self.weightMatrix[-1]) + self.biasMatrix[-1]))
            outputActivation.append(outputStorage[-1])

        return outputStorage, outputActivation

    # Stochastic Gradient Descent. (classification)
    #def sgd(self,outputStorage):
    #    cost = T.mean(T.nnet.categorical_crossentropy(outputStorage,self.targets))
    #    params = [self.ihMatrix,self.hoMatrix,self.hBiasMatrix,self.oBiasMatrix]
    #    gradient = T.grad(cost=cost, wrt=params)
    #    updates = []
    #    for p, g in zip(params, gradient):
    #        updates.append([p, p - g * self.lr])
    #    return cost, updates

    # Stochastic Gradient Descent. (curvefitting)

    def sgd2(self,outputStorage):

        cost = T.mean(T.pow(outputStorage[-1]-self.targets,2))/2
        params = []

        # Set the parameters
        for w in self.weightMatrix[::-1]: params.append(w)
        for b in self.biasMatrix[:0:-1]: params.append(b)

        gradient = T.grad(cost=cost,wrt=params)
        updates = []
        for p, q in zip(params, gradient):
            updates.append([p, p - q * self.finelr])

        return cost, updates

class StackRBM(object):

    def __init__(self,inputData,preTrainEpoch,preLearningRate,weightMatrix,biasMatrix,batchSize,momentum):
        self.inputPattern = inputData
        self.inputNumber = len(inputData)
        self.preTrainEpoch = preTrainEpoch
        self.prelr = preLearningRate
        self.weightMatrix = weightMatrix
        self.biasMatrix = biasMatrix
        self.batchSize = batchSize
        self.momentum = momentum
        self.hiddenNumber = len(weightMatrix)-1

    def trainRBM(self):

        # The number of hidden layers for training
        for hidden in range(self.hiddenNumber):

            # Assign input data, weight and bias
            vhMatrix = self.weightMatrix[hidden]
            hBiasMatrix = self.biasMatrix[hidden+1]
            vBiasMatrix = self.biasMatrix[hidden]
            inputSave = []

            # training the data for given epochs
            for epoch in range(self.preTrainEpoch):

                # Randomize all the training data.
                layerForPT = self.inputPattern
                batchIndex = shakeBatch(self.inputNumber,self.batchSize,'train')

                # training each data for updating weights and biases(unsupervised learning)
                for num in range(len(batchIndex)):

                    # Bias replication.
                    batchInputNumber = len(batchIndex[num])
                    batch_hBiasMatrix = np.tile(hBiasMatrix,[batchInputNumber,1])
                    batch_vBiasMatrix = np.tile(vBiasMatrix,[batchInputNumber,1])

                    ### visual0
                    visual0Array = layerForPT[batchIndex[num]]

                    # hidden0
                    hidden0 = np.dot(visual0Array,vhMatrix) + batch_hBiasMatrix
                    hidden0Array = binarySigmoid(self.momentum,hidden0)

                    ### visual1
                    visual1 = np.dot(hidden0Array,vhMatrix.T) + batch_vBiasMatrix
                    visual1Array = binarySigmoid(self.momentum,visual1)

                    ### hidden1
                    hidden1 = np.dot(visual1Array,vhMatrix) + batch_hBiasMatrix
                    hidden1Array = binarySigmoid(self.momentum,hidden1)

                    # update weights and biases
                    vhMatrix += self.prelr * (np.dot(visual0Array.T,hidden0Array) - np.dot(visual1Array.T,hidden1Array))
                    vBiasMatrix += np.mean(self.prelr * (visual0Array - visual1Array),axis=0)
                    hBiasMatrix += np.mean(self.prelr * (hidden0Array - hidden1Array),axis=0)
                    if epoch+1 == self.preTrainEpoch:

                        for line in hidden0Array:
                            inputSave.append(line)

                print('{}/{} Hidden Layer, {}/{} epoch'.format(hidden+1,self.hiddenNumber,epoch+1,self.preTrainEpoch))

            self.weightMatrix[hidden] = vhMatrix
            self.biasMatrix[hidden+1] = hBiasMatrix
            self.biasMatrix[hidden] = vBiasMatrix
            if epoch+1 == self.preTrainEpoch:
                self.inputPattern = np.asarray(inputSave)

        return self.weightMatrix, self.biasMatrix, self.inputPattern





