# -*- coding: utf-8 -*-
'''


                                                                    Written by Hyungwon Yang
                                                                                2016. 03. 09
                                                                                   EMCS Labs
'''
import numpy as np
import random


class StackRBM(object):

    def __init__(self,inputData,preTrainEpoch,weightMatrix,biasMatrix):
        self.InputPattern = inputData
        self.preTrainEpoch = preTrainEpoch
        self.weighMatrix = weightMatrix
        self.biasMatrix = biasMatrix
        self.hiddenNumber = len(weightMatrix)

    def trainRBM(self):

        # The number of hidden layers for training
        for hidden in range(hiddenNumber):

            vhMatrix = weightMatrix[hidden]
            hBiasMatrix = biasMatrix[hidden+1]
            vBiasMatrix = biasMatrix[hidden]
            inputSave = []

            # training the data for given epochs
            for epoch in range(preTrainEpoch):

                if type(inputPattern) is not list:
                    rand_num = np.random.permutation(len(inputPattern))
                    layerForPT = inputPattern[rand_num]
                else:
                    layerForPT = [ran for ran in inputPattern]
                    random.shuffle(layerForPT)


                # training each data for updating weights and biases(unsupervised learning)
                for num in range(len(inputPattern)):

                    ### visual0
                    if type(layerForPT) is not list:
                        visual0Array = np.array([layerForPT[num]])
                    else:
                        visual0Array = layerForPT[num]
                    # hidden0
                    hidden0 = np.dot(visual0Array,vhMatrix) + hBiasMatrix
                    hidden0Array = logistic.binarySigmoid(momentum,hidden0)

                    ### visual1
                    visual1 = np.dot(hidden0Array,vhMatrix.T) + vBiasMatrix
                    visual1Array = logistic.binarySigmoid(momentum,visual1)

                    ### hidden1
                    hidden1 = np.dot(visual1Array,vhMatrix) + hBiasMatrix
                    hidden1Array = logistic.binarySigmoid(momentum,hidden1)

                    # update weights and biases
                    vhMatrix += lr * (np.dot(visual0Array.T,hidden0Array) - np.dot(visual1Array.T,hidden1Array))
                    vBiasMatrix += lr * (visual0Array - visual1Array)
                    hBiasMatrix += lr * (hidden0Array - hidden1Array)
                    if epoch+1 == pre_epoch:

                        inputSave.append(hidden0Array)

                print('{}/{} Hidden Layer, {}/{} epoch'.format(hidden+1,hidden_num,epoch+1,pre_epoch))

            weightMatrix[hidden] = vhMatrix
            biasMatrix[hidden+1] = hBiasMatrix
            biasMatrix[hidden] = vBiasMatrix
            inputPattern = inputSave