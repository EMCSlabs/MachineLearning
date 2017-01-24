# -*- coding: utf-8 -*-
'''
Deep Neural Network for Curve-Fitting and Classification Problems.


                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 10
                                                                                   EMCS Labs
'''
import numpy as np
import theano
import loadfile
import errortools as et

from SetValues import SetDNN, shakeBatch
from NetworkModel import StackRBM, DNN_model

#def dnn(inputData,targetData,training,testing,fineTrainEpoch,fineLearningRate,
#        momentum,batchSize,normalize,hiddenLayers,errorMethod,hiddenActivation,
#        outputActivation,plotOption,preTrainEpoch,preLearningRate):

    ##### tmp values

inputData, targetData, test_in, test_out = loadfile.readmnist()
training = 'on'
testing = 'on'
fineTrainEpoch = 5
fineLearningRate = 0.001
momentum = 0.9
batchSize = 10
normalize = 'off'
hiddenLayers = [50,50,50]
errorMethod = 'MSE'
hiddenFunction= 'sigmoid'
outputFunction = 'softmax'
plotOption = 'off'
preTrainEpoch = 1
preLearningRate = 0.1
error_epoch = 5

#####

print('Loading the data and setting default values...')

inputNumber,inputUnit = np.shape(inputData)
targetNumber,targetUnit = np.shape(targetData)



print('Constructing the hidden layers...')

# Setting default values by using SetValues.
values = SetDNN(inputData=inputData,
                targetData=targetData,
                fineLearningRate=fineLearningRate,
                preLearningRate=preLearningRate,
                batchSize=batchSize,
                momentum=momentum,
                preTrainEpoch=preTrainEpoch,
                fineTrainEpoch=fineTrainEpoch,
                hiddenUnits=hiddenLayers,
                W=None,
                b=None)

# Setting hidden layers: weightMatrix and biasMatrix
weightMatrix = values.genWeight()
biasMatrix = values.genBias()
inX, outY = values.genMatrices()

# Setting parameters
preTrainEpoch, fineTrainEpoch = values.epochs()
finelr, prelr, momentum, batchSize = values.params()


################################
#####   PRETRAIN SESSION   #####
################################

print('running pre-training...')

RBM = StackRBM(inputData=inputData,
               preTrainEpoch=preTrainEpoch,
               preLearningRate=preLearningRate,
               weightMatrix=weightMatrix,
               biasMatrix=biasMatrix,
               batchSize=batchSize,
               momentum=momentum)

weightMatrix, biasMatrix, inputPattern = RBM.trainRBM()

# Sharing the weight and bias
weightMatrix, biasMatrix = values.sharing(weightMatrix,biasMatrix)


################################
#####   FINETUNE SESSION   #####
################################

# Fine turning
print('Fine_tuning the data...')

print('Constructing DBN_model...')
models = DNN_model(inX=inX,
                   outY=outY,
                   weightMatrix=weightMatrix,
                   biasMatrix=biasMatrix,
                   fineLearningRate=fineLearningRate,
                   )

outputStorage, outputActivation = models.fnn(hiddenFunction,outputFunction)
cost, updates = models.sgd2(outputStorage)

# Train and Predict function.
train = theano.function(inputs=[inX,outY],outputs=cost,updates=updates,allow_input_downcast=True)
predict = theano.function(inputs=[inX],outputs=outputActivation[-1],allow_input_downcast=True)


print('Training the data...')
'''
for epoch in range(fineTrainEpoch):

    # Shuffling inputs and outputs
    batchIndex = shakeBatch(inputNumber,batchSize,'train')
    error_history = []

    #total_inputs = range(train_in.shape[0])
    for batch_input,batch_target in zip(range(len(batchIndex)),range(len(batchIndex))):

        error = train(np.array(inputData)[batchIndex[batch_input]],np.array(targetData)[batchIndex[batch_target]])
        error_history.append(error)
    error_sum = np.mean(error_history)

    # Print error.
    if epoch%error_epoch == 0:

        print 'Epoch: {}, error: {}.'.format(epoch+1,error_sum)
'''
########################################################
for epoch in range(fineTrainEpoch):

    # Shuffling inputs and outputs
    batchIndex = shakeBatch(inputNumber,batchSize,'train')
    rand_num = np.random.permutation(len(inputData))
    inputData = inputData[rand_num]
    targetData = targetData[rand_num]
    error_history = []

    #total_inputs = range(train_in.shape[0])
    for batch_in,batch_out in zip(range(0,len(inputData)+1,batchSize),range(1,len(inputData)+1,batchSize)):

        #error = train(train_in[batch_in:batch_out],train_out[batch_in:batch_out])
        error = train(np.array(inputData[batch_in:batch_out]),np.array(targetData[batch_in:batch_out]))
        error_history.append(error)
    #if batch_out
    error_sum = np.mean(error_history)

    # Print error.
    if epoch%error_epoch == 0:

        print('Epoch: {}, error: {}.'.format(epoch+1,error_sum))
#########################################################

print('\nFine_tuning completed. Testing the result...')



###############################
#####   TESTING SESSION   #####
###############################


