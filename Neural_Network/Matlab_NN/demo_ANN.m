clear;clc;close all

% Import data and set parameters.
%%% STEP 1. %%%
% DEMONSTRATION: CLASSIFICATION
load datasets/train_mnist
training = 'on';
testing = 'off';

% PARAMETER SETTINGS
trainRatio = 80; % Percentage of train data, rest of them for validation.
epochTrain = 'off'; 
fineTrainEpoch = 50;
fineLearningRate = 0.01;
momentum = 0.9;
batchSize = 100;
normalize = 'off';
hiddenLayers = [100];
errorMethod = 'CE'; % MSE / CE
hiddenActivation = 'sigmoid'; % sigmoid / tanh
outputActivation = 'softmax'; % linear / sigmoid / softmax
plotOption = 'on'; % on / off

% pre-training

preTrainEpoch = 0;
preLearningRate = 0.01;


%% Build a network with data and parameters.
%%% STEP 2. %%%

N = Netbuild(inputData, targetData, training, testing,trainRatio,epochTrain,...
                      fineTrainEpoch, fineLearningRate, momentum, batchSize,...
                      normalize, hiddenLayers, errorMethod, hiddenActivation,...
                      outputActivation, plotOption, preTrainEpoch, preLearningRate);
                  
%% Train the network.
%%% STEP 3. %%%

N_updated = DNN(N,1);

%% Testing

load datasets/large_test_mnist
N_updated.inputData = inputData;
N_updated.targetData = targetData;
N_updated.training = 'off';
N_updated.testing = 'on';

DNN(N_updated);

