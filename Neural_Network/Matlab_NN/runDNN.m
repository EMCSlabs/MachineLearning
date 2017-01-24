% runDNN
% DNN trial script. 
%                                                             Hyungwon Yang
%                                                             2016. 02. 26
%                                                             EMCS labs


% USAGE
% 1. Select curve-fitting or classification problems.
% 2. Run the STEP 1. (You should choose one problem and run only that part.)
% 3. Run the STEP 2. (Run this part no matter what problems you chose.)
% 4. Run the STEP 3. and 4. (You should run the problem related parts.)
% EXAMPLE
% CURVE-FITTING: (click the line and 'run selection')
% > Run as follows: line number 21 > 83 > 92 > 97
% CLASSIFICATION:
% > Run as follows: line number 51 > 83 > 113 > 118
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear command window and workspace.
clear;clc;close all
% Set path
pathOrganizer()

%%% STEP 1. %%%
% DEMONSTRATION: CURVE-FITTING
load datasets/train_mfcc
training = 'on';
testing = 'off';

% PARAMETER SETTINGS
trainRatio = 80; % Percentage of train data, rest of them for validation.
epochTrain = 'on'; 
fineTrainEpoch = 100;
fineLearningRate = 0.001;
momentum = 0.9;
batchSize = 10;
normalize = 'on';
hiddenLayers = [100 100 100];
errorMethod = 'MSE'; % MSE / CE
hiddenActivation = 'sigmoid'; % sigmoid / tanh
outputActivation = 'linear'; % linear / sigmoid / softmax
plotOption = 'off'; % on / off

% pre-training
preTrainEpoch = 5;
preLearningRate = 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Clear command window and workspace.
clear;clc;close all
% Set path
pathOrganizer()
%%% STEP 1. %%%
% DEMONSTRATION: CLASSIFICATION
load datasets/train_mnist
training = 'on';
testing = 'off';

% PARAMETER SETTINGS
trainRatio = 80; % Percentage of train data, rest of them for validation.
epochTrain = 'off'; 
fineTrainEpoch = 100;
fineLearningRate = 0.01;
momentum = 0.9;
batchSize = 10;
normalize = 'off';
hiddenLayers = [10];
errorMethod = 'CE'; % MSE / CE
hiddenActivation = 'sigmoid'; % sigmoid / tanh
outputActivation = 'softmax'; % linear / sigmoid / softmax
plotOption = 'off'; % on / off

% pre-training

preTrainEpoch = 0;
preLearningRate = 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%% STEP 2. %%%

N = Netbuild(inputData, targetData, training, testing,trainRatio,epochTrain,...
                      fineTrainEpoch, fineLearningRate, momentum, batchSize,...
                      normalize, hiddenLayers, errorMethod, hiddenActivation,...
                      outputActivation, plotOption, preTrainEpoch, preLearningRate);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training data : curve-fitting
%%% STEP 3. %%%

N_updated = DNN(N,1);
save('N_updated_mfcc','N_updated')

%% Testing data : curve-fitting
%%% STEP 4. %%%
clear;clc;close all

load datasets/test_mfcc
load N_updated_mfcc
N_updated.inputData = inputData;
N_updated.targetData = targetData;
N_updated.training = 'off';
N_updated.testing = 'on';
N_updated.plotOption = 'on';

DNN(N_updated);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training data : classification
%%% STEP 3. %%%

N_updated = DNN(N,1);
save('N_updated_mnist','N_updated')

%% Testing data : classification
%%% STEP 4. %%%
clear;clc;close all

load datasets/test_mnist
load N_updated_mnist
N_updated.inputData = inputData;
N_updated.targetData = targetData;
N_updated.training = 'off';
N_updated.testing = 'on';

DNN(N_updated);