% Network Building Function
% This is the function for building a 'N' structure for running DNN
%
%                                                             Hyungwon Yang                                                            
%                                                             2016. 02. 26
%                                                             EMCS labs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function N = Netbuild(inputData, targetData, training, testing, trainRatio,...
                      epochTrain,fineTrainEpoch, fineLearningRate, momentum, batchSize,...
                      normalize, hiddenLayers, errorMethod, hiddenActivation,...
                      outputActivation, plotOption, preTrainEpoch, preLearningRate)

% Format Check
if trainRatio <= 0 || trainRatio > 100
    error('trainRatio value should range from 0 to 100')
elseif trainRatio == 100
    warning(['trainRatio value is 100 which means every datasets will be allocated to training sets.'...
             'validation process will not be operated.'])
elseif any(rem(trainRatio,1))
    error('trainRatio value should be integer not float.')
end
                  
% Input and output infromation
[~,inputUnit] = size(inputData);
[~,outputUnit] = size(targetData);

% Initializing weights and biases.
% Hidden Layer Information
hiddenStructure = hiddenLayers;

% the number of hidden layers and units
layerStructure{1} = inputUnit;
for hls = 2:length(hiddenStructure)+1
    layerStructure{hls} = hiddenStructure(hls-1);
end
layerStructure{length(hiddenStructure)+2} = outputUnit;

hiddenLayerNumber = length(layerStructure);

visualBiasValue = log10((1/layerStructure{1}) / (1-(1/layerStructure{1})));
biasMatrix{1} = repmat(visualBiasValue,1,layerStructure{1});

range = 0.1;
for i = 1:hiddenLayerNumber-1
    
    % Weight Matrix
    weightMatrix{i} = randn(layerStructure{i},layerStructure{i+1})* range * 2 - range;
    %weightMatrix{i} = randn(layerStructure{i},layerStructure{i+1});
    % Bias Matrix
    visualBiasValue = log10((1/layerStructure{i+1}) / (1-(1/layerStructure{i+1})));
    biasMatrix{i+1} = repmat(visualBiasValue,1,layerStructure{i+1});
    layerError{i} = zeros(1,1);
end

ERROR{1} = layerError;
WEIGHT{1} = weightMatrix;
BIAS{1} = biasMatrix;
error_history = [];

% Structuring
N = struct('inputData',inputData,'targetData',targetData,'training',training,'testing',testing,...
    'trainRatio',trainRatio,'epochTrain',epochTrain,'fineTrainEpoch',fineTrainEpoch,...
    'fineLearningRate',fineLearningRate,'momentum',momentum,'batchSize',batchSize,...
    'normalize',normalize,'hiddenLayers',hiddenLayers,'errorMethod',errorMethod,...
    'plotOption',plotOption,'preTrainEpoch',preTrainEpoch,'preLearningRate',preLearningRate,...
    'hiddenActivation',hiddenActivation,'outputActivation',outputActivation,...
    'layerError',ERROR,'weight',WEIGHT,'bias',BIAS,'errorInfo',error_history);
