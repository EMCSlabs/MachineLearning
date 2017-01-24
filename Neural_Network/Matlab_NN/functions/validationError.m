% validationError
% This function calculates MSE for error comparision with training datasets.
%
%                                                             Hyungwon Yang                                                             
%                                                             2016. 03. 23
%                                                             EMCS labs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function valMSE = validationError(inputPattern,outputPattern,inputNumber,batchSize,...
                                  hiddenLayerNumber,hiddenAct,outputAct,...
                                  momentum,weightMatrix,biasMatrix)

batchIndex = shakeBatch(inputNumber,batchSize,'test');
for num = 1:length(batchIndex)
    % Assign val input
    valLayerActivation{1} = inputPattern(batchIndex{num},:);
    valBatchInputNumber = length(batchIndex{num});
    
    for hid = 1:hiddenLayerNumber - 2
        % Assign weight and bias
        weight = weightMatrix{hid};
        bias = repmat(biasMatrix{hid+1},valBatchInputNumber,1);
        
        valLayerStore = valLayerActivation{hid} * weight + bias;
        if strcmp(hiddenAct,'sigmoid')
            valLayerActive = BinarySigmoid(momentum,valLayerStore);
        elseif strcmp(hiddenAct,'tanh')
            valLayerActive = tanh(valLayerStore);
        end
        valLayerActive = BinarySigmoid(momentum,valLayerStore);
        valLayerMemory{hid} = valLayerActive;
        valLayerStorage{hid+1} = valLayerStore';
        
        valLayerActivation{hid+1} = valLayerMemory{hid};
    end
    
    % Last up (hidden to output)
    weight = weightMatrix{end};
    bias = repmat(biasMatrix{end},valBatchInputNumber,1);
    
    valLayerStore = valLayerActivation{hiddenLayerNumber-1} * weight + bias;
    
    % Method selection
    if strcmp(outputAct,{'sigmoid'}) % classification
        valLayerMemory{hiddenLayerNumber-1} = BinarySigmoid(valLayerStore);
        
    elseif strcmp(outputAct,{'softmax'}) % classification
        valLayerMemory{hiddenLayerNumber-1} = ml_softmax(valLayerStore);
        
    elseif strcmp(outputAct,'linear') % regression
        valLayerMemory{hiddenLayerNumber-1} = valLayerStore;
    end
    
    valLayerStorage{hiddenLayerNumber} = valLayerStore;
    valLayerActivation{hiddenLayerNumber} = valLayerMemory{end};
    valStackedOutput(batchIndex{num},:) = valLayerActivation{hiddenLayerNumber};
end

if strcmp(outputAct,'linear') 
    valOutputError = (outputPattern - valStackedOutput).^2;
    resultArray = sum(valOutputError) / 2;
    valMSE = floor(sum(resultArray)/inputNumber);
    
elseif sum(strcmp(outputAct,{'sigmoid','softmax'})) 
    valMSE = sum(-sum(outputPattern .* log(valStackedOutput)))/inputNumber;
end

 
