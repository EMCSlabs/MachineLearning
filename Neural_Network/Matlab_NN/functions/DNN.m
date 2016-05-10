% Machine Learning Toolbox
%    for classification and curve-fitting problems.
%
%                                                             Hyungwon Yang                                                             
%                                                             2016. 02. 29
%                                                             EMCS labs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function N_updated = DNN(N,varargin)
% Check input variables.
if nargin < 1
    error(['Input network is not defined. Please use ''Netbuild'''...
           'to constrcut the input network first.'])
end
if length(varargin) > 1
    error(['Too many optional variables. Please check which '...
           'specific optional variables are available.'])
elseif ~isempty(varargin)
    % Option: force process or not. if you check 'forceProcess', DNN
    % function will not ask you for data validity.
    processOption(1) = varargin{1};
else
    processOption(1) = 0;
end

%%%%% RETRIEVE THE DATA AND DISPLAY ITS INFORMATION %%%%%
if processOption(1) == 0
    fprintf(['\t#####                     #####\n'...
             '\t##### Deep Neural Network #####\n'...
             '\t#####                     #####\n'...
             '\t#####           EMCS LABS #####\n\n'])
    pause(1)
    fprintf(['  Data setting information will be displayed.\n'...
             '  Please check all the parameters thoroughly before running DNN.\n'])
    pause(2)
elseif processOption(1) == 1
    fprintf(['\t#####                     #####\n'...
             '\t##### Deep Neural Network #####\n'...
             '\t#####                     #####\n'...
             '\t#####           EMCS LABS #####\n\n'])
end

% Retrieve the train and test data information 
INdata = N.inputData;
OUTdata = N.targetData;

% Divide the data into training and validation sets.
ratioValue = N.trainRatio;
trainRatio = ceil(length(INdata) * ratioValue * 0.01);
inputData = INdata(1:trainRatio,:);
outputData = OUTdata(1:trainRatio,:);
valInputData = INdata(trainRatio+1:end,:);
valOutputData = OUTdata(trainRatio+1:end,:);
if isempty(valInputData) 
    epochMode = 'on';
end

% Data Normalization: zscoring.
if strcmp(N.normalize,'on')
    % Data normalization
    inputPattern{1} = zscore(inputData);
    normalize = 'on';
else
    inputPattern{1} = inputData;
    normalize = 'off';
end
outputPattern = outputData;

[inputNumber,inputUnit] = size(inputPattern{1});
[outputNumber,outputUnit] = size(outputPattern);
[valInputNumber,~] = size(valInputData);
[valOutputNumber,~] = size(valOutputData);

% Retrieve parameters.
trainRatio = N.trainRatio;
validationRatio = 100 - N.trainRatio;
trainMode = N.training;
testMode = N.testing;
if strcmp(trainMode,'on'); 
    learningMode = 'Training Mode'; else learningMode = 'Testing Mode';
end
batchSize = N.batchSize;
fineTrainEpoch = N.fineTrainEpoch;
fineLearningRate = N.fineLearningRate;
momentum = N.momentum;
hiddenAct = N.hiddenActivation;
outputAct = N.outputActivation;
epochMode = N.epochTrain;
if strcmp(epochMode,'on')
    warning(['  Epoch mode is activated.\n Turning this mode off is preferred to achieve'...
        'better training results.\n'])
elseif strcmp(epochMode,'off')
    fineTrainEpoch = 10e5;
end

% Pretraining Mode
if N.preTrainEpoch > 0
    preTrain = 'on';
else
    preTrain = 'off';
end
preTrainEpoch = N.preTrainEpoch;
preLearningRate = N.preLearningRate;

% If preTrain mode is on: rescale the data between 0 to 1.
if strcmp(preTrain,'on')
    min_val = min(min(inputPattern{1}));
    max_val = max(max(inputPattern{1}));
    
    % Check whether input data is rescaled or not.
    if min_val < 0 || max_val > 1
        % Data rescaling. (0 to 1)
        up = bsxfun(@minus,inputPattern{1},min(inputPattern{1}));
        down = max(inputPattern{1}) - min(inputPattern{1});
        inputPattern{1} = bsxfun(@rdivide,up,down);
        normalize = 'on';
        fprintf('Input data is rescaled between 0 to 1.\n')
    end
end

% Hidden Layer Information
hiddenStructure = N.hiddenLayers;
hiddenInfo = num2str(hiddenStructure);

% the number of hidden layers and units
layerStructure{1} = inputUnit;
for hls = 2:length(hiddenStructure)+1
    layerStructure{hls} = hiddenStructure(hls-1);
end
layerStructure{length(hiddenStructure)+2} = outputUnit;
hiddenLayerNumber = length(layerStructure);

% Error calculation and plotting parameters
errorMethod = N.errorMethod;
plotOption = N.plotOption;

% Parameter error check
assert(sum(strcmp(N.training,{'on','off'}))==1,'Training value should be on or off.')
assert(sum(strcmp(N.testing,{'on','off'}))==1,'Testing value should be on or off.')
assert(sum(strcmp(N.epochTrain,{'on','off'}))==1,'epochTrain value should be on or off.')
assert(sum(strcmp(N.normalize,{'on','off'}))==1,'normalize value should be on or off.')
assert(sum(strcmp(N.plotOption,{'on','off'}))==1,'plotOption value should be on or off.')
assert(sum(strcmp(N.errorMethod,{'MSE','CE'}))==1,'errorMethod value should be MSE or CE.')
assert(sum(strcmp(N.hiddenActivation,{'sigmoid','tanh'}))==1,'hiddenActivation value should be sigmoid or tanh.')
assert(sum(strcmp(N.outputActivation,{'sigmoid','softmax','linear'}))==1,'outputActivation value should be sigmoid, softmax, or linear.')

% Information description before running DNN
fprintf(['\n'...
    '----- DATA INFORMATION -----\n\n'...
    '* DATA: \n'...
    '   Input data     : %d\n'...
    '   Target data    : %d\n'...
    '   Validation data: %d\n'...
    '   Train mode     : %s\n'...
    '   Test mode      : %s\n'...
    '   Epoch mode     : %s\n'...
    '* SIMULATION SETTINGS: \n'...
    '   Hidden structure  : [%d %s %d]\n'...
    '   Train epoch       : %d\n'...
    '   Learning rate     : %.6f\n'...
    '   Momentum          : %.2f\n'...
    '   Batch size        : %d\n'...
    '   Data Normalization: %s\n'...
    '   Hidden function   : %s\n'...
    '   Output function   : %s\n'...
    '   Pretrain          : %s\n'...
    '   Pretrain epoch    : %d\n'...
    '   Pre-learning rate : %.6f\n'...
    '   Error calculation : %s\n'...
    '   Plotting          : %s\n'...
    '----- DATA INFORMATION -----\n'...
    ],...
    inputNumber,outputNumber,valInputNumber,trainMode,...
    testMode,epochMode,inputUnit,hiddenInfo,outputUnit,...
    fineTrainEpoch,fineLearningRate,momentum,batchSize,...
    normalize,hiddenAct,outputAct,preTrain,preTrainEpoch,...
    preLearningRate,errorMethod,plotOption...
    )
if strcmp(normalize,'on')
    % Data normalization notice
    fprintf(['   *** NOTICE ***\n',...
            '   Your input data has been normalized: scaled between (0 ~ 1).\n\n'])
end

% Inquire for processing it.
while processOption(1) == 0
    ANS = input('\n   Do you want to proceed? (y/n): ','s');
    if sum(strcmp(ANS,{'y','Y'}))
        fprintf('\nData and parameters have been loaded.\nActivating Deep Neural Network...\n')
        %AnsCor = 0;
        break
    elseif sum(strcmp(ANS,{'n','N'}))
        fprintf('   Process interrupted. Please restart the machine.\n\n')
        N_updated = [];
        return
    else
        fprintf('\n   Wrong argument. Argument should be ''y'' or ''n''.');
    end
end
clear ANS
%%
%%%%% DATA ASSIGNMENT %%%%%
tic
fprintf('Initializing weights and biases...\n')
% Value assignment: initial weights and biases.
weightMatrix = N.weight;
biasMatrix = N.bias;
layerError = N.layerError;

% The number of weight matrix for pre-training / last weight matrix is 
% reserved for fine-tuning.
WeightMatrixRange =length(weightMatrix) - 1;
currentVisualLayer = 1:WeightMatrixRange;
currentHiddenLayer = 2:WeightMatrixRange + 1;
weightIndex = 1:hiddenLayerNumber -1;
hiddenIndex = 1:hiddenLayerNumber; 

% Batch Management
% batch size
batchSize = N.batchSize;
batchNumber= ceil(inputNumber/batchSize);

% batch index 
batchIndex = shakeBatch(inputNumber,batchSize,'train');

% Activate epoch mode if validation dataset is not allocated.
if isempty(valInputData) 
    epochMode = 'on';
end
% error and recording
sse_history = [];
val_history = [];
epoch_history = 0;
sse_keep = [];
valBox = [];
pre_valBox = 0;
min_valBox = 10e5;
valCheck = 0;
valHis = 0;
epochVisual = 50;
op = 1;

%%
%%%%% PRETRAINING %%%%%
if strcmp(trainMode,'on')
    if strcmp(preTrain,'on')
        fprintf('Pre-training the model...\nTraining Information:\n')
        fprintf('   Input features: %d, Output features: %d, examples: %d\n',...
            inputUnit,outputUnit,inputNumber)
        % Hidden layers
        for hid = 1:WeightMatrixRange
            fprintf('   %d Hidden layer units: %d\n',...
                hid,layerStructure{1+hid})
            
            % Pre-training epochs
            for epoch = 1:preTrainEpoch
                
                % Assign weight and bias.
                vhMatrix = weightMatrix{hid};
                vBiasMatrix = biasMatrix{currentVisualLayer(hid)};
                hBiasMatrix = biasMatrix{currentHiddenLayer(hid)};
                
                % Training all examples (shuffling the data)
                batchIndex = shakeBatch(inputNumber,batchSize,'train');
                
                for num = 1:length(batchIndex)
                    
                    % Bias replication.
                    batchInputNumber = length(batchIndex{num});
                    batch_hBiasMatrix = repmat(hBiasMatrix,batchInputNumber,1);
                    batch_vBiasMatrix = repmat(vBiasMatrix,batchInputNumber,1);
                    
                    % Assign input.
                    layerForPretrain = inputPattern{hid}(batchIndex{num},:);
                    % First: visual to hidden
                    visual0Array = layerForPretrain;
                    % First: hidden to activation
                    hidden0 = visual0Array * vhMatrix + batch_hBiasMatrix;
                    hidden0Array = BinarySigmoid(momentum,hidden0);
                    
                    % Second: hidden to visual
                    visual1 = vhMatrix * hidden0Array' + batch_vBiasMatrix';
                    visual1Array = BinarySigmoid(momentum,visual1);
                    % Second: visual to activation
                    hidden1 = visual1Array' * vhMatrix + batch_hBiasMatrix;
                    hidden1Array = BinarySigmoid(momentum,hidden1);
                    
                    % update weights and biases
                    vhMatrix = vhMatrix + preLearningRate * (visual0Array' * hidden0Array - visual1Array * hidden1Array);
                    vBiasMatrix = mean(batch_vBiasMatrix + preLearningRate * (visual0Array - visual1Array'));
                    hBiasMatrix = mean(batch_hBiasMatrix + preLearningRate * (hidden0Array - hidden1Array));
                    
                    % input update
                    inputPattern{hid+1}(batchIndex{num},:) = hidden0Array;
                    
                end
                
                % Store weights and biases that have been updated.
                weightMatrix{hid} = vhMatrix;
                biasMatrix{hid} = vBiasMatrix;
                biasMatrix{hid+1} = hBiasMatrix;
                
                fprintf('%d/%d Hidden layer, %d/%d Epoch\n',...
                    hid,WeightMatrixRange,epoch,preTrainEpoch)
            end
            
        end
        fprintf('Pre-training complete.\nPrepare for Fine-tuning.\n\n')
    end

%%
%%%%% FINETUNING %%%%%
    if strcmp(preTrain,'on'); fprintf('Fine-tuning the model...\n')
    else fprintf('Training DNN model...\n'); end

% Feedforward process
    for epoch = 1:fineTrainEpoch
    
        % Training all examples (shuffling the data)
        batchIndex = shakeBatch(inputNumber,batchSize,'train');
        for num = 1:length(batchIndex)
            
            layerActivation{1} = inputPattern{1}(batchIndex{num},:);
            batchInputNumber = length(batchIndex{num});
            
            for hid = 1:hiddenLayerNumber - 2
                
                % Retrieve the trained weights and biases
                weight = weightMatrix{hid};
                bias = repmat(biasMatrix{hid+1},batchInputNumber,1);
            
                % FNN learning
                layerStore = layerActivation{hid} * weight + bias;
                if strcmp(hiddenAct,'sigmoid')
                    layerActive = BinarySigmoid(momentum,layerStore);
                elseif strcmp(hiddenAct,'tanh')
                    layerActive = tanh(layerStore);
                end
                layerMemory{hid} = layerActive;
                layerStorage{hid+1} = layerStore;
                
                layerActivation{hid+1} = layerMemory{hid};
            end
            
            % Last up (hidden to output)
            weight = weightMatrix{end};
            bias = repmat(biasMatrix{end},batchInputNumber,1);
            
            layerStore = layerActivation{hiddenLayerNumber-1} * weight + bias;
            
            % Last output activation selection
            if strcmp(outputAct,'sigmoid')
                layerMemory{hiddenLayerNumber-1} = BinasrySigmoid(layerStore);
                
            elseif strcmp(outputAct,'softmax')
                layerMemory{hiddenLayerNumber-1} = ml_softmax(layerStore);
                
            elseif strcmp(outputAct,'linear')
                layerMemory{hiddenLayerNumber-1} = layerStore;
            end
            
            layerStorage{hiddenLayerNumber} = layerStore;
            layerActivation{hiddenLayerNumber} = layerMemory{end};
            
            % Back propagation: error calculation
            % Last layer
            outputLayerIndex = hiddenLayerNumber - 1;
            layerError{outputLayerIndex} = ...
                outputPattern(batchIndex{num},:) - layerActivation{outputLayerIndex+1};
            
            if strcmp(outputAct,'sigmoid')
                layerError{outputLayerIndex-1} = ...
                    weightMatrix{outputLayerIndex} * layerError{outputLayerIndex}'...
                    .* ((layerActivation{outputLayerIndex+1} .* (1 - layerActivation{outputLayerIndex+1})) * momentum);
                
            elseif sum(strcmp(outputAct,{'linear','softmax'}))
                layerError{outputLayerIndex-1} = layerError{outputLayerIndex} * weightMatrix{outputLayerIndex}';
            end
            
            % Hidden to down layers
            for i = length(layerError) - 1: -1: 1
                if strcmp(hiddenAct,'sigmoid')
                    layerError{i} = layerError{i+1} * weightMatrix{i+1}' ...
                        .* ((layerActivation{i+1} .* (1 - layerActivation{i+1})) * momentum);
                elseif strcmp(hiddenAct,'tanh')
                    layerError{i} = layerError{i+1} * weightMatrix{i+1}' ...
                        .* ((1 - layerActivation{i+1}.^2) * momentum);
                end
            end
                
            % Update weights and biases.
            for i = 1:length(weightMatrix)
                weightMatrix{i} = weightMatrix{i}...
                    + fineLearningRate * layerActivation{i}' * layerError{i};
            end
            for i = 2:length(biasMatrix)
                biasMatrix{i} = biasMatrix{i} + fineLearningRate * mean(layerError{i-1});
            end
            
            % Error calculation
            if strcmp(errorMethod,'MSE')
                % Sum of square errors
                sse = trace(layerError{outputLayerIndex}' * layerError{outputLayerIndex});
            elseif strcmp(errorMethod,'CE')
                % Cross entropy
                sse = -sum(outputPattern(batchIndex{num},:) .* log(layerMemory{end}));
            end
            sse_keep = [sse_keep sse];
                
        end
        
        % Visualize weight matrices.
        if strcmp(plotOption,'on')
            figwm = figure(1);
            for pw = 1:hiddenLayerNumber-1
                subplot(1,hiddenLayerNumber-1,pw)
                visualize(weightMatrix{pw});
                title(sprintf('Epoch: %d',epoch))
            end
            drawnow
        end
       
        % error check and display
        sse_sum = sum(sse_keep)/inputNumber;
        sse_history = [sse_history sse_sum];
        %fprintf('Epoch %d: Error %f\n',epoch,sse_sum)
        sse_keep = [];
        
        % Visualize Error rates: training and validation errors.
        if strcmp(epochMode,'off')
            valMSE = validationError(valInputData,valOutputData,valInputNumber,batchSize,...
                hiddenLayerNumber,hiddenAct,outputAct,...
                momentum,weightMatrix,biasMatrix);
        end
        
        if strcmp(plotOption,'on')
            
            val_history = [val_history valMSE];
            figmse = figure(2);
            plot(1:epoch,sse_history,'o-r',...
                1:epoch,val_history,'o-g')
            xlabel('Epoch Number','fontsize',12)
            ylabel('Error','fontsize',12)
            axis([0 epochVisual 0 max(sse_history(1),val_history(1))])
            title('Error change','fontsize',15)
            legend('Training Error','Validation Error')
            drawnow
            if epochVisual-2 < epoch
                epochVisual = epochVisual + 50;
            end
        end
        
        % Error comparision.
        if strcmp(epochMode,'off')
            valBox = [valBox valMSE];
            if length(valBox) > 5
                valBox(1) = [];
            end
            now_valBox = mean(valBox);
            
            % keep global minimum value.
            if now_valBox < min_valBox;
                min_valBox = now_valBox;
            end
                
            if now_valBox < valMSE
                valCheck = valCheck + 1;
                
            elseif valMSE < min_valBox
                 valCheck = 0;
            end
            
            % Break if valCheck is 10
            if valCheck >= 10
                break
            end
            %fprintf('Epoch %d: Error: %f Validation Error / Check: %f / %d min_value: %f \n',epoch,sse_sum,valMSE,valCheck,min_valBox)
            fprintf('Epoch %d: Error: %f, Validation Error & Check: %f / %d \n',epoch,sse_sum,valMSE,valCheck)
            epoch_history = epoch_history + 1;
        else
            fprintf('Epoch %d: Error %f \n',epoch,sse_sum)
            epoch_history = epoch_history + 1;
        end
        
        
    end

    % Result: weight, bias, layerError, errorhistory.
    N.weight = weightMatrix;
    N.bias = biasMatrix;
    N.layerError = layerError;
    N.errorHistory = sse_history;
    N.epochHistory = epoch_history;
    if strcmp(preTrain,'on');
        fprintf('Fine-tuning complete.\nDNN process has been finisehd.\n\n')
    else fprintf('DNN process has been finisehd.\n\n');
    end

    N_updated = N;

%%%%% TRAINING REPORT %%%%%
    total_time = toc;
    fprintf([...
        '----- Report Summary -----\n\n'...
        'Larning Mode     : %s\n'...
        'Input data       : %d\n'...
        'Total epoch      : %d\n'...
        'PreTrain         : %s\n'...
        'Learning time    : %d seconds\n'...
        'Error calculation: %s\n'...
        'Error rate       : %.2f\n\n'...
        '----- Training Report -----\n\n'...
        ],...
        learningMode,inputNumber,epoch_history,preTrain,...
        round(total_time),errorMethod,sse_sum)
end
%%
%%%%% TESTING %%%%%
if strcmp(testMode,'on')

    fprintf('Testing the trained model...\nTesting Information:\n')
    fprintf('   Input features: %d, Output features: %d, examples: %d\n',...
        inputUnit,outputUnit,inputNumber)
    
    % Retrieve Parameters
    sse_history = N.errorHistory;
    epoch_history = N.epochHistory;
    
    % Testing data with trained weights and biases
    batchIndex = shakeBatch(inputNumber,batchSize,'test');
    for num = 1:batchNumber
        % Assign test input
        testLayerActivation{1} = inputPattern{1}(batchIndex{num},:);
        testBatchInputNumber = length(batchIndex{num});
        
        for hid = 1:hiddenLayerNumber - 2
            % Assign weight and bias
            weight = weightMatrix{hid};
            bias = repmat(biasMatrix{hid+1},testBatchInputNumber,1);
            
            testLayerStore = testLayerActivation{hid} * weight + bias;
            if strcmp(hiddenAct,'sigmoid')
                testLayerActive = BinarySigmoid(momentum,testLayerStore);
            elseif strcmp(hiddenAct,'tanh')
                testLayerActive = tanh(testLayerStore);
            end
            testLayerActive = BinarySigmoid(momentum,testLayerStore);
            testLayerMemory{hid} = testLayerActive;
            testLayerStorage{hid+1} = testLayerStore';
            
            testLayerActivation{hid+1} = testLayerMemory{hid};
        end
        
        % Last up (hidden to output)
        weight = weightMatrix{end};
        bias = repmat(biasMatrix{end},testBatchInputNumber,1);
        
        testLayerStore = testLayerActivation{hiddenLayerNumber-1} * weight + bias;
        
        % Method selection
        if strcmp(outputAct,{'sigmoid'}) % classification
            testLayerMemory{hiddenLayerNumber-1} = BinarySigmoid(testLayerStore);
            
        elseif strcmp(outputAct,{'softmax'}) % classification
            testLayerMemory{hiddenLayerNumber-1} = ml_softmax(testLayerStore);
            
        elseif strcmp(outputAct,'linear') % regression
            testLayerMemory{hiddenLayerNumber-1} = testLayerStore;
        end
        
        testLayerStorage{hiddenLayerNumber} = testLayerStore;
        testLayerActivation{hiddenLayerNumber} = testLayerMemory{end};
        testStackedOutput(batchIndex{num},:) = testLayerActivation{hiddenLayerNumber};
        
        if sum(strcmp(outputAct,{'sigmoid','softmax'}))
            % Find the result index
            max_idx = max(testLayerActivation{hiddenLayerNumber},[],2);
            % obtain each predicted value to compare it with target.
            for check = 1:length(batchIndex{num})
                
                % one to one matching
                pred_result = find(testLayerActivation{hiddenLayerNumber}(check,:) == max_idx(check));
                test_result = find(outputPattern(op,:) == 1);
            
            % Calculating accuracy
                if pred_result == test_result;
                    resultArray(op) = 1;
                    op = op+1;
                else
                    resultArray(op) = 0;
                    op = op+1;
                end
            end
        end
        
    end
    fprintf('Testing complete.\n')
    fprintf('Prepare for result reporting.\n')
    %%
%%%%% RESULT DISPLAY %%%%%
    % Curve-fitting
    if strcmp(outputAct,'linear') 
        % Calculating MSE
        testOutputError = (outputPattern - testStackedOutput).^2;
        resultArray = sum(testOutputError) / 2;
        MSE = floor(sum(resultArray)/inputNumber);
        
        % correlation coefficient r and relation plotting
        fprintf('Correlation coefficient r calculating...\n')
        corval = corrcoef(outputPattern,testStackedOutput);
        Rvalue = corval(1,2);
        dispCor = num2str(Rvalue);
        
        if strcmp(plotOption,'on')
            
            fprintf('Plotting error change...\n')
            % Error trace plotting
            fig1 = figure('position',[320 250 830 450]);
            errorChangeAx = axes('Parent', fig1, 'units','pixels',...
            'position',[60 70 310 320],...
            'fontsize',9,...
            'nextplot','replacechildren');
            plot([1:epoch_history],sse_history(1:epoch_history),'o-k')
            xlabel('Epoch Number','fontsize',12)
            ylabel('Error','fontsize',12)
            title('Error change','fontsize',15)
            
            % Correlation plot
            corrPlotAx = axes('Parent', fig1, 'units','pixels',...
            'position',[450 70 320 320],...
            'fontsize',9,...
            'nextplot','replacechildren');
            plot(outputPattern,testStackedOutput,'ok')
            xlabel('Target Data','fontsize',12)
            ylabel('Output Data','fontsize',12)
            title(sprintf('r = %f',Rvalue),'fontsize',15);
        
        end

        dispAccuracy = '...';
        learnType = 'Curve-fitting (regression)';
    
    % Classification
    elseif sum(strcmp(outputAct,{'sigmoid','softmax'})) 
        
        % Calculating cross entropy
        MCE = sum(-sum(outputPattern .* log(testStackedOutput)))/inputNumber;
        % For displaying the error value.
        MSE = MCE;
        if strcmp(plotOption,'on')
            
            fprintf('Plotting error change...\n')
            % Error trace plotting
            figure('position',[50 280 450 450])
            plot([1:epoch_history],sse_history(1:epoch_history),'o-k')
            xlabel('Epoch Number','fontsize',12)
            ylabel('Error','fontsize',12)
            title('Error change','fontsize',15)
            
            fprintf('Plotting confusion matrix...\n')
            % Plotting confusion maxtrix
            figure
            DNN_conf(outputPattern',testStackedOutput')
            
            fprintf('Plotting ROC...\n')
            % Plotting roc 
            figure
            DNN_roc(outputPattern',testStackedOutput')

        end
        testAccuracy = round((sum(resultArray) / inputNumber)*100,2);
        dispAccuracy = num2str(testAccuracy);
        fprintf('Testing complete.')
        
        dispCor = '...';
        learnType = 'Classification';

    end
    
%%%%% TESTING REPORT %%%%%
    fprintf([...
        'DNN testing process complete.\n'...
        '----- Report Summary -----\n\n'...
        'Learning Mode     : %s\n'...
        'Input data       : %d\n'...
        'Learning type    : %s\n'...
        'Total epoch      : %d\n'...
        'PreTrain         : %s\n'...
        'Error calculation: %s\n'...
        'Error rate       : %.2f\n'...
        'Accuracy         : %s\n'...
        'Correlation r    : %s\n\n'...
        '----- Testing Report -----\n'
        ],...
        learningMode,inputNumber,learnType,epoch_history,preTrain,...
        errorMethod,MSE,dispAccuracy,dispCor)
end

