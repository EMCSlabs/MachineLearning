% For randomizing input orders, this function shuffles all the
% batch indices. 
%                                                             Hyungwon Yang
%                                                             2016. 02. 26
%                                                             EMCS labs
                                                            
function shakedBatch = shakeBatch(inputNumber,batchSize,option)

batchNumber= ceil(inputNumber/batchSize);
if strcmp(option,'train')
    % Generate mixed batch indices.
    batchBox = randperm(inputNumber);
    
elseif strcmp(option,'test')
    
    batchBox = 1:inputNumber;
end

start = 0;
final = 0;
for i = 1:batchNumber
    start = final + 1;
    final = final + batchSize;
    if final > inputNumber
        final = final - (final - inputNumber);
    end
    batchIndex{i,1}= batchBox(start:final);
end

% Output
shakedBatch = batchIndex;