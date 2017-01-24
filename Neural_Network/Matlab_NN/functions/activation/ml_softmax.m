% softmax function                                          hyungwon Yang

function values = ml_softmax(inputs)

    numerator = exp(bsxfun(@minus,inputs,max(inputs,[],2)));
    denominator = sum(numerator,2);
    values = bsxfun(@rdivide, numerator, denominator);

end
