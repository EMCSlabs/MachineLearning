% softmax function                                          hyungwon Yang

function y = BinarySigmoidPrime(momentum,x)

y = momentum * BinarySigmoid(momentum,x) .* (1-BinarySigmoid(momentum,x));