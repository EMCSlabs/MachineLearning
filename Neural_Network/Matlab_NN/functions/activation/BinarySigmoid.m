% softmax function                                          hyungwon Yang

function [y] = BinarySigmoid(momentum,x)

y = 1./(1 + exp(momentum*-x));