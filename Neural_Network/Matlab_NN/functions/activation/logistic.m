%Sigmoid function                                          hyungwon Yang

function[y] = logistic(x)
y = 1./(1 + exp(-x));