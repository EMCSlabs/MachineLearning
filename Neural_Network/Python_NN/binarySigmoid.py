"""
binarySigmoid function
This sigmoid function decays the values by multiplying momentum.

                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 24
                                                                                    EMCS Lab
"""
import numpy as np

def binarySigmoid(momentum,X):
    return 1.0/(1.0 + np.exp(momentum * -X))