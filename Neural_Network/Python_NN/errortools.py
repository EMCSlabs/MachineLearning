# -*- coding: utf-8 -*-
"""
Errortools for calculating the errors of your networks and visualizing the results.

                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 25
                                                                                   EMCS Labs
"""

'''
1. MSE
2. Pearson R
3. correlationplot
4. confusionplot

'''
import numpy as np
import matplotlib.pyplot as plt

# Calculating Mean Sequare Error
def MSE(target_y,hat_y):

    hat_out = np.concatenate(hat_y,axis=0)
    hat_line = np.concatenate(hat_out,axis=0)
    target_out = target_y[0:len(target_y)]
    target_line = np.concatenate(target_out,axis=0)

    # MSE
    mse = np.sum((target_line - hat_line)**2)/(len(target_y)*2)

    return mse

# Calculating Pearson R correlation coefficients.
def pearsonR(target_y,hat_y):

    hat_out = np.concatenate(hat_y,axis=0)
    hat_line = np.concatenate(hat_out,axis=0)
    target_out = target_y[0:len(target_y)]
    target_line = np.concatenate(target_out,axis=0)

    # Correlation coefficients
    cor_coefficients = np.corrcoef(hat_line,target_line)[0,1]
    return cor_coefficients


# Correlation plot for curvefitting problem (regression)
def correlationplot(target_y,hat_y):

    hat_out = np.concatenate(hat_y,axis=0)
    hat_line = np.concatenate(hat_out,axis=0)
    target_out = target_y[0:len(target_y)]
    target_line = np.concatenate(target_out,axis=0)

    plt.plot(hat_line,target_line,'o')
    plt.xlabel('Predicted data')
    plt.ylabel('Target data')
    plt.title('Correlation between Target and Predicted data')
    limspace = 5
    plt.xlim(hat_line.min()-limspace,hat_line.max()+limspace)
    plt.ylim(target_line.min()-limspace,target_line.max()+limspace)
    plt.show()

# confusion plot for classification problem (needs to be made...)


