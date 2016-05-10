# -*- coding: utf-8 -*-
'''
This script is ANN tutorial.
Please run the 'runDNN' function to activate ANN and try to
understand its structure and usage.

                                                                    Written by Hyungwon Yang
                                                                                2016. 03. 09
                                                                                   EMCS Labs
'''
import loadfile



def runDNN():

    # Import data
    train_in, train_out, test_in, test_out = loadfile.readartmfcc()

