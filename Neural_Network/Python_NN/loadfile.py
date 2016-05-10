# -*- coding: utf-8 -*-
'''
loadfile.py is written for importing data conveniently.
It supports mat file data as follows.

- Curvefitting
	1. bodyData.mat
	2. buildingData.mat
	3. art_mfcc.mat

- Classification
	1. MNIST
	2. cancerData.mat

																	Written by Hyungwon Yang
																				2016. 02. 10
																					EMCS Lab
'''

import scipy.io as sio
import numpy as np
import os

data_path = './train_data'

################################################## CurveFitting Data ##################################################

# Import bodyfatData
def readbody():

	source_path = os.path.join(data_path,'bodyData.mat')
	data = sio.loadmat(source_path)

	bodyfatInputs = data['bodyfatInputs'].transpose()
	bodyfatTargets = data['bodyfatTargets'].transpose()

	train_out = bodyfatTargets[0:200]
	train_in = bodyfatInputs[0:200]
	test_out = bodyfatTargets[200:253]
	test_in = bodyfatInputs[200:253]

	print('Data Information')
	print('Train data: 200 examples, Test data: 52 examples\nInputs: 13 features, Outputs: 1 feature\n')
	return train_in, train_out, test_in, test_out

# Import buildingData
def readbuilding():

	source_path = os.path.join(data_path,'buildingData.mat')
	data = sio.loadmat(source_path)
	buildingInputs = data['buildingInputs'].transpose()
	buildingTargets = data['buildingTargets'].transpose()

	train_out = buildingTargets[0:3400]
	train_in = buildingInputs[0:3400]
	test_out = buildingTargets[3400:4209]
	test_in = buildingTargets[3400:4209]

	print('Data Information')
	print('Train data: 3,400 examples, Test data: 808 examples\nInputs: 14 features, Outputs: 3 features\n')
	return train_in, train_out, test_in, test_out

# Import art_mfcc Data

def readartmfcc():

	source_path = os.path.join(data_path,'art_mfcc.mat')
	data = sio.loadmat(source_path)
	art_V = data['art_V'].transpose()
	mfcc_V = data['mfcc_V'].transpose()

	train_out = art_V[0:12000]
	train_in = mfcc_V[0:12000]
	test_out = art_V[12000:16001]
	test_in = mfcc_V[12000:16001]

	print('Data Information')
	print('Train data: 12,000 examples, Test data: 4,000 examples\nInputs: 39 features, Outputs: 16 features(x&y pairs, total 8 pallet locations)\n')
	return train_in, train_out, test_in, test_out


################################################# Classification Data #################################################

# Import MNIST
def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def readmnist(ntrain=60000,ntest=10000,onehot=True):
	fd = open(os.path.join(data_path,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_path,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_path,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_path,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	train_x = trX[:ntrain]
	trY = trY[:ntrain]

	test_x = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		train_y = one_hot(trY, 10)
		test_y = one_hot(teY, 10)
	else:
		train_y = np.asarray(trY)
		test_y = np.asarray(teY)

	print('Data Information')
	print('Train data:60,000 examples, Test data: 10,000 examples\nInputs: 784 features, Outputs: 10 classification(0-9 digits)\n')
	return train_x, train_y, test_x, test_y

# Import cancerData
def readcancer():

	source_path = os.path.join(data_path,'cancerData.mat')
	data = sio.loadmat(source_path)
	inputs = data['cancerInputs']
	outputs = data['cancerTargets']

	inputs = np.float64(inputs)
	outputs = np.float64(outputs)

	train_in = inputs[:,0:550].transpose()
	train_out = outputs[:,0:550].transpose()
	test_in = inputs[:,550:701].transpose()
	test_out = outputs[:,550:701].transpose()

	print('Data Information')
	print('Train data: 550 examples, Test data: 150 examples\nInputs: 9 features, Outputs: 2 classification(cancer or not)\n')
	return train_in, train_out, test_in, test_out
