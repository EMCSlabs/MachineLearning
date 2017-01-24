# -*- coding: utf-8 -*-
"""
Just note for testing...
Please skip this script.

@author: hyungwonyang
"""
'''
class jeagu():
    
    def respect(self):
        print '존경해요~'


class sunghah(jeagu):
    
    def jodenge(self):
        print 'jojo~'


def nicesum(A):
    B = []
    for i in range(1,len(A)+1):
        B.append(i**i)
    C = sum(B)
    return C


# For checking GPU or CPU usage

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

A_number = [63,234,76,43,23,88,55,46,16,88,245]
number_box = []
number_list = [1,4,5]
for check in number_list:
    number_box.append(A_number[check])



import random
intGen = []
for numbers in range(10):
    intGen.append(random.random())
    print '%d cost: ' % numbers, intGen[numbers]

fileName = 'Obama_Address.txt'
text = open(fileName,'r')
data = text.readlines()
allwords = []

for line in data:
    toss = line.split()
    for putin in toss:
        allwords.append(putin)


class Sum(object):

    def summation(self,x,y):
        z = x+y
        return z

class Div(object):
    def division(self,x,y):
        z = x/y
        return z


class Together(Sum,Div):
    def multiplication(self,x,y):
        z = x*y
        return z

class Numbers():

    def justsum(self,a,b):
        a = 10
        c = a+b
        return c
    def selfsum(self,a,b):
        self.a = 10
        c = self.a+b
        return c


class Walk():
    def setValue(self,personName,oneStep):
	    self.name = personName
	    self.step = oneStep
    def shortwalk(self):
	    print 'my name is %s' % self.name
	    return self.step * 3
    def longwalk(self):
	    print 'my name is %s' % self.name
	    return self.step * 7


class Gretting():
    def __init__(self):
        print 'welcome to python'
        self.good = 'good afternoon'
    def writing(self):
        print self.good


# '/Users/hyungwonyang/Google_Drive/Python/ML_sean'
### playing sound ###
import pyglet
song = pyglet.media.load('thesong.ogg')
song.play()


##
import time, wave, pymedia.audio.sound as sound
f= wave.open( 'YOUR FILE NAME', 'rb' )
sampleRate= f.getframerate()
channels= f.getnchannels()

format= sound.AFMT_S16_LE

snd= sound.Output( sampleRate, channels, format )
s= f.readframes( 300000 )
snd.play( s )

while snd.isPlaying():
    time.sleep( 0.05 )

### playing sound

import pygame

import time

pygame.init()

pygame.mixer.music.load("test.wav")

pygame.mixer.music.play()

time.sleep(10)

### resampling ###
import time, numpy, pygame.mixer, pygame.sndarray
from scikits.samplerate import resample

pygame.mixer.init(44100,-16,2,4096)

# choose a file and make a sound object
sound_file = "tone.wav"
sound = pygame.mixer.Sound(sound_file)

# load the sound into an array
snd_array = pygame.sndarray.array(sound)

# resample. args: (target array, ratio, mode), outputs ratio * target array.
# this outputs a bunch of garbage and I don't know why.
snd_resample = resample(snd_array, 1.5, "sinc_fastest")

# take the resampled array, make it an object and stop playing after 2 seconds.
snd_out = pygame.sndarray.make_sound(snd_resample)
snd_out.play()
time.sleep(2)



import numpy as np
import matplotlib.pyplot as plt

Age = [15, 16, 17, 18, 19, 20]
Tim = [150, 157, 164, 170, 178, 182]
Ninna = [145, 150, 152, 155, 165, 171]

xtotal = np.arange(len(Age))
width = 0.3

bar1 = plt.bar(xtotal, Tim, width, color='b')
bar2 = plt.bar(xtotal+width, Ninna, width, color='r')

plt.ylabel('Height')
plt.xlabel('Age')
plt.title('Height of Time and Ninna by their age')
plt.xticks(xtotal + width, Age)


from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import axes3d

fig = figure()
ax = fig.gca(projection='3d')

# plot points in 3D
class1 = 0.6 * random.standard_normal((200,3))
ax.plot(class1[:,0],class1[:,1],class1[:,2],'o')
class2 = 1.2 * random.standard_normal((200,3)) + array([5,4,0])
ax.plot(class2[:,0],class2[:,1],class2[:,2],'o')
class3 = 0.3 * random.standard_normal((200,3)) + array([0,3,2])
ax.plot(class3[:,0],class3[:,1],class3[:,2],'o')



import numpy as np
import matplotlib.pyplot as plt


N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2 * np.random.rand(N)
colors = theta

ax = plt.subplot(111, projection='polar')
c = plt.scatter(theta, r, c=colors, s=area, cmap=plt.cm.hsv)
c.set_alpha(0.75)

plt.show()

import numpy as np


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self):
        self.lastind = 0

        self.text = ax.text(0.05, 0.95, 'selected: none',
                            transform=ax.transAxes, va='top')
        self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
                                 color='yellow', visible=False)

    def onpress(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p'):
            return
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(xs) - 1)
        self.update()

    def onpick(self, event):

        if event.artist != line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - xs[event.ind], y - ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        ax2.cla()
        ax2.plot(X[dataind])

        ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
                 transform=ax2.transAxes, va='top')
        ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(xs[dataind], ys[dataind])

        self.text.set_text('selected: %d' % dataind)
        fig.canvas.draw()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.random.rand(100, 200)
    xs = np.mean(X, axis=1)
    ys = np.std(X, axis=1)

    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.set_title('click on point to plot time series')
    line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance

    browser = PointBrowser()

    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()

import numpy as np
from binarySigmoid import binarySigmoid
from SetValues import shakeBatch

hiddenNumber=3
inputPattern = inputData
# The number of hidden layers for training
for hidden in range(hiddenNumber):

    # Assign input data, weight and bias
    vhMatrix = weightMatrix[hidden]
    hBiasMatrix = biasMatrix[hidden+1]
    vBiasMatrix = biasMatrix[hidden]
    inputSave = []

    # training the data for given epochs
    for epoch in range(preTrainEpoch):

        # Randomize all the training data.
        layerForPT = inputPattern
        batchIndex = shakeBatch(inputNumber,batchSize,'train')

        # training each data for updating weights and biases(unsupervised learning)
        for num in range(len(batchIndex)-1):

            # Bias replication.
            batchInputNumber = len(batchIndex[num])
            batch_hBiasMatrix = np.tile(hBiasMatrix,[batchInputNumber,1])
            batch_vBiasMatrix = np.tile(vBiasMatrix,[batchInputNumber,1])

            ### visual0
            visual0Array = layerForPT[batchIndex[num]]

            # hidden0
            hidden0 = np.dot(visual0Array,vhMatrix) + batch_hBiasMatrix
            hidden0Array = binarySigmoid(momentum,hidden0)

            ### visual1
            visual1 = np.dot(hidden0Array,vhMatrix.T) + batch_vBiasMatrix
            visual1Array = binarySigmoid(momentum,visual1)

            ### hidden1
            hidden1 = np.dot(visual1Array,vhMatrix) + batch_hBiasMatrix
            hidden1Array = binarySigmoid(momentum,hidden1)

            # update weights and biases
            vhMatrix += prelr * (np.dot(visual0Array.T,hidden0Array) - np.dot(visual1Array.T,hidden1Array))
            vBiasMatrix += np.mean(prelr * (visual0Array - visual1Array),axis=0)
            hBiasMatrix += np.mean(prelr * (hidden0Array - hidden1Array),axis=0)
            if epoch+1 == preTrainEpoch:

                for line in hidden0Array:
                    inputSave.append(line)

        print '{}/{} Hidden Layer, {}/{} epoch'.format(hidden+1,hiddenNumber,epoch+1,preTrainEpoch)

    weightMatrix[hidden] = vhMatrix
    biasMatrix[hidden+1] = hBiasMatrix
    biasMatrix[hidden] = vBiasMatrix
    if epoch+1 == preTrainEpoch:
        inputPattern = np.asarray(inputSave)



        for epoch in range(preTrainEpoch):

                # Randomize all the training data.
                layerForPT = inputPattern
                batchIndex = shakeBatch(inputNumber,batchSize,'train')

                # training each data for updating weights and biases(unsupervised learning)
                for num in range(len(batchIndex)-1):

                    # Bias replication.
                    batchInputNumber = len(batchIndex[num])
                    batch_hBiasMatrix = np.tile(hBiasMatrix,[batchInputNumber,1])
                    batch_vBiasMatrix = np.tile(vBiasMatrix,[batchInputNumber,1])

                    ### visual0
                    visual0Array = layerForPT[batchIndex[num]]

                    # hidden0
                    hidden0 = np.dot(visual0Array,vhMatrix) + batch_hBiasMatrix
                    hidden0Array = binarySigmoid(momentum,hidden0)

                    ### visual1
                    visual1 = np.dot(hidden0Array,vhMatrix.T) + batch_vBiasMatrix
                    visual1Array = binarySigmoid(momentum,visual1)

                    ### hidden1
                    hidden1 = np.dot(visual1Array,vhMatrix) + batch_hBiasMatrix
                    hidden1Array = binarySigmoid(momentum,hidden1)

                    # update weights and biases
                    vhMatrix += prelr * (np.dot(visual0Array.T,hidden0Array) - np.dot(visual1Array.T,hidden1Array))
                    vBiasMatrix += np.mean(prelr * (visual0Array - visual1Array),axis=0)
                    hBiasMatrix += np.mean(prelr * (hidden0Array - hidden1Array),axis=0)
                    if epoch+1 == preTrainEpoch:

                        for line in hidden0Array:
                            inputSave.append(line)

                print '{}/{} Hidden Layer, {}/{} epoch'.format(hidden+1,hiddenNumber,epoch+1,preTrainEpoch)

            weightMatrix[hidden] = vhMatrix
            biasMatrix[hidden+1] = hBiasMatrix
            biasMatrix[hidden] = vBiasMatrix
            if epoch+1 == preTrainEpoch:
                inputPattern = np.asarray(inputSave)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
###
import numpy as np
from scipy.optimize import minimize

X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([3,6,9,12,15,18,21,24,27,30])
W = np.random.rand(2)
equation = lambda X: np.array([X**0,X**1])
optim_equ = lambda W: 1/2.*sum((np.dot(equation(X).transpose(),W) - Y))**2

res = minimize(optim_equ,W,method='CG',options={'disp':True})
optimal_W = res.x
'''
