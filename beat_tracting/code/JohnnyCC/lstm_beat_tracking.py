
# coding: utf-8

# In[1]:


#%matplotlib inline

#import librosa
#import librosa.display
#import IPython.display as ipd
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

#TRAINING_PATH = 'dataset/mirex_beat_tracking_2016/train/'


# In[2]:


BATCH_SIZE = 431
INPUT_SIZE = 2049
BATCH_START= 0
TIME_STEPS = 1 
OUTPUT_SIZE= 1
CELL_SIZE = 100


# In[3]:


trainX = np.load('train1_x.npy')
trainY = np.load('train1_y.npy')


# In[4]:


trainX = np.tile(trainX,40).reshape(BATCH_SIZE*40,INPUT_SIZE)


# In[5]:


print('trainX.shape: '+ str(trainX.shape))
print('trainY.shape: '+ str(trainY.shape))


# In[6]:



model = Sequential()
# build a LSTM RNN
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    return_sequences=True,      # True: output at all steps. False: output as last step.
    stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[7]:


def get_batch():
    global BATCH_START
    # xs shape (50batch, 20steps)
    batch_x_ret = trainX[BATCH_START:BATCH_START+BATCH_SIZE,:].reshape(BATCH_SIZE,1,INPUT_SIZE)
    batch_y_ret = trainY[0,BATCH_START:BATCH_START+BATCH_SIZE].reshape(BATCH_SIZE,1,OUTPUT_SIZE)
    
    BATCH_START += BATCH_SIZE
    if BATCH_START>=trainX.shape[0]: 
        BATCH_START = 0
        
    return [batch_x_ret,batch_y_ret]


# In[ ]:


print('Training ------------')
for step in range(501):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    print('step: '+str(step)+'\tcost: '+str(cost))
    #plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    #plt.ylim((-1.2, 1.2))
    #plt.draw()
    #plt.pause(0.1)
    #if step % 10 == 0:
    #    print('train cost: ', cost)


# In[ ]:




