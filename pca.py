import numpy as np
import pylab as plt
import seaborn as sns;

sns.set_theme()

from sklearn.decomposition import PCA
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam

import gzip
import sys
import pickle

# In[basic iris example]
# https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com/229093

# In[setup MINST data]
# You might need to Download data from https://s3.amazonaws.com/img-datasets/mnist.npz
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# Reshape to flat matrix of 60k images 28x28=784 pixels long
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

# In[Run PCA vs AE]
n_comp = 2 #2 or 3 = 2

# PCA (Classic )

mu = x_train.mean(axis=0)
pca = PCA()
pca.fit(x_train)

# X-hat
Zpca = pca.transform(x_train)# [:,:n_comp] #np.dot(pca.transform(x_train)[:,:n_comp], pca.components_[:n_comp,:])
Rpca = np.dot(Zpca[:,:n_comp], pca.components_[:n_comp,:]) + mu # reconstruction

errpca = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with %i PCs: %0.3f'%(n_comp,errpca))
#This outputs: PCA SVD reconstruction error with 2 PCs: 0.056

# PCA (SVD = Single Value Decomposition)

mu = x_train.mean(axis=0)
U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)
Zsvd = np.dot(x_train - mu, V.transpose())

Rsvd = np.dot(Zsvd[:,:2], V[:2,:]) + mu    # reconstruction
errsvd = np.sum((x_train-Rsvd)**2)/Rsvd.shape[0]/Rsvd.shape[1]
print('PCA reconstruction error with %i PCs: %0.3f'%(n_comp,errsvd))
#This outputs: PCA SVD reconstruction error with 2 PCs: 0.056

# Training the autoencoder
# 28x28 pixels = 784 vector >> 512 >> 128 >> 2 << 128 << 512 << 784
# n_comp = 3 #2 or 3

m = Sequential()
m.add(Dense(1024,  activation='elu', input_shape=(784,)))
m.add(Dense(512,  activation='elu'))
m.add(Dense(256,  activation='elu'))
m.add(Dense(n_comp ,    activation='linear', name="bottleneck"))  # was 2
m.add(Dense(256,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(1024,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())

history = m.fit(x_train, x_train, batch_size=128, epochs=5, verbose=1, 
                validation_data=(x_test, x_test))

encoder = Model(m.input, m.get_layer('bottleneck').output)
Zenc = encoder.predict(x_train)  # bottleneck representation
Renc = m.predict(x_train)        # reconstruction

errnlpca = np.sum((x_train-Renc)**2)/Renc.shape[0]/Renc.shape[1]
print('AE reconstruction error with %i PCs: %0.3f'%(n_comp,errnlpca))

# Wts
w = encoder.get_weights()
#wt = encoder.weights #c.f. encoder.summary()

print('Keras Neural Net Model details: ')
m.summary()
