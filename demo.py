# import sys, os
# sys.path.insert(0, os.getcwd())
import tensorflow as tf
from LstmVAE import LSTM_Var_Autoencoder
from LstmVAE import preprocess
import pandas as pd
import numpy as np
import ipdb

nrows, ncols = 1000, 2
data = np.random.random((nrows, ncols))

df = pd.DataFrame(data=data)
df = preprocess(df)  #return standardized and normalized df, check NaN values replacing it with 0

timesteps, n_dim = 5, ncols
df = df.reshape(-1,timesteps, n_dim) #use 3D input, n_dim = 1 for 1D time series.

vae = LSTM_Var_Autoencoder(intermediate_dim = 15, z_dim = 3, n_dim= n_dim, stateful = True) #default stateful = False

vae.fit(df, learning_rate=0.001, batch_size = 100, num_epochs = 200, opt = tf.train.AdamOptimizer, REG_LAMBDA = 0.01,
                    grad_clip_norm=10, optimizer_params=None, verbose = True)

"""
   REG_LAMBDA is the L2 loss lambda coefficient, should be set to 0 if not desired.
   optimizer_param : pass a dict = {}
"""

x_reconstructed, recons_error = vae.reconstruct(df, get_error = True) #returns squared error

x_reduced = vae.reduce(df) #latent space representation
ipdb.set_trace()
