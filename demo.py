# import sys, os
# sys.path.insert(0, os.getcwd())
import tensorflow as tf
from LstmVAE import LSTM_Var_Autoencoder
from LstmVAE import preprocess
import pandas as pd
import numpy as np
import ipdb
import matplotlib.pyplot as plt

data_path = "/home/hongminwu/baxter_ws/src/SPAI/smach_based_introspection_framework/introspection_data_folder.AC_offline_test/temp_folder_prediction_for_error_prevention_wrench_norm/anomaly_detection_feature_selection_folder/No.0 filtering scheme/whole_experiment/experiment_at_2018y05m19d15H46M44S/experiment_at_2018y05m19d15H46M44S.csv"

df = pd.read_csv(data_path, header=0, index_col=0)
print (df.shape)
n_obs, n_dim = df.shape[0], df.shape[1]
# n_obs, n_dim = 1000, 2
# data = np.random.random((n_obs, n_dim))

df = preprocess(df)  #return standardized and normalized df, check NaN values replacing it with 0

timesteps = 1
df = df.reshape(-1,timesteps, n_dim) #use 3D input, n_dim = 1 for 1D time series.

'''
    intermediate_dim : LSTM cells dimension.
    z_dim : dimension of latent space.
    n_dim : dimension of input data.
    statefull : if true, keep cell state through batches.
'''
vae = LSTM_Var_Autoencoder(intermediate_dim = 15, z_dim = 3, n_dim= n_dim, stateful = True) #default stateful = False

vae.fit(df, learning_rate=0.001, batch_size = 50, num_epochs = 2000, opt = tf.train.AdamOptimizer, REG_LAMBDA = 0.01,
                    grad_clip_norm=10, optimizer_params=None, verbose = True)

"""
   REG_LAMBDA is the L2 loss lambda coefficient, should be set to 0 if not desired.
   optimizer_param : pass a dict = {}
"""

x_reconstructed, recons_error = vae.reconstruct(df, get_error = True) #returns squared error

x_reduced = vae.reduce(df) #latent space representation

#--------------plot----------------------------
fig, axarr = plt.subplots(nrows = 1, ncols=2)
axarr[0].plot(df.reshape(n_obs, n_dim), c = 'black')
axarr[0].plot(x_reconstructed.reshape(n_obs, n_dim), c = 'blue')
axarr[1].plot(recons_error.reshape(n_obs, n_dim), c='r', label='error')
plt.legend()
plt.show()
