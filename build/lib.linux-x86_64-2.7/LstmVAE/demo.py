import sys,os, ipdb

ipdb.set_trace()
sys.path.insert(0, os.getcwd())

from LstmVAE import LSTM_Var_Autoencoder
from LstmVAE import preprocess
import pandas as pd
import numpy as np

nrows, ncols = 1000, 2
data = np.random.random((nrows, ncols))

df = pd.DataFrame(data=data)

preprocess(df)
