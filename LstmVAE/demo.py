# import sys, os
# sys.path.insert(0, os.getcwd())

from LstmVAE import LSTM_Var_Autoencoder
from LstmVAE import preprocess
import pandas as pd
import numpy as np
import ipdb

nrows, ncols = 1000, 2
data = np.random.random((nrows, ncols))

df = pd.DataFrame(data=data)
df = preprocess(df)
ipdb.set_trace()
