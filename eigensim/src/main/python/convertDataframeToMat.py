import pandas as pd
import numpy, scipy.io

print('### Reading /media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/pkl_all/all.binetflow')
df = pd.read_pickle('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/pkl_all/all.binetflow')

print('### Saving into .csv')
df.to_csv('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/pkl_all/all.csv', sep='\t')

print('### Done!')