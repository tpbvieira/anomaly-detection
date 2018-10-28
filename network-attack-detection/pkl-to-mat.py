# coding=utf-8
import scipy.io as sio
import pandas as pd
import os
import gc
import warnings
import time
warnings.filterwarnings(action='once')

start_time = time.time()

# pickle files - have the same names but different directory
pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/pkl_fast/')
pkl_directory = os.fsencode(pkl_path)
pkl_files = os.listdir(pkl_directory)
print("### Pickle Directory: ", pkl_directory)

# mat files
mat_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere-botnet-2011/ctu-13/mat_fast/')
mat_directory = os.fsencode(mat_path)

# for each file
for sample_file in pkl_files:
    # read pickle file
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    print('### Converting: ',pkl_file_path)
    data_df = pd.read_pickle(pkl_file_path)

    # save mat file
    mat_file_path = os.path.join(mat_directory, sample_file).decode('utf-8')
    data_dict = {'data_np': data_df.values}
    sio.savemat(mat_file_path, data_dict)

    # garbage collector
    gc.collect()

print("--- Execution time: %s seconds ---" % (time.time() - start_time))