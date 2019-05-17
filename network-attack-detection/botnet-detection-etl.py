# coding=utf-8
import os
import gc
import time
import warnings
import pandas as pd
from botnet_detection_utils import ctu13_data_cleasing, ctu13_raw_column_types
warnings.filterwarnings("ignore")


start_time = time.time()
raw_path = os.path.join('data/ctu_13/raw_fast/')
raw_directory = os.fsencode(raw_path)
file_list = os.listdir(raw_directory)
pkl_path = os.path.join('data/ctu_13/raw_clean_pkl_fast/')
pkl_directory = os.fsencode(pkl_path)

# from raw to raw_clean_pkl
for sample_file in file_list:
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    if os.path.isfile(pkl_file_path):
        print("## File Already Exist: ", pkl_file_path)
    else:
        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
        print("## Raw File: ", raw_file_path)
        raw_df = pd.read_csv(raw_file_path, header=0, dtype=ctu13_raw_column_types)
        clean_df = ctu13_data_cleasing(raw_df)
        clean_df.to_pickle(pkl_file_path)
    gc.collect()
print("--- %s seconds ---" % (time.time() - start_time))
