# coding=utf-8
import os, gc, time, warnings
import pandas as pd
from botnet_detection_utils import data_cleasing
warnings.filterwarnings("ignore")


start_time = time.time()

raw_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/raw/')
raw_directory = os.fsencode(raw_path)
file_list = os.listdir(raw_directory)

pkl_path = os.path.join('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/')
pkl_directory = os.fsencode(pkl_path)

# for each file
for sample_file in file_list:

    # read pickle file with pandas or...
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    if os.path.isfile(pkl_file_path):
        print("## PKL File Already Exist: ", pkl_file_path)
    else:  # load raw file and save clean data into pickles
        raw_file_path = os.path.join(raw_directory, sample_file).decode('utf-8')
        print("## Raw File: ", raw_file_path)
        raw_df = pd.read_csv(raw_file_path, header=0, dtype=column_types)
        clean_df = data_cleasing(raw_df)
        clean_df.to_pickle(pkl_file_path)
    gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))