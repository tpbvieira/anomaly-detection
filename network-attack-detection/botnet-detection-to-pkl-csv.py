# coding=utf-8
import os, gc, time, warnings, pickle
import pandas as pd
from utils import get_feature_labels, get_feature_order
warnings.filterwarnings("ignore")


start_time = time.time()

pkl_path = os.path.join('/home/thiago/dev/projects/discriminative-sensing/network-attack-detection/BinetflowTrainer-master/saved_data')
pkl_directory = os.fsencode(pkl_path)
file_list = os.listdir(pkl_directory)

# for each file/case
for sample_file in file_list:
    print(sample_file)
    # read pickle file with pandas or...
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')

    if os.path.isfile(pkl_file_path):
        print("## Sample File: ", pkl_file_path)
        with open(pkl_file_path, 'rb') as f:
            summaries = pickle.load(f)
            feature, label = get_feature_labels(summaries)
            df_data = pd.DataFrame(data=feature, columns=get_feature_order())
            df_label = pd.DataFrame(data=label, columns=['Label'])
            df = pd.concat([df_data, df_label], axis=1)

            new_pkl_file_path = os.path.join(pkl_directory, sample_file[6:]).decode('utf-8')
            df.to_pickle(new_pkl_file_path)

            new_csv_file_path = os.path.splitext(new_pkl_file_path)[0] + '.csv'
            df.to_csv(new_csv_file_path, sep=',')

    gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))