# coding=utf-8
import pandas as pd
import os, gc, time

# track execution time
start_time = time.time()

# pickle files have the same names
pkl_path = os.path.join('/opt/data/pkl/')
pkl_directory = os.fsencode(pkl_path)
pkl_files = os.listdir(pkl_directory)
print("### Pkl Directory: ", pkl_directory)
print("### Files: ", pkl_files)

csv_path = os.path.join('/opt/data/csv/')
csv_directory = os.fsencode(csv_path)

for sample_file in pkl_files:
    print('Loading: ', sample_file)
    pkl_file_path = os.path.join(pkl_directory, sample_file).decode('utf-8')
    pkl_df = pd.read_pickle(pkl_file_path)
    print('Loaded: ', pkl_df.shape)

    csv_file_path = os.path.join(csv_directory, sample_file).decode('utf-8')
    print('Saving: ', csv_file_path)
    pkl_df.to_csv(csv_file_path)
    print('Saved!')

    gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))