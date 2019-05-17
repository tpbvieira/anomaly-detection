import pickle
import pandas as pd


def get_dataset(prefix):

    file_list = [open('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110818-2.binetflow',
                      'rb'),
                 open('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110816-2.binetflow',
                      'rb'),
                 open('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110815-2.binetflow',
                      'rb'),
                 open('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110819.binetflow',
                      'rb'),
                 open('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110816.binetflow',
                      'rb'),
                 open('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110815.binetflow',
                      'rb'),
                 open('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110818.binetflow',
                      'rb'),
                 open(
                     '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110811.binetflow',
                     'rb'),
                 open(
                     '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110815-3.binetflow',
                     'rb'),
                 open(
                     '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110817.binetflow',
                     'rb'),
                 open(
                     '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110810.binetflow',
                     'rb'),
                 open(
                     '/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110816-3.binetflow',
                     'rb'),

                 open('/media/thiago/ubuntu/datasets/network/stratosphere_botnet_2011/ctu_13/pkl/capture20110812.binetflow',
                      'rb')]

    # file_list = [open('../data/ctu_13/agg_pkl/%ss_10.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_11.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_12.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_15.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_15-2.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_15-3.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_16.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_16-2.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_16-3.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_17.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_18.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_18-2.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_19.pk1' % prefix, 'rb')]

    # file_list = [open('../data/cicids2017/raw/pkl/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'rb'),
    #              open('../data/cicids2017/raw/pkl/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'rb'),
    #              open('../data/cicids2017/raw/pkl/Friday-WorkingHours-Morning.pcap_ISCX.csv', 'rb'),
    #              # open('../data/cicids2017/raw/pkl/Monday-WorkingHours.pcap_ISCX.csv', 'rb'),
    #              open('../data/cicids2017/raw/pkl/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', 'rb'),
    #              open('../data/cicids2017/raw/pkl/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', 'rb'),
    #              open('../data/cicids2017/raw/pkl/Tuesday-WorkingHours.pcap_ISCX.csv', 'rb'),
    #              open('../data/cicids2017/raw/pkl/Wednesday-workingHours.pcap_ISCX.csv', 'rb')]

    # file_list = [
    #     # open('../data/ctu_13/agg_pkl/%ss_10.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_11.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_12.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_15.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_15-2.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_15-3.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_16.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_16-2.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_16-3.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_17.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_18.pk1' % prefix, 'rb'),
    # #              open('../data/ctu_13/agg_pkl/%ss_19.pk1' % prefix, 'rb'),
    #              open('../data/ctu_13/agg_pkl/%ss_18-2.pk1' % prefix, 'rb')]

    return [pickle.load(file) for file in file_list], file_list


def split_dataset(dataset_list):
    splitted_dataset = {'testing': [],
                        'cross_validation': [],
                        'training': []}
    for dataset in dataset_list:
        df_normal = dataset[dataset['Label'] == 0]
        df_anomaly = dataset[dataset['Label'] == 1]

        num_normal_rows = df_normal.shape[0]
        num_anomaly_rows = df_anomaly.shape[0]

        normal_half = num_normal_rows // 2
        # 50% of the normal traffic for testing
        df_testing = df_normal[:normal_half]
        splitted_dataset['training'].append(df_testing)

        # 25% of the normal + 50% of the anomaly traffic for Cross-Validation
        normal_quarter = normal_half // 2
        df_cv = df_normal[normal_half:normal_half + normal_quarter]
        anomaly_half = num_anomaly_rows // 2
        df_cv = pd.concat([df_cv, df_anomaly[:anomaly_half]])
        splitted_dataset['cross_validation'].append(df_cv)

        # 25% of the normal + 50% of the anomaly traffic for Testing
        df_testing = df_normal[normal_half + normal_quarter:]
        df_testing = pd.concat([df_testing, df_anomaly[anomaly_half:]])
        splitted_dataset['testing'].append(df_testing)

    return splitted_dataset

