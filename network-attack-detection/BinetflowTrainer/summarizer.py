import statistics
import numpy as np

class Summarizer:

    def __init__(self):
        self.data = {
            'n_conn': 0,
            'avg_duration': 0,
            'std_duration': 0,
            'mdn_duration': 0,
            'p95_duration': 0,
            'p05_duration': 0,
            'n_udp': 0,
            'n_tcp': 0,
            'n_icmp': 0,
            'n_sports>1024': 0,
            'n_sports<1024': 0,
            'n_dports>1024': 0,
            'n_dports<1024': 0,
            'n_s_a_p_address': 0,
            'n_s_b_p_address': 0,
            'n_s_c_p_address': 0,
            'n_s_na_p_address': 0,
            'n_d_a_p_address': 0,
            'n_d_b_p_address': 0,
            'n_d_c_p_address': 0,
            'n_d_na_p_address': 0,
            'normal_flow_count': 0,
            'background_flow_count': 0
        }
        self.is_attack = 0  # would be 1 if it is an attack, set 0 by default
        self._duration = 0
        self.used = False
        self.dur_list = []
        self.totPkts_list = []
        self.totBytes_list = []
        self.srcBytes_list = []

    def add(self, item):
        self.used = True

        # n_conn
        self.data['n_conn'] += 1

        # 'n_udp, 'n_tcp' and 'n_icmp'
        proto = 'n_%s' % item['proto']
        if proto in self.data:
            self.data[proto] += 1

        # 'avg_duration'
        self._duration += float(item['dur'])
        self.data['avg_duration'] = self._duration / self.data['n_conn']

        # sometimes ports are in a weird format so exclude them for now (Exception: pass)
        # 'n_sports>1024' and 'n_sports<1024'
        try:
            if int(item['sport']) < 1024:
                self.data['n_sports<1024'] += 1
            else:
                self.data['n_sports>1024'] += 1
        except Exception:
            pass

        # sometimes ports are in a weird format so exclude them for now (Exception: pass)
        # 'n_dports>1024' and 'n_dports<1024'
        try:
            if int(item['dport']) < 1024:
                self.data['n_dports<1024'] += 1
            else:
                self.data['n_dports>1024'] += 1
        except Exception:
            pass

        # Label, 'normal_flow_count' and 'background_flow_count'
        if 'Botnet' in item['label']:
            self.is_attack = 1
        elif 'Normal' in item['label']:
            self.data['normal_flow_count'] += 1
        elif 'Background' in item['label']:
            self.data['background_flow_count'] += 1

        # 'n_s_a_p_address','n_d_a_p_address','n_s_b_p_address','n_d_b_p_address','n_s_c_p_address','n_d_c_p_address',
        # 'n_s_na_p_address','n_d_na_p_address'
        self.data['n_s_%s_p_address' % classify(item['srcaddr'])] += 1
        self.data['n_d_%s_p_address' % classify(item['dstaddr'])] += 1

        self.dur_list.append(float(item['dur']))
        if len(self.dur_list) > 2:
            self.data['mdn_duration'] = statistics.median(self.dur_list)
            self.data['std_duration'] = statistics.stdev(self.dur_list)
            self.data['avg_duration'] = statistics.mean(self.dur_list)
            self.data['p95_duration'] = np.percentile(self.dur_list, 95)
            self.data['p05_duration'] = np.percentile(self.dur_list, 5)


def classify(ip):
    parts = ip.split('.')

    try:
        first = int(parts[0])
    except Exception:
        return 'na'

    # TODO: write a better way to classify this.
    if 1 <= first <= 126:
        return 'a'
    elif 128 <= first <= 191:
        return 'b'
    elif 192 <= first <= 223:
        return 'c'
    return 'na'
