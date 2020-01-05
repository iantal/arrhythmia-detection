import wfdb

if __name__ == "__main__":

    ##################################
    # records
    ##################################

    # record = wfdb.rdrecord('../data/101', channels=[0, 1])
    """
    {'record_name': '101', 'n_sig': 2, 'fs': 360, 'counter_freq': None, 'base_counter': None, 'sig_len': 650000, 'base_time': None, 'base_date': None, 'comments': ['75 F 1011 654 x1', 'Diapres'], 'sig_name': ['MLII', 'V1'], 'p_signal': array([[-0.345, -0.16 ],
       [-0.345, -0.16 ],
       [-0.345, -0.16 ],
       ...,
       [-0.295, -0.11 ],
       [-0.29 , -0.11 ],
       [ 0.   ,  0.   ]]), 'd_signal': None, 'e_p_signal': None, 'e_d_signal': None, 'file_name': ['101.dat', '101.dat'], 'fmt': ['212', '212'], 'samps_per_frame': [1, 1], 'skew': [None, None], 'byte_offset': [None, None], 'adc_gain': [200.0, 200.0], 'baseline': [1024, 1024], 'units': ['mV', 'mV'], 'adc_res': [11, 11], 'adc_zero': [1024, 1024], 'init_value': [955, 992], 'checksum': [29832, 19589], 'block_size': [0, 0]}

    """
    # record2 = wfdb.rdrecord('../data/102', channels=[0, 1])
    """
    {'record_name': '102', 'n_sig': 2, 'fs': 360, 'counter_freq': None, 'base_counter': None, 'sig_len': 650000, 'base_time': None, 'base_date': None, 'comments': ['84 F 1525 167 x1', 'Digoxin', 'The rhythm is paced with a demand pacemaker.  The PVCs are multiform.'], 'sig_name': ['V5', 'V2'], 'p_signal': array([[-0.2  ,  0.005],
       [-0.2  ,  0.005],
       [-0.2  ,  0.005],
       ...,
       [-0.17 ,  0.2  ],
       [-0.195,  0.195],
       [ 0.   ,  0.   ]]), 'd_signal': None, 'e_p_signal': None, 'e_d_signal': None, 'file_name': ['102.dat', '102.dat'], 'fmt': ['212', '212'], 'samps_per_frame': [1, 1], 'skew': [None, None], 'byte_offset': [None, None], 'adc_gain': [200.0, 200.0], 'baseline': [1024, 1024], 'units': ['mV', 'mV'], 'adc_res': [11, 11], 'adc_zero': [1024, 1024], 'init_value': [984, 1025], 'checksum': [-28574, 13743], 'block_size': [0, 0]}
m
    """
    # record3 = wfdb.rdrecord('../data/104', channels=[0, 1])
    """
    {'record_name': '104', 'n_sig': 2, 'fs': 360, 'counter_freq': None, 'base_counter': None, 'sig_len': 650000, 'base_time': None, 'base_date': None, 'comments': ['66 F 1567 694 x1', 'Digoxin, Pronestyl', 'The rate of paced rhythm is close to that of the underlying sinus rhythm,', 'resulting in many pacemaker fusion beats.  The PVCs are multiform.  Several', 'bursts of muscle noise occur, but the signals are generally of good quality.'], 'sig_name': ['V5', 'V2'], 'p_signal': array([[-0.15 ,  0.2  ],
       [-0.15 ,  0.2  ],
       [-0.15 ,  0.2  ],
       ...,
       [-0.065,  0.2  ],
       [-0.06 ,  0.205],
       [ 0.   ,  0.   ]]), 'd_signal': None, 'e_p_signal': None, 'e_d_signal': None, 'file_name': ['104.dat', '104.dat'], 'fmt': ['212', '212'], 'samps_per_frame': [1, 1], 'skew': [None, None], 'byte_offset': [None, None], 'adc_gain': [200.0, 200.0], 'baseline': [1024, 1024], 'units': ['mV', 'mV'], 'adc_res': [11, 11], 'adc_zero': [1024, 1024], 'init_value': [994, 1064], 'checksum': [-14371, 17572], 'block_size': [0, 0]}
    """
    # print(record.__dict__)
    # print(record2.__dict__)
    # print(record3.__dict__)

    ################################
    # signals, fields, annotations
    ################################
    sig, fields = wfdb.rdsamp('../data/105', channels=[0], sampto=15000)
    ann_ref = wfdb.rdann('../data/105', 'atr', sampto=15000)

    print(sig)
    print(fields)
    print(ann_ref.__dict__)

    """
    [[-0.445]
     [-0.445]
     [-0.445]
     ...
     [-0.07 ]
     [-0.06 ]
     [-0.06 ]]
    {'fs': 360, 'sig_len': 15000, 'n_sig': 1, 'base_date': None, 'base_time': None, 'units': ['mV'], 'sig_name': ['MLII'], 'comments': ['73 F 1624 1629 x1', 'Digoxin, Nitropaste, Pronestyl', 'The PVCs are uniform.  The predominant feature of this tape is', 'high-grade noise and artifact.']}
    {'record_name': '105', 'extension': 'atr', 'sample': array([   67,   197,   459,   708,   965,  1222,  1479,  1741,  2015,
            2287,  2550,  2803,  3052,  3303,  3563,  3835,  4102,  4371,
            4635,  4901,  5154,  5407,  5561,  5923,  6195,  6468,  6736,
            6996,  7246,  7506,  7767,  7922,  8290,  8564,  8829,  9093,
            9350,  9600,  9845, 10105, 10380, 10645, 10908, 11168, 11424,
           11676, 11931, 12187, 12449, 12625, 12987, 13252, 13507, 13755,
           14006, 14253, 14517, 14786]), 'symbol': ['+', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'V', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'V', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'V', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'], 'subtype': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 'chan': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'num': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'aux_note': ['(N\x00', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'fs': 360, 'label_store': None, 'description': None, 'custom_labels': N
    """
