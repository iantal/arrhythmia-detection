"""
The data is provided by
https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.
Two or more cardiologists independently annotated each record; disagreements were resolved to obtain the computer-readable
reference annotations for each beat (approximately 110,000 annotations in all) included with the database.

    Code		Description
    N		Normal beat (displayed as . by the PhysioBank ATM, LightWAVE, pschart, and psfd)
    L		Left bundle branch block beat
    R		Right bundle branch block beat
    B		Bundle branch block beat (unspecified)
    A		Atrial premature beat
    a		Aberrated atrial premature beat
    J		Nodal (junctional) premature beat
    S		Supraventricular premature or ectopic beat (atrial or nodal)
    V		Premature ventricular contraction
    r		R-on-T premature ventricular contraction
    F		Fusion of ventricular and normal beat
    e		Atrial escape beat
    j		Nodal (junctional) escape beat
    n		Supraventricular escape beat (atrial or nodal)
    E		Ventricular escape beat
    /		Paced beat
    f		Fusion of paced and normal beat
    Q		Unclassifiable beat
    ?		Beat not classified during learning
"""

from __future__ import division, print_function

from glob import glob

import deepdish as dd
import numpy as np
from config import get_config
from scipy.signal import find_peaks
from sklearn import preprocessing
from tqdm import tqdm
from utils import add_noise
from wfdb import rdrecord, rdann


def get_records():
    paths = glob('../data/*.atr')
    paths = [path[:-4].rsplit("/", 1)[1] for path in paths]
    paths.sort()
    return paths


class Preprocess(object):
    def __init__(self, split):
        self.split = split
        self.nums = get_records()  # file names without extension
        self.features = ['MLII', 'V1', 'V2', 'V4', 'V5']  # signal names

    def split_data(self):
        if self.split:
            test_set = ['101', '105', '114', '118', '124', '201', '210', '217']
            train_set = [x for x in self.nums if x not in test_set]
            self.data_saver(train_set, '../data/train.hdf5', '../data/trainlabel.hdf5')
            self.data_saver(test_set, '../data/test.hdf5', '../data/testlabel.hdf5')
        else:
            self.data_saver(self.nums, '../data/targetdata.hdf5', '../data/labeldata.hdf5')

    def data_saver(self, dataset, dataset_name, labels_name):
        classes = ['N', 'V', '/', 'A', 'F', '~']
        classes_length = len(classes)  # used for creating masks
        datadict, data_label = dict(), dict()

        """
        {
            "N": [],
            "V": [],
            "/": [],
            "A": [],
            "F": [],
            "~": []
        }
        """
        for feature in self.features:
            datadict[feature] = list()
            data_label[feature] = list()

        self.data_process(dataset, classes, datadict, data_label)
        self.add_noise_to_dataset(classes_length, data_label, datadict)

        dd.io.save(dataset_name, datadict)
        dd.io.save(labels_name, data_label)

    @staticmethod
    def data_process(dataset, classes, datadict, data_label):
        input_size = config.input_size
        for num in tqdm(dataset):
            record = rdrecord('../data/' + num, smooth_frames=True)

            signals_channel_0 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 0])).tolist()
            signals_channel_1 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 1])).tolist()

            peaks, _ = find_peaks(signals_channel_0, distance=150)

            feature0, feature1 = record.sig_name[0], record.sig_name[1]

            # skip the first and last peaks to have enough range of the sample
            for peak in tqdm(peaks[1:-1]):
                start, end = peak - input_size // 2, peak + input_size // 2
                annotation = rdann('../data/' + num, extension='atr', sampfrom=start, sampto=end,
                                   return_label_elements=['symbol'])

                # remove some of "N" which breaks the balance of dataset
                if len(annotation.symbol) == 1 and (annotation.symbol[0] in classes) and (
                        annotation.symbol[0] != "N" or np.random.random() < 0.15):
                    y = [0] * len(classes)
                    y[classes.index(annotation.symbol[0])] = 1
                    data_label[feature0].append(y)
                    data_label[feature1].append(y)
                    datadict[feature0].append(signals_channel_0[start:end])
                    datadict[feature1].append(signals_channel_1[start:end])

    @staticmethod
    def add_noise_to_dataset(classes_length, data_label, datadict):
        noises = add_noise(config)
        for feature in ["MLII", "V1"]:
            d = np.array(datadict[feature])
            if len(d) > 15 * 10 ** 3:
                n = np.array(noises["trainset"])
            else:
                n = np.array(noises["testset"])

            datadict[feature] = np.concatenate((d, n))
            size, _ = n.shape
            l = np.array(data_label[feature])
            noise_label = [0] * classes_length
            noise_label[-1] = 1

            noise_label = np.array([noise_label] * size)
            data_label[feature] = np.concatenate((l, noise_label))


def main(config):
    p = Preprocess(config.split)
    return p.split_data()


if __name__ == "__main__":
    config = get_config()
    main(config)
    # print(get_records())
