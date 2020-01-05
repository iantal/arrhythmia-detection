"""
The CINC data is provided by https://physionet.org/challenge/2017/
"""
from __future__ import division, print_function

import csv
import os

import numpy as np
from config import get_config
from utils import *
from scipy.io import loadmat
from keras.models import load_model

from utils import mkdir_recursive, uploadedData, preprocess


def cincData(config):
    __download_cinc_data(config)
    num = config.num
    testlabel = []
    __read_data(testlabel)

    if num is None:
        high = len(testlabel) - 1
        num = np.random.randint(1, high)
    filename, label = testlabel[num - 1]
    filename = 'training2017/' + filename + '.mat'

    data = loadmat(filename)
    print("The record of " + filename)
    if not config.upload:
        data = data['val']
        _, size = data.shape
        data = data.reshape(size, )
    else:
        data = np.array(data)
    return data, label


def __download_cinc_data(config):
    if config.cinc_download:
        cmd = "curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
        os.system(cmd)
        os.system("unzip training2017.zip")


def __read_data(testlabel):
    with open('training2017/REFERENCE.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            testlabel.append([row[0], row[1]])
            line_count += 1
        print(f'Processed {line_count} lines.')


def predict(data, label, peaks):
    classesM = ['N', 'Ventricular', 'Paced', 'A', 'F', 'Noise']
    predicted, result = predictByPart(data, peaks)
    sumPredict = sum(predicted[x][1] for x in range(len(predicted)))
    avgPredict = sumPredict / len(predicted)
    print("The average of the predict is:", avgPredict)
    print("The most predicted label is {} with {:3.1f}% certainty".format(classesM[avgPredict.argmax()],
                                                                          100 * max(avgPredict[0])))
    sec_idx = avgPredict.argsort()[0][-2]
    print("The second predicted label is {} with {:3.1f}% certainty".format(classesM[sec_idx],
                                                                            100 * avgPredict[0][sec_idx]))
    print("The original label of the record is " + label)
    print("Result:")
    print(result)


def predictByPart(data, peaks):
    classesM = ['N', 'Ventricular', 'Paced', 'A', 'F', 'Noise']
    predicted = list()
    result = ""
    counter = [0] * len(classesM)

    model = load_model('models/MLII-latest.hdf5')
    config = get_config()
    for i, peak in enumerate(peaks[3:-1]):
        total_n = len(peaks)
        start, end = peak - config.input_size // 2, peak + config.input_size // 2
        prob = model.predict(data[:, start:end])
        prob = prob[:, 0]
        ann = np.argmax(prob)
        counter[ann] += 1
        if classesM[ann] != "N":
            print("The {}/{}-record classified as {} with {:3.1f}% certainty".format(i, total_n, classesM[ann],
                                                                                     100 * prob[0, ann]))
        result += "(" + classesM[ann] + ":" + str(round(100 * prob[0, ann], 1)) + "%)"
        predicted.append([classesM[ann], prob])
        if classesM[ann] != 'N' and prob[0, ann] > 0.95:
            import matplotlib.pyplot as plt
            plt.plot(data[:, start:end][0, :, 0], )
            mkdir_recursive('results')
            plt.savefig('results/hazard-' + classesM[ann] + '.png', format="png", dpi=300)
            plt.close()
    result += "{}-N, {}-Venticular, {}-Paced, {}-A, {}-F, {}-Noise".format(counter[0], counter[1], counter[2],
                                                                           counter[3], counter[4], counter[5])
    return predicted, result


if __name__ == '__main__':
    config = get_config()
    data, label = cincData(config)
    data, peaks = preprocess(data, config)
    predict(data, label, peaks)

