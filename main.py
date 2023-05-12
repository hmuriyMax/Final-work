import json
import os
import time

import numpy as np
from matplotlib import use as mpl_use
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open("output.json", "r") as my_file:
        experiments = my_file.read()

    exp = json.loads(experiments)

    neurons = {}
    iters = {}
    prec = []

    # lastNeuron = 0
    # lastIter = 0
    for el in exp["Experiments"]:
        neurons[el["Neurons"]] = True
        iters[el["Iterations"]] = True

    neurons = dict(sorted(neurons.items()))
    iters = dict(sorted(iters.items()))

    itr = 0
    for key in neurons:
        neurons[key] = itr
        itr += 1

    itr = 0
    for key in iters:
        iters[key] = itr
        itr += 1

    prec = [[0 for _ in iters] for _ in neurons]

    for el in exp["Experiments"]:
        prec[neurons[el["Neurons"]]][iters[el["Iterations"]]] = el["Accuracy"]
        # while
        # neurons.append(el["Neurons"])
        # iters.append(el["Iterations"])
        # prec.append(el["Accuracy"])

    plt.figure()
    plt.imshow(prec, cmap='rainbow', aspect='equal', vmin=0, vmax=100, origin="lower")
    plt.colorbar()
    plt.yticks(ticks=np.arange(len(neurons.keys())), labels=neurons.keys())
    plt.xticks(ticks=np.arange(len(iters.keys())), labels=iters.keys(), rotation=90)
    plt.show()
    # time.sleep(10)
    # plt.plot([i + 1 for i in range(14)], [34443, 19503, 13959, 10443, 9400, 8528, 8800, 8400,
    #                                       13839, 10682, 10781, 10155, 10520, 10804])
    # plt.show()