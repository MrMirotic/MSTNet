import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import os
from matplotlib import cm


path = "data\\"
outpath = path #output directory

if __name__ == "__main__":
    draw_comparision = True
    draw_data_sources = False

    if draw_comparision:
        name_list = [
            'MSTNet',
            'MSTNet-S',
            'MSTNet-R',
            'MSTNet-L',
            'ConvLSTM',
            'PredCNN',
            'StepDeep',
            'PredRNN',
            'LightningNet'
        ]
        score_list = ['TS', 'ETS', 'PMSE']
        marker_list = ['o', 's', 'p', 'P', 'X', 'H', '*', '^', 'D']
        myColors = cm.Set1(range(9))
        choice = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        val_arr = np.zeros(shape=(len(name_list), len(score_list), 10), dtype=float)

        with open(path+'result.txt', 'r') as f:
            data = f.readlines()
            pos = 0
            for i in range(len(name_list)):
                for j in range(10):
                    tmp = data[pos].split('\t')
                    pos+=1
                    for k in range(len(score_list)):
                        val_arr[i][k][j] = float(tmp[k])

        print(val_arr[0])
        print(val_arr[1])
        scale = 1.6
        x = np.arange(1, 11, 1)
        for j in range(len(score_list)):
            fig = plt.figure(figsize=(8 * scale, 6 * scale))
            for i in range(len(name_list)):
                if i== 8 and j == 2:
                    continue
                plt.plot(x, val_arr[i, j, :], marker=marker_list[i], markersize=10, fillstyle='none', ls='--', lw=3,
                         color=myColors[choice[i]])

            plt.title(score_list[j], fontsize=50)
            plt.legend(name_list, loc='best', fontsize=25)
            plt.xlabel('frame',fontdict={'size': 30})
            plt.tick_params(labelsize=25)
            plt.savefig(outpath + 'comparision_' + score_list[j] + '.png')
            plt.close()