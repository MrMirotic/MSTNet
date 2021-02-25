from PIL import Image
import numpy as np
import math


printnum = 512 #要可视化的数据组数
num = 30 #每组数据的图片个数
row = 64 #行数
col = 64 #列数
fs = 1 #可视化放缩比例
a = np.zeros((num, row, col), dtype=int)
dir = "data\\"
fileName = 'example'
outDir = dir+'out\\'

def tran(x):
    x = int(math.pow(x, 0.3333333)*math.pow(255.0, 0.66666667))
    if x>255:
        x = 255
    tmp = np.zeros((3))
    if x<=51:
        tmp[0] = 0
        tmp[1] = x*5
        tmp[2] = 255
    elif x<=102:
        x -= 51
        tmp[0] = 0
        tmp[1] = 255
        tmp[2] = 255 - x*5
    elif x<=153:
        x -= 102
        tmp[0] = x*5
        tmp[1] = 255
        tmp[2] = 0
    elif x<=204:
        x -= 153
        tmp[0] = 255
        tmp[1] = 255-int(128.0 * x/51.0+0.5);
        tmp[2] = 0
    else:
        x -= 204
        tmp[0] = 255
        tmp[1] = 127 - int(127.0 * x / 51.0 + 0.5);
        tmp[2] = 0
    return tmp

def fun():
    fk = np.zeros((num, row*fs, col*fs, 3), dtype=int)
    mx = 0
    for t in range(num):
        for i in range(row):
            for j in range(col):
                if(a[t][i][j]>mx):
                    mx = a[t][i][j]
    for t in range(num):
        print(t)
        for i in range(row*fs):
            for j in range(col*fs):
                fk[t][i][j] = tran(int(a[t][int(i / fs)][int(j / fs)] * 1.0 * 254 / mx))
    return fk

with open(dir+fileName,'r') as f:
    data = f.readlines()
    pos = 1
    for i in range(printnum):
        for t in range(num):
            for j in range(row):
                tmp = data[pos].split()
                pos+=1
                for k in range(col):
                    a[t][j][k] = int(float(tmp[k]))
                    if(a[t][j][k]<0):
                        a[t][j][k] = 0
            pos+=1
        tmpdir = outDir
        print(tmpdir)
        fk = fun()
        for t in range(num):
            p = Image.fromarray(fk[t].astype('uint8'))
            print(tmpdir+str(i)+"_"+str(t//10)+"_"+str(t%10)+".jpg")
            p.save(tmpdir+str(i)+"_"+str(t//10)+"_"+str(t%10)+".jpg")

