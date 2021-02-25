import numpy as np

ind = "example"

filepath = "data\\"+ind
outpath = "data\\"+ind+".csv"
row = 64
col = 64
num = 512
threshold = 0.99

a = np.zeros((10,row,col))
b = np.zeros((10,row,col))

n1 = np.zeros((10))
n2 = np.zeros((10))
n3 = np.zeros((10))
n4 = np.zeros((10))
se = np.zeros((10))
sen = np.zeros((10))

def fun():
    for t in range(10):
        for i in range(row):
            for j in range(col):
                if(a[t][i][j] > 0.1 and b[t][i][j] >= threshold):
                    n1[t] += 1
                elif(a[t][i][j] <= 0.1 and b[t][i][j] >= threshold):
                    n2[t] += 1
                elif (a[t][i][j] > 0.1 and b[t][i][j] < threshold):
                    n3[t] += 1
                else:
                    n4[t] += 1
            if(a[t][i][j] > 0.1):
                se[t] += (a[t][i][j]-b[t][i][j])*(a[t][i][j]-b[t][i][j])
                sen[t] += 1
    for i in range(10):
        if (n1[i] == 0):
            n1[i] = 1
        if (n2[i] == 0):
            n2[i] = 1
        if (n3[i] == 0):
            n3[i] = 1
        if (n4[i] == 0):
            n4[i] = 1
        if (sen[i] == 0):
            sen[i] = 1


def calTS(p):
    return round(n1[p] / (n1[p] + n2[p] + n3[p]), 3)
def calETS(p):
    R = (n1[p]+n2[p])*(n1[p]+n3[p])/(n1[p]+n2[p]+n3[p]+n4[p])
    return round((n1[p] - R) / (n1[p] + n2[p] + n3[p] - R), 3)
def calPOD(p):
    return round(n1[p] / (n1[p] + n3[p]), 3)
def calFAR(p):
    return round(n2[p] / (n1[p] + n2[p]), 3)
def calMAR(p):
    return round(n3[p] / (n1[p] + n3[p]), 3)
def calBS(p):
    return round((n1[p] + n2[p]) / (n1[p] + n3[p]), 3)
def calMSE(p):
    return round(se[p]/sen[p], 3)

with open(filepath,'r') as f:
    data = f.readlines()
    pos = 1
    for i in range(num):
        print(i)
        for t in range(30):
            for j in range(row):
                #print(data[pos])
                tmp = data[pos].split()
                pos += 1
                for k in range(col):
                    fk = float(tmp[k])
                    if(fk<0):
                        fk = 0
                    if(t>=10 and t<20):
                        a[t-10][j][k] = fk
                    elif(t>=20):
                        b[t-20][j][k] = fk
            pos += 1
        fun()

if outpath == "screen":
    for i in range(10):
        print("第"+str(i)+"帧")
        print("TS：" + str(calTS(i)))
        print("ETS：" + str(calETS(i)))
        print("命中率：" + str(calPOD(i)))
        print("虚警率：" + str(calFAR(i)))
        print("漏报率：" + str(calMAR(i)))
        print("偏差：" + str(calBS(i)))
        print("均方误差:" + str(calMSE(i)))
    for i in range(1, 10):
        n1[0] += n1[i]
        n2[0] += n2[i]
        n3[0] += n3[i]
        n4[0] += n4[i]
        se[0] += se[i]
        sen[0] += sen[i]
    print("总的")
    print("TS：" + str(calTS(0)))
    print("ETS：" + str(calETS(0)))
    print("命中率：" + str(calPOD(0)))
    print("虚警率：" + str(calFAR(0)))
    print("漏报率：" + str(calMAR(0)))
    print("偏差：" + str(calBS(0)))
    print("均方误差:" + str(calMSE(0)))
else:
    with open(outpath,"w") as f:
        f.write(".,")
        f.write("TS,")
        f.write("ETS,")
        f.write("命中率,")
        f.write("虚警率,")
        f.write("漏报率,")
        f.write("偏差,")
        f.write("均方误差,")
        f.write("\n")
        for i in range(10):
            f.write(str(i) + ",")
            f.write(str(calTS(i)) + ",")
            f.write(str(calETS(i)) + ",")
            f.write(str(calPOD(i)) + ",")
            f.write(str(calFAR(i)) + ",")
            f.write(str(calMAR(i)) + ",")
            f.write(str(calBS(i)) + ",")
            f.write(str(calMSE(i)) + ",")
            f.write("\n")
        for i in range(1, 10):
            n1[0] += n1[i]
            n2[0] += n2[i]
            n3[0] += n3[i]
            n4[0] += n4[i]
            se[0] += se[i]
            sen[0] += sen[i]
        f.write("all,")
        f.write(str(calTS(0)) + ",")
        f.write(str(calETS(0)) + ",")
        f.write(str(calPOD(0)) + ",")
        f.write(str(calFAR(0)) + ",")
        f.write(str(calMAR(0)) + ",")
        f.write(str(calBS(0)) + ",")
        f.write(str(calMSE(0)) + ",")