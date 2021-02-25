def prepare(startind, filename, sf, tmp, trainlx, trainly, trainrx, trainry):
    with open(filename, 'r') as f:
        data = f.readlines()
        pos = startind * 27301#==================
        for i in range(trainnum):
            pos = pos + 1
            for j in range(20):
                for k1 in range(row):
                    tmpstr = data[pos].split()
                    for k2 in range(col):
                        tmp[i][j][0][k1][k2] = float(tmpstr[k2])
                    pos = pos + 1
                pos = pos + 1
                for k0 in range(9):
                    pos = pos + 65
                for k0 in range(height):
                    for k1 in range(row):
                        tmpstr = data[pos].split()
                        for k2 in range(col):
                            tmp[i][j][k0+1][k1][k2] = float(tmpstr[k2])
                        pos = pos + 1
                    pos = pos + 1
                for k0 in range(11-height):
                    pos = pos + 65
    if sf:
        np.random.shuffle(tmp)
    for i in range(trainnum):
        for j in range(10):
            for k0 in range(height+1):
                for k1 in range(row):
                    for k2 in range(col):
                        if(k0>0):
                            trainrx[i // batch_size][j][i % batch_size][0][k0-1][k1][k2] = tmp[i][j][k0][k1][k2]
                            trainry[i // batch_size][j][i % batch_size][0][k0-1][k1][k2] = tmp[i][j+10][k0][k1][k2]
                        else:
                            trainlx[i // batch_size][j][i % batch_size][0][k1][k2] = tmp[i][j][k0][k1][k2]
                            trainly[i // batch_size][j][i % batch_size][0][k1][k2] = tmp[i][j + 10][k0][k1][k2]