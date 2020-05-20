import numpy as np 
import pandas as pd 
from scipy import stats
import os


file = os.listdir('./cwru')

for name in file:
    DE_var = []
    FE_var = []
    DE_ppm = []
    FE_ppm = []
    DE_skew = []
    FE_skew = []
    DE_X1 = []
    DE_X2 = []
    DE_X3 = []
    FE_X1 = []
    FE_X2 = []
    FE_X3 = []
    print(name)
    data = pd.read_csv('./cwru/' + name,encoding='gbk')
    L = int(len(data)/1024)
    for i in range(L):
        tmp_DE = data["DE_time"][i*1024:(i+1)*1024]
        tmp_FE = data["FE_time"][i*1024:(i+1)*1024]
        # tmp_BA = data["BA_time"][i*1000:(i+1)*1000]
        DE_var.append(np.var(tmp_DE))
        FE_var.append(np.var(tmp_FE))
        # BA_var.append(np.var(tmp_BA))
        DE_ppm.append(np.max(tmp_DE)-np.min(tmp_DE))
        FE_ppm.append(np.max(tmp_FE)-np.min(tmp_FE))
        # BA_ppm.append(np.max(tmp_BA)-np.min(tmp_BA))
        DE_skew.append(stats.skew(tmp_DE))
        FE_skew.append(stats.skew(tmp_FE))
        # BA_skew.append(stats.skew(tmp_BA))
        d = np.abs(np.fft.fft(tmp_DE,1024))
        f = np.abs(np.fft.fft(tmp_DE,1024))
        DE_X1.append(d[32])
        DE_X2.append(d[64])
        DE_X3.append(d[96])
        FE_X1.append(f[32])
        FE_X2.append(f[64])
        FE_X3.append(f[96])
    df = pd.DataFrame({"DE_var":DE_var,"FE_var":FE_var,"DE_ppm":DE_ppm,"FE_ppm":FE_ppm,
            "DE_skew":DE_skew,"FE_skew":FE_skew,"DE_X1":DE_X1,"FE_X1":FE_X1,"DE_X2":DE_X2,"FE_X2":FE_X2,"DE_X3":DE_X3,"FE_X3":FE_X3})
    df.to_csv('./feature/' + name,index=False)
    #print(name)