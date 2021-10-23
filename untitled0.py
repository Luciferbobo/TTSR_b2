import numpy as np
import torch



def auto_S(S):
    S=S.cpu().detach()
    for i in range(S.size()[0]):             #对每一个batch处理
        l = S[i][0].numpy().reshape(1,1600).tolist()[0]
        data = sorted(l)
        size = len(data)
        if size % 2 == 0:
            median = (data[size//2]+data[size//2-1])/2
        if size % 2 == 1:
            median = data[(size-1)//2]
        #median=np.median(l)
        for j in range(1,S.size()[2]-1):
            for k in range(1,S.size()[3]-1):
                if S[i][0][j][k].item()>median and S[i][0][j-1][k].item()>median and S[i][0][j+1][k].item()>median and S[i][0][j][k-1].item()>median and S[i][0][j][k+1].item()>median:
                   S[i][0][j][k]+=0.1
                   if S[i][0][j][k]>0.95:
                       S[i][0][j][k]=0.95
    return S
     