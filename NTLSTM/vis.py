import os
import cv2
import numpy as np
from math import sqrt
import pdb
import matplotlib.pyplot as plt 

cm = plt.cm.get_cmap('RdYlBu')

for f in os.listdir('/data/zlx/gn_attention/huaweicloud/prediction/30_30/'):
    if not f.endswith('.png'):
        continue
    raw = cv2.imread(os.path.join('/data/zlx/gn_attention/huaweicloud/prediction/30_30/', f),-1)
    pred = raw[:1024,:]
    gt = raw[1024:,:]

    cha = np.array(pred, dtype=np.int16) - np.array(gt, dtype=np.int16)
    #pdb.set_trace()
    plt.imshow(cha, cmap=plt.cm.get_cmap('seismic'))
    plt.title('Year {}'.format(int(f.split('.')[0])+2004))
    plt.colorbar()
    plt.clim(-40, 40)
    plt.savefig('/data/zlx/gn_attention/huaweicloud/prediction/cha_'+f.split('.')[0]+'.png' , bbox_inches='tight', dpi=400,pad_inches=0)
    plt.close()
    
    mse_c = np.sum((pred - gt)**2)/(gt.size)
    rmse_c = sqrt(mse_c)
    r2_c = 1-mse_c/gt.var()
    print('Year{} RMSE: {}, R2:{}'.format(int(f.split('.')[0])+2004, rmse_c, r2_c))
