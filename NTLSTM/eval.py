import torch
from math import sqrt
import torch.nn as nn
import sys
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from model import *
from loss import *
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from dataset import *
from tqdm import tqdm
from tqdm import trange
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from scipy import stats
import pdb
#matplotlib.use('Agg')

dataDir = './'
savemodelDir = './'
bestmodelDir = './min49.pth'

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def save_model(model, step):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, savemodelDir + str(step) + '.pth')


def load_model(modelType, modelDir=bestmodelDir, clsLoss=False):
    if modelType == 'convLSTM':
        ## convLSTM
        encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
        encoder_forecaster = EF(encoder, forecaster).to(device)
    else:
        print('model no exists')
        return

    encoder_forecaster.load_state_dict(
        torch.load(modelDir, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))

    return encoder_forecaster.to(device)


def cal_eval(pred, gt):
    # cal each statisc for each year
    len_year = gt.shape[0]
    #pdb.set_trace()
    assert len_year==8

    rmse = []
    r2 = []
    slope = []
    intercept = []
    r_value = []
    p_value = []
    std_err = []
    for y in range(len_year):
        pred_n = pred[y,:,:]
        gt_n = gt[y,:,:]
        mse_c = mean_squared_error(gt_n, pred_n)
        rmse.append(sqrt(mse_c))
        r2.append(r2_score(gt_n, pred_n))

        regression = stats.linregress(gt_n.reshape(-1,1).squeeze(), pred_n.reshape(-1,1).squeeze())
        #pdb.set_trace()
        slope.append(regression.slope)
        intercept.append(regression.intercept)
        r_value.append(regression.rvalue)
        p_value.append(regression.pvalue)
        std_err.append(regression.stderr)
    return rmse, r2, slope, intercept, r_value, p_value, std_err
        
def draw_eval(data, name):
    data = np.array(data) 
    df=pd.DataFrame(data, columns=['2005','2006','2007','2008','2009','2010','2011','2012'])
    if not os.path.exists('./'+name):
        os.mkdir('./'+name)
    df.to_csv('./'+name+'/'+name+'.csv')
    year_len = data.shape[1]
    x = range(data.shape[0])
    assert year_len==8
    color = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'black']
    plt.figure(facecolor='#FFFFFF', figsize=(50,4))
    for y in range(year_len):
        data_year = data[:,y]
        plt.plot(x, data_year, marker='.', ls='-', label=('Year:'+str(y+2005)), c=color[y], linewidth=1.0, ms=6, mfc=color[y], mec=color[y], mew=3)
    plt.ylabel(name+' of year:2005-2012')
    plt.legend(['2005','2006','2007','2008','2009','2010','2011','2012'])
    plt.savefig('./'+name+'/'+name+'.png')
    plt.close()        

def test(dataloader, model, std):
    with torch.no_grad():
        model.eval()
        f = open("eval.txt", "w") 
        out_all = []
        target_all = []

        rmse = []
        r2 = []
        slope = []
        intercept = []
        r_value = []
        p_value = []
        std_err = []
        
        for names, rawInputs, targets in tqdm(dataloader):
            # input: 5D S*B*I*H*W
            rawInputs = rawInputs.to(device).permute(1, 0, 2, 3, 4).float()
            targets = targets.to(device).permute(1, 0, 2, 3, 4).float()
            #masks = masks.to(device).permute(1, 0, 2, 3, 4).float()
            outputs = model(rawInputs)
                
            output_numpy = np.clip(np.array(outputs.detach().cpu().numpy().squeeze()*std), 0.0, 6300.0)
            out_all.append(output_numpy)
            target_numpy = np.array(targets.detach().cpu().numpy().squeeze()*std)
            target_all.append(np.array(targets.detach().cpu().numpy().squeeze()*std))
            
            prevent = np.array(target_numpy/100.0,dtype=np.uint8)
            #pdb.set_trace()
            judge = len(prevent[prevent>0]) < 0.05*targets.detach().cpu().numpy().size
            if judge: 
                continue
            else:
                #pdb.set_trace()
                rmse_c, r2_c, slope_c, intercept_c, r_value_c, p_value_c, std_err_c = cal_eval(np.array(output_numpy/100.0,dtype=np.uint8), np.array(target_numpy/100.0,dtype=np.uint8))
                #mse_c = np.sum((output_numpy - targets.detach().cpu().numpy().squeeze()*std)**2)/(targets.detach().cpu().numpy().size)
                #rmse_c = sqrt(mse_c)
                rmse.append(rmse_c)
                r2.append(r2_c)
                slope.append(slope_c)
                intercept.append(intercept_c)
                r_value.append(r_value_c)
                p_value.append(p_value_c)
                std_err.append(std_err_c) 
                #r2_c = 1-mse_c/(np.array(targets.detach().cpu().numpy()*std).var())
            #print('data:{},RMSE:{},R2:{}'.format(names, rmse_c, r2_c),file=f)
            #pdb.set_trace()
            #save_gifs(np.concatenate((output_numpy,np.array(targets.detach().cpu().numpy()*std)), axis=3), names)
            #save_imgs(np.concatenate((output_numpy,np.array(targets.detach().cpu().numpy()*std)), axis=3), names)
        #mse_all = np.sum((np.array(out_all) - np.array(target_all))**2)/(np.array(target_all).size)
        #rmse_all = sqrt(mse_all)
        #r2_all = 1-mse_all/(np.array(target_all).var())
        draw_eval(rmse, 'rmse')
        draw_eval(r2, 'r2')
        draw_eval(slope, 'slope')
        draw_eval(intercept, 'intercept')
        draw_eval(r_value, 'r_value')
        draw_eval(p_value, 'p_value')
        draw_eval(std_err, 'std_err')
        print('all evaluation done, RMSE:{},R2:{}'.format(rmse_all, r2_all),file=f)
        print('all evaluation done, RMSE:{},R2:{}'.format(rmse_all, r2_all))


def main_test(modelType, clsLoss=False):
    ###
    Std = 353.269822
    datadir = '/data/zlx/global_ntl/'
    starttime = time.time()
    test_data = ntlDataset(dataDir=datadir,train=False, std=Std)
    print(test_data.__len__())
    print('timeCost is: ' + str(time.time() - starttime))
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0, pin_memory=True)

    model = load_model(modelType, clsLoss=clsLoss)
    test(test_loader, model, Std)


if __name__ == '__main__':
    starttime = time.time()
    #main_train(modelType=cfg.GLOBAL.MODEL_CONVLSTM, clsLoss=True)
    main_test(modelType=cfg.GLOBAL.MODEL_CONVLSTM, clsLoss=True)
    print('timeCost is: ' + str(time.time() - starttime))
