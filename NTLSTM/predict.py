import torch
from math import sqrt
import torch.nn as nn
import sys
import copy
import time
import cv2
from model import *
from dataset import *
from tqdm import tqdm
from tqdm import trange
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from scipy import stats
import pdb
import gdal
#matplotlib.use('Agg')

dataDir = './'
savemodelDir = './'
bestmodelDir = './min49.pth'

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def load_model(modelType, modelDir=bestmodelDir, clsLoss=False):
    if modelType == 'convLSTM':
        ## convLSTM
        encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
        encoder_forecaster = EF(encoder, forecaster).to(device)

    elif modelType == 'trajGRU':
        ## trajGRU
        encoder = Encoder(trajGRU_encoder_params[0], trajGRU_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(trajGRU_forecaster_params[0], trajGRU_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
        encoder_forecaster = EF(encoder, forecaster).to(device)

    elif modelType == 'conv2D':
        ## conv2D
        encoder_forecaster = Predictor(conv2d_params).to(cfg.GLOBAL.DEVICE)
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
              
    
def weight_map(crop_size, stride):
    weight = np.ones((crop_size, crop_size), dtype=np.float32)
    border = crop_size - stride
    for i in range(crop_size):
        weight_i = 1.0
        if i < border:
            weight_i *= (1.0 * (i + 1)/border)
        if i > stride:
            weight_i *= (1.0 * (crop_size - i)/border)
        for j in range(crop_size):
            weight_j = 1.0 * weight_i
            if j < border:
                weight_j *= (1.0 * (j + 1)/border)
            if j > stride:
                weight_j *= (1.0 * (crop_size - j)/border)
            weight[i][j] *= weight_j
    return weight


def main_test(modelType, clsLoss=False):
    ###
    Std = 353.269822
    datadir = '/data/zlx/ntl_dataset/'

    # store the raw data using gdal
    raw_arr = gdal.Open('/home/zlx/NTL/gn_attention/China1992.tif')
    im_geotrans = raw_arr.GetGeoTransform()
    im_proj = raw_arr.GetProjection()
    input_year = range(1992, 2004)
    all_arr = []
    crop_size = 512
    stride = int(0.8 * crop_size)
    crop_weight = weight_map(crop_size, stride)
    
    for i in input_year:
        input_name = '/home/zlx/NTL/gn_attention/Projection/China'+str(i)+'.tif'
        input_arr = cv2.imread(input_name,-1)
        arr = input_arr/Std
        all_arr.append(arr)
    input_ = np.array(all_arr)
    starttime = time.time()
    model = load_model(modelType, clsLoss=clsLoss)
    
    H=input_arr.shape[0]
    W=input_arr.shape[1]
        #imgs = imgs.transpose((1,2,0)).reshape(-1, H, W)
        #imgs = torch.from_numpy(imgs).unsqueeze(0).float()


    h = int(np.ceil(1.0*(H - crop_size)/stride) + 1)
    w = int(np.ceil(1.0*(W - crop_size)/stride) + 1)
    feat = np.zeros((13, H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    
    with torch.no_grad():
        model.eval()
        for crop_h in range(h):
            h_start = min(crop_h * stride, H - crop_size)
            h_end = h_start + crop_size
            for crop_w in range(w):
                w_start = min(crop_w * stride, W - crop_size)
                w_end = w_start + crop_size
                image = input_[:, h_start:h_end, w_start:w_end]
                crop_input = torch.from_numpy(image).unsqueeze(1).unsqueeze(1).to(device).float()
                outputs = model(crop_input)                
                output_numpy = np.clip(np.array(outputs.detach().cpu().numpy().squeeze()*Std), 0.0, 6300.0)
                output_numpy = np.array(output_numpy, dtype=np.float32)
                #pdb.set_trace()
                feat[:,h_start:h_end, w_start:w_end] += (output_numpy * crop_weight)
                weights[h_start:h_end, w_start:w_end] += crop_weight

    #pdb.set_trace()
    feat = feat / weights
    feat = np.array(feat, dtype=np.float32)
    
    for y in range(13):
        im_bands = 1
        im_height = feat.shape[1]
        im_width = feat.shape[2]
        driver = gdal.GetDriverByName("GTiff")
        datatype = gdal.GDT_Float32
        filename = './predict/raw_' + str(y+2013)+'.tif'
        #pdb.set_trace()
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)      
        dataset.GetRasterBand(1).WriteArray(feat[y,:,:])
        del dataset
        
        im_bands = 1
        im_height = feat.shape[1]
        im_width = feat.shape[2]
        compare_name = '/data/zlx/Projection/China'+str(y+2013)+'.tif'
        compare = cv2.imread(compare_name,-1)
        driver = gdal.GetDriverByName("GTiff")
        datatype = gdal.GDT_Float32
        filename = './predict/compare_' + str(y+2013)+'.tif'
        #pdb.set_trace()
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)      
        out = feat[y,:,:] - compare
        dataset.GetRasterBand(1).WriteArray(out)
        del dataset
    

if __name__ == '__main__':
    starttime = time.time()
    #main_train(modelType=cfg.GLOBAL.MODEL_CONVLSTM, clsLoss=True)
    main_test(modelType=cfg.GLOBAL.MODEL_CONVLSTM, clsLoss=True)
    print('timeCost is: ' + str(time.time() - starttime))
