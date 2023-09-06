import torch.nn as nn
import sys
import copy
import time
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
import pdb
import math

dataDir = './'
savemodelDir = './'
bestmodelDir = './min49.pth'

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class SegMetric(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        assert label_trues.shape == label_preds.shape
        self.confusion_matrix += self._fast_hist(label_trues, label_preds, self.n_classes)

    def get_scores(self):

        """
        Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc

        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



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
        if clsLoss:
            encoder_forecaster.forecaster.stage1.conv3_3 = \
                nn.Conv2d(8, len(cfg.HKO.EVALUATION.MIDDLE_VALUE), kernel_size=(1, 1), stride=(1, 1)).to(
                    cfg.GLOBAL.DEVICE)

    else:
        print('model no exists')
        return

    encoder_forecaster.load_state_dict(
        torch.load(modelDir, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))

    return encoder_forecaster.to(device)


def train_and_val(dataloaders, model, optimizer, criterion, lr_scheduler, n_epoch, train_size, val_size, std):
    ### 
    writer = SummaryWriter()
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    f = open("log_{}.txt".format(now), "w") 
    iterNum = 0
    minVal_epoch = 0
    minVal_loss = sys.maxsize
    bestVal_model_dict = None
    for epoch in trange(n_epoch):
        starttime = time.time()
        print('Epoch {}/{}'.format(epoch + 1, n_epoch),file=f)
        learningRate = optimizer.state_dict()['param_groups'][0]['lr']
        print('learning rate: {:.6f}'.format(learningRate),file=f)

        for phase in ['train', 'val']:
            if phase == 'train':
                lr_scheduler.step()
                model.train()
            else:
                model.eval()

            # 
            runningLoss = 0.0
            eval_rmse = 0.0
            eval_r2 = 0.0
            val_size2 = val_size
            #pdb.set_trace()
            for masks, rawInputs, targets in dataloaders[phase]:
                # iterNum += rawInputs.shape[0]
                # print(iterNum)
                # input: 5D S*B*I*H*
                #pdb.set_trace()
                rawInputs = rawInputs.to(device).permute(1, 0, 2, 3, 4).float()
                targets = targets.to(device).permute(1, 0, 2, 3, 4).float()
                #masks = masks.to(device).permute(1, 0, 2, 3, 4).float()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    #pdb.set_trace()
                    outputs = model(rawInputs)
                    loss = criterion(outputs, targets)#, masks.float())
                    #pdb.set_trace()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        dbsize = train_size
                    else:
                        if np.array(targets.detach().cpu().numpy()*std).var()<1e-6:
                            val_size2 = val_size2 - 1
                            rmse_c = 0
                            r2_c = 0
                        else:
                            #pdb.set_trace()
                            out = np.clip(np.array(outputs.detach().cpu().numpy().squeeze()*std),0.0, 6300.0)
                            tar = np.array(targets.detach().cpu().numpy().squeeze()*std)
                            mse_c = np.sum((out - tar)**2)/(targets.detach().cpu().numpy().size)
                            rmse_c = math.sqrt(mse_c)
                            if rmse_c<0:
                                pdb.set_trace()
                            r2_c = 1-mse_c/tar.var()
                        eval_rmse = eval_rmse + rmse_c
                        eval_r2 = eval_r2 + r2_c
                        dbsize = val_size2
                        # HHS = cal_HSS(outputs, targets, focus4Pic=False)
                runningLoss = runningLoss + loss.item() * rawInputs.shape[1]
                #evalScore = evalScore
                #pdb.set_trace()

            epoch_loss = runningLoss / dbsize
            epoch_rmse = eval_rmse / dbsize
            epoch_r2 = eval_r2 / dbsize
            print('{} Loss: {}, best loss{}'.format(phase, epoch_loss, minVal_loss),file=f)
            print('{} Loss: {}, best loss{}'.format(phase, epoch_loss, minVal_loss))
            if phase == 'val':
                print('Epoch{} RMSE: {}, R2:{}'.format(epoch, epoch_rmse, epoch_r2),file=f)
                print('Epoch{} RMSE: {}, R2:{}'.format(epoch, epoch_rmse, epoch_r2))
            if phase == 'train': writer.add_scalar('scalar/trainLoss', epoch_loss, epoch + 1)
            if phase == 'val':
                writer.add_scalar('scalar/valLoss', epoch_loss, epoch + 1)
                if epoch_loss < minVal_loss:
                    minVal_epoch = epoch
                    minVal_loss = epoch_loss
                    bestVal_model_dict = copy.deepcopy(model.state_dict())
                if epoch % 8 == 0:
                    save_model(model, epoch)

        print('timeCost is: ' + str(time.time() - starttime))
        print('-' * 10)
    model.load_state_dict(bestVal_model_dict)
    save_model(model, 'min' + str(minVal_epoch))
    writer.close()


def main_train(modelType, clsLoss=False):
    batch_size = 2
    n_epoch = 50
    learning_rate = 0.001
    datadir = '/home/zlx/NTL/ntl_xian_dataset_512/'

    ### 
    train_data = ntlDataset(dataDir=datadir, train=True)
    val_data = ntlDataset(dataDir=datadir, train=False)  #train_db, val_db = torch.utils.data.random_split(train_data, [traindb_size, valdb_size])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=0, pin_memory=True)
    dataloaders = {'train': train_loader, 'val': val_loader}

    if modelType == 'convLSTM':
        ## convLSTM
        encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
        encoder_forecaster = EF(encoder, forecaster).to(device)
    else:
        print('model no exists')
        return

    ### Loss
    if clsLoss:
        #criterion = torch.nn.MSELoss(reduction='mean').to(device)
        #encoder_forecaster.forecaster.stage1.conv3_3 = \
        #    nn.Conv2d(8, len(cfg.HKO.EVALUATION.MIDDLE_VALUE), kernel_size=(1, 1), stride=(1, 1)).to(cfg.GLOBAL.DEVICE)
        criterion = nn.NLLLoss(weight=np.array([1,1,5,5,1])) #WeightedCrossEntropyLoss(cfg.HKO.EVALUATION.THRESHOLDS, cfg.HKO.EVALUATION.BALANCING_WEIGHTS,LAMBDA=0.1).to(device)
    else:
        criterion = Weighted_mse_mae(LAMBDA=0.1).to(device)
    parameters = list(encoder_forecaster.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.00005)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    encoder_forecaster.load_state_dict(
        torch.load(bestmodelDir, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))

    train_and_val(dataloaders, encoder_forecaster, optimizer, criterion, exp_lr_scheduler, n_epoch,
                  len(train_loader), len(val_loader), Std)


if __name__ == '__main__':
    starttime = time.time()
    main_train(modelType=cfg.GLOBAL.MODEL_CONVLSTM, clsLoss=True)
    #main_test(modelType=cfg.GLOBAL.MODEL_CONVLSTM,weatherType=None, focus4Pic=False, clsLoss=True)
    print('timeCost is: ' + str(time.time() - starttime))
