import numpy as np
import os
import torch
import cv2
import pandas as pd
from PIL import Image
from torch import nn
from collections import OrderedDict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

class Statistics:

    @staticmethod
    def cal_rangeNum(image: np.ndarray):
        res = [0] * 6
        rangeList = [[-1, 20], [20, 30], [30, 40], [40, 50], [50, 80], [80, 256]]
        for i in np.nditer(image):
            for j in range(len(rangeList)):
                if i > rangeList[j][0] and i <= rangeList[j][1]:
                    res[j] += 1
                    break
        res = np.array(res)
        return res

    @staticmethod
    def cal_num2prop(rangeNum: np.ndarray):
        res_prop = rangeNum / rangeNum.sum()
        return res_prop


class Pretreatment:
    @staticmethod
    def removeNoise(images: np.ndarray):
        rawimage = images.copy()
        hAndW = rawimage.shape[0]

        images = images.reshape((-1, images.shape[-1]))
        u = np.mean(images, axis=0)
        S = np.matmul((images - u).T, (images - u)) / (images.shape[0] - 1)

        D = np.sqrt(np.sum(np.matmul(images - u, np.linalg.pinv(S)) * (images - u), axis=1))

        D = D.reshape(hAndW, hAndW)
        mask = np.ones_like(D)
        Threshold = np.mean(D) + 3 * np.std(D)
        noise_loc = np.where(D >= Threshold)
        mask[noise_loc] = 0
        res = rawimage[:, :, 0] * mask
        return rawimage[:, :, 0], res, mask

    @staticmethod
    def cleanDataset(dataDir, Thred=0.2):
        def getLeftAndRight(nowIdx, flag):
            def getIdx(idx, l):
                if idx >= 0 and idx < l:
                    return idx
                elif idx < 0:
                    return l - idx
                else:
                    return idx - l

            l = len(flag)
            leftOff = 1
            rightOff = 1
            leftIdx = None
            rightIdx = None
            while leftIdx is None:
                tmpIdx = getIdx(nowIdx - leftOff, l)
                if flag[tmpIdx]:
                    leftIdx = tmpIdx
                else:
                    leftOff += 1
            while rightIdx is None:
                tmpIdx = getIdx(nowIdx + rightOff, l)
                if flag[tmpIdx]:
                    rightIdx = tmpIdx
                else:
                    rightOff += 1
            return leftIdx, rightIdx

        with open('./abnormal_train.txt', 'w') as abTxt:
            # abTxt.write(
            #     'RAD_id, idx, leftIdx, rightIdx, score_Structural_1, score_Structural_2, score_Hist_1, score_Hist_2, score_Mse_1, score_Mse_2 \n')
            abTxt.write('RAD_id,idx\n')
            names = []
            assholes = []
            sonDirs = os.listdir(dataDir)
            for sonDir in sonDirs:
                sonDir = os.path.join(dataDir, sonDir)
                areaDirs = os.listdir(sonDir)
                for areaDir in areaDirs:
                    names.append(areaDir)
                    areaDir = os.path.join(sonDir, areaDir)
                    imgFiles = os.listdir(areaDir)
                    chnlList = []
                    size = []
                    for imgFile in imgFiles:
                        if imgFile.split('.')[-1] != 'png':
                            continue
                        else:
                            imgFile = os.path.join(areaDir, imgFile)
                            img = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
                            chnlList.append(img)
                            size.append(os.path.getsize(imgFile))
                    # H * W * I
                    image = cv2.merge(chnlList)
                    mean_image = image.mean(axis=2).squeeze()
  
                    size = np.array(size)
                    b = 1.0
                    mad = b * np.median(np.abs(size - np.median(size)))
                    lower_limit = np.median(size) - 3 * (mad)
                    upper_limit = np.median(size) + 3 * (mad)
                    flag_head = False
                    flag_tail = False

                    if (size[0] < lower_limit or size[0] > upper_limit) and np.abs(size[0] - size[1]) / size[1] > Thred:
                        image[:, :, 0] = mean_image
                        flag_head = True
                    if (size[-1] < lower_limit or size[-1] > upper_limit) and np.abs(size[-1] - size[-2]) / size[
                        -2] > Thred:
                        image[:, :, -1] = mean_image
                        flag_tail = True

                    abnormal_flag = [True] * image.shape[2]
                    abnormal_index = []
                    for i in range(1, image.shape[2] - 1):
                        # test_image = image[:, :, i]
                        leftIdx, rightIdx = getLeftAndRight(i, abnormal_flag)
                        # left_image = image[:, :, leftIdx]
                        # right_image = image[:, :, rightIdx]

                        proportion_1 = np.abs(size[leftIdx] - size[i]) / size[leftIdx]
                        proportion_2 = np.abs(size[rightIdx] - size[i]) / size[rightIdx]

                        if proportion_1 > Thred and proportion_2 > Thred:
                            abnormal_flag[i] = False
                            if names[-1] not in assholes:
                                assholes.append(names[-1])
                            abnormal_index.append(i)
                            # abTxt.write(str(names[-1]) + ',' + str(i) + ',' + str(leftIdx) + ',' + str(rightIdx) + '\n')
                            print(names[-1], i, leftIdx, rightIdx)
                    if names[-1] in assholes:
                        if flag_head:
                            abnormal_index.append(0)
                        if flag_tail:
                            abnormal_index.append(image.shape[2] - 1)
                        abnormal_index.sort(reverse=False)
                        abnormal_index = [str(k) for k in abnormal_index]
                        abnormal_index_str = ';'.join(abnormal_index)
                        abTxt.write(str(names[-1]) + ',' + abnormal_index_str + '\n')
                        print(len(assholes), len(names))
        print(len(assholes))
        abTxt.close()
        # score_1 = mean_squared_error(test_image, left_image)
        # score_2 = mean_squared_error(test_image, right_image)
        #
        # H1 = cv2.calcHist([image], [i], None, [256], [0, 256])
        # H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 
        #

        # H2 = cv2.calcHist([image], [leftIdx], None, [256], [0, 256])
        # H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
        #
        # H3 = cv2.calcHist([image], [rightIdx], None, [256], [0, 256])
        # H3 = cv2.normalize(H3, H3, 0, 1, cv2.NORM_MINMAX, -1)
        #
        # similarity_1 = cv2.compareHist(H1, H2, 0)
        # similarity_2 = cv2.compareHist(H1, H3, 0)
        #
        # s1 = structural_similarity(test_image, left_image, data_range=80)
        # s2 = structural_similarity(test_image, right_image, data_range=80)
        #
        # if (
        #         s1 < 0.5 and s2 < 0.5 and similarity_1 < 0.999 and similarity_2 < 0.999 and score_1 > 25 and score_2 > 25) or \
        #         (similarity_1 < 0.995 and similarity_2 < 0.995) or \
        #         (score_1 > 40 and score_2 > 40) or \
        #         (s1 < 0.6 and s2 < 0.6 and similarity_1 < 0.998 and
        #          similarity_2 < 0.998 and score_1 > 30 and score_2 > 30):
        #     # if s1 < 0.6 and s2 < 0.6 and similarity_1 < 0.998 and similarity_2 < 0.998 and score_1 > 30 and score_2 > 30:
        #     abnormal_flag[i] = False
        #     if names[-1] not in assholes:
        #         assholes.append(names[-1])
        #         abTxt.write(str(names[-1]) + '\n')
        #     # abTxt.write(
        #     #     str(names[-1]) + ', ' + str(i) + ', ' + str(leftIdx) + ', ' + str(rightIdx) + ', ' + str(
        #     #         s1) + ', ' + str(s2) + ', ' + str(similarity_1) + ', ' + str(similarity_2) + ', ' + str(
        #     #         score_1) + ', ' + str(score_2) + '\n')


class image_clustering:
    def __init__(self, folder_path="data", n_clusters=2, use_pca=True):
        paths = os.listdir(folder_path)
        paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.n_clusters = n_clusters
        self.folder_path = folder_path
        self.image_paths = paths
        self.use_pca = use_pca

    def load_images(self):
        self.images = []
        for image in self.image_paths:
            self.images.append(cv2.resize(cv2.imread(self.folder_path + "/" + image), (128, 128)))
        self.images = np.float32(self.images).reshape(len(self.images), -1)
        # self.images /= 255
        # print("\n " + str(self.max_examples) + " images from the \"" + self.folder_path + "\" folder have been loaded in a random order.")

    def get_new_imagevectors(self):
        model2 = PCA(n_components=2)
        self.images_new = model2.fit_transform(self.images)
        print(self.images.shape)
        # self.images_new = self.images

    def clustering(self):
        model = KMeans(n_clusters=self.n_clusters, random_state=782)
        model.fit(self.images_new)
        predictions = model.predict(self.images_new)
        # predictions = tsne(self.images_new)

        return predictions


class ProbToPixel(object):
    def __init__(self, middle_value, requires_grad=False, NORMAL_LOSS_GLOBAL_SCALE=0.00005):
        '''

        :param middle_value:
        '''
        middle_value = np.array([dBZ_to_pixel(ele) for ele in middle_value])
        if requires_grad:
            self._middle_value = torch.from_numpy(middle_value, requires_grad=True)
        else:
            self._middle_value = torch.from_numpy(middle_value)
        self.requires_grad = requires_grad
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE

    def __call__(self, prediction, ground_truth, mask, lr):
        '''

        :param prediction:
        :return:
        '''
        from config import cfg

        # prediction: S*B*C*H*W
        result = np.argmax(prediction, axis=2)[:, :, np.newaxis, ...]
        prediction_result = np.zeros(result.shape, dtype=np.float32)
        if not self.requires_grad:
            for i in range(len(self._middle_value)):
                prediction_result[result == i] = self._middle_value[i]

        else:
            balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
            weights = torch.ones_like(prediction_result) * balancing_weights[0]
            thresholds = [dBZ_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
            for i, threshold in enumerate(thresholds):
                weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (
                        ground_truth >= threshold).float()
            weights = weights * mask.float()

            loss = torch.zeros(1, requires_grad=True).float()
            for i in range(len(self._middle_value)):
                m = (result == i)
                prediction_result[m] = self._middle_value.data[i]
                tmp = (ground_truth[m] - self._middle_value[i])
                mse = torch.sum(weights[m] * (tmp ** 2), (2, 3, 4))
                mae = torch.sum(weights[m] * (torch.abs(tmp)), (2, 3, 4))
                loss = self.NORMAL_LOSS_GLOBAL_SCALE * (torch.mean(mse) + torch.mean(mae))
            loss.backward()
            self._middle_value -= lr * self._middle_value.grad

        return prediction_result


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            if v[1] != 1:
                gn = nn.GroupNorm(num_channels = v[1] ,num_groups = 4)
                layers.append((layer_name, transposeConv2d))
                layers.append(('gn_'+layer_name, gn))
            else:
                layers.append((layer_name, transposeConv2d))

            #layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            if v[1] != 1:
                gn = nn.GroupNorm(num_channels = v[1] ,num_groups = 4)
                layers.append((layer_name, conv2d))
                layers.append(('gn_'+layer_name, gn))
            else:
                layers.append((layer_name, conv2d))

            #layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


# input: B, C, H, W
# flow: [B, 2, H, W]
def wrap(input, flow):
    from config import cfg
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(cfg.GLOBAL.DEVICE)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(cfg.GLOBAL.DEVICE)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(input, vgrid)
    return output


def pixel_to_dBZ(img):
    """
    Parameters
    ----------
    img : np.ndarray or float
    Returns
    -------
    """
    return np.clip(img * 80.0, a_min=0.0, a_max=255.0)


def dBZ_to_pixel(dBZ_img):
    """
    Parameters
    ----------
    dBZ_img : np.ndarray
    Returns
    -------
    """
    return np.clip((dBZ_img) / 80.0, a_min=0.0, a_max=1.0)


def save_gif(single_seq, fname):
    """Save a single gif consisting of image sequence in single_seq to fname."""
    img_seq = [Image.fromarray(img, 'F').convert("L") for img in single_seq]
    img = img_seq[0]
    img.save(fname + '/all.gif', save_all=True, append_images=img_seq[1:], duration=500)


def save_img(single_seq, fname):
    """Save a single gif consisting of image sequence in single_seq to fname."""
    #img_seq = [Image.fromarray(img, 'F').convert("L") for img in single_seq]
    for i in range(single_seq.shape[0]):
        img = np.array(single_seq[i]/100.0,dtype=np.uint16)
        pred = img[:1024,:]
        gt = img[1024:,:]
    
        cha = np.array(pred, dtype=np.int16) - np.array(gt, dtype=np.int16)
        plt.imshow(cha, cmap=plt.cm.get_cmap('seismic'))
        plt.title('Year {}'.format((i+2005)))
        plt.colorbar()
        plt.clim(-40, 40)
        plt.savefig(fname + '/cha_{}.png'.format((i + 1)), bbox_inches='tight', dpi=400,pad_inches=0)
        plt.close()
        cv2.imwrite(fname + '/{}.png'.format((i + 1)),img)


def save_gifs(seq, fdirs):
    """Save several gifs.
    Args:
      seq: Shape (num_gifs, IMG_SIZE, IMG_SIZE)
      prefix: prefix-idx.gif will be the final filename.
    """
    #seq = seq.cpu().detach().numpy()
    for i in range(len(fdirs)):
        outputDir = '/data/zlx/gn_attention/huaweicloud/prediction/' + str(fdirs[i])

        if os.path.exists(outputDir):
            for root, dirs, files in os.walk(outputDir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(outputDir)
        os.makedirs(outputDir)
        save_gif(seq[:, i].squeeze(), outputDir)


def save_imgs(seq, fdirs):
    root = '/data/zlx/gn_attention/huaweicloud/prediction/'
    for i in range(len(fdirs)):
        outputDir = root + str(fdirs[i])
        save_img(seq[:, i].squeeze(), outputDir)


def read_abnormalFile(fileDir='./', train=True):
    res = {}

    if train:
        fileDir = fileDir + 'abnormal_train.txt'
    else:
        fileDir = fileDir + 'abnormal_test.txt'

    with open(fileDir, 'r') as f:
        for line in f:
            context = line.strip().split(',')
            if context[0] == 'RAD_id':
                continue
            else:
                radID = context[0]
                abnormalIDs = context[1].split(';')
                abnormalIDs = [int(v) for v in abnormalIDs]
                res[radID] = abnormalIDs
    return res


def read_rainsplitFile(fileDir='./', train=True):
    res = {}

    if train:
        fileDir = fileDir + 'trainset_split_by_rain.txt'
    else:
        fileDir = fileDir + 'testset_split_by_rain.txt'

    with open(fileDir, 'r') as f:
        for line in f:
            context = line.strip().split(',')
            weatherType = context[0]
            radID = context[1]
            if weatherType not in res:
                res[weatherType] = [radID]
            else:
                res[weatherType].append(radID)
    return res
