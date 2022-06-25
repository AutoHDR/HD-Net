from re import X
import torch
import torch.nn as nn
import numpy as np
# from zmq import device
from models import *
from utils_tools import *
from data_loading import *
from torch.autograd import Variable
import time
from torch.utils.tensorboard import SummaryWriter
import os, tarfile
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.optim.lr_scheduler as PyLR

def TestImg(HD_Model, data_path, step_flag, device, batch_size = 8, num_epochs = 20, log_path = './results/HD-Net/', lr = 0.001, wt_decay=0.005):

    # data processing
    test_csv  = data_path + '/test.csv'
    # Load Synthetic dataset
    test_dataset, _ = get_dataset_OT(read_from_csv=test_csv, validation_split=0)
    test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    with torch.no_grad():
        for bix, data in enumerate(test_dl):
            face_1, face_2, mask, index = data
            b,c,w,h = face_1.shape
        
            mask   = mask.to(device)
            face_1   = face_1.to(device)
            face_2   = face_2.to(device)

            mask = mask.to(device)

            face_1 = face_1 * mask
            face_2 = face_2 * mask

            pA_1, pS_1, pN_1, pL_1, pA_2, pS_2, pN_2, pL_2 = HD_Model(face_1,face_2)

            pS_1 = ImageBatchNormalization(pS_1)*mask
            pS_2 = ImageBatchNormalization(pS_2)*mask
            pA_1 = ImageBatchNormalization(pA_1)*mask
            pA_2 = ImageBatchNormalization(pA_2)*mask

            pN_1 = F.normalize(pN_1) * mask
            pN_2 = F.normalize(pN_2) * mask

            rec_F_1 = pA_1 * pS_1 * mask 
            rec_F_2 = pA_2 * pS_2 * mask

            rec_S_1 = get_shading_DPR_B(pN_1, pL_1, VisLight=False) * mask
            rec_S_2 = get_shading_DPR_B(pN_2, pL_2, VisLight=False) * mask

        
            file_name = './data/results/'
            print(index)
            Spnormal = Sphere_DPR(pN_1, pL_1)
            s_immg1 = torch.cat([face_1*mask, rec_F_1*mask, rec_S_1*pA_1*mask, pA_1*mask, pS_1*mask, rec_S_1*mask, get_shading_DPR_B(Spnormal, pL_1, VisLight=True).expand([b, 3, w,h]), get_normal_in_range(normal_DPR2SFS(pN_1))*mask], dim=0)
            save_image(s_immg1, file_name + '_img1.png', nrow=b, normalize=False)

            s_immg2 = torch.cat([face_2*mask, rec_F_2*mask, rec_S_2*pA_2*mask, pA_2*mask, pS_2*mask, rec_S_2*mask, get_shading_DPR_B(Spnormal, pL_2, VisLight=True).expand([b, 3, w,h]), get_normal_in_range(normal_DPR2SFS(pN_2))*mask], dim=0)
            save_image(s_immg2, file_name + '_img2.png', nrow=b, normalize=False)
            print()




def Smooth_kind(shading):
    [b,c,w,h] = shading.shape
    result = torch.zeros_like(shading)
    epsilon = 0.01*torch.ones(1).to(shading.device)
    for i in range(b):
        tp = shading[i,:,:,:]
        if tp.max() > epsilon:
            result[i,:,:,:] = tp / tp.max()
        else:
            result[i,:,:,:] = tp / epsilon
    return result

def SmoothKind(input):
    [b,c,w,h] = input.shape
    result = torch.zeros_like(input)
    epsilon = 0.01*torch.ones(1).to(input.device)
    tp_max = input.max(dim=1).values.max(dim=1).values.max(dim=1).values.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand([b,c,w,h]) + 0.0000001

    tp_up = torch.where(tp_max > epsilon, input/tp_max, input) + 0.0000001
    tp_low = torch.where(tp_max < epsilon, tp_up/tp_up, tp_up)

    return tp_low

def gradient(x):
    gradient_model = Gradient_Net().to(x.device)
    g = gradient_model(x)
    return g

def normal_normalization(normal_decoder1):
    b, c, w, h = normal_decoder1.shape
    sqrt_sum = torch.sqrt(normal_decoder1[:, 0, :, :] * normal_decoder1[:, 0, :, :] +
                          normal_decoder1[:, 1, :, :] * normal_decoder1[:,1, :, :] +
                          normal_decoder1[:, 2, :, :] * normal_decoder1[:, 2, :, :] + 0.0000001)

    normal_decoder1 = normal_decoder1 / sqrt_sum.reshape([b,1,w,h]).expand(b,c,w,h)
    return normal_decoder1

def Image_Batch_Normalization(albedo):
    [b,c,w,h] = albedo.size()
    result = torch.zeros_like(albedo)
    for i in range(b):
        tp = albedo[i,:,:,:]
        dev = tp.max()- tp.min()
        result[i,:,:,:] = (tp - tp.min())/(dev+0.000001)
    return result

def ImageBatchNormalization(input):
    [b,c,w,h] = input.size()
    tp_max = input.max(dim=1).values.max(dim=1).values.max(dim=1).values.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand([b,c,w,h])
    tp_min = input.min(dim=1).values.min(dim=1).values.min(dim=1).values.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand([b,c,w,h])
    
    tp_data =  (input - tp_min) / (tp_max - tp_min + 0.00001)

    return tp_data



def Sphere_DPR(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal_sp = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).to(normal.device).permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = get_shading_DPR_B_0802(normalBatch, lighting)
    return normalBatch

def normal_DPR2SFS(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 1, :, :]
    tt[:, 1, :, :] = normal[:, 2, :, :]
    tt[:, 2, :, :] = -normal[:, 0, :, :]
    return tt

def normal_SFS2DPR(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = -normal[:, 2, :, :]
    tt[:, 1, :, :] = normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 1, :, :]
    return tt

def normal_DPR2SHTool(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 0, :, :]
    tt[:, 1, :, :] = normal[:, 2, :, :]
    tt[:, 2, :, :] = -normal[:, 1, :, :]
    return tt

def PIL2CV2SHTool(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 2, :, :]
    tt[:, 1, :, :] = normal[:, 0, :, :]
    tt[:, 2, :, :] = -normal[:, 1, :, :]
    return tt



def test(test_dl, HD_Model, testpath, epoch, device):
    sum_a_mse = 0
    sum_a_rmse = 0
    sum_std = 0
    sum_mean = 0
    sum_n20 = 0
    sum_n25 = 0
    sum_n30 = 0
    flag = 0
 
    for bix, data in enumerate(test_dl):
        face_1, face_2, normal, albedo,  mask, index = data
        b, c, w, h = face_1.shape
        mask   = mask.to(device)
        face_1   = face_1.to(device)
        face_2   = face_2.to(device)
        albedo = albedo.to(device)
        mask = mask.to(device)

        normal = normal_SFS2DPR(normal).to(device)


        face_1 = face_1 * mask
        face_2 = face_2 * mask
        normal = F.normalize(normal) * mask

        pA_1, pS_1, pN_1, pL_1, pA_2, pS_2, pN_2, pL_2 = HD_Model(face_1,face_2)


        pS_1 = ImageBatchNormalization(pS_1)*mask
        pS_2 = ImageBatchNormalization(pS_2)*mask

        pA_1 = ImageBatchNormalization(pA_1)*mask
        pA_2 = ImageBatchNormalization(pA_2)*mask

        pN_1 = F.normalize(pN_1) * mask
        pN_2 = F.normalize(pN_2) * mask

        rec_F_1 = pA_1 * pS_1 * mask 
        rec_F_2 = pA_2 * pS_2 * mask

        rec_S_1 = get_shading_DPR_B(pN_1, pL_1) * mask
        rec_S_2 = get_shading_DPR_B(pN_2, pL_2) * mask


        file_name = testpath + str(epoch) + '_'
        Spnormal = Sphere_DPR(pN_1, pL_1)

        s_immg1 = torch.cat([face_1, rec_F_1, rec_S_1*pA_1, pA_1, pS_1, rec_S_1, get_shading_DPR_B(Spnormal, pL_1).expand([b, 3, w,h]), get_normal_in_range(normal_DPR2SFS(pN_1))*mask], dim=0)
        save_image(s_immg1, file_name + '_img1.png', nrow=b, normalize=False)

        s_immg2 = torch.cat([face_2, rec_F_2, rec_S_2*pA_2, pA_2, pS_2, rec_S_2, get_shading_DPR_B(Spnormal, pL_2).expand([b, 3, w,h]), get_normal_in_range(normal_DPR2SFS(pN_2))*mask], dim=0)
        save_image(s_immg2, file_name + '_img2.png', nrow=b, normalize=False)

        std, mean, n20, n25, n30 = Normal_Std_Mean(pN_1, normal, mask)
        a_mae, a_rmse = Test_MAE_RMSE(albedo, pA_1, mask)
        sum_a_mse += a_mae
        sum_a_rmse += a_rmse
        sum_std = sum_std + std
        sum_mean = sum_mean + mean
        sum_n20 += n20
        sum_n25 += n25
        sum_n30 += n30
        flag = flag + 1
        print('testing....{:4}'.format(bix))
        if flag == 200:
            break
    return sum_mean / flag, sum_std / flag, sum_n20 / flag, sum_n25 / flag, sum_n30 / flag, sum_a_mse / flag, sum_a_rmse / flag


def Test_MAE_RMSE(p, g, mask):
    b, c, w, h = p.shape
    device = p.device
    mask = torch.where(mask > 0.8, torch.ones(1).to(device), torch.zeros(1).to(device))

    mae = torch.sum(torch.abs(p * mask - g * mask)) / torch.sum(mask)
    rmse = torch.sqrt(torch.sum((p * mask - g * mask) ** 2) / torch.sum(mask))
    # similar = cos(p * mask, g * mask)
    # Ones = torch.ones([b, 1, w, h]).to(device) * mask
    # mean = torch.sum(torch.abs(Ones-similar))/torch.sum(mask)
    return mae, rmse  # , mean


def Normal_Std_Mean(normal, gtnormal, mask):
    pi = 3.1415926
    device = normal.device
    b, c, w, h = normal.shape
    mask = torch.where(mask > 0.8, torch.ones(1).to(device), torch.zeros(1).to(device))
    normal = normal * mask
    gtnormal = gtnormal * mask
    x1 = normal[:, 0, :, :]
    y1 = normal[:, 1, :, :]
    z1 = normal[:, 2, :, :]

    gx1 = gtnormal[:, 0, :, :]
    gy1 = gtnormal[:, 1, :, :]
    gz1 = gtnormal[:, 2, :, :]

    up = (x1 * gx1 + y1 * gy1 + z1 * gz1)
    low = torch.sqrt(x1 * x1 + y1 * y1 + z1 * z1) * torch.sqrt(gx1 * gx1 + gy1 * gy1 + gz1 * gz1)
    mask = torch.where(mask > 0.8, torch.ones(1).to(device), torch.zeros(1).to(device))
    pixel_number = torch.sum(mask[:, 0, :, :])

    normal_angle_pi = (up / (low + 0.000001))
    # cos_angle_err = torch.where(normal_angle_pi>0.98, torch.ones(1).to(device), torch.zeros(1).to(device))
    cos_angle_err = normal_angle_pi

    angle_err_hudu = torch.acos(cos_angle_err)
    angle_err_du = (angle_err_hudu / pi * 180) * mask[:, 0, :, :]

    mean = torch.sum((angle_err_du)) / pixel_number
    std = torch.sqrt(torch.sum((angle_err_du - mean) * (angle_err_du - mean) * mask[:, 0, :, :]) / pixel_number)

    ang20 = torch.ones(1).to(device) * 20
    ang25 = torch.ones(1).to(device) * 25
    ang30 = torch.ones(1).to(device) * 30
    ang0 = torch.ones(1).to(device) * 0
    ang1 = torch.ones(1).to(device)
    count_20 = torch.sum(torch.where(angle_err_du < ang20, ang1, ang0) * mask[:, 0, :, :]) / pixel_number
    count_25 = torch.sum(torch.where(angle_err_du < ang25, ang1, ang0) * mask[:, 0, :, :]) / pixel_number
    count_30 = torch.sum(torch.where(angle_err_du < ang30, ang1, ang0) * mask[:, 0, :, :]) / pixel_number

    return std, mean, count_20, count_25, count_30

