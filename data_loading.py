import shutil
from pyparsing import lineStart
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from utils_tools import *
import glob
from random import randint
import random
import os
from PIL import Image
import pandas as pd
from random import shuffle
from utils_tools import save_image, denorm, get_normal_in_range
import numpy as np
from torchvision.utils import save_image

IMAGE_SIZE = 256

def generate_data_crop_face_csv(dir):

    floder = dir + '*'
    floderlist = glob.glob(floder)
    floderlist.sort()
    face_list_1 = []
    face_list_2 = []
    albedo_list = []
    normal_list = []
    mask_list = []
    for i in range(len(floderlist)):
        subfloder = floderlist[i] +'/*'
        sublist = glob.glob(subfloder)
        sublist.sort()
        for j in range(len(sublist)):
            dofloder = sublist[j]
            im0 = dofloder + '/im0.jpg'
            im1 = dofloder + '/im1.jpg'
            im2 = dofloder + '/im2.jpg'
            im3 = dofloder + '/im3.jpg'
            mask = dofloder + '/_mask.png'
            albedo = dofloder + '/albedo.jpg'
            normal = dofloder + '/normal.jpg'
            face_list_1.append(im0)
            face_list_2.append(im1)
            albedo_list.append(albedo)
            normal_list.append(normal)
            mask_list.append(mask)

            face_list_1.append(im0)
            face_list_2.append(im2)
            albedo_list.append(albedo)
            normal_list.append(normal)
            mask_list.append(mask)

            face_list_1.append(im0)
            face_list_2.append(im3)
            albedo_list.append(albedo)
            normal_list.append(normal)
            mask_list.append(mask)

            face_list_1.append(im1)
            face_list_2.append(im2)
            albedo_list.append(albedo)
            normal_list.append(normal)
            mask_list.append(mask)

            face_list_1.append(im1)
            face_list_2.append(im3)
            albedo_list.append(albedo)
            normal_list.append(normal)
            mask_list.append(mask)

            face_list_1.append(im2)
            face_list_2.append(im3)
            albedo_list.append(albedo)
            normal_list.append(normal)
            mask_list.append(mask)

            # print('')
    data_len = len(face_list_1)

    print('face1 len is '+str(len(face_list_1)) + ' face2 '+ str(len(face_list_2))+
          ' albedo '+ str(len(albedo_list))+ ' mask '+ str(len(mask_list)))
    if len(face_list_1)!=len(albedo_list):
        print('the dataset has some problem!')
    train_flag = int(len(face_list_1) - len(face_list_1)*0.2)
    face_list_1_train = face_list_1[0:train_flag:1]
    face_list_2_train = face_list_2[0:train_flag:1]
    albedo_list_train = albedo_list[0:train_flag:1]
    mask_list_train = mask_list[0:train_flag:1]
    normal_list_train = normal_list[0:train_flag:1]

    face_list_1_test = face_list_1[train_flag:len(face_list_1):1]
    face_list_2_test = face_list_2[train_flag:len(face_list_1):1]
    albedo_list_test = albedo_list[train_flag:len(face_list_1):1]
    mask_list_test = mask_list[train_flag:len(face_list_1):1]
    normal_list_test = normal_list[train_flag:len(face_list_1):1]

    name_to_list = {'face_1': face_list_1_train,'face_2':face_list_2_train,'mask':mask_list_train,
                    'albedo': albedo_list_train, 'normal':normal_list_train}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(dir+'../train.csv')


    name_to_list = {'face_1': face_list_1_test,'face_2':face_list_2_test,'mask':mask_list_test,
                    'albedo': albedo_list_test, 'normal':normal_list_test}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(dir+'../test.csv')

    print('train and test csv file is saved.')

def generate_data_DPR_csv(dir):

    floder = dir + '*'
    floderlist = glob.glob(floder)
    floderlist.sort()
    face_list_1 = []
    face_list_2 = []
    normal_list = []
    mask_list = []
    for i in range(len(floderlist)):
        dofloder = floderlist[i]
        dofloder.split('/')
        im0 = dofloder + '/'+ dofloder.split('/')[-1]+'_00.png'
        im1 = dofloder + '/'+ dofloder.split('/')[-1]+'_01.png'
        im2 = dofloder + '/'+ dofloder.split('/')[-1]+'_02.png'
        im3 = dofloder + '/'+ dofloder.split('/')[-1]+'_03.png'
        im4 = dofloder + '/'+ dofloder.split('/')[-1]+'_04.png'
        normal = dofloder + '/normal.png'

        face_list_1.append(im0)
        face_list_2.append(im1)
        normal_list.append(normal)

        face_list_1.append(im0)
        face_list_2.append(im2)
        normal_list.append(normal)

        face_list_1.append(im0)
        face_list_2.append(im3)
        normal_list.append(normal)

        face_list_1.append(im0)
        face_list_2.append(im4)
        normal_list.append(normal)

        face_list_1.append(im1)
        face_list_2.append(im2)
        normal_list.append(normal)

        face_list_1.append(im1)
        face_list_2.append(im3)
        normal_list.append(normal)

        face_list_1.append(im1)
        face_list_2.append(im4)
        normal_list.append(normal)

        face_list_1.append(im2)
        face_list_2.append(im3)
        normal_list.append(normal)

        face_list_1.append(im2)
        face_list_2.append(im4)
        normal_list.append(normal)


        face_list_1.append(im3)
        face_list_2.append(im4)
        normal_list.append(normal)


            # print('')

    print('face1 len is '+str(len(face_list_1)) + ' face2 '+ str(len(face_list_2)))
    if len(face_list_1)!=len(normal_list):
        print('the dataset has some problem!')
    train_flag = int(len(face_list_1) - len(face_list_1)*0.2)
    face_list_1_train = face_list_1[0:train_flag:1]
    face_list_2_train = face_list_2[0:train_flag:1]
    normal_list_train = normal_list[0:train_flag:1]

    face_list_1_test = face_list_1[train_flag:len(face_list_1):1]
    face_list_2_test = face_list_2[train_flag:len(face_list_1):1]
    normal_list_test = normal_list[train_flag:len(face_list_1):1]

    name_to_list = {'face_1': face_list_1_train,'face_2':face_list_2_train, 'normal':normal_list_train}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(dir+'../train.csv')


    name_to_list_test = {'face_1': face_list_1_test,'face_2':face_list_2_test, 'normal':normal_list_test}
    df = pd.DataFrame(data=name_to_list_test)
    df.to_csv(dir+'../test.csv')

    print('train and test csv file is saved.')

def generate_data_GT_csv(dir):

    floder = dir + '*'
    floderlist = glob.glob(floder)
    floderlist.sort()
    face_list_1 = []
    face_list_2 = []
    normal_list = []
    mask_list = []
    for i in range(len(floderlist)):
        dofloder = floderlist[i]
        sublist = glob.glob(dofloder + '/texture/*')

        for t in range(len(sublist)):
            random.shuffle(sublist)
            face_list_1 += sublist
            random.shuffle(sublist)
            face_list_2 += sublist

    name_to_list = {'face_1': face_list_1,'face_2':face_list_2}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(dir+'../train.csv')

    print('train and test csv file is saved.')

def get_dataset_DPR(syn_dir=None, read_from_csv=None, read_celeba_csv=None, read_first=None, validation_split=0):
    mask = []
    face = []

    df = pd.read_csv(read_from_csv)
    df = df[:read_first]
    face_1 = list(df['face_1'])
    face_2 = list(df['face_2'])
    normal = list(df['normal'])

    assert (len(face_1) == len(face_2) == len(normal))
    dataset_size = len(face_1)
    validation_count = int(validation_split * dataset_size / 100)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = SfSNetDataset_DPR( face_1, face_2, normal, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class SfSNetDataset_DPR(Dataset):
    def __init__(self, face_1, face_2, normal, albedo, mask, transform=None):

        self.face_1 = face_1
        self.face_2 = face_2
        self.normal = normal
        self.mask = mask
        self.albedo = albedo

        self.transform = transform
        self.dataset_len = len(self.face_1)
        self.mask_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self.normal_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        face_1 = self.transform(Image.open(self.face_1[index]))
        face_2 = self.transform(Image.open(self.face_2[index]))

        normal = self.normal_transform(Image.open(self.normal[index])) # PIL
        normal = 2.0*(normal - 0.5)

        mask = self.transform(Image.open(self.mask[index]))
        albedo = self.transform(Image.open(self.albedo[index]))

        # result = {'face1' : face_1, 'face2' : face_2, 'normal' : normal, 'albedo' : albedo, 'mask' : mask,}
        # return result
        return face_1, face_2, normal, albedo, mask, self.face_1[index]

    def __len__(self):
        return self.dataset_len

def generate_data_paired_sy_csv(dir):

    floder = dir + '*'
    floderlist = glob.glob(floder)
    floderlist.sort()
    face_list_1 = []
    face_list_2 = []
    albedo_list = []
    normal_list = []
    mask_list = []
    for i in range(len(floderlist)):
        subfloder = floderlist[i] + '/??????_albedo_?_?.png'
        sublist = glob.glob(subfloder)
        sublist.sort()
        for j in range(len(sublist)):
            dofloder = sublist[j]
            im0 = dofloder[:-4] + '-imgHQ*****_light_**.png'
            tpface = glob.glob(im0)
            tpface.sort()
            if (len(tpface) == 2):
                face_list_1.append(tpface[0])
                face_list_2.append(tpface[1])
                albedo_list.append(dofloder)
                normal_list.append(dofloder.replace('albedo', 'normal')) 
                mask_list.append(dofloder.replace('albedo', 'mask'))

    train_flag = int(len(face_list_1) - len(face_list_1)*0.2)
    face_list_1_train = face_list_1[0:train_flag:1]
    face_list_2_train = face_list_2[0:train_flag:1]

    face_list_1_test = face_list_1[train_flag:len(face_list_1):1]
    face_list_2_test = face_list_2[train_flag:len(face_list_1):1]

    albedo_list_train = albedo_list[0:train_flag:1]
    albedo_list_test = albedo_list[train_flag:len(face_list_1):1]

    normal_list_train = normal_list[0:train_flag:1]
    normal_list_test = normal_list[train_flag:len(face_list_1):1]

    mask_list_train = mask_list[0:train_flag:1]
    mask_list_test = mask_list[train_flag:len(face_list_1):1]

    name_to_list = {'face_1': face_list_1_train,'face_2':face_list_2_train, 'albedo':albedo_list_train, 'normal':normal_list_train, 'mask':mask_list_train}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(dir+'/../train.csv')


    name_to_list = {'face_1': face_list_2_test,'face_2':face_list_1_test, 'albedo':albedo_list_test, 'normal':normal_list_test, 'mask':mask_list_test}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(dir+'/../test.csv')

    print('train.py and test csv file is saved.')

def get_dataset_sy(syn_dir=None, read_from_csv=None, read_celeba_csv=None, read_first=None, validation_split=0):

    df = pd.read_csv(read_from_csv)
    df = df[:read_first]
    face_1 = list(df['face_1'])
    face_2 = list(df['face_2'])
    normal = list(df['normal'])
    mask = list(df['mask'])
    albedo = list(df['albedo'])

    assert (len(face_1) == len(face_2) == len(normal) == len(albedo) == len(mask))
    dataset_size = len(face_1)
    validation_count = int(validation_split * dataset_size / 100)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = SfSNetDataset_DPR( face_1, face_2, normal, albedo, mask, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset



def get_dataset_GT(syn_dir=None, read_from_csv=None, read_celeba_csv=None, read_first=None, validation_split=0):
    df = pd.read_csv(read_from_csv)
    df = df[:read_first]
    face_1 = list(df['face_1'])
    face_2 = list(df['face_2'])

    assert (len(face_1) == len(face_2))
    dataset_size = len(face_1)
    validation_count = int(validation_split * dataset_size / 100)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = SfSNetDataset_GT( face_1, face_2, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class SfSNetDataset_GT(Dataset):
    def __init__(self, face_1, face_2, transform=None):

        self.face_1 = face_1
        self.face_2 = face_2

        self.transform = transform
        self.dataset_len = len(self.face_1)
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.normal_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        face_1 = Image.open(self.face_1[index])
        face_2 = Image.open(self.face_2[index])

        normal1 = Image.open(self.face_1[index].replace('texture', 'normal')) # PIL
        normal2 = Image.open(self.face_2[index].replace('texture', 'normal')) # PIL


        mask1 = Image.open(self.face_1[index].replace('texture', 'mask'))
        mask2 = Image.open(self.face_2[index].replace('texture', 'mask'))

        tpf_1 = Image.new('RGB', (256,256))
        tpf_1.paste(face_1, (16, 27))
        tpf_2 = Image.new('RGB', (256,256))
        tpf_2.paste(face_2, (16, 27))

        tpm_1 = Image.new('RGB', (256,256))
        tpm_1.paste(mask1, (16, 27))
        tpm_2 = Image.new('RGB', (256,256))
        tpm_2.paste(mask2, (16, 27))


        tpn_1 = Image.new('RGB', (256,256))
        tpn_1.paste(normal1, (16, 27))
        tpn_2 = Image.new('RGB', (256,256))
        tpn_2.paste(normal2, (16, 27))

        fa1 = self.transform(tpf_1)
        fa2 = self.transform(tpf_2)
        mk1 = self.transform(tpm_1)
        mk2 = self.transform(tpm_2)
        nm1 = self.transform(tpn_1)
        nm2 = self.transform(tpn_2)

        normal1 = 2.0*(nm1 - 0.5)
        normal2 = 2.0*(nm2 - 0.5)


        return fa1, fa2, normal1, normal2, mk1, mk2, self.face_1[index]

    def __len__(self):
        return self.dataset_len


# dataset_path = '/media/xteam1/6699a23c-17ee-4ba0-a18f-d5b89d4917ee/dataset/GT/nonlinear/'
# generate_data_GT_csv(dataset_path)