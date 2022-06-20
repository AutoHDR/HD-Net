import numpy as np
import cv2, glob, os
from imutils import face_utils
import shutil
    

ImgSize = 256
imgPath = '/home/toughlife/data/DPR_dataset/'
NormalList = glob.glob(imgPath + '*')
NormalList.sort()
for i in range(len(NormalList)):
    flag = 0
    tp = NormalList[i] + '/' + NormalList[i][33:] + '_mask_68.png'
    if not os.path.exists(tp):
        print(NormalList[i])
        shutil.rmtree(NormalList[i])
        continue
    cv2.imwrite(tp, cv2.resize(cv2.imread(tp),(ImgSize,ImgSize)))
    # pathI = tp.replace(NormalList[i][33:] + '_mask_68', 'normal_n')
    # cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    # pathI = tp.replace(NormalList[i][33:] + '_mask_68', 'normal_f')
    # cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    pathI = tp.replace(NormalList[i][33:] + '_mask_68', 'normal')
    cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    pathI = tp.replace(NormalList[i][33:] + '_mask_68', 'full_normal')
    cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    pathI = tp.replace(NormalList[i][33:] + '_mask_68', NormalList[i][33:] + '_00')
    cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    pathI = tp.replace(NormalList[i][33:] + '_mask_68', NormalList[i][33:] + '_01')
    cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    pathI = tp.replace(NormalList[i][33:] + '_mask_68', NormalList[i][33:] + '_02')
    cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    pathI = tp.replace(NormalList[i][33:] + '_mask_68', NormalList[i][33:] + '_03')
    cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    pathI = tp.replace(NormalList[i][33:] + '_mask_68', NormalList[i][33:] + '_04')
    cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))
    pathI = tp.replace(NormalList[i][33:] + '_mask_68', 'ori_shading')
    cv2.imwrite(pathI, cv2.resize(cv2.imread(pathI),(ImgSize,ImgSize)))

    if os.path.exists(tp.replace(NormalList[i][33:] + '_mask_68.png', 'ori_shading.exr')):
        os.remove(tp.replace(NormalList[i][33:] + '_mask_68.png', 'ori_shading.exr'))
        os.remove(tp.replace(NormalList[i][33:] + '_mask_68.png', 'relighting_mask.png'))
    print(tp)
    
    
    

        
