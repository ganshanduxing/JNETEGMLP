## generate cipherimage
import os
import multiprocessing as mul
import numpy as np
from JPEG.jacdecColorHuffman import jacdecColor
from JPEG.jdcdecColorHuffman import jdcdecColor
from JPEG.invzigzag import invzigzag
import cv2
from JPEG.rgbandycbcr import ycbcr2rgb, rgb2ycbcr
import glob
import tqdm
from JPEG.DCT import idctJPEG
from JPEG.Quantization import iQuantization
from encryption_utils import loadEncBit


def deEntropy(acall, dcall, row, col, type, N=8, QF=100):
    accof = acall
    dccof = dcall
    kk, acarr = jacdecColor(accof, type)
    kk, dcarr = jdcdecColor(dccof, type)
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)

    Eob = np.where(acarr == 999)
    Eob = Eob[0]
    count = 0
    kk = 0
    ind1 = 0
    xq = np.zeros([row, col])
    for m in range(0, row, N):
        for n in range(0, col, N):
            ac = acarr[ind1: Eob[count]]
            ind1 = Eob[count] + 1
            count = count + 1
            acc = np.append(dcarr[kk], ac)
            az = np.zeros(64 - acc.shape[0])
            acc = np.append(acc, az)
            temp = invzigzag(acc, 8, 8)
            temp = iQuantization(temp, QF, type)
            temp = idctJPEG(temp)
            xq[m:m + N, n:n + N] = temp + 128
            kk = kk + 1
    return xq


cipher_path = '../data/cipherimages/ukbench/'


def subImages(k, QF=90):
    bitstream_dic = loadEncBit(k).item()
    dcallY = bitstream_dic['dccofY']
    acallY = bitstream_dic['accofY']
    dcallCb = bitstream_dic['dccofCb']
    acallCb = bitstream_dic['accofCb']
    dcallCr = bitstream_dic['dccofCr']
    acallCr = bitstream_dic['accofCr']
    row, col = bitstream_dic['size']
    cipher_Y = deEntropy(acallY, dcallY, row, col, 'Y', QF=QF)
    cipher_cb = deEntropy(acallCb, dcallCb, int(row / 2), int(col / 2), 'C', QF=QF)
    cipher_cr = deEntropy(acallCr, dcallCr, int(row / 2), int(col / 2), 'C', QF=QF)

    cipherimage = np.zeros([row, col, 3])
    cipher_cb = cv2.resize(cipher_cb,
                           (col, row),
                           interpolation=cv2.INTER_CUBIC)
    cipher_cr = cv2.resize(cipher_cr,
                           (col, row),
                           interpolation=cv2.INTER_CUBIC)
    cipherimage[:, :, 0] = cipher_Y
    cipherimage[:, :, 1] = cipher_cb
    cipherimage[:, :, 2] = cipher_cr
    cipherimage = np.round(cipherimage)
    cipherimage = cipherimage.astype(np.uint8)
    cipherimage = ycbcr2rgb(cipherimage)

    if not os.path.exists(cipher_path): #+ k.split('\\')[-2]):
        os.mkdir(cipher_path) #+ k.split('\\')[-2])

    # np.save(srcFiles + srcFiles[k].split('/')[-2] + srcFiles[k].split('/')[-1], cipherimage)

    merged = cv2.merge([cipherimage[:, :, 2], cipherimage[:, :, 1], cipherimage[:, :, 0]])
    cv2.imwrite(cipher_path + k.split('\\')[-1].split('.')[0] + '.jpg', # + k.split('\\')[-2] + '/' + k.split('\\')[-1].split('.')[0] + '.jpg',
                merged,
                [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    # 6`    print('Generate Cipher-image ' + k.split('\\')[-1].split('.')[0] + 'success')


def Gen_cipher_images(QF):
    srcFiles = glob.glob('../data/JPEGBitStream/ukbench/*')  # 所有文件夹类别名称
    cipher_path = '../data/cipherimages/ukbench'
    for category_path in tqdm.tqdm(srcFiles):

        # # 读取子文件
        # sub_name = glob.glob(category_path + '/*')
        #
        # # 创建线程池
        # # pool = mul.Pool(5)
        # # rel = pool.map(subImages, sub_name)
        #
        # print('class: ' + category_path)
        # for k in tqdm.tqdm(sub_name):
        #     subImages(k, QF)

        subImages(category_path, QF)