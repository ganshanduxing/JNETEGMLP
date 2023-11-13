## laod plain-images and secret keys
import os
import multiprocessing as mul
import time

import scipy.io as scio
from encryption_utils import ksa
from encryption_utils import prga
from encryption_utils import yates_shuffle
import tqdm
from encryption_utils import loadImageSet, loadImageFiles
from JPEG.rgbandycbcr import rgb2ycbcr
import cv2
import copy
from JPEG.jdcencColor import jdcencColor
from JPEG.zigzag import zigzag
from JPEG.invzigzag import invzigzag
from JPEG.jacencColor import jacencColor
from JPEG.Quantization import *
from cipherimageRgbGenerate import Gen_cipher_images
import hashlib


def encryption_each_component(image_component, keys, transform_stage4change, type, row, col, N, QF):
    # generate block permutation vector
    block8_number = int((row * col) / (8 * 8))
    data = [i for i in range(0, block8_number)]
    p_blockY = yates_shuffle(data, keys)
    keys = keys[64:]

    allblock8 = np.zeros([8, 8, int(row * col / (8 * 8))])
    allblock8_number = 0
    count = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            t = image_component[m:m + N, n:n + N] - 128

            # block rotation
            rotation_num = int('0b' + keys[:2], 2)
            keys = keys[2:]
            for k in range(0, rotation_num):
                t = np.rot90(t, -1)

            # new orthogonal transforms
            choice = keys[count * 63:(count + 1) * 63]
            dec_row = int('0b' + choice[0:7], 2)
            dec_column = []
            for i in range(7, 63, 7):
                dec_column.append(int('0b' + choice[i:i + 7], 2))
            rtransform = transform_stage4change[:, :, dec_row]
            ctransform = np.zeros([8, 8, 8])
            for i in range(0, 8):
                ctransform[:, :, i] = transform_stage4change[:, :, dec_column[i]]
            y = np.dot(t, rtransform.T)
            for i in range(0, 8):
                y[:, i] = np.dot(ctransform[:, :, i], y[:, i])

            temp = Quantization(y, QF, type=type)  # Quanlity

            allblock8[:, :, allblock8_number] = temp
            allblock8_number = allblock8_number + 1
            count = count + 1

    # block permutation
    permuted_blocks = copy.copy(allblock8)
    for i in range(len(p_blockY)):
        permuted_blocks[:, :, i] = allblock8[:, :, p_blockY[i]]

    # Huffman coding
    dccof = []
    accof = []
    for i in range(0, allblock8_number):
        temp = copy.copy(permuted_blocks[:, :, i])
        if i == 0:
            dc = temp[0, 0]
            dc_component = jdcencColor(dc, type)
            dccof = np.append(dccof, dc_component)
        else:
            dc = temp[0, 0] - dc
            dc_component = jdcencColor(dc, type)
            dccof = np.append(dccof, dc_component)
            dc = temp[0, 0]
        acseq = []
        aczigzag = zigzag(temp)
        eobi = 0
        for j in range(63, -1, -1):
            if aczigzag[j] != 0:
                eobi = j
                break
        if eobi == 0:
            acseq = np.append(acseq, [999])
        else:
            acseq = np.append(acseq, aczigzag[1: eobi + 1])
            acseq = np.append(acseq, [999])
        ac_component = jacencColor(acseq, type)
        accof = np.append(accof, ac_component)

    return dccof, accof


def encryption(img, keyY, keyCb, keyCr, transforms, QF, N=8):
    # N: block size
    # QF: quality factor
    row, col, _ = img.shape
    plainimage = rgb2ycbcr(img)
    plainimage = plainimage.astype(np.float64)
    Y = plainimage[:, :, 0]
    Cb = plainimage[:, :, 1]
    Cr = plainimage[:, :, 2]

    for i in range(0, int(16 * np.ceil(col / 16) - col)):
        Y = np.c_[Y, Y[:, -1]]
        Cb = np.c_[Cb, Cb[:, -1]]
        Cr = np.c_[Cr, Cr[:, -1]]

    for i in range(0, int(16 * np.ceil(row / 16) - row)):
        Y = np.r_[Y, [Y[-1, :]]]
        Cb = np.r_[Cb, [Cb[-1, :]]]
        Cr = np.r_[Cr, [Cr[-1, :]]]

    [row, col] = Y.shape

    Cb = cv2.resize(Cb,
                    (int(col / 2), int(row / 2)),
                    interpolation=cv2.INTER_CUBIC)
    Cr = cv2.resize(Cr,
                    (int(col / 2), int(row / 2)),
                    interpolation=cv2.INTER_CUBIC)

    # Y component
    dccofY, accofY = encryption_each_component(Y, keyY, transforms, type='Y', row=row, col=col, N=N, QF=QF)
    ## Cb and Cr component
    dccofCb, accofCb = encryption_each_component(Cb, keyCb, transforms, type='Cb', row=int(row / 2), col=int(col / 2),
                                                 N=N, QF=QF)
    dccofCr, accofCr = encryption_each_component(Cr, keyCr, transforms, type='Cr', row=int(row / 2), col=int(col / 2),
                                                 N=N, QF=QF)

    accofY = accofY.astype(np.int8)
    dccofY = dccofY.astype(np.int8)
    accofCb = accofCb.astype(np.int8)
    dccofCb = dccofCb.astype(np.int8)
    accofCr = accofCr.astype(np.int8)
    dccofCr = dccofCr.astype(np.int8)
    return accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr, row, col


# read plain-images
def read_plain_images():
    plainimage_all = loadImageSet('../data/plainimages/*.jpg')
    # save size information
    img_size = []
    for temp in plainimage_all:
        row, col, _ = temp.shape
        img_size.append((row, col))
    np.save("../data/plainimages.npy", plainimage_all)
    np.save("../data/img_size.npy", img_size)
    return plainimage_all


# generate encryption key and embedding key
# keys are independent from plainimage
# encryption key generation - RC4
def generate_keys(control_length=512 * 512):
    # secret keys
    data_lenY = np.ones([1, int(control_length)])
    keyY = scio.loadmat('../data/keys/key.mat')  # Y component encryption key
    keyY = keyY['key'][0]
    keyCb = scio.loadmat('../data/keys/keyCb.mat')  # Cb component encryption key
    keyCb = keyCb['keyCb'][0]
    keyCr = scio.loadmat('../data/keys/keyCr.mat')  # Cr component encryption key
    keyCr = keyCr['keyCr'][0]
    # keys stream
    s = ksa(keyY)
    r = prga(s, data_lenY)
    encryption_keyY = ''
    for i in range(0, len(r)):
        temp1 = str(r[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyY = encryption_keyY + temp2

    data_lenC = np.ones([1, int(control_length // 4)])
    s1 = ksa(keyCb)
    r1 = prga(s1, data_lenC)
    encryption_keyCb = ''
    for i in range(0, len(r1)):
        temp1 = str(r1[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyCb = encryption_keyCb + temp2

    s2 = ksa(keyCr)
    r2 = prga(s2, data_lenC)
    encryption_keyCr = ''
    for i in range(0, len(r2)):
        temp1 = str(r2[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyCr = encryption_keyCr + temp2

    transform_stage4change = scio.loadmat('../data/keys/transform_stage4change.mat')
    transform_stage4change = transform_stage4change['transform_stage4change']

    return encryption_keyY, encryption_keyCb, encryption_keyCr, transform_stage4change


bit_path = '../data/JPEGBitStream/ukbench'
QF = 90

def main(imageFile):
    # read plain-image
    img = cv2.imread(imageFile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 196))
    row, col, _ = img.shape
    encryption_keyY, encryption_keyCb, encryption_keyCr, transform_stage4change = generate_keys(row * col)
    accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr, row, col = encryption(img, encryption_keyY,
                                                                              encryption_keyCb,
                                                                              encryption_keyCr,
                                                                              transform_stage4change, QF,
                                                                              N=8)

    if not os.path.exists(bit_path):  # + imageFile.split('\\')[-2]):
        os.mkdir(bit_path) #+ imageFile.split('\\')[-2])
    img_size = (row, col)
    np.save(bit_path + '/' + imageFile.split('\\')[-1].split('.')[0] + '.npy', # + imageFile.split('\\')[-2] + '/' + imageFile.split('\\')[-1].split('.')[0] + '.npy',
            {'accofY': accofY, 'dccofY': dccofY, 'accofCb': accofCb, 'dccofCb': dccofCb, 'accofCr': accofCr,
             'dccofCr': dccofCr, 'size': img_size})
    # print('Generate BitStream ' + imageFile.split('\\')[-1].split('.')[0] + 'Success')


if __name__ == '__main__':
    # image encryption
    QF = 90
    encryption_keyY, encryption_keyCb, encryption_keyCr, transform_stage4change = generate_keys()
    bit_path = '../data/JPEGBitStream/ukbench'
    imageFiles = loadImageFiles('../data/plainimages/ukbench/*.jpg')

    for k in tqdm.tqdm(imageFiles):
        main(k)

    # 创建线程池
    # pool = mul.Pool(5)
    # pool.map(main, imageFiles)
    # del pool

    # generate cipher-images
    Gen_cipher_images(QF=QF)
