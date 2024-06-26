from __future__ import print_function
from skimage.draw import circle, line_aa, polygon
import cv2
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
import imageio

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

    def make_colorwheel():
        '''
        Generates a color wheel for optical flow visualization as presented in:
            Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
            URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun
        '''
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros((ncols, 3))
        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
        col = col + RY
        # YG
        colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
        colorwheel[col:col + YG, 1] = 255
        col = col + YG
        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
        col = col + GC
        # CB
        colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
        colorwheel[col:col + CB, 2] = 255
        col = col + CB
        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
        col = col + BM
        # MR
        colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
        colorwheel[col:col + MR, 0] = 255
        return colorwheel


class flow2color():
    # code from: https://github.com/tomrunia/OpticalFlow_Visualization
    # MIT License
    #
    # Copyright (c) 2018 Tom Runia
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to conditions.
    #
    # Author: Tom Runia
    # Date Created: 2018-08-03
    def __init__(self):
        self.colorwheel = make_colorwheel()

    def flow_compute_color(self, u, v, convert_to_bgr=False):
        '''
        Applies the flow color wheel to (possibly clipped) flow components u and v.
        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun
        :param u: np.ndarray, input horizontal flow
        :param v: np.ndarray, input vertical flow
        :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
        :return:
        '''
        flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
        ncols = self.colorwheel.shape[0]

        rad = np.sqrt(np.square(u) + np.square(v))
        a = np.arctan2(-v, -u) / np.pi
        fk = (a + 1) / 2 * (ncols - 1)
        k0 = np.floor(fk).astype(np.int32)
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0

        for i in range(self.colorwheel.shape[1]):
            tmp = self.colorwheel[:, i]
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1 - f) * col0 + f * col1

            idx = (rad <= 1)
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            col[~idx] = col[~idx] * 0.75  # out of range?

            # Note the 2-i => BGR instead of RGB
            ch_idx = 2 - i if convert_to_bgr else i
            flow_image[:, :, ch_idx] = np.floor(255 * col)

        return flow_image

    def __call__(self, flow_uv, clip_flow=None, convert_to_bgr=False):
        '''
        Expects a two dimensional flow image of shape [H,W,2]
        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun
        :param flow_uv: np.ndarray of shape [H,W,2]
        :param clip_flow: float, maximum clipping value for flow
        :return:
        '''
        if len(flow_uv.size()) != 3:
            flow_uv = flow_uv[0]
        flow_uv = flow_uv.permute(1, 2, 0).cpu().detach().numpy()

        assert flow_uv.ndim == 3, 'input flow must have three dimensions'
        assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

        if clip_flow is not None:
            flow_uv = np.clip(flow_uv, 0, clip_flow)

        u = flow_uv[:, :, 1]
        v = flow_uv[:, :, 0]

        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)

        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        image = self.flow_compute_color(u, v, convert_to_bgr)
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0 * 2 - 1
        return image


def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])
    #image_numpy = cv2.resize(image_numpy, (176,256))

    imageio.imwrite(image_path, image_numpy)


LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
MISSING_VALUE = 10
# yx
def draw_pose_from_cords(poses, img=(256, 256), radius=2, draw_joints=True, text=None, backinfo=False, output_dir=None):
    if isinstance(img,str) and img is not None and backinfo:
        colors = cv2.imread(img)
    else:
        colors = np.zeros(shape=img + (3,), dtype=np.uint8)
    if len(poses) == 0:
        return colors
    poses = poses[0].astype(np.int)
    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = poses[f][0] <= MISSING_VALUE or poses[f][1] <= MISSING_VALUE
            to_missing = poses[t][0] <= MISSING_VALUE or poses[t][1] <= MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(poses[f][0], poses[f][1], poses[t][0], poses[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255

    for i, joint in enumerate(poses):
        if poses[i][0] == MISSING_VALUE or poses[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img)
        colors[yy, xx] = COLORS[i]
    if isinstance(text,str) and text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(colors, text, (30, 30), font, 0.5, (255, 255, 255), 1)
    if isinstance(output_dir,str) and output_dir is not None:
        cv2.imwrite(output_dir, colors)
    return colors
def generate_image(target,pred,imgname,imgshape=(256,256,3),save_path=None,text = None,draw_pose=None):
    assert target.shape==pred.shape
    assert len(target.shape) == 4
    assert draw_pose is not None
    B,Seqlen,kp,dim = target.shape
    if target.shape[-1] == 3:
        target = target[...,:2]
        pred = pred[...,:2]
    if isinstance(target,torch.Tensor):
        try:
            target = target.numpy()
        except:
            target = target.cpu().detach().numpy()
    if isinstance(pred,torch.Tensor):
        try:
            pred = pred.numpy()
        except:
            pred = pred.cpu().detach().numpy()

    if text is None:
        top = 'Gt'
        down = 'Gen'
    else:
        top = text[0]
        down = text[1]
    if imgshape[0] == imgshape[1]:
        norm = np.ones_like(target) * imgshape[0]  # 256
        target = target * norm
        pred = pred * norm
    else:
        norm =  np.ones_like(target) * np.array([225,400])
        target = target * norm
        pred = pred * norm
    gt = []
    for i in range(B):
        tar = target[i]
        for k in range(Seqlen):
            img_t = draw_pose(poses=np.expand_dims(tar[k],axis=0),text=top) # 需要统一  img:unit8?
            gt.append(img_t)
    gtimg = np.concatenate(gt,axis=1)

    gen = []
    for i in range(B):
        pre = pred[i]
        for k in range(Seqlen):
            img_p = draw_pose(poses=np.expand_dims(pre[k], axis=0),text=down)  # 需要统一  img:unit8?
            gen.append(img_p)
    genimg = np.concatenate(gen, axis=1)

    res = np.concatenate((gtimg, genimg), axis=0)

    if save_path is None:
        save_path = r'/home/xtf/sshcode/Pose_pro/images'
    os.makedirs(save_path,exist_ok=True)
    save_path += '/%s'% imgname
    cv2.imwrite(save_path,res)
