import random
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def get_random_params(size, scale_param):
    w, h = size
    scale = random.random() * scale_param

    new_w = int(w * (1.0 + scale))
    new_h = int(h * (1.0 + scale))
    x = random.randint(0, np.maximum(0, new_w - w))
    y = random.randint(0, np.maximum(0, new_h - h))
    return {'crop_param': (x, y, w, h), 'scale_size': (new_h, new_w)}


def get_transform(param, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if 'scale_size' in param and param['scale_size'] is not None:
        osize = param['scale_size']
        transform_list.append(transforms.Resize(osize, interpolation=method))

    if 'crop_param' in param and param['crop_param'] is not None:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, param['crop_param'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __crop(img, pos):
    x1, y1, tw, th = pos
    return img.crop((x1, y1, x1 + tw, y1 + th))


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def trans_keypoins(keypoints, param, img_size=(256,256)):
    missing_keypoint_index = keypoints == -1

    # crop the white line in the original dataset
    keypoints[:, 0] = (keypoints[:, 0] - 40)

    # resize the dataset
    img_h, img_w = img_size
    scale_w = 1.0 / 176.0 * img_w
    scale_h = 1.0 / 256.0 * img_h

    if 'scale_size' in param and param['scale_size'] is not None:
        new_h, new_w = param['scale_size']
        scale_w = scale_w / img_w * new_w
        scale_h = scale_h / img_h * new_h

    if 'crop_param' in param and param['crop_param'] is not None:
        w, h, _, _ = param['crop_param']
    else:
        w, h = 0, 0

    keypoints[:, 0] = keypoints[:, 0] * scale_w - w
    keypoints[:, 1] = keypoints[:, 1] * scale_h - h
    keypoints[missing_keypoint_index] = -1
    return keypoints
