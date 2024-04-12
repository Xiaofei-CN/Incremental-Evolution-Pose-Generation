import glob
import os.path
from data.base_dataset import BaseDataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from .dataset_basefunction import *
import random
from collections import Counter

class FashionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 256))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        return parser
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.fashiondir = opt.fashiondir
        self.scale_param = opt.scale_param if opt.phase == 'train' else 0
        self.datapath = os.path.join(self.root, '%s%s' % (self.phase,self.fashiondir))
        pairLst = os.path.join(self.root, '%s.txt' % self.phase)
        self.name_pairs = self.init_categories(pairLst)
        self.size = len(self.name_pairs)

        self.load_size = (opt.loadSize,opt.loadSize)
        # prepare for transformation

    # def __getitem__(self, index):
    #     # prepare for source image Xs and target image Xt
    #     if self.opt.phase == 'train':
    #         index = random.randint(0, self.size-1)
    #     imgdir, Xs_name,Xt_name,flag = self.name_pairs[index]
    #     imglist = self.gen_imglist(int(Xs_name), int(Xt_name), eval(flag))
    #     imglistlen = len(imglist)
    #     Xs_name = imgdir + '[' + Xs_name
    #     imagelist = []
    #     poselist = []
    #     maplist = []
    #     seglist = []
    #     for im in imglist:
    #         image = self.trans(F.resize(Image.open(os.path.join(self.datapath,imgdir,'%s.png'%im)).convert('RGB'),self.load_size))
    #         imagelist.append(image)
    #         map,pose = self.obtain_bone(imgdir,im)
    #         seglist.append(self.gen_onehot(os.path.join(self.datapath,imgdir,'%s_gray.png'%im)))
    #         poselist.append(pose)
    #         maplist.append(map)
    #
    #     if len(imglist) < 8:
    #         imagelist += [imagelist[-1]] * (7-imglistlen)
    #         poselist += [poselist[-1]] * (7-imglistlen)
    #         maplist += [maplist[-1]] * (7-imglistlen)
    #         seglist += [seglist[-1]] * (7-imglistlen)
    #     imagelist = torch.stack(imagelist)
    #     poselist = torch.stack(poselist)
    #     maplist = torch.stack(maplist)
    #     seglist = torch.stack(seglist) #str(imglist[-1])
    #     return {'imagelist': imagelist, 'poselist': poselist, 'maplist': maplist,'seglist':seglist,
    #             'Xs_path': Xs_name, 'Xt_path': Xt_name, 'reverse': '1', 'imglistlen': imglistlen}
    def __getitem__(self, index):
        # prepare for source image Xs and target image Xt
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)
        return self.get_mata_data_from_list(index) if self.fashiondir == 'data' else self.get_mata_data_from_video(index)

    def get_mata_data_from_list(self,index):
        imgdir, Xs_name, Xt_name, flag = self.name_pairs[index]
        imglist = self.gen_imglist(int(Xs_name), int(Xt_name), eval(flag))
        imglist.append(eval(Xt_name))
        imglistlen = len(imglist)
        if imglistlen > 7:
            imglist.pop(-1)
            imglistlen = len(imglist)
        Xs_name = imgdir + '[' + Xs_name
        imagelist = []
        poselist = []
        maplist = []
        seglist = []
        param = get_random_params(self.load_size, self.scale_param)
        for im in imglist:
            image = self.get_image_tensor(os.path.join(self.datapath, imgdir, '%s.png' % im), param)
            imagelist.append(image)
            map, pose = self.obtain_bone(dir=imgdir, name=im, param=param)
            seglist.append(self.gen_onehot(os.path.join(self.datapath, imgdir, '%s_gray.png' % im), param))
            poselist.append(pose)
            maplist.append(map)

        if len(imglist) < 8:
            imagelist += [imagelist[-1]] * (7 - imglistlen)
            poselist += [poselist[-1]] * (7 - imglistlen)
            maplist += [maplist[-1]] * (7 - imglistlen)
            seglist += [seglist[-1]] * (7 - imglistlen)
        imagelist = torch.stack(imagelist)
        poselist = torch.stack(poselist)
        maplist = torch.stack(maplist)
        seglist = torch.stack(seglist)  # str(imglist[-1])
        return {'imagelist': imagelist, 'poselist': poselist, 'maplist': maplist, 'seglist': seglist,
                'Xs_path': Xs_name, 'Xt_path': str(imglist[-1]), 'reverse': '1', 'imglistlen': imglistlen}

    def get_mata_data_from_video(self,index):
        imgdir = self.name_pairs[index]
        if isinstance(imgdir,list):
            imgdir,imglist = imgdir
            imglist = imglist.split(']')
            imglistlen = len(imglist)
            sourcename,targetname = imglist[0],imglist[-1]
            imglist = [os.path.join(self.datapath, imgdir,f'{i}.npy') for i in imglist]
        else:
            imgdirpath = os.path.join(self.datapath, imgdir)
            filelist = glob.glob(os.path.join(imgdirpath, '*.npy'))
            filelist = sorted(filelist)
            fileNumber = len(filelist)
            sourceNumber = random.randint(0, fileNumber - 40 * 6 - 1)
            imglistlen = random.randint(2, 7)
            imglist = [filelist[sourceNumber + i * 40] for i in range(imglistlen)]
            # assert imglist[-1] <= fileNumber
            sourcename = os.path.basename(imglist[0])[:-4]
            targetname = os.path.basename(imglist[-1])[:-4]
        Xs_name = imgdir + '[' + str(sourcename) + ']' + str(targetname)

        imagelist = []
        poselist = []
        maplist = []
        seglist = []
        param = get_random_params(self.load_size, self.scale_param)
        for im in imglist:
            image = self.get_image_tensor(im[:-3]+'png', param)
            imagelist.append(image)
            map, pose = self.obtain_bone(im, param)
            seglist.append(self.gen_onehot('%s_gray.png'%im[:-4], param))
            poselist.append(pose)
            maplist.append(map)

        if len(imglist) < 8:
            imagelist += [imagelist[-1]] * (7 - imglistlen)
            poselist += [poselist[-1]] * (7 - imglistlen)
            maplist += [maplist[-1]] * (7 - imglistlen)
            seglist += [seglist[-1]] * (7 - imglistlen)
        imagelist = torch.stack(imagelist)
        poselist = torch.stack(poselist)
        maplist = torch.stack(maplist)
        seglist = torch.stack(seglist)  # str(imglist[-1])
        return {'imagelist': imagelist, 'poselist': poselist, 'maplist': maplist, 'seglist': seglist,
                'Xs_path': Xs_name, 'Xt_path': targetname, 'reverse': '1', 'imglistlen': imglistlen}

    def get_image_tensor(self, path,param=None):
        img = Image.open(path)
        if param:
            trans = get_transform(param, normalize=True, toTensor=True)
        else:
            trans = get_transform({}, normalize=True, toTensor=True)
        img = trans(img)
        return img
    def check(self,data):
        for index,number in data:
            if index > 19:
                continue
            else:
                return index
        return 0
    def gen_onehot(self, path,param=None):
        img = Image.open(path)
        if param:
            trans = get_transform(param,normalize=False,toTensor=False)
        else:
            trans = get_transform({},normalize=False,toTensor=False)
        img = trans(img)
        s1np = np.expand_dims(np.array(img), -1)
        where = np.where(s1np > 19)
        windowSize = 5
        size = (windowSize-1) // 2
        for x,y,z in zip(*where):
            x1,x2 = x-size,x+size
            y1,y2 = y-size,y+size
            window = s1np[x1:x2+1,y1:y2+1,z]
            window = window.reshape(1,-1)[0]
            counter = Counter(window).most_common(20)
            pi = self.check(counter)
            s1np[x,y,z] = pi

        s1np = np.concatenate([s1np, s1np, s1np], -1)
        SPL1_img = Image.fromarray(np.uint8(s1np))
        SPL1_img = np.expand_dims(np.array(SPL1_img)[:, :, 0], 0)
        num_class = 20
        _, h, w = SPL1_img.shape
        tmp = torch.from_numpy(SPL1_img).view(-1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        onehot = ones.view([h, w, num_class])
        onehot = onehot.permute(2, 0, 1)
        bk = onehot[0, ...]
        hair = torch.sum(onehot[[1, 2], ...], dim=0)
        l1 = torch.sum(onehot[[5,11,7, 12], ...], dim=0)
        l2 = torch.sum(onehot[[4, 13], ...], dim=0)
        l3 = torch.sum(onehot[[9, 10], ...], dim=0)
        l4 = torch.sum(onehot[[16, 17,8, 18, 19], ...], dim=0)
        l5 = torch.sum(onehot[[3,14,15], ...], dim=0)
        l6 = torch.sum(onehot[[6], ...], dim=0)
        CL8 = torch.stack([bk, hair, l1, l2, l3, l4, l5, l6])

        return CL8

    def gen_imglist(self, start, end, reverse):
        def getList(start, end):
            diff = end - start
            if diff > 0:
                return list(range(start, end))
            else:
                return list(range(start, 8)) + list(range(0, end))

        def getReverseList(start, end):
            res = getList(start, end)
            res.pop(0)
            res.append(end)
            return res[::-1]

        if reverse:
            return getReverseList(end, start)
        else:
            return getList(start, end)

    def __len__(self):
        if self.opt.phase == 'train':
            return 6000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'FashionDataset'

    def cords_to_map(self, cords, img_size, sigma=6):
        result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
        for i, point in enumerate(cords):
            if point[0] == 0 or point[1] == 0:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
        return result

    def obtain_bone(self, dir, name=None,param=None):
        try:
            array = np.load(os.path.join(self.datapath, dir, f'{name}.npy'))[0,:,:2]
        except:
            array = np.load(dir)[:,:2]
        if param:
            array = self.trans_keypoins(array,param)

        x = array[:, :1]
        y = array[:, 1:]
        yxarray = np.concatenate((y, x), axis=1)
        pose = self.cords_to_map(yxarray, self.load_size)
        pose = np.transpose(pose, (2, 0, 1))
        # import matplotlib.pyplot as plt
        # rp = np.zeros((256,256))
        # for k,p in enumerate(pose):
        #     rp += p
        # plt.imshow(rp)
        # plt.show()
        # plt.savefig('%s-%s.jpg'%(dir,name))
        pose = torch.Tensor(pose)
        array = torch.Tensor(array)
        return pose, array

    def trans_keypoins(self,keypoints, param, img_size=(256, 256)):
        missing_keypoint_index = keypoints == 0

        # crop the white line in the original dataset
        # keypoints[:, 0] = (keypoints[:, 0] - 40)

        # resize the dataset
        img_h, img_w = img_size
        scale_w = 1.0 / 256.0 * img_w
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
        keypoints[missing_keypoint_index] = 0
        return keypoints

    def init_categories(self, pairLst):
        print('Loading data pairs ...')
        if self.fashiondir == 'data':
            with open(pairLst, 'r') as f:
                data = [da.strip('\n').split('[') for da in f.readlines()]
        else:
            if self.phase == 'train':
                data = os.listdir(self.datapath)
            else:
                with open(os.path.join(os.path.dirname(pairLst),'test-256.txt'),'r') as f:
                    data = [da.strip('\n').split('[') for da in f.readlines()]
            # check data
            # data = []
            # for dir in listdir:
            #     filelist = glob.glob(os.path.join(self.datapath,dir,'*.npy'))
            #     if len(filelist) == 0:
            #         continue
            #     s = 0
            #     for file in filelist:
            #         if os.path.isfile(file) and os.path.isfile(file[:-4]+'_vis.png') and os.path.join(file[:-3]+'png'):
            #             s += 1
            #     if s == len(filelist):
            #         data.append(dir)
            print(f'Total {len(data)} file has loaded')
        print('Loading data pairs finished ...')
        return data



