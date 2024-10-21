import os
import torch
import sys
from torchvision import utils
from collections import OrderedDict
from util import util
from util import pose_utils
import numpy as np
import ntpath
import cv2
from torchvision import utils as torchutils
from PIL import Image

class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    def get_current_visuals(self,savepath,total_steps):
        """Return visualization images"""
        visualBigGenImage = torch.cat([self.imagelist[:,-1,...], self.fake_image_t], dim=0)
        bs,c,h,w = visualBigGenImage.shape
        if bs < 7:
            nrow = bs
        else:
            nrow = 7
        grid = torchutils.make_grid(visualBigGenImage,nrow=nrow, padding=30, normalize=True)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        name = f'{total_steps}.png'
        savepath = os.path.join(savepath,name)
        im.save(savepath)


    def convert2im(self, value, name):
        if 'label' in name:
            convert = getattr(self, 'label2color')
            value = convert(value)

        if 'flow' in name: # flow_field
            convert = getattr(self, 'flow2color')
            value = convert(value)

        if value.size(1) == 18: # bone_map
            value = np.transpose(value[0].detach().cpu().numpy(),(1,2,0))
            value = pose_utils.draw_pose_from_map(value)[0]
            result = value

        elif value.size(1) == 21: # bone_map + color image
            value = np.transpose(value[0,-3:,...].detach().cpu().numpy(),(1,2,0))
            # value = pose_utils.draw_pose_from_map(value)[0]
            result = value.astype(np.uint8)

        else:
            result = util.tensor2im(value.data)
        return result

    def get_current_errors(self):
        """Return training loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def save(self, label):
        pass

    # save model
    def save_networks(self, which_epoch):
        """Save all the networks to the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

    # load models
    def load_networks(self, which_epoch):
        """Load all the networks from the disk"""
        device = torch.device('cuda:0')
        for name in self.model_names:
            if isinstance(name, str):
                filename = '%s_net_%s.pth' % (which_epoch, name)
                if self.opt.isTrain and self.opt.continue_train:
                    path = os.path.join(self.opt.checkpoints_dir,self.opt.name, filename)
                    # path = os.path.join(self.opt.load_pretrain,filename)
                else:
                    path = os.path.join(self.opt.checkpoints_dir,self.opt.name, filename)
                net = getattr(self, 'net_' + name)
                try:
                    '''
                    new_dict = {}
                    pretrained_dict = torch.load(path)
                    for k, v in pretrained_dict.items():
                        if 'transformer' in k:
                            new_dict[k.replace('transformer', 'PTM')] = v
                        else:
                            new_dict[k] = v

                    net.load_state_dict(new_dict)
                    '''
                    # net.load_state_dict(torch.load(path,map_location=device))
                    net.load_state_dict(torch.load(path))
                    print('load %s from %s' % (name, filename))
                except FileNotFoundError:
                    print('do not find checkpoint for network %s' % name)
                    continue
                except:
                    # pretrained_dict = torch.load(path,map_location=device)
                    pretrained_dict = torch.load(path)
                    model_dict = net.state_dict()
                    try:
                        pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        nop = {k: v for k, v in model_dict.items() if k not in pretrained_dict}
                        if len(pretrained_dict_) == 0:
                            pretrained_dict_ = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if
                                                k.replace('module.', '') in model_dict}
                        if len(pretrained_dict_) == 0:
                            pretrained_dict_ = {('module.' + k): v for k, v in pretrained_dict.items() if
                                                'module.' + k in model_dict}

                        pretrained_dict = pretrained_dict_
                        net.load_state_dict(pretrained_dict)
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % name)
                    except:
                        print('Pretrained network %s has fewer layers; The following are not initialized:' % name)
                        not_initialized = set()
                        for k, v in pretrained_dict.items():
                            if v.size() == model_dict[k].size():
                                model_dict[k] = v

                        for k, v in model_dict.items():
                            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                                # not_initialized.add(k)
                                not_initialized.add(k.split('.')[0])
                        print(sorted(not_initialized))
                        net.load_state_dict(model_dict)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()
                if not self.isTrain:
                    net.eval()

    def update_learning_rate(self, epoch=None):
        """Update learning rate"""
        for scheduler in self.schedulers:
            if epoch == None:
                scheduler.step()
            else:
                scheduler.step(epoch)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate=%.7f' % lr)

    def get_current_learning_rate(self):
        lr_G = self.optimizers[0].param_groups[0]['lr']
        lr_D = self.optimizers[1].param_groups[0]['lr']
        return lr_G, lr_D

    def save_results(self, save_data, old_size, data_name='none', data_ext='jpg'):
        """Save the training or testing results to disk"""
        img_paths = self.get_image_paths()

        for i in range(len(img_paths)):
            print('process image ...... %s' % img_paths[i])
            # short_path = ntpath.basename(img_paths[i])  # get image path
            # name = os.path.splitext(short_path)[0]
            img_name = '%s.%s' % (img_paths[i], data_ext)

            util.mkdir(self.opt.results_dir)
            img_path = os.path.join(self.opt.results_dir, img_name)
            utils.save_image(save_data, img_path, nrow=10, padding=30, normalize=True)
            # self.save_images(img_path,*save_data)


    def save_images(self,PATH, src_img, tgt_img, fake_img):
        n, c, h, w = src_img.size()
        samples = torch.FloatTensor(3 * n, c, h, w).zero_()
        for i in range(n):
            samples[3 * i + 0] = src_img[i].data
            samples[3 * i + 1] = tgt_img[i]
            samples[3 * i + 2] = fake_img[i].data

        utils.save_image(samples,PATH , nrow=8, padding=30, normalize=True)


    def save_chair_results(self, save_data, old_size, img_path, data_name='none', data_ext='jpg'):
        """Save the training or testing results to disk"""
        img_paths = self.get_image_paths()
        print(save_data.shape)
        for i in range(save_data.size(0)):
            print('process image ...... %s' % img_paths[i])
            short_path = ntpath.basename(img_paths[i])  # get image path
            name = os.path.splitext(short_path)[0]
            img_name = '%s_%s.%s' % (name, data_name, data_ext)

            util.mkdir(self.opt.results_dir)
            img_numpy = util.tensor2im(save_data[i].data)
            img_numpy = cv2.resize(img_numpy, (old_size[1], old_size[0]))
            util.save_image(img_numpy, img_path)
