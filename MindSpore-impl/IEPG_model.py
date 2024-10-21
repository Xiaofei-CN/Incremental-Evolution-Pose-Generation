import random
from PIL import Image
import numpy as np
import torch
import os
import itertools
from .base_model import BaseModel
from . import networks
from . import external_function
from . import base_function
from util import util, pose_utils,distributed_utils,face_util
from torchvision import utils as torchutils
import time
label_colours = [(0,0,0),(255,0,0),(255,0,0),(255,85,0),(0,0,255),(255,85,0),
                 (0,0,85),(255,85,0),(85,255,170),(0,85,85),(0,85,85),
                 (51,170,221),(255,85,0), (0,0,255),(51,170,221),(51,170,221),
                 (85,255,170),(85,255,170),(85,255,170),(85,255,170)]
class IEPGModel(BaseModel):
    def name(self):
        return 'IEPGModel'

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--init_type', type=str, default='orthogonal', help='initial type')
        parser.add_argument('--use_spect_g', action='store_false', help='use spectual normalization in generator')
        parser.add_argument('--use_spect_d', action='store_false', help='use spectual normalization in generator')
        parser.add_argument('--use_coord', action='store_true', help='use coordconv')
        parser.add_argument('--lambda_style', type=float, default=500, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_content_face', type=float, default=1, help='weight for the VGG19 content loss')
        parser.add_argument('--layers_g', type=int, default=3, help='number of layers in G')
        parser.add_argument('--save_input', action='store_true', help="whether save the input images when testing")
        parser.add_argument('--num_blocks', type=int, default=3, help="number of resblocks")
        parser.add_argument('--affine', action='store_true', default=True, help="affine in PTM")
        parser.add_argument('--nhead', type=int, default=2, help="number of heads in PTM")
        parser.add_argument('--num_SFEBs', type=int, default=2, help="number of CABs in PTM")
        parser.add_argument('--num_TPKFBs', type=int, default=2, help="number of CABs in PTM")
        parser.add_argument('--Dmodel',type=str,default='D')
        parser.add_argument('--one_hot_channel',type=int,default=8)
        parser.add_argument('--FSBlock_nums', type=int, default=0)
        parser.add_argument('--encoder_layer', type=int, default=3)
        parser.add_argument('--PDecay', action='store_true', default=False)

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--t_s_ratio', type=float, default=0.5, help='loss ratio between dual tasks')
        parser.add_argument('--dis_layers', type=int, default=4, help='number of layers in D')
        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.old_size = opt.old_size
        self.t_s_ratio = opt.t_s_ratio
        self.loss_names = ['app_gen_s', 'content_gen_s','G', 'style_gen_s', 'content_gen_face_s',
                           'content_gen_face_t','app_gen_t', 'ad_gen_t', 'dis_img_gen_t', 'content_gen_t', 'style_gen_t']
        self.model_names = ['G']
        self.decay = 0.3 if opt.PDecay else 0

        stepdict = {'0':1,'1':2,'2':3,'5':6,'d1':5,'d2':5,'d3':5,'d4':5,'d5':5,'6':7}
        self.step = stepdict[opt.step]
        self.net_G = networks.define_G(opt, image_nc=opt.image_nc, pose_nc=opt.structure_nc, ngf=64, img_f=512,
                                       encoder_layer=opt.encoder_layer, norm=opt.norm, activation='LeakyReLU',
                                       use_spect=opt.use_spect_g, use_coord=opt.use_coord, output_nc=3, num_blocks=3,
                                       affine=True, nhead=opt.nhead, num_SFEBs=opt.num_SFEBs, num_TPKFBs=opt.num_TPKFBs)
        if opt.dataset_mode == 'fashion':
            self.FACE_MISSING_VALUE = 0
        else:
            self.FACE_MISSING_VALUE = -1
        # Discriminator networ
        if self.isTrain:
            self.model_names = ['G', 'D']
            self.net_D = networks.define_D(opt, ndf=32, img_f=128, layers=opt.dis_layers, use_spect=opt.use_spect_d)
            self.net_D.train(),self.net_G.train()

        if self.opt.verbose:
                print('---------- Networks initialized -------------')
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            #self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            self.GANloss = external_function.GANLoss(opt.gan_mode).cuda()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function.VGGLoss().cuda()

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G.parameters())),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                lr=opt.lr * opt.ratio_g2d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            self.schedulers = [base_function.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        else:
            self.net_G.eval()

        if not self.isTrain or opt.continue_train:
            print('model resumed from latest')
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        if len(self.gpu_ids) > 0:
            self.imagelist = distributed_utils.to_cuda(input['imagelist'])
            self.seglist = distributed_utils.to_cuda(input['seglist'])
            self.poselist = distributed_utils.to_cuda(input['poselist'])
            self.maplist = distributed_utils.to_cuda(input['maplist'])
        else:
            self.imagelist = input['imagelist']
            self.seglist = input['seglist']
            self.poselist = input['poselist']
            self.maplist = input['maplist']

        self.image_paths = []
        for i in range(self.imagelist.size(0)):
            self.image_paths.append(input['Xs_path'][i] + ']' + input['Xt_path'][i])
    def sourceImageListDecay(self,epcoh):
        if (epcoh+1) % 30 == 0:
            self.decay -= 0.1
    def forward(self):
        fake_image_t_list = []
        fake_image_s_list = []
        fake_image_t_list.append(self.imagelist[:, 0, ...])
        fake_image_s_list.append(self.imagelist[:, 0, ...])
        for i in range(self.imagelist.shape[1]-1):
            p = np.random.uniform(low=0,high=1) + self.decay
            if self.opt.isTrain and p > 0.5:
                source_image_list = self.imagelist[:,:i+1,...].reshape(self.opt.batchSize,-1,self.opt.loadSize,self.opt.loadSize)
            else:
                source_image_list = torch.cat(fake_image_t_list[:i+1],dim=1)
            source_image_list = torch.cat([source_image_list,
                                           torch.zeros(self.opt.batchSize,(self.step*3)-source_image_list.shape[1],
                                            self.opt.loadSize,self.opt.loadSize).to(self.imagelist.device)],dim=1)
            fake_image_t,fake_image_s = self.net_G(source=self.imagelist[:, 0, ...], source_B=self.maplist[:, 0, ...],
                                                   target_B=self.maplist[:, i+1, ...],source_imagelist=source_image_list,
                                                   Xs_onehot=self.seglist[:, 0, ...],Xt_onehot=self.seglist[:, i+1, ...])
            fake_image_t_list.append(fake_image_t)
            fake_image_s_list.append(fake_image_s)

        fake_image_t = torch.stack(fake_image_t_list, dim=1)
        fake_image_s = torch.stack(fake_image_s_list, dim=1)

        self.visualBigGenImage = torch.cat([self.imagelist[0],fake_image_t[0]],dim=0)
        self.fake_image_t = fake_image_t[:, 1:, ...]
        self.fake_image_s = fake_image_s[:, 1:, ...]

    def process_loss(self,loss):
        if self.opt.distributed:
            return distributed_utils.reduce_value(loss)
        else:
            return loss

    def test(self):
        """Forward function used in test time"""
        fake_image_t_list = []
        fake_image_t_list.append(self.imagelist[:, 0, ...])
        for i in range(self.maplist.shape[1]-1):
            source_image_list = torch.cat(fake_image_t_list[:i + 1], dim=1)
            source_image_list = torch.cat([source_image_list,
                                           torch.zeros(self.opt.batchSize, (self.step*3) - source_image_list.shape[1],
                                                       self.opt.loadSize, self.opt.loadSize).cuda()], dim=1)
            fake_image_t, fake_image_s = self.net_G(source=self.imagelist[:, i, ...], source_B=self.maplist[:, i, ...],
                                                   target_B=self.maplist[:, i+1, ...],source_imagelist=source_image_list,
                                                   Xs_onehot=self.seglist[:, i, ...],Xt_onehot=self.seglist[:, i+1, ...])

            fake_image_t_list.append(fake_image_t)
        # save latesd image
        self.saveSingleImage(fake_image_t, self.image_paths[0])
        fake_image_t_list = torch.cat(fake_image_t_list, dim=0)
        # save Gt and generated images
        save_data = torch.cat([self.imagelist.squeeze(0),
                               fake_image_t_list], dim=0)
        self.saveImages(save_data, self.poselist[0][0], self.poselist[0][-1])
    def saveSingleImage(self,image,imagepath):
        image_numpy = util.tensor2im(image[0].data)
        os.makedirs(self.opt.results_dir[:-1]+'s',exist_ok=True)
        img_path = os.path.join(self.opt.results_dir[:-1]+'s', imagepath+'_fake_vis.bmp')#[:-6]+imagepath[-2:]
        util.save_image(image_numpy, img_path)
        print(imagepath)
    def saveImages(self,tensor,source_pose,target_pose):
        # source_pose = torch.stack([source_pose[:, 1], source_pose[:, 0]], dim=-1)
        # target_pose = torch.stack([target_pose[:, 1], target_pose[:, 0]], dim=-1)
        # source_pose = source_pose.detach().cpu().numpy()
        # target_pose = target_pose.detach().cpu().numpy()
        # source_poseimage,_ = pose_utils.draw_pose_from_cords(source_pose)
        # target_poseimage,_ = pose_utils.draw_pose_from_cords(target_pose)

        poselist = torch.stack([self.poselist[..., 1], self.poselist[..., 0]], dim=-1).squeeze(0).detach().cpu().numpy()
        imglist = []
        for pose in poselist:
            imglist.append(pose_utils.draw_pose_from_cords(pose)[0])
        imglist = np.concatenate(imglist, axis=1)
        results = []
        for i in range(7):
            img = torch.zeros((3, 256, 256)).cuda()
            for index, si in enumerate(self.seglist[0][i]):
                r, g, b = label_colours[index]
                img[0] += si * r
                img[1] += si * g
                img[2] += si * b
            results.append(img.permute(1,2,0))
        segtensor = torch.cat(results,dim=1).cpu().numpy()
        grid = torchutils.make_grid(tensor,nrow=self.step+1, padding=0, normalize=True)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # w = np.zeros((602, 572, 3))
        # w[173:429, 30:286, :] = source_poseimage
        # w[173:429, 316:, :] = target_poseimage
        ndarr = np.concatenate([imglist,segtensor, ndarr], axis=0).astype(np.uint8)
        im = Image.fromarray(ndarr)

        img_name = '%s.png' % self.image_paths[0]
        os.makedirs(self.opt.results_dir,exist_ok=True)
        img_path = os.path.join(self.opt.results_dir, img_name)
        im.save(img_path)

    def backward_D_basic(self, netD, real,fake):
        #resha
        real = real.reshape(-1,self.opt.output_nc,self.opt.loadSize,self.opt.loadSize)
        fake = fake.reshape(-1,self.opt.output_nc,self.opt.loadSize,self.opt.loadSize)
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty
        return D_loss

    def backward_D(self):
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen_t = self.backward_D_basic(self.net_D, self.imagelist[:, 1:, ...], self.fake_image_t)
        D_loss = self.process_loss(self.loss_dis_img_gen_t)
        D_loss.backward()
    def backward_G_basic(self, fake_image, target_image,kp, use_d):
        #reshape
        if len(fake_image.shape)==5:
            fake_image = fake_image.reshape(-1,self.opt.output_nc,self.opt.loadSize,self.opt.loadSize)
            target_image = target_image.reshape(-1,self.opt.output_nc,self.opt.loadSize,self.opt.loadSize)
        # Calculate reconstruction loss
        if target_image.shape[0] != fake_image.shape[0]:
            target_image = target_image.unsqueeze(1).repeat(1,self.step,1,1,1).reshape(*fake_image.shape)
        assert target_image.shape[0] == fake_image.shape[0]
        loss_app_gen = self.L1loss(fake_image, target_image)
        loss_app_gen = loss_app_gen * self.opt.lambda_rec  #self.opt.lambda_rec= 2.5

        # Calculate GAN loss
        loss_ad_gen = None
        if use_d:
            base_function._freeze(self.net_D)
            D_fake = self.net_D(fake_image)
            loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g  # 2

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(fake_image, target_image)
        loss_style_gen = loss_style_gen * self.opt.lambda_style
        loss_content_gen = loss_content_gen * self.opt.lambda_content
        if self.opt.face:
            face_location = self.get_face_location(self.poselist)
            loss_content_gen_face, loss_style_gen_face = self.Vggloss(face_util.crop_face_from_output(fake_image,face_location),
                                                            face_util.crop_face_from_output(target_image,face_location))
            loss_content_gen_face = loss_content_gen_face * self.opt.lambda_content_face
        else:

            loss_content_gen_face = torch.zeros(1)
        return loss_app_gen, loss_ad_gen, loss_style_gen, loss_content_gen,loss_content_gen_face
    def get_face_location(self,pose):
        pose_view = pose.view(-1, 18, 2)
        bl,_,_ = pose_view.shape # bs * length
        face_location = []
        for i in pose_view:
            if int(i[14, 0]) != self.FACE_MISSING_VALUE and int(i[15, 0]) != self.FACE_MISSING_VALUE:
                y0, x0 = i[14, 0:2]
                y1, x1 = i[15, 0:2]
                face_location.append(torch.tensor([y0, x0, y1, x1]).float())
            else:
                face_location.append(torch.tensor([self.FACE_MISSING_VALUE, self.FACE_MISSING_VALUE,
                                                   self.FACE_MISSING_VALUE, self.FACE_MISSING_VALUE]).float())
        return torch.stack(face_location).to(pose.device)
    def backward_G(self):
        base_function._unfreeze(self.net_D)

        self.loss_app_gen_t, self.loss_ad_gen_t, self.loss_style_gen_t, self.loss_content_gen_t,\
        self.loss_content_gen_face_t = \
            self.backward_G_basic(self.fake_image_t, self.imagelist[:, 1:, ...],kp=self.poselist[:, 1:, ...], use_d = True)

        self.loss_app_gen_s, self.loss_ad_gen_s, self.loss_style_gen_s, self.loss_content_gen_s, \
        self.loss_content_gen_face_s = \
            self.backward_G_basic(self.fake_image_s, self.imagelist[:, 0, ...], kp=self.poselist[:, 0, ...],use_d = False)

        loss_G = self.t_s_ratio*(self.loss_app_gen_t+self.loss_style_gen_t+self.loss_content_gen_t+self.loss_content_gen_face_t) \
                      + (1-self.t_s_ratio)*(self.loss_app_gen_s+self.loss_style_gen_s+self.loss_content_gen_s+self.loss_content_gen_face_s)+self.loss_ad_gen_t
        self.loss_G = self.process_loss(loss_G)
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

