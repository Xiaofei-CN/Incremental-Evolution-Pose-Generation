import os
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from .PoseTrans import *
# from PoseTrans import *
# from PTM import PTM
import cv2
import math

###############################################################################
# Functions
###############################################################################

def define_G(opt, image_nc=3, pose_nc=18, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
             activation='ReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3, affine=True, nhead=2,
             num_SFEBs=2, num_TPKFBs=2):
    print(opt.model)
    print('build Generator %s' % opt.model)
    if opt.model in ['IEPG','IEPGN']:
        netG = PoseImageGenerator(image_nc, pose_nc, ngf, img_f, encoder_layer, norm, activation, use_spect,
                                use_coord, output_nc, num_blocks, affine, nhead, num_SFEBs, num_TPKFBs, opt.step,
                                opt.one_hot_channel,opt.FSBlock_nums,opt.use_CS,opt.use_skc)
    elif opt.model == 'GEC':
        netG = PoseGenerator(batch_size=opt.batchSize, use_ln=opt.use_ln, layers=opt.layers,istrain=opt.phase,datasetmode=opt.dataset_mode)
    elif opt.model == 'GECF':
        netG = PoseFGenerator(batch_size=opt.batchSize, use_ln=opt.use_ln, layers=opt.layers,istrain=opt.phase,datasetmode=opt.dataset_mode)
    elif opt.model == 'GECT':
        netG = PoseTransformer()
    else:
        raise ('generator not implemented!')
    return init_net(netG,opt, opt.init_type, opt.gpu_ids)


def define_D(opt, input_nc=3, ndf=64, img_f=1024, layers=3, norm='none', activation='LeakyReLU', use_spect=True, ):
    if opt.model in ['IEPG','IEPGN']:
        netD = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect)
    elif opt.model in ['GEC','GECT']:
        seqlen = 7 if opt.dataset_mode in ['tr','deepfashion'] else 8
        netD = SeqDiscriminator(seqlen=seqlen)
    elif opt.model == 'GECF':
        seqlen = 1#7 if opt.dataset_mode == 'tr' else 8
        netD = SeqDiscriminator(seqlen=seqlen)
    else:
        raise ('Discriminator not implemented!')
    return init_net(netD, opt,opt.init_type, opt.gpu_ids)


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# facemodel
##############################################################################
class InpaintingGenerator1(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, n_res=6, norm='in', activation='relu'):
        super(InpaintingGenerator1, self).__init__()

        self.encoder = nn.Sequential(  # input: [4, 256, 256]
            B.conv_block(in_nc, nf, 5, stride=1, padding=2, norm='none', activation=activation),  # [64, 256, 256]
            B.conv_block(nf, 2 * nf, 3, stride=2, padding=1, norm=norm, activation=activation),  # [128, 128, 128]
            B.conv_block(2 * nf, 2 * nf, 3, stride=1, padding=1, norm=norm, activation=activation),
            B.conv_block(2 * nf, 4 * nf, 3, stride=2, padding=1, norm=norm, activation=activation)# [128, 128, 128]
        )
        blocks = []
        for _ in range(n_res):
            block = B.ResBlock_new(4 * nf)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            B.conv_block(4 * nf,4 * nf, 3, stride=1, padding=1, norm=norm, activation=activation),  # [256, 64, 64]
            B.upconv_block(4 * nf, 2 * nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),
            # [128, 128, 128]
            B.upconv_block(2 * nf, nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),
            # [64, 256, 256]
            B.conv_block(nf, out_nc, 3, stride=1, padding=1, norm='none', activation='tanh')  # [3, 256, 256]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_nc, nf, norm='bn', activation='lrelu'):
        super(Discriminator, self).__init__()
        global_model = []
        global_model += [B.conv_block(in_nc, nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [64, 128, 128]
                         B.conv_block(nf, 2 * nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [128, 64, 64]
                         B.conv_block(2 * nf, 4 * nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [256, 32, 32]
                         B.conv_block(4 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [512, 16, 16]
                         B.conv_block(8 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [512, 8, 8]
                         B.conv_block(8 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm,
                                      activation=activation)]  # [512, 4, 4]
        self.global_model = nn.Sequential(*global_model)


        self.local_fea1 = B.conv_block(in_nc, nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [64, 64, 64]
        self.local_fea2 = B.conv_block(nf, 2 * nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [128, 32, 32]
        self.local_fea3 = B.conv_block(2 * nf, 4 * nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [256, 16, 16]
        self.local_fea4 = B.conv_block(4 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [512, 8, 8]
        self.local_fea5 = B.conv_block(8 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [512, 4, 4]

        self.global_classifier = nn.Linear(512 * 4 * 4, 512)
        self.local_classifier = nn.Linear(512 * 4 * 4, 512)
        self.classifier = nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(1024, 1))

    def forward(self, x_local, x_global):
        out_local_fea1 = self.local_fea1(x_local)
        out_local_fea2 = self.local_fea2(out_local_fea1)
        out_local_fea3 = self.local_fea3(out_local_fea2)
        out_local_fea4 = self.local_fea4(out_local_fea3)
        out_local_fea5 = self.local_fea5(out_local_fea4)
        out_local = out_local_fea5.view(out_local_fea5.size(0), -1)
        out_local = self.local_classifier(out_local)

        out_global = self.global_model(x_global)
        out_global = out_global.view(out_global.size(0), -1)
        out_global = self.global_classifier(out_global)

        out = torch.cat([out_local, out_global], dim=1)
        out = self.classifier(out)
        return out, [out_local_fea1, out_local_fea2, out_local_fea3, out_local_fea4, out_local_fea5]
##############################################################################
# Generator
##############################################################################

class SourceEncoder(nn.Module):
    """
    Source Image Encoder (En_s)
    :param image_nc: number of channels in input image
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param encoder_layer: encoder layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """

    def __init__(self, image_nc, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False):
        super(SourceEncoder, self).__init__()

        self.encoder_layer = encoder_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = image_nc

        self.block0 = EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                            nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.channelatten = ChannelAttention(in_planes=ngf)
        self.spatialatten = SpatialAttention()

    def forward(self, source):
        inputs = source
        out = self.block0(inputs)

        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out

class IncrementalEvolutionEncoder(nn.Module):
    """
    Incremental Evolution Encoder (En_s)
    :param image_nc: number of channels in input image
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param encoder_layer: encoder layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """

    def __init__(self, image_nc, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False,use_4=True,N=0,use_skc=False):
        super(IncrementalEvolutionEncoder, self).__init__()

        self.encoder_layer = encoder_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = image_nc

        self.block0 = nn.Sequential(IEBlockOptimized(input_nc, ngf, norm_layer,
                                            nonlinearity, use_spect, use_coord,use_4,use_skc),
                                    *nn.ModuleList([copy.deepcopy(
                                        SameIEBlock(ngf, ngf, norm_layer,
                                            nonlinearity, use_spect, use_coord,use_skc)) for i in range(N)])
                                    )
        mult = 1
        for i in range(encoder_layer - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = nn.Sequential(IEBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord,use_4,use_skc),
                                  *nn.ModuleList([copy.deepcopy(
                                      SameIEBlock(ngf * mult, ngf * mult, norm_layer,
                                                        nonlinearity, use_spect, use_coord,use_skc)) for i in range(N)])
                                  )
            setattr(self, 'encoder' + str(i), block)

    def forward(self, source):
        inputs = source
        out = self.block0(inputs)

        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out

class PoseImageGenerator(nn.Module):
    """

    :param image_nc: number of channels in input image
    :param pose_nc: number of channels in input pose
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    :param output_nc: number of channels in output image
    :param num_blocks: number of ResBlocks
    :param affine: affine in Pose Transformer Module
    :param nhead: number of heads in attention module
    :param num_SFEBs: number of CABs
    :param num_TPKFBs: number of TTBs
    """

    def __init__(self, image_nc=3, pose_nc=18, ngf=64, img_f=512, layers=3, norm='instance',
                 activation='LeakyReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3, affine=True, nhead=2,
                 num_SFEBs=2, num_TPKFBs=2, step='6', one_hot_channel=8,FSBlock_nums=0,use_CS=False,use_skc=False):
        super(PoseImageGenerator, self).__init__()

        use_4 = True
        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        input_nc = 2 * pose_nc + image_nc + one_hot_channel * 2

        # Encoder En_c
        self.block0 = EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                            nonlinearity, use_spect, use_coord,use_4,use_CS)

        mult = 1
        for i in range(self.layers - 1):  # self.layers = 3
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord,use_4,use_CS)
            setattr(self, 'encoder' + str(i), block)

        # self.channel2 = nn.Sequential(norm_layer(128),nonlinearity,nn.Conv2d(128,64,kernel_size=1))
        # self.channel1 = nn.Sequential(norm_layer(256),nonlinearity,nn.Conv2d(256,128,kernel_size=1))
        # self.channel0 = nn.Sequential(norm_layer(512),nonlinearity,nn.Conv2d(512,256,kernel_size=1))

        # ResBlocks
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            block = ResBlock(ngf * mult, ngf * mult, norm_layer=norm_layer,
                             nonlinearity=nonlinearity, use_spect=use_spect, use_coord=use_coord)
            setattr(self, 'mblock' + str(i), block)

        # Pose Transformer Module (PTM)
        self.PTM = PoseTrans(d_model=ngf * mult, nhead=nhead, num_SFEBs=num_SFEBs,
                       num_TPKFBs=num_TPKFBs, dim_feedforward=ngf * mult,
                       activation="LeakyReLU", affine=affine, norm=norm)

        stepdict = {'0':1,'1':2,'2':3,'5':6,'d1':5,'d2':5,'d3':5,'d4':5,'d5':5,'6':7}
        lenimg = stepdict[step]
        self.source_encoder = IncrementalEvolutionEncoder(image_nc * lenimg, ngf, img_f, layers, norm, activation, use_spect,
                                            use_coord,use_4,FSBlock_nums,use_skc)

        # Decoder
        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), img_f // ngf) if i != self.layers - 1 else 1
            up = ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord,use_CS=True)
            setattr(self, 'decoder' + str(i), up)
            cn = nn.Sequential(norm_layer(ngf * mult_prev * 2), nonlinearity, nn.Conv2d(ngf * mult_prev * 2, ngf * mult_prev, kernel_size=1))
            setattr(self, 'channel' + str(i), cn)

        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)


    def draw_featuremap(self, path, feature):
        assert feature.shape[0] == 1 and len(feature.shape) == 4
        os.makedirs(path, exist_ok=True)
        for k, i in enumerate(feature[0]):
            i = i.detach().cpu().numpy()
            cv2.imwrite(f'{path}/r_{k}.png', i * 255)

    def forward(self, source, source_B, target_B, source_imagelist, Xs_onehot, Xt_onehot, is_train=True):
        # Self-reconstruction Branch
        # Source-to-source Inputs
        input_s_s = torch.cat((source, source_B, source_B, Xs_onehot, Xs_onehot), 1)
        # Source-to-source Encoder
        FssList = []
        F_s_s = self.block0(input_s_s)
        FssList.append(F_s_s)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            F_s_s = model(F_s_s)
            FssList.append(F_s_s)
        # Source-to-source Resblocks
        for i in range(self.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_s_s = model(F_s_s)

        # Transformation Branch
        # Source-to-target Inputs
        input_s_t = torch.cat((source, source_B, target_B, Xs_onehot, Xt_onehot), 1)
        # Source-to-target Encoder
        FstList = []
        F_s_t = self.block0(input_s_t)
        FstList.append(F_s_t)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            F_s_t = model(F_s_t)
            FstList.append(F_s_t)

        # Source-to-target Resblocks
        for i in range(self.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_s_t = model(F_s_t)

        # Source Image Encoding
        F_s = self.source_encoder(source_imagelist)

        # Pose Transformer Module for Dual-task Correlation
        F_s_t = self.PTM(F_s_s, F_s_t, F_s)

        # Source-to-source Decoder (only for training)
        out_image_s = None
        if is_train:
            for i in range(self.layers):
                model = getattr(self, 'decoder' + str(i))
                # a = FssList.pop()
                # F_s_s = torch.cat([F_s_s,a],dim=1)
                F_s_s = getattr(self,'channel' + str(i))(torch.cat([F_s_s,FssList.pop()],dim=1))
                F_s_s = model(F_s_s)

            out_image_s = self.outconv(F_s_s)

        # Source-to-target Decoder
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            F_s_t = getattr(self, 'channel' + str(i))(torch.cat([F_s_t, FstList.pop()], dim=1))
            F_s_t = model(F_s_t)

        out_image_t = self.outconv(F_s_t)
        return out_image_t, out_image_s

class PoseGenerator(nn.Module):
    def __init__(self, batch_size=8, hidden_units=1024, layers=3, drop_prob=0.3, use_ln=True,
                 use_cuda=True,istrain=None,datasetmode=None,teachforce=True):
        super(PoseGenerator, self).__init__()
        self.hidden_dim = hidden_units
        self.input_dim = 128
        self.use_cuda = use_cuda
        self.layers = layers
        self.batch_size = batch_size
        self.istrain = True if istrain == 'train' else False
        self.init_hidden(batch_size,True)

        self.tf = False if datasetmode in ['deepfashion'] else teachforce
        self.Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.seqlen = 7 if datasetmode in ['tr','deepfashion'] else 8
        self.gen_embed_block = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            *[nn.LayerNorm(512) if use_ln else nn.Identity()],
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Linear(self.hidden_dim, 128)

        self.genlstm = nn.LSTM(input_size=512, hidden_size=self.hidden_dim, num_layers=self.layers,
                               dropout=drop_prob, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.act = nn.Tanh()
        self.apply(self.weight_init)
        self.class_name = self.__class__.__name__

    def forward(self, x):
        bs = x.shape[0]
        z = Variable(self.Tensor(np.random.normal(
            0, 1, (bs,self.seqlen-2, 128))))
        state = self.gen_state
        first = x[:, 0, :].unsqueeze(1)
        target = x[:, -1, :].unsqueeze(1)
        results = [first]
        data = torch.cat((first,z,target),dim=1)
        modelInputs = self.gen_embed_block(data)
        rnn_out,state = self.genlstm(modelInputs,state)
        genData = self.act(self.fc(rnn_out[:,1,:])).unsqueeze(1)
        result = self.out(genData)
        results.append(result)

        for i in range(self.seqlen-3):
            p = torch.rand(1) if self.istrain and self.tf else 0
            if p > 0.5:
                modelInputs = self.gen_embed_block(x[:,i+2,:]).unsqueeze(1)
                # data = x[:,:i+2,:]
            else:
                modelInputs = self.gen_embed_block(results[i+1])
                # data = torch.cat(results,dim=1)
            # data = torch.cat([data,z[:,i+1:,:],target],dim=1)
            # modelInputs = self.gen_embed_block(data)
            rnn_out, state = self.genlstm(modelInputs,state)
            genData = self.act(self.fc(rnn_out)) #[:, i+2, :]
            result = self.out(genData)#.unsqueeze(1)
            results.append(result)
        results.append(target)

        results = torch.cat(results,dim=1)
        return results

    def init_hidden(self, bs1,bidirectional=False):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        if bidirectional:
            layers = self.layers * 2
        else:
            layers = self.layers
        if self.use_cuda:
            self.gen_state = (nn.Parameter(torch.zeros(layers, bs1, self.hidden_dim)).cuda(),
                            nn.Parameter(torch.zeros(layers, bs1, self.hidden_dim)).cuda())
        else:
            self.gen_state = (nn.Parameter(torch.zeros(layers, bs1, self.hidden_dim)),
                              nn.Parameter(torch.zeros(layers, bs1, self.hidden_dim)))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param)
                else:
                    nn.init.zeros_(param)
class PoseFGenerator(nn.Module):
    def __init__(self, batch_size=8, hidden_units=1024, layers=3, drop_prob=0.3, use_ln=True,
                 use_cuda=True,istrain=None,datasetmode=None,teachforce=True):
        super(PoseFGenerator, self).__init__()
        self.hidden_dim = hidden_units
        self.input_dim = 128
        self.use_cuda = use_cuda
        self.layers = layers
        self.batch_size = batch_size
        self.istrain = True if istrain == 'train' else False
        self.init_hidden(batch_size,True)

        self.tf = False if datasetmode in ['deepfashion'] else teachforce
        self.Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.seqlen = 7 if datasetmode in ['tr','deepfashion'] else 8
        self.gen_embed_block = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            *[nn.LayerNorm(512) if use_ln else nn.Identity()],
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Linear(self.hidden_dim, 128)

        self.genlstm = nn.LSTM(input_size=512, hidden_size=self.hidden_dim, num_layers=self.layers,
                               dropout=drop_prob, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.act = nn.Tanh()
        self.apply(self.weight_init)
        self.class_name = self.__class__.__name__

    def forward(self, x):
        bs = x.shape[0]
        z = Variable(self.Tensor(np.random.normal(
            0, 1, (bs,self.seqlen-1, 128))))
        state = self.gen_state
        first = x[:, 0, :].unsqueeze(1)
        #target = x[:, -1, :].unsqueeze(1)
        results = [first]
        data = torch.cat((first,z),dim=1)
        modelInputs = self.gen_embed_block(data)
        rnn_out,state = self.genlstm(modelInputs,state)
        genData = self.act(self.fc(rnn_out[:,1,:])).unsqueeze(1)
        result = self.out(genData)
        results.append(result)

        for i in range(self.seqlen-2):
            p = torch.rand(1) if self.istrain and self.tf else 0
            if p > 0.5:
                modelInputs = self.gen_embed_block(x[:,i+2,:]).unsqueeze(1)
                # data = x[:,:i+2,:]
            else:
                modelInputs = self.gen_embed_block(results[i+1])
                # data = torch.cat(results,dim=1)
            # data = torch.cat([data,z[:,i+1:,:],target],dim=1)
            # modelInputs = self.gen_embed_block(data)
            rnn_out, state = self.genlstm(modelInputs,state)
            genData = self.act(self.fc(rnn_out)) #[:, i+2, :]
            result = self.out(genData)#.unsqueeze(1)
            results.append(result)

        results = torch.cat(results,dim=1)
        return results

    def init_hidden(self, bs1,bidirectional=False):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        if bidirectional:
            layers = self.layers * 2
        else:
            layers = self.layers
        if self.use_cuda:
            self.gen_state = (nn.Parameter(torch.zeros(layers, bs1, self.hidden_dim)).cuda(),
                            nn.Parameter(torch.zeros(layers, bs1, self.hidden_dim)).cuda())
        else:
            self.gen_state = (nn.Parameter(torch.zeros(layers, bs1, self.hidden_dim)),
                              nn.Parameter(torch.zeros(layers, bs1, self.hidden_dim)))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param)
                else:
                    nn.init.zeros_(param)

class PoseTransformer(nn.Module):
    def __init__(self, batch_size=8, hidden_units=1024, layers=3, drop_prob=0.3, use_ln=True,
                 use_cuda=True,istrain=None,datasetmode=None,teachforce=True):
        super(PoseTransformer, self).__init__()
        self.hidden_dim = hidden_units
        self.input_dim = 128
        self.use_cuda = use_cuda
        self.layers = layers
        self.batch_size = batch_size

        encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transencoder = nn.TransformerEncoder(encoder, num_layers=8)
        decoder = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transdecoder = nn.TransformerDecoder(decoder, num_layers=8)

        self.PE = PositionalEncoding(512,0.2)
        self.patchEmbeding = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256,512),
        )
        self.output = nn.Sequential(
            nn.Linear(512,128),
            nn.Tanh()
        )

        self.class_name = self.__class__.__name__
    def forward(self, x):
        x = self.patchEmbeding(x)
        fisrt = x[:,0,:].unsqueeze(1)
        end = x[:,-1,:].unsqueeze(1)
        zeros = torch.zeros((x.shape[0],5,x.shape[-1])).cuda()
        modelInput = torch.cat([fisrt, zeros[:, :5, :], end],dim=1)
        modelInput = self.PE(modelInput)
        memory = self.transencoder(modelInput)

        for i in range(5):
            modelInput = self.transdecoder(modelInput,memory)

        out = self.output(modelInput)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):


        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)
class Cle_Ae(nn.Module):
    def __init__(self, indim=36, latentdim=128):
        super(Cle_Ae, self).__init__()
        self.pose_shape = [18, 2]
        self.latentdim = latentdim
        self.hidden_dim = 512
        self.layers = 3
        self.encoder = nn.Sequential(
            nn.Linear(indim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(self.hidden_dim, latentdim),
            nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Linear(latentdim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(self.hidden_dim, indim),
            nn.Sigmoid())

    def encode(self, input):
        res = self.encoder(input)
        return res
    def decode(self, input):
        return self.decoder(input)

##############################################################################
# Discriminator
##############################################################################
class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """

    def __init__(self, input_nc=3, ndf=64, img_f=256, layers=3, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False):
        super(ResDiscriminator, self).__init__()

        self.layers = layers

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf, ndf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            block = ResBlockEncoder(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, use_spect,
                                    use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 1))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out

class SeqDiscriminator(nn.Module):
    def __init__(self,seqlen=7):
        super(SeqDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128*seqlen, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.apply(self.weight_init)
        self.class_name = self.__class__.__name__

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, data):
        inputs = data
        out = self.model(inputs)
        return out

