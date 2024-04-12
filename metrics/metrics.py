import os
import pathlib
import torch
import numpy as np
from imageio import imread
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import glob
import argparse
from PIL import Image
import tqdm
import lpips
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear')

        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

class FID():
    """docstring for FID
    Calculates the Frechet Inception Distance (FID) to evalulate GANs
    The FID metric calculates the distance between two distributions of images.
    Typically, we have summary statistics (mean & covariance matrix) of one
    of these distributions, while the 2nd distribution is given by a GAN.
    When run as a stand-alone program, it compares the distribution of
    images that are stored as PNG/JPEG at a specified location with a
    distribution given by summary statistics (in pickle format).
    The FID is calculated by assuming that X_1 and X_2 are the activations of
    the pool_3 layer of the inception net for generated samples and real world
    samples respectivly.
    See --help to see further details.
    Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
    of Tensorflow
    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    def __init__(self,datamode):
        self.dims = 2048
        self.batch_size = 64
        self.cuda = True
        self.verbose=False
        self.datamode = datamode

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx])
        if self.cuda:
            # TODO: put model into specific GPU
            self.model.cuda()

    def __call__(self, images, gt_path):
        """ images:  list of the generated image. The values must lie between 0 and 1.
            gt_path: the path of the ground truth images.  The values must lie between 0 and 1.
        """
        if not os.path.exists(gt_path):
            raise RuntimeError('Invalid path: %s' % gt_path)


        print('calculate gt_path statistics...')
        m1, s1 = self.compute_statistics_of_path(gt_path, self.verbose)
        print('calculate generated_images statistics...')
        m2, s2 = self.calculate_activation_statistics(images, self.verbose)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


    def calculate_from_disk(self, generated_path, gt_path):
        """
        """
        if not os.path.exists(gt_path):
            raise RuntimeError('Invalid path: %s' % gt_path)
        if not os.path.exists(generated_path):
            raise RuntimeError('Invalid path: %s' % generated_path)

        print('calculate gt_path statistics...')
        m1, s1 = self.compute_statistics_of_path(gt_path, self.verbose,True)
        print('calculate generated_path statistics...')
        m2, s2 = self.compute_statistics_of_path(generated_path, self.verbose,False)
        print('calculate frechet distance...')
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        print('fid_distance %f' % (fid_value))
        return fid_value


    def compute_statistics_of_path(self, path, verbose,duoceng):
        npz_file = os.path.join(path, 'statistics.npz')
        if os.path.exists(npz_file) and duoceng: # only gt data not need reading again
            f = np.load(npz_file)
            m, s = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            m, s = self.calculate_activation_statistics(path, verbose,duoceng)
            np.savez(npz_file, mu=m, sigma=s)
        return m, s

    def calculate_activation_statistics(self, path, verbose,duoceng=False):
        """Calculation of the statistics used by the FID.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the
                         number of calculated batches is reported.
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(path, verbose,duoceng)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma



    def get_activations(self, path, verbose=False,duoceng=False):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : the images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size depends
                         on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the number
                         of calculated batches is reported.
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        self.model.eval()

        # path = pathlib.Path(path)
        if duoceng:
            if self.datamode == 'fashion':
                filenames = list(glob.glob(path + '/*/*.jpg')) + list(glob.glob(path + '/*/*.png'))
                filenames = [f for f in filenames if not f.endswith('_gray.png') and not f.endswith('_vis.png')
                             and not f.endswith('_pose.png')]
            elif self.datamode == 'taichi':
                filenames = []
                for i in list(glob.glob(path + '/*')):
                    filenames += list(glob.glob(os.path.join(i, '*.npy')))[:14]
                filenames = [f[:-3] + 'png' for f in filenames]

            # filenames = [f for f in filenames if len(os.path.basename(f)) <= 5]
        else:
            filenames = list(glob.glob(path+'/*.jpg')) + list(glob.glob(path+'/*.png'))+ list(glob.glob(path+'/*.bmp'))
        # filenames = os.listdir(path)

        d0 = len(filenames)

        n_batches = d0 // self.batch_size
        n_used_imgs = n_batches * self.batch_size
        import tqdm
        pred_arr = np.empty((n_used_imgs, self.dims))
        for i in tqdm.tqdm(range(n_batches)):

            start = i * self.batch_size
            end = start + self.batch_size

            imgs = np.array([imread(str(fn)).astype(np.float32) for fn in filenames[start:end]])

            # Bring images to shape (B, 3, H, W)
            imgs = imgs.transpose((0, 3, 1, 2))

            # Rescale images to be between 0 and 1
            imgs /= 255

            batch = torch.from_numpy(imgs).type(torch.FloatTensor)
            # batch = Variable(batch, volatile=True)
            if self.cuda:
                batch = batch.cuda()

            pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.batch_size, -1)

        if verbose:
            print(' done')

        return pred_arr


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

def get_image_list(flist,duoceng):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            if duoceng:
                flist = list(glob.glob(flist + '/*/*.jpg')) + list(glob.glob(flist + '/*/*.png'))
            else:
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))+ list(glob.glob(flist + '/*.bmp'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list'%flist)
    return []


class LPIPS():
    def __init__(self, use_gpu=True):
        self.model =lpips.LPIPS(net='alex').cuda()
        self.use_gpu=use_gpu

    def __call__(self, image_1, image_2):
        """
            image_1: images with size (n, 3, w, h) with value [-1, 1]
            image_2: images with size (n, 3, w, h) with value [-1, 1]
        """
        result = self.model.forward(image_1, image_2)
        return result

    def calculate_from_disk(self, path_1, path_2, batch_size=64, verbose=False, sort=True):
        if sort:
            files_1 = sorted(get_image_list(path_1,False))
            files_2 = sorted(get_image_list(path_2,True))
        else:
            files_1 = get_image_list(path_1,False)
            files_2 = get_image_list(path_2,True)

        result=[]

        d0 = len(files_1)
        assert len(files_1)  == len(files_2)
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0

        n_batches = d0 // batch_size
        n_used_imgs = n_batches * batch_size

        for i in tqdm.tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
                # end='', flush=True)
            start = i * batch_size
            end = start + batch_size
            imgs_1 = np.array([imread(str(fn)).astype(np.float32) / 127.5 - 1 for fn in files_1[start:end]])
            imgs_2 = np.array([imread(str(fn)).astype(np.float32) / 127.5 - 1 for fn in files_2[start:end]])

            # Bring images to shape (B, 3, H, W)
            imgs_1 = imgs_1.transpose((0, 3, 1, 2))
            imgs_2 = imgs_2.transpose((0, 3, 1, 2))
            img_1_batch = torch.from_numpy(imgs_1).type(torch.FloatTensor)
            img_2_batch = torch.from_numpy(imgs_2).type(torch.FloatTensor)

            if self.use_gpu:
                img_1_batch = img_1_batch.cuda()
                img_2_batch = img_2_batch.cuda()
            result.append(self.model.forward(img_1_batch, img_2_batch).detach().cpu().numpy())
        distance = np.average(result)
        print('lpips: %.4f'%distance)
        return distance

def load_image_paths(generated, gt_path,datasetmode='taichi'):

    distorted_image_list = sorted(get_image_list(generated, False))
    distorated_list = []
    gt_list = []
    for distorted_image in distorted_image_list:
        image = os.path.basename(distorted_image)

        if datasetmode =='taichi':
            image = image[:-4].split(']')
            dir = image[0]
            filelist = [da for da in os.listdir(os.path.join(gt_path,dir)) if da.endswith('.npy')]
            filelist = sorted(filelist)
            if eval(image[-1].strip('_fake_vis')): #.strip('_fake_vis')
                filelist = filelist[1::2]
            else:
                filelist = filelist[::2]
            image = filelist[7][:-4] + '.png'
        else:
            image = image[:-4].strip('_fake_vis').split('_2_')
            dir = image[0][:-2]
            image = image[-1][0] + '.png'
            # image = image[-1] + '.png'
        gt_image = os.path.join(gt_path, dir, image)
        if not os.path.isfile(gt_image):
            print("hhhhhhhhh")
            print(gt_image)
            continue
        gt_list.append(gt_image)
        distorated_list.append(distorted_image)
    return distorated_list,gt_list

def ssimAndPsnr(generated, gt_path):
    ssim_list,psnr_list = [],[]
    print("start reading images")
    for ge,gt in tqdm.tqdm(zip(generated,gt_path)):
        generated_images = imread(ge).astype(np.float32) / 255.0
        target_images = imread(gt).astype(np.float32) / 255.0
        ssim = compare_ssim(target_images,generated_images,data_range=1,
                            win_size=51,multichannel=True)
        psnr = compare_psnr(target_images,generated_images,data_range=1)
        # ssim = compare_ssim(target_images, generated_images, gaussian_weights=True, sigma=1.5,
        #                     use_sample_covariance=False, multichannel=True,
        #                     data_range=generated_images.max() - generated_images.min())
        # psnr = compare_psnr(target_images, generated_images)
        ssim_list.append(ssim)
        psnr_list.append(psnr)

    print("Finish reading images")

    return np.mean(ssim_list), np.mean(psnr_list)

def metrics(fake_image_path,fid_real_path,Gt_path,dataset_mode='fashion',save_file=None):
    print(dataset_mode)
    print('load start')

    lpips = LPIPS()
    print('load LPIPS')

    fid = FID(datamode=dataset_mode)
    print('load FID')

    print('calculate fid metric...')
    fid_score = fid.calculate_from_disk(fake_image_path, fid_real_path,)

    print('calculate lpips metric...')
    distorated_path, gt_path = load_image_paths(fake_image_path, Gt_path,dataset_mode)

    lpips_score = lpips.calculate_from_disk(distorated_path, gt_path, sort=False)

    ssim_result, psnr_result = ssimAndPsnr(distorated_path, gt_path)

    resl = "SSIM score: %s \nPSNR score: %s\n FID score: %s \nLPIPS score: %s" % (ssim_result,psnr_result,fid_score,lpips_score)
    print(resl)
    if save_file is not None:
        assert os.path.exists(save_file)
        name = os.path.basename(fake_image_path)
        reslpath = os.path.join(save_file,'%s.txt'%name)
        with open(reslpath,'w') as f:
            f.write(resl)


if __name__ == "__main__":
    # fid_real_path = r'/home/xtf/data/fashionvideo/traindata'
    # Gt_path = r'/home/xtf/data/fashionvideo/testdata'
    # fake_image_path = r'../results/IEPG_fashionface70_s'
    fid_real_path = r'E:\BaiduNetdiskDownload\fashionvideo\traindata'
    Gt_path = r'E:\BaiduNetdiskDownload\fashionvideo\testdata'
    fake_image_path = r'E:\paper_result\OUR\images result\DPTNASUSKAD_fashionvideo_L_s'
    metrics(fake_image_path=fake_image_path,
            fid_real_path=fid_real_path,
            Gt_path=Gt_path)
