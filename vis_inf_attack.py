"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
from torch import autograd
from torchvision import transforms
#from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time

from loguru import logger
logger.add('record.txt')

import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.plots import Annotator
from utils.torch_utils import select_device
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
import cv2
import PIL
from utils.plots import Annotator, colors, save_one_box



class Yolov5interface:
    def __init__(self,weights,data):
        self.model = DetectMultiBackend(weights, device=select_device(""), dnn=False, data=data, fp16=False)
        self.model.eval()
        self.counter=0
    def dorun(self, im):
        im = im.float()
        return self.model(im, augment=False, visualize=False)

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        #self.yolov5=Yolov5interface()
        self.prob_extractor_yolov5 = yolov5_MaxProbExtractor(0, 80, self.config).cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.patch_applier = PatchApplier().cuda()
    def train(self):
        img_size = 640
        batch_size = 4
        n_epochs = 1000
        max_lab = 14
        time_str = time.strftime("%Y%m%d-%H%M%S")
        inf_weights="./weights/inf.pt"
        vis_weights="./weights/vis.pt"
        vis_inf_yaml="./weights/vis_inf.yaml"
        img_vis="./data_vis_inf/visible/val" #D:\flir-adv-mine\data_vis_inf\infrared\train
        img_inf="./data_vis_inf/infrared/val"
        label_vis="./data_vis_inf/infrared/val"
        label_inf="./data_vis_inf/infrared/val"
        train_loader = torch.utils.data.DataLoader(LLVIPdataset(img_vis,img_inf,label_vis, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)

        yolov5_vis=Yolov5interface(vis_weights,vis_inf_yaml)
        yolov5_inf=Yolov5interface(inf_weights,vis_inf_yaml)

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        adv_patch_grey = torch.full((1, self.config.patch_size, self.config.patch_size), 0.5)
        adv_patch_grey.requires_grad_(True)
        optimizer = optim.Adam([adv_patch_grey], lr=0.1 , amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (visimg_batch,infimg_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):

                    adv_patch_grey_two=adv_patch_grey.clone()
                    adv_patch_grey_three=adv_patch_grey.clone()
                    adv_patch_cpu=torch.cat((adv_patch_grey, adv_patch_grey_two, adv_patch_grey_three), dim=0)
                    visimg_batch = visimg_batch.cuda()
                    infimg_batch = infimg_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    adv_patch = adv_patch_cpu.cuda()

                    vis_adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    inf_adv_batch_t = self.patch_transformer(1-adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)

                    visp_img_batch = self.patch_applier(visimg_batch, vis_adv_batch_t)
                    infp_img_batch = self.patch_applier(infimg_batch, inf_adv_batch_t)

                    visp_img_batch = F.interpolate(visp_img_batch, (img_size, img_size))
                    infp_img_batch = F.interpolate(infp_img_batch, (img_size, img_size))

                    output_vis=yolov5_vis.dorun(visp_img_batch)
                    output_inf=yolov5_inf.dorun(infp_img_batch)
                    max_prob_vis=self.prob_extractor_yolov5(output_vis)
                    max_prob_inf=self.prob_extractor_yolov5(output_inf)


                    det_loss_vis = torch.mean(max_prob_vis)
                    det_loss_inf = torch.mean(max_prob_inf)
                    det_loss=(det_loss_vis+det_loss_inf)/2
                    loss = det_loss

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_loss += loss


                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del vis_adv_batch_t,inf_adv_batch_t, det_loss, visp_img_batch,infp_img_batch, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            print("generate once!")
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            if epoch%3==0:
                im.save(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('EPOCH TIME: ', et1-et0)
                logger.debug("epoch:{} detloss:{} ".format(epoch,ep_det_loss))
                del vis_adv_batch_t,inf_adv_batch_t,  det_loss, visp_img_batch,infp_img_batch, loss
                torch.cuda.empty_cache()
            et0 = time.time()

    def generate_patch(self, type):
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        if type == 'black':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.0)
        if type == 'white':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 1.0)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):

        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    trainer = PatchTrainer("paper_obj")
    trainer.train()

if __name__ == '__main__':
    main()