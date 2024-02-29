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
from yolov3.utils.general import non_max_suppression
#from tensorboardX import SummaryWriter
import subprocess
from models.common import DetectMultiBackend
from yolov7.utils.torch_utils import select_device
import patch_config
import sys
import time
import torch
torch.autograd.set_detect_anomaly(True)
from ultralytics import YOLO

from  Yolov8interface import Yolov8interface,yolov8_MaxProbExtractor
from loguru import logger
logger.add('record.txt')


class Yolov5interface:
    def __init__(self,weights,data):
        self.model = DetectMultiBackend(weights, device=select_device(""), dnn=False, data=data, fp16=False)
        self.model.eval()
        self.counter=0
    def dorun(self, im):
        im = im.float()
        return self.model(im, augment=False, visualize=False)

from yolov3interface import Yolov3interface




class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.prob_extractor_yolov5 = yolov5_MaxProbExtractor(0, 80, self.config).cuda()
        self.prob_extractor_yolov3 = MaxProbExtractor(0, 80, self.config).cuda()
        self.prob_extractor_yolov8 = yolov8_MaxProbExtractor(0, 80, self.config).cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.patch_applier = PatchApplier().cuda()


    def train(self):

        img_size = 1280
        batch_size = 4
        n_epochs = 1000
        max_lab = 14
        time_str = time.strftime("%Y%m%d-%H%M%S")
        img_path="./frame"
        label_path="./frame/labels"
        train_loader = torch.utils.data.DataLoader(InriaDataset(img_path,label_path, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        self.epoch_length = len(train_loader)

        #yolov3 = Yolov3interface("./weights/yolov3_best.pt", 'data/coco.yaml')
        yolov5 = Yolov5interface("./weights/mine_best.pt", 'data/coco.yaml')
        #yolov8 = Yolov8interface("yolov8n.pt")

        print(f'One epoch is {len(train_loader)}')

        adv_patch_grey = torch.full((1, 4, 4), 0.5)
        adv_patch_grey.requires_grad_(True)
        optimizer = optim.Adam([adv_patch_grey], lr=0.1 , amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)


        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):

                    adv_patch_grey_one = adv_patch_grey.unsqueeze(0).float()
                    adv_patch_grey_one = F.interpolate(adv_patch_grey_one, (300, 300))
                    adv_patch_grey_one=adv_patch_grey_one.squeeze(0)
                    adv_patch_grey_two=adv_patch_grey_one.clone()
                    adv_patch_grey_three=adv_patch_grey_one.clone()
                    adv_patch_cpu=torch.cat((adv_patch_grey_one, adv_patch_grey_two, adv_patch_grey_three), dim=0)


                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    adv_patch = adv_patch_cpu.cuda()
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)

                    watch_patch=0
                    if watch_patch==1:
                        patch = adv_batch_t[0][0]
                        im = transforms.ToPILImage('RGB')(patch)
                        im.save("patch_demo.png")

                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))


                    #output = yolov8.dorun(p_img_batch)
                    #max_prob = torch.max(output[0][:, 4, :], dim=1).values
                    #det_loss = torch.mean(max_prob)

                    output = yolov5.dorun(p_img_batch)
                    max_prob = self.prob_extractor_yolov5(output)
                    det_loss = torch.mean(max_prob)

                    #output = yolov3.dorun(p_img_batch)
                    #max_prob = torch.max(output[0][:, :, 4], dim=1).values
                    #det_loss = torch.mean(max_prob)

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    loss=det_loss
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)

                    bt1 = time.time()
                    if i_batch % 3 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            print("generate once!")
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            if epoch % 3 == 0:
                im.save(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('EPOCH TIME: ', et1 - et0)
                logger.debug("epoch:{} detloss:{} ".format(epoch, ep_det_loss))
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, loss
                torch.cuda.empty_cache()
            et0 = time.time()

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