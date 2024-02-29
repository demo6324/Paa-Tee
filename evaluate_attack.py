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

def get_name(path):

    folder_path = path

    # 列出目标文件夹中的所有文件
    files = os.listdir(folder_path)

    # 创建一个空的列表来存储图片文件的文件名
    image_files = []

    # 遍历目标文件夹中的所有文件
    for file in files:
        # 检查文件扩展名是否表示为图像文件（可以根据需要扩展这个列表）
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 将文件名添加到列表中
            image_files.append(file)

    # 打印所有图片文件的文件名
    return image_files


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        #self.yolov5=Yolov5interface()
        self.prob_extractor_yolov5 = yolov5_MaxProbExtractor(0, 80, self.config).cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.patch_applier = PatchApplier().cuda()
    def train(self):
        img_size = 1280
        batch_size = 1

        max_lab = 14
        img_path="./frame"
        label_path="./frame/labels"
        black_save="./result/"
        adv_patch="./pics/20240208-114101_ObjectOnlyPaper_18.png"
        origin_name=get_name(img_path)


        train_loader = torch.utils.data.DataLoader(InriaDataset(img_path,label_path, max_lab, img_size,
                                                                shuffle=False),
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=0)
        save_path=black_save
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        adv_patch_grey = self.read_image(adv_patch)


        for i_batch, (img_batch, lab_batch) in (enumerate(train_loader)):

                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                adv_patch = adv_patch_grey.cuda()

                adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)

                watch_patch = 1
                if watch_patch == 1:
                    patch = adv_batch_t[0][0]
                    im = transforms.ToPILImage('RGB')(patch)
                    im.save("patch_demo.png")

                p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))
                for i in range(0,batch_size):
                    img = torch.squeeze(p_img_batch[i], dim=0)
                    img = transforms.ToPILImage()(img.detach().cpu())
                    img.save(save_path+origin_name[i_batch*batch_size+i])


    def read_image(self, path):

        patch_img = Image.open(path).convert('RGB')
        #tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        #patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    trainer = PatchTrainer("paper_obj")
    trainer.train()

if __name__ == '__main__':
    main()