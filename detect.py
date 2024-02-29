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
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time
from YV5interface import Yolov5interface

from loguru import logger
logger.add('record.txt')



class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()


        ################################
        # self.darknet_model = Darknet(self.config.cfgfile)
        # self.darknet_model.load_weights(self.config.weightfile)
        # self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.yolov5=Yolov5interface()

        ##################################
        # self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.prob_extractor_yolov5 = yolov5_MaxProbExtractor(0, 80, self.config).cuda()

        ###################################################
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        #生成一个随机的patch
        self.patch_transformer = PatchTransformer().cuda()

        #贴图
        self.patch_applier = PatchApplier().cuda()


    def train(self):


        img_size = 1280
        batch_size = self.config.batch_size
        batch_size = 1
        n_epochs = 10000
        max_lab = 14


        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        self.epoch_length = len(train_loader)
        et0 = time.time()
        for epoch in range(n_epochs):
            ep_loss =0
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):

                    img_batch = img_batch.cuda()
                    output=self.yolov5.dorun(img_batch)
                    max_prob=self.prob_extractor_yolov5(output)
                    det_loss = torch.mean(max_prob)
                    loss = det_loss
                    ep_loss += loss

            ep_loss = ep_loss / len(train_loader)
            print("loss : " ,ep_loss)


    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
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
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():

  #  if len(sys.argv) != 2:
    #    print('You need to supply (only) a configuration mode.')
     #   print('Possible modes are:')
     #   print(len(sys.argv))
      #  print(patch_config.patch_configs)


    trainer = PatchTrainer("paper_obj")
    trainer.train()

if __name__ == '__main__':
    main()


