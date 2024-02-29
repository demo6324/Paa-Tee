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
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        #446
        # img_size = self.darknet_model.height

        img_size = 1280
        batch_size = self.config.batch_size
        batch_size = 1
        n_epochs = 1000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        img_black="./FLIR_data"

        label_balck="./FLIR_data/label"

        train_loader = torch.utils.data.DataLoader(InriaDataset(img_black,label_balck, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        adv_patch_grey = torch.full((1, self.config.patch_size, self.config.patch_size), 0.5)
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
                # with autograd.detect_anomaly():
                    #adv_patch_cpu = adv_patch_grey.repeat(3, 1, 1)
                    adv_patch_grey_two=adv_patch_grey.clone()
                    adv_patch_grey_three=adv_patch_grey.clone()
                    adv_patch_cpu=torch.cat((adv_patch_grey, adv_patch_grey_two, adv_patch_grey_three), dim=0)
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    ###################################################
                    #生成一堆ttt.png,补丁是随机旋转过的好像
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)



                    if epoch == 0:
                        p_img_batch = adv_batch_t.squeeze(0)
                        for i in range(0,14):
                            a = p_img_batch[i]
                            im = transforms.ToPILImage('RGB')(a)
                            im.save("./runs/"+str(i)+".png")









                        ############################################################################
                        # 把每张图上的每个人身上放一个补丁，如t2.png
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    ###########################################################################
                    # p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))


                    output=self.yolov5.dorun(p_img_batch)
                    #max_prob 是分类为人的最大的概率值
                    # max_prob = self.prob_extractor(output)
                    max_prob=self.prob_extractor_yolov5(output)
                    #计算可打印loss
                    nps = self.nps_calculator(adv_patch)
                    #计算tv_loss，为相邻像素点的欧式距离，表示图像的平滑程度，平滑变换的图像看上去显得比较真实，也能增加攻击的鲁棒性。
                    tv = self.total_variation(adv_patch)


                    nps_loss = nps*0.01
                    # tv_loss = tv*2.5
                    tv_loss = tv
                    det_loss = torch.mean(max_prob)
                    #det_Loss= det_loss*2
                    #loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    loss=det_loss

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss


                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            print("generate once!")
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            if epoch%10==0:
                im.save(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            # plt.imshow(im)
            # plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                logger.debug("epoch:{} detloss:{} npsloss:{} tvloss:{}".format(epoch,ep_det_loss,ep_nps_loss,ep_tv_loss))
                #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                #plt.imshow(im)
                #plt.show()
                #im.save("saved_patches/patchnew1.jpg")
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

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