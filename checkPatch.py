import matplotlib.pyplot as plt
from torchvision import transforms

import patch_config
from YV5interface import Yolov5interface
from load_data import *
from utils import *

# from utilsadv import get_region_boxes,nms
import utilsadv
config = patch_config.patch_configs['paper_obj']()
#darknet_model = Darknet(config.cfgfile)
#darknet_model.load_weights(config.weightfile)
#darknet_model = darknet_model.eval().cuda()

patch_applier = PatchApplier().cuda()
patch_transformer = PatchTransformer().cuda()
prob_extractor_yolov5 = yolov5_MaxProbExtractor(0, 80, config).cuda()

yolov5 = Yolov5interface()
yolov5_ori = Yolov5interface()

patch_size = 300
# img_size = darknet_model.height
img_size = 1280

count=0

img_dir_v = "inria/Train/pos"
lab_dir_v = "inria/Train/pos/yolo-labels"


adv_patch = Image.open("1.png").convert('RGB')
adv_patch = adv_patch.resize((300, 300))

transform = transforms.ToTensor()

adv_patch = transform(adv_patch).cuda()

train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir_v, lab_dir_v, 14, img_size, shuffle=True),
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=10)

for i_batch, (img_batch, lab_batch) in enumerate(train_loader):

    count = count + 1

    # yolov5_ori.test(img_batch.cuda(), savedir='ori')

    img_size = img_batch.size(-1)
    adv_batch_t = patch_transformer(adv_patch, lab_batch.cuda(), img_size, do_rotate=True, rand_loc=False)
    p_img = patch_applier(img_batch.cuda(), adv_batch_t)
    # p_img = F.interpolate(p_img, (darknet_model.height, darknet_model.width))
    p_img = F.interpolate(p_img, (img_size, img_size))

    # p_img = transforms.ToPILImage('RGB')(p_img)

    # p_img.save("exp/{}.png".format(count))

    yolov5.test(p_img, savedir='exp')
    # output = darknet_model(p_img)
    #
    #
    # boxes = utilsadv.get_region_boxes(output, 0.5, darknet_model.num_classes,
    #                          darknet_model.anchors, darknet_model.num_anchors)[0]
    #
    # boxes = utilsadv.nms(boxes, 0.4)
    # class_names = utilsadv.load_class_names('data/coco.names')
    # squeezed = p_img.squeeze(0)
    # print(squeezed.shape)
    # img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
    # plotted_image = utilsadv.plot_boxes(img, boxes, class_names=class_names)
    # plt.imshow(plotted_image)
    # plt.show()
    # print(1)

'''
transforms.ToPILImage()(p_img[0].detach().cpu()).save('/show1.png')
# apply an image as patch
patch_size = adv_patch.size(-1)
horse = Image.open("data/horse.jpg").convert('RGB')
tf = transforms.Resize((patch_size,patch_size))
horse = tf(horse)
transform = transforms.ToTensor()
horse = transform(horse)

adv_batch_t = patch_transformer(horse.cuda(), label.cuda(), img_size)
p_img = patch_applier(image.cuda(), adv_batch_t)
p_img = F.interpolate(p_img,(darknet_model.height, darknet_model.width))
output = darknet_model(p_img)
boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
                         darknet_model.anchors, darknet_model.num_anchors)[0]
boxes = nms(boxes,0.4)
class_names = load_class_names('data/coco.names')
squeezed = p_img.squeeze(0)
im = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
plotted_image = plot_boxes(im, boxes, class_names=class_names)
plt.imshow(plotted_image)
plt.show()
'''
