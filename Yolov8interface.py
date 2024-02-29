# code_name
# 作者：赵海淇
# 时间
import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.ops import (LOGGER, non_max_suppression,  scale_coords, xyxy2xywh)
import cv2
import PIL
from ultralytics.utils.plotting import Annotator, colors, save_one_box
# model = YOLO("yolov8.yaml").load('yolov8n.pt')
data = 'ultralytics/cfg/datasets/coco128.yaml'
weights = 'yolov8n.pt'
dnn = False
half = False
device = select_device("")
visualize = False
augment = False
import torch.nn as nn

conf_thres=0.5
iou_thres=0.45
classes=None
agnostic_nms=False
max_det=300
line_thickness=3

class Yolov8interface:
    def __init__(self,weights):
        self.model = AutoBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.model.eval()
        self.counter = 0

    def dorun(self, im):
        im = im.float()

        return self.model(im, augment=augment, visualize=visualize)






class yolov8_MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(yolov8_MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        # self.loss_target = loss_target

    def forward(self, YOLOoutput):

        output = YOLOoutput[0]  #(4,84,8400)
        output_objectness = torch.sigmoid(output[:, 4, :])
        output = output[:,3:3+self.num_cls,:]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return  max_conf

