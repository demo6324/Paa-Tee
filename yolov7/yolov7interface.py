
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import TracedModel
import torch
from yolov7.models.common import *
from yolov7.utils.torch_utils import select_device
visualize = False
augment = False
device = select_device("")
class Yolov7interface:
    def __init__(self,weights,data):
        imgsz = 640
        trace = True
        half = True
        model = attempt_load(weights, map_location=device)  # load FP32 model
        if trace:
            model = TracedModel(model, device, imgsz)
        if half:
            model.half()  # to FP16
        self.model.eval()
        self.counter=0

    def dorun(self, im):
        im = im.float()
        return self.model(im, augment=False)
