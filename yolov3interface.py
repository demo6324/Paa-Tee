from yolov3.models.common import DetectMultiBackend
import torch
from utils.torch_utils import select_device
visualize = False
augment = False
device = select_device("")
class Yolov3interface:
    def __init__(self,weights,data):
        self.model = DetectMultiBackend(weights, device=device, dnn="False", data=data, fp16=False)
        self.model.eval()
        self.counter=0
    def dorun(self, im):
        im = im.float()
        return self.model(im, augment=augment, visualize=visualize)
def prob_extractor_yolov3(output,batch_size):
    try :
        for i in range (batch_size):
            torch.max(output[i][:, :, 5]).item()

    except:
        pass