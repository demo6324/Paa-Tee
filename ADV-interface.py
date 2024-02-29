import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.torch_utils import select_device

data = 'data/coco128.yaml'
weights = 'weights/best_black.pt'
dnn = False
half = False
device = select_device("")
visualize = False
augment = False


class Yolov5interface:
    def __init__(self):
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

    def dorun(self, im):
        im = im.float()
        return self.model(im, augment=augment, visualize=visualize)



if __name__ == '__main__':
    y5 = Yolov5interface()
    # im=transforms.ToTensor()(im)

    im = np.random.randn(8, 3, 448, 448)
    im = torch.from_numpy(im).to(device)
    #
    # if len(im.shape) == 3:
    #     im = im[None]

    # im /= 255

    pred = y5.dorun(im)

    print(1)
