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
data = 'data/coco.yaml'
weights = 'weights/yolov5s.pt'
dnn = False
half = False
device = select_device("")
visualize = False
augment = False


conf_thres=0.5
iou_thres=0.45
classes=None
agnostic_nms=False
max_det=1000
line_thickness=3

class Yolov5interface:
    def __init__(self):
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.model.eval()
        self.counter=0
    def dorun(self, im):
        im = im.float()
        return self.model(im, augment=augment, visualize=visualize)

    def test(self, im,savedir):

        pred=self.dorun(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        names=self.model.names
        s=""
        # Process predictions

        for i, det in enumerate(pred):  # per image
            t_img=im[0].transpose(2, 0).transpose(0, 1).detach().cpu()
            t_img=t_img.numpy()
            t_img=np.ascontiguousarray(t_img)*255
            annotator = Annotator(t_img, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    # label = names[c]
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            PIL.Image.fromarray(np.uint8(im0)).save('{}/{}.png'.format(savedir,self.counter))
            # cv2.imwrite('{}/{}.png'.format(savedir,self.counter),im0)
            self.counter+=1
            print(self.counter)



if __name__ == '__main__':

    # 图像文件的路径
    from PIL import Image

    # 图像文件的路径
    image_path = '0.jpg'

    # 使用PIL打开图像
    image = Image.open(image_path)
    image = image.convert('RGB')
    y5 = Yolov5interface()
    im = np.expand_dims( image, axis=0)

    im = torch.from_numpy(im).to(device)
    pred = y5.dorun(im)

    print(1)
