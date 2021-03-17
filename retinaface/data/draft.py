import glob
import os
search_path = "widerface/*/*/*.jpg"
for idx, img_path in enumerate(glob.glob(search_path)):
    os.rename(img_path, '/home/pdd/Desktop/workspace/Pytorch_Retinaface/data/widerface/val/images/{}.jpg'.format(idx))