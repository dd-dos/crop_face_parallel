import os
import glob
import tqdm
import numpy as np
import shutil
from multiprocessing import Pool
import multiprocessing
import torch
import cv2
import torchvision

from PIL import Image
import time
import torch

def custom_crop(img, bbox, ratio=1/4):
    if not isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().detach().numpy()

    bb_width, bb_height = bbox[2]-bbox[0], bbox[3]-bbox[1]
    img_width, img_height = img.size

    x1 = int(np.max([0, bbox[0]-bb_width*ratio]))
    x2 = int(np.min([img_width, bbox[2]+bb_width*ratio]))
    y1 = int(np.max([0, bbox[1]-bb_height*ratio]))
    y2 = int(np.min([img_height, bbox[3]+bb_height*ratio]))

    return img.crop((x1,y1,x2,y2))


class Raw_CelebA_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size=448):
        img_list = glob.glob(os.path.join(root, "*/*/*/*.jpg"))
        self.dataset = []
        for img in img_list:
            if "live" in img:
                self.dataset.append([img, "live"])
            elif "spoof" in img:
                self.dataset.append([img, "spoof"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset[idx]
        img = torch.from_numpy(cv2.imread(image_path)).to(torch.uint8)
        return img, label, idx


def custom_collate(batch):
    imgs = []
    labels = []
    idxes = []
    for item in batch:
        imgs.append(item[0])
        labels.append(item[1])
        idxes.append(item[2])

    return [imgs, labels, idxes]


def task(bag):
    global OUTDIR
    bboxes, img, label, img_id = bag
    for bbox_id, bbox in enumerate(bboxes):
        if bbox[-1] >= 0.96:
            img_crop = custom_crop(img, bbox)
            img_name = '{}.{}.jpg'.format(img_id, bbox_id)
            if label=="live":
                outdir = os.path.join(OUTDIR, "live")
            elif label=="spoof":
                outdir = os.path.join(OUTDIR, "spoof")
            img_save_path = os.path.join(outdir, img_name)
            img_crop.save(img_save_path)

if __name__=="__main__":
    OUTDIR = "./cropped_face"
    shutil.rmtree(OUTDIR, ignore_errors=True)
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "live"), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "spoof"), exist_ok=True)

    dtset = Raw_CelebA_Dataset("./celebA")
    dataloader = torch.utils.data.DataLoader(dtset, batch_size=128, num_workers=4, collate_fn=custom_collate)
    with torch.no_grad():
        for idx, data in tqdm.tqdm(enumerate(dataloader), total=int(np.ceil(len(dtset)/128))):
            imgs, labels, imgs_id = data
            detector = torch.jit.load("./retinaface_torchscript/model/scripted_model.pt")
            batch_res = detector.forward_batch(imgs)
            bbox_list = [batch_res[i][0] for i in range(len(batch_res))]
            bbox_list = [[bbox.cpu().numpy() for bbox in bbox_list[i]] for i in range(len(bbox_list))]

            pool = Pool(multiprocessing.cpu_count())
            bag = [(bbox_list[i], imgs[i], labels[i], imgs_id[i]) for i in range(len(imgs))]
            pool.map(func=task, iterable=bag)
            pool.close()



