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


# def miles_crop(img, bbox):
#     width, height = img.size
#     x1 = int(np.max([0, bbox[0]-width*1/3]))
#     x2 = int(np.min([width, bbox[2]+width*1/3]))
#     y1 = int(np.max([0, bbox[1]-height*1/2]))
#     y2 = int(np.min([height, bbox[3]+height*1/3]))

#     return img.crop((x1,y1,x2,y2))



# @torch.no_grad()    
# def crop_img(input_dir, OUTDIR, detector, folder_id):
#     img_search_path = os.path.join(input_dir, "*.jpg")

#     img_path_list = [path for path in glob.glob(img_search_path)]
#     num_batches = int(len(img_path_list)/128)

#     for i in tqdm.tqdm(range(num_batches)):
#         img_list = [cv2.imread(path) for path in img_path_list[i:128+i]]
#         tensor_list = [torch.from_numpy(img).to(torch.uint8) for img in img_list]

#         batch_res = detector.forward_batch(tensor_list)
#         for res_id, res in enumerate(batch_res):
#             bboxes = res[0]
#             img = img_list[res_id]
#             for bbox_id, bbox in enumerate(bboxes):
#                 if bbox[-1] >= 0.95:
#                     img_crop = custom_crop(img, bbox)
#                     img_name = '{}.{}.{}.{}.jpg'.format(folder_id, i, res_id, bbox_id)
#                     img_save_path = os.path.join(OUTDIR, img_name)
#                     img_crop.save(img_save_path)

#     img_list = [cv2.imread(path) for path in img_path_list[num_batches*128:]]
#     tensor_list = [torch.from_numpy(img).to(torch.uint8) for img in img_list]
#     batch_res = detector.forward_batch(tensor_list)
#     for res_id, res in enumerate(batch_res):
#         bboxes = res[0]
#         img = img_list[res_id]
#         for bbox_id, bbox in enumerate(bboxes):
#             if bbox[-1] >= 0.95:
#                 img_crop = custom_crop(img, bbox)
#                 img_name = '{}.{}.{}.{}.jpg'.format(folder_id, num_batches, res_id, bbox_id)
#                 img_save_path = os.path.join(OUTDIR, img_name)
#                 img_crop.save(img_save_path)


# def task(img_folder, folder_id, detector):
#     OUTDIR = OUTPUT

#     live_folder = os.path.join(img_folder, "*/live")
#     spoof_folder = os.path.join(img_folder, "*/spoof")

#     print("==> Process live imgs:")
#     crop_img(live_folder, os.path.join(OUTDIR, "live"), detector, folder_id)
#     print("==> Process spoof imgs:")
#     crop_img(spoof_folder, os.path.join(OUTDIR, "spoof"), detector, folder_id)


# def main_process(input_dir, OUTDIR, folder_id, detector, parallel=False):
#     global OUTPUT
#     OUTPUT = OUTDIR
#     os.makedirs(OUTDIR, exist_ok=True)
#     os.makedirs(os.path.join(OUTDIR, "live"), exist_ok=True)
#     os.makedirs(os.path.join(OUTDIR, "spoof"), exist_ok=True)

#     if parallel:
#         pool = Pool(multiprocessing.cpu_count())
#         pool.map(func=task, iterable=glob.glob(folder_search_path))
#         pool.close()
#     else:
#         task(input_dir, folder_id, detector)
    

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
    detech_res, img, label, img_id = bag
    bboxes = detech_res[0]
    for bbox_id, bbox in enumerate(bboxes):
        if bbox[-1] >= 0.95:
            img_crop = custom_crop(img, bbox)
            img_name = '{}.{}.jpg'.format(img_id, bbox_id)
            if label=="live":
                outdir = os.path.join(OUTDIR, "live")
            elif label=="spoof":
                outdir = os.path.join(OUTDIR, "spoof")
            img_save_path = os.path.join(outdir, img_name)
            img_crop.save(img_save_path)

if __name__=="__main__":
    # os.makedirs("./cropped_face", exist_ok=True)
    # # for folder_id, img_folder in enumerate(glob.glob("./celebA/sub_folder_*[!.tar.gz]")):
    # detector = torch.jit.load("./retinaface_torchscript/model/scripted_model.pt")
    # for folder_id, img_folder in enumerate(glob.glob("./celebA/sub_folder_*")):
    #     print("=> Process {}:".format(img_folder.split("/")[-1]))
    #     out_path = os.path.join("./cropped_face", img_folder.split("/")[-1])
    #     main_process(img_folder, out_path, folder_id, detector)
    #     print("====================================================")
    # main_process("../hello/sub_folder_0", "./test_crop_face_parallel")
    # model = torch.jit.load("./weight/scripted_model.pt")
    # img = [torch.from_numpy(cv2.imread("./sample.jpg")).to(torch.uint8)]
    # res = model.forward_batch(img)
    # import ipdb; ipdb.set_trace()
    OUTDIR = "./cropped_face"
    shutil.rmtree(OUTDIR, ignore_errors=True)
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "live"), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "spoof"), exist_ok=True)

    dtset = Raw_CelebA_Dataset("./celebA")
    dataloader = torch.utils.data.DataLoader(dtset, batch_size=128, num_workers=8, collate_fn=custom_collate)
    for idx, data in tqdm.tqdm(enumerate(dataloader), total=int(np.ceil(len(dtset)/128))):
        imgs, labels, imgs_id = data
        detector = torch.jit.load("./retinaface_torchscript/model/scripted_model.pt")
        with torch.no_grad():
            batch_res = detector.forward_batch(imgs)

        # pool = Pool(multiprocessing.cpu_count())
        pool = Pool(multiprocessing.cpu_count())
        bag = [[batch_res[i], imgs[i], labels[i], imgs_id[i]] for i in range(len(imgs))]
        pool.map(func=task, iterable=bag)
        pool.close()



