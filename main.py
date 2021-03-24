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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def miles_crop(img, bbox):
    width, height = img.size
    x1 = int(np.max([0, bbox[0]-width*1/3]))
    x2 = int(np.min([width, bbox[2]+width*1/3]))
    y1 = int(np.max([0, bbox[1]-height*1/2]))
    y2 = int(np.min([height, bbox[3]+height*1/3]))

    return img.crop((x1,y1,x2,y2))


# def crop_img(input_dir, output_dir, detector, folder_id):
#     img_search_path = os.path.join(input_dir, "*.jpg")
#     tensor_list = [torch.from_numpy(cv2.imread(path)).to(torch.uint8) for path in glob.glob(img_search_path)]
#     for idx, img_file in tqdm.tqdm(enumerate(glob.glob(img_search_path)), total=len(glob.glob(img_search_path))):
#         img = Image.open(img_file)
#         bboxes = detector.detect_from_image(np.array(img))
#         for box_id, bbox in enumerate(bboxes):
#             if bbox[-1] >= 0.95:
#                 magic_list = [1/4]
#                 for magic_id, magic in enumerate(magic_list):
#                     img_crop = custom_crop(img, bbox, ratio=magic)
#                     img_name = '{}.{}.{}.{}.jpg'.format(folder_id, idx, box_id, magic_id)
#                     img_save_path = os.path.join(output_dir, img_name)
#                     img_crop.save(img_save_path)

@torch.no_grad()    
def crop_img(input_dir, output_dir, detector, folder_id):
    img_search_path = os.path.join(input_dir, "*.jpg")

    img_path_list = [path for path in glob.glob(img_search_path)]
    num_batches = int(len(img_path_list)/64)

    for i in tqdm.tqdm(range(num_batches)):
        img_list = [cv2.imread(path) for path in img_path_list[i:64+i]]
        tensor_list = [torch.from_numpy(img).to(torch.uint8) for img in img_list]

        batch_res = detector.forward_batch(tensor_list)
        for res_id, res in enumerate(batch_res):
            bboxes = res[0]
            img = img_list[res_id]
            for bbox_id, bbox in enumerate(bboxes):
                if bbox[-1] >= 0.95:
                    img_crop = custom_crop(img, bbox)
                    img_name = '{}.{}.{}.{}.jpg'.format(folder_id, i, res_id, bbox_id)
                    img_save_path = os.path.join(output_dir, img_name)
                    img_crop.save(img_save_path)

    img_list = [cv2.imread(path) for path in img_path_list[num_batches*64:]]
    tensor_list = [torch.from_numpy(img).to(torch.uint8) for img in img_list]
    batch_res = detector.forward_batch(tensor_list)
    for res_id, res in enumerate(batch_res):
        bboxes = res[0]
        img = img_list[res_id]
        for bbox_id, bbox in enumerate(bboxes):
            if bbox[-1] >= 0.95:
                img_crop = custom_crop(img, bbox)
                img_name = '{}.{}.{}.{}.jpg'.format(folder_id, num_batches, res_id, bbox_id)
                img_save_path = os.path.join(output_dir, img_name)
                img_crop.save(img_save_path)


def task(img_folder, folder_id, detector):
    output_dir = OUTPUT

    live_folder = os.path.join(img_folder, "*/live")
    spoof_folder = os.path.join(img_folder, "*/spoof")

    print("==> Process live imgs:")
    crop_img(live_folder, os.path.join(output_dir, "live"), detector, folder_id)
    print("==> Process spoof imgs:")
    crop_img(spoof_folder, os.path.join(output_dir, "spoof"), detector, folder_id)


def main_process(input_dir, output_dir, folder_id, detector, parallel=False):
    global OUTPUT
    OUTPUT = output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "live"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "spoof"), exist_ok=True)

    if parallel:
        pool = Pool(multiprocessing.cpu_count())
        pool.map(func=task, iterable=glob.glob(folder_search_path))
        pool.close()
    else:
        task(input_dir, folder_id, detector)
    

if __name__=="__main__":
    os.makedirs("./cropped_face", exist_ok=True)
    # for folder_id, img_folder in enumerate(glob.glob("./celebA/sub_folder_*[!.tar.gz]")):
    detector = torch.jit.load("./retinaface_torchscript/model/scripted_model.pt")
    for folder_id, img_folder in enumerate(glob.glob("./celebA/sub_folder_*")):
        print("=> Process {}:".format(img_folder.split("/")[-1]))
        out_path = os.path.join("./cropped_face", img_folder.split("/")[-1])
        main_process(img_folder, out_path, folder_id, detector)
        print("====================================================")
    # main_process("../hello/sub_folder_0", "./test_crop_face_parallel")
    # model = torch.jit.load("./weight/scripted_model.pt")
    # img = [torch.from_numpy(cv2.imread("./sample.jpg")).to(torch.uint8)]
    # res = model.forward_batch(img)
    # import ipdb; ipdb.set_trace()