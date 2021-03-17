import os
import glob
import tqdm
import numpy as np
import shutil
from multiprocessing import Pool
import multiprocessing

from PIL import Image
from retinaface.retina_detector import RetinaDetector
import time

def custom_crop(img, bbox, ratio=1/3):
    width, height = img.size
    x1 = int(np.max([0, bbox[0]-width*ratio]))
    x2 = int(np.min([width, bbox[2]+width*ratio]))
    y1 = int(np.max([0, bbox[1]-height*ratio]))
    y2 = int(np.min([height, bbox[3]+height*ratio]))

    return img.crop((x1,y1,x2,y2))


def miles_crop(img, bbox):
    width, height = img.size
    x1 = int(np.max([0, bbox[0]-width*1/3]))
    x2 = int(np.min([width, bbox[2]+width*1/3]))
    y1 = int(np.max([0, bbox[1]-height*1/2]))
    y2 = int(np.min([height, bbox[3]+height*1/3]))

    return img.crop((x1,y1,x2,y2))


def crop_img(input_dir, output_dir, detector):
    img_search_path = os.path.join(input_dir, "*.jpg")
    for idx, img_file in enumerate(glob.glob(img_search_path)):
        img = Image.open(img_file)
        bboxes = detector.detect_from_image(np.array(img))
        for box_id, bbox in enumerate(bboxes):
            if bbox[-1] >= 0.8:
                magic_list = [1/4, 1/14]
                for magic_id, magic in enumerate(magic_list):
                    img_crop = custom_crop(img, bbox, ratio=magic)
                    img_name = input_dir.split("/")[-1] + '_{}.{}.{}.jpg'.format(idx, box_id, magic_id)
                    img_save_path = os.path.join(output_dir, img_name)
                    img_crop.save(img_save_path)


def task(img_folder):
    output_dir = "./crop_faces_liveness"
    detector = RetinaDetector(device='cpu', 
                            path_to_detector="/home/pdd/Downloads/dataset/retinaface/weights/mobilenet0.25_Final.pth",
                            mobilenet_pretrained="/home/pdd/Downloads/dataset/retinaface/weights/mobilenetV1X0.25_pretrain.tar")
    
    live_folder = os.path.join(img_folder, "live")
    spoof_folder = os.path.join(img_folder, "spoof")

    crop_img(live_folder, os.path.join(output_dir, "live"), detector)
    crop_img(spoof_folder, os.path.join(output_dir, "spoof"), detector)


def main_process(input_dir, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "live"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "spoof"), exist_ok=True)

    folder_search_path = os.path.join(input_dir, "*") 

    print("Processing...")
    
    pool = Pool(multiprocessing.cpu_count())
    pool.map(func=task, iterable=glob.glob(folder_search_path))
    

if __name__=="__main__":
    main_process("./CelebA_sample", "./crop_faces_liveness")