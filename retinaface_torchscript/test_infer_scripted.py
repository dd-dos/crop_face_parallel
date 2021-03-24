
import time

import cv2
import numpy as np
import torch
import torch.autograd.profiler as profiler
from tqdm import tqdm

# from model import RetinaNetDetector

# model = RetinaNetDetector()
# model.load_weights('models/mobilenet0.25_Final.pth')
scripted_model = torch.jit.load('models/scripted_model.pt')
model = scripted_model


# Must check if preprocess batch is correct compare to non batched version
# processed_batch_tensor = model.preprocess_batch([test_tensor], target_size=1920)[0][0]
# processed_single_tensor = model.preprocess(test_tensor_2)[0][0]
# print(torch.all(processed_batch_tensor == processed_single_tensor))
# # NOTE: WOW, the result is actually different. Let's see why ?
# # Fixed. It's good now

# First dummy run for warmup
for i in range(10):
    image = torch.rand((640, 480, 3)).to(torch.uint8)
    _ = model.forward(image)

# First dummy run for warmup batch
for i in range(3):
    images = []
    for _ in range(32):
        img = np.random.randint(0, 255, (640, 480, 3)).astype(np.uint8)
        test_tensor = torch.from_numpy(img).to(torch.uint8)
        images.append(test_tensor)
    _ = model.forward_batch(images)


print("***************************")

batch_size = 32
all_time = []
with torch.no_grad():
    # img = cv2.imread('sample.png')
    for _ in range(batch_size):
        img = np.random.randint(0, 255, (640, 480, 3)).astype(np.uint8)
        test_tensor = torch.from_numpy(img).to(torch.uint8)
        st = time.time()
        output = model.forward(test_tensor)
        taken = time.time() - st
        all_time.append(taken)
    print("Single batch tooks", np.mean(all_time)*1000, "ms")
    print("Total tooks", np.sum(all_time)*1000, "ms")
    # print(output)

    # img = cv2.imread('sample.png')
    # test_tensor_2 = torch.from_numpy(img).to(torch.uint8)

print('==================================')
batch_size = 32
with torch.no_grad():
    images = []
    for i in range(batch_size):
        img = np.random.randint(0, 255, (640, 480, 3)).astype(np.uint8)
        test_tensor = torch.from_numpy(img).to(torch.uint8)
        images.append(test_tensor)

    st = time.time()
    res = model.forward_batch(images, target_size=640)
    print("Batched", batch_size, "tooks", (time.time() - st)*1000, "ms")
    # print(res)

# image = torch.rand((640, 480, 3)).to(torch.uint8)
# _ = scripted_model.forward(image)

runs = 3000

# for _ in tqdm(range(runs), desc="Normal model"):
#     image = torch.rand((640, 480, 3)).to(torch.uint8)
#     _ = model.forward(image)

img = cv2.imread('sample.png')
for _ in tqdm(range(runs), desc="Scripted model"):
    # img = np.random.randint(0, 255, (640, 480, 3)).astype(np.uint8)
    # image = torch.rand((640, 480, 3)).to(torch.uint8)
    test_tensor = torch.from_numpy(img).to(torch.uint8)
    _ = model.forward(test_tensor)
