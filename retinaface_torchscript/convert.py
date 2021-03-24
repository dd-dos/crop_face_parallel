import cv2
import torch
from model import RetinaNetDetector

model = RetinaNetDetector()
model.load_weights('models/mobilenet0.25_Final.pth')

img = cv2.imread('sample.jpg')
input_tensor = torch.from_numpy(img).to(torch.uint8)

with torch.no_grad():
    res = model.forward(input_tensor)
    print(res)
    print('=============================')

    # scripted_model = torch.jit.script(model.model)
    scripted_model = torch.jit.script(model)
    scripted_model.save('models/scripted_model.pt')

    res = scripted_model.forward(input_tensor)
    print(res)

    # print("Start trace")
    # traced_model = torch.jit.trace(model, img)
    # # print(traced_model.code)
    # print('Traced sucessfully')
    # traced_model.save('traced_model.zip')
