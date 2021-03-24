import cv2
from model import RetinaNetDetector


model = RetinaNetDetector(model_path='mobilenet0.25_Final.pth')

sample_image_path = 'sample.png'
image = cv2.imread(sample_image_path)

res = model.predict(image)
print(res)
