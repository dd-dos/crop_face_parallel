import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

img = Image.open('sample.png')
img = ToTensor()(img)
img = img.unsqueeze(0)
print(img.shape)
out = F.interpolate(img, size=(720, 128))  # The resize operation on tensor.
print(out.shape)
out = out[0]
ToPILImage()(out).save('test.png', mode='png')