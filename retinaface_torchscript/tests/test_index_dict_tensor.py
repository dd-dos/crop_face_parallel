import time
import torch

big_ass_tensor = torch.ones((10000, 4)).cuda()
big_ass_tensor_2 = torch.ones((10000, 4)).cuda()
torch.cuda.synchronize()

dict = {
    'abc': big_ass_tensor,
    'abd': big_ass_tensor_2
}
torch.cuda.synchronize()


st = time.time()
res = dict['abc']
torch.cuda.synchronize()
print("Index dict actually tooks:", (time.time()-st)*1000, "ms")
