import torch

batch_size = 2
score = torch.tensor([[0.1, 0.9, 0.2],
                  [0.4, 0.2, 0.3]])

box = torch.tensor([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]],
                  [[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]]])

print("score", score.shape)
print("box", box.shape)
idx = torch.argsort(score, -1, descending=True)
print(idx)

sorted_score = score.gather(1, idx)
print(sorted_score)

print('==============')

def apply_indices_2d_to_3d(idx, _3d_tensor):
    nLast3Dim, nLast2Dim, nLastDim = _3d_tensor.shape[-3:]
    lastDimCounter = torch.arange(0, nLastDim, dtype=torch.long)
    last3DimCounter = torch.arange(0, nLast3Dim, dtype=torch.long)
    return _3d_tensor.reshape(-1)[(idx*nLastDim+(last3DimCounter*nLastDim*nLast2Dim).unsqueeze(-1)
                        ).unsqueeze(-1).expand(-1, -1, nLastDim) + lastDimCounter]


sorted_box = apply_indices_2d_to_3d(idx, box)
print(sorted_box)