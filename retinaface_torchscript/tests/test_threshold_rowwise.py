import torch

a = torch.tensor([
    [0.1, 0.2, 0.9],
    [0.1, 0.7, 0.6]
])

mask = a > 0.5 and a < 0.8
print(a[mask])


for i in range(a.shape[0]):
    print(a[i][mask[i]])

print(a.shape)
import ipdb; ipdb.set_trace()

# Expected res
"""
b = [
    [0.9],
    [0.7, 0.6]
]
"""