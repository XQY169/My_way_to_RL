import torch
import sys
import numpy as np
import torch.nn.functional as F
# a = torch.tensor([1.0,2.0],dtype = torch.float32,requires_grad=True)
# a2 = np.array([3.0,4.2],dtype = np.float32)
# with torch.no_grad():
#     a2_tensor = torch.from_numpy(a2)
#     c = a*a2
#     print(c.sum())
a = np.array([1,2,3.1])
print(np.max(a))