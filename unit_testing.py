'''
import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from general import PosEncoder
import torch
import torch.nn as nn
import torchtext_folder.torchtext
from encoder import Encoder
import torch.autograd as THag
'''

'''
P = PosEncoder(max_seq_len=10, model_dimension=512)
print(numpy.shape(P.PE))
print(P.PE[0, 8, 100].item())   # getting the value of a tensor at a certain position
'''
'''
# testing what the x.size() call will return
x = torch.zeros(1, 2)
print(x.size(1))
'''
'''
# testing how the slice of the positional encoding works
P = PosEncoder(max_seq_len=2, model_dimension=512)
x = torch.zeros(1, 2, 512)
y = x + THag.Variable(P.PE[:, :x.size(1)])
r = x + THag.Variable(P.PE[:x.size(1)])
print(np.shape(y), np.shape(r))
print(y)
print(r)
'''
'''
x = torch.zeros(1, 20)
print(x)
print(x.unsqueeze(-1))
'''

'''
x = torch.zeros(1, 5, 512)
x[0, 3, 0] = 1
print(np.shape(x))
P = Encoder(max_seq_len=20, num_of_words=100, model_dimension=512)
msk = P.second_attempt_mask_input(x)
msk = P.mask_target(x)
print(msk)
'''
'''
def create_trg_self_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention.
    ones = torch.ones(1, target_len, target_len)
    t_self_mask = torch.triu(ones, diagonal=1)

    return t_self_mask


print(create_trg_self_mask(10))
'''
'''
k = nn.Linear(10, 10)
print(k.bias)
'''
'''
x = torch.zeros(1, 2, 512)
x[0, 1, 0] = 1
print("x is :")
print(x)
print('mask is: ')
# print(create_trg_self_mask(target_len=100))
'''

k = 512
h = 7
assert not k % h