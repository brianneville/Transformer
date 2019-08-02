import torch.autograd as THag
import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
import numpy as np
from copy import deepcopy
import math
from torchtext_folder import torchtext
from torchtext_folder_copy import vocab
from collections import Counter


def copy_modules(module, N):
    # create the Nx copies of the encoder and decoder modules
    modules = nn.ModuleList([deepcopy(module) for i in range(0, N)])
    return modules


def xavier_init(tensor):
    nn.init.xavier_uniform_(tensor, gain=1)  # xavier init as done in the tensor2tensor for encoder/decoder layers
    nn.init.constant_(tensor, val=0)    # set mean to zero


def mask_input(input_sentence):
    # create a mask to be used when inputting into the attention layer, with 0's in positions of the '<PAD>' string
    input_padding, input_mask = '<pad>', np.ones_like(input_sentence)
    useless_required_counter = Counter()
    vocab_obj = vocab.Vocab(counter=useless_required_counter)
    # ^ this obj is not needed beyond its function to convert string to int
    padding_encoded = vocab_obj.stoi[input_padding]

    # zero the inputs which correspond to the encoding of the keyword '<pad>'
    # currently this DOES NOT set the the entire dim with size=512 to have a value of zero
    # print(f'padding encoded = {padding_encoded} shape ={np.shape(input_sentence)[1]}')
    for i in range(0, np.shape(input_sentence)[1]):
        input_mask[:, i, 0] = (input_sentence[0, i, 0] != padding_encoded)
    return input_mask


def mask_target(target_sentence_len):
    # Prevent leftward information flow in self-attention. (only attend to subsequent parts of the sentence)
    ones = torch.ones(1, target_sentence_len, target_sentence_len)
    t_self_atten_mask = torch.triu(ones, diagonal=1).unsqueeze(1)

    return t_self_atten_mask


class FinalLinear(nn.Module):
    def __init__(self, model_dimension, target):
        super().__init__()
        self.linear = nn.Linear(model_dimension, target)

    def forward(self, x):
        # perform one final linear and softmax layer to return a single output
        x = self.linear(x)
        return log_softmax(x, dim=-1)


class PosEncoder(nn.Module):
    def __init__(self, max_seq_len, model_dimension):
        # max_seq_len = the length of the longest sequence that will ever need to be encoded
        # model dimension = 512 <- the dimension of the input/output vectors of one of the encoders
        super().__init__()      # initialise the nn.module superclass
        self.model_dimension = model_dimension

        # build PE for odd and even dimensions.
        self.PE = torch.zeros(max_seq_len, self.model_dimension)
        for i in range(0, self.model_dimension):
            for pos in range(0, max_seq_len):
                exponent = 2 * i / self.model_dimension
                if not i % 2:
                    # i is a multiple of 2. use the sin formula
                    self.PE[pos, i] = math.sin(pos/(10000**exponent))
                else:
                    # i is not a multiple of 2. use the cos formula for odd powers
                    self.PE[pos, i] = math.cos(pos/(10000**exponent))

        # include an extra dimension
        self.PE = self.PE.unsqueeze(0)
        self.register_buffer('PE_withparams', self.PE)

    def forward(self, x):   # where x is an input tensor of shape(x) = (1, sentence_len, model_dimension)
        # get length of input sequence along axis 1.
        # corresponding with the max_seq_len dimension of the posEncoder,
        # (as the 0 dimension has been un-squeezed, so the words are on dim=1)
        seq_len = x.size(1)
        # slice the self.PE tensor such that it returns the encoding of the relevant dimensions along the dim=1 that the
        # words are on.
        # add this encoding to the input to encode it
        x = x + THag.Variable(self.PE[:, :seq_len]) # take : (all) of dim=0, and :seq_len (start to seq_len) of dim=1
        # randomly select to drop out some of the words:
        drop = nn.Dropout(p=0.05)
        return drop(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 model_dimension,
                 head_count=8,
                 dropout=0.0):
        super().__init__()
        self.model_dimension = model_dimension
        self.dropout_rate = dropout
        self.head_count = head_count
        self.d_k = round(model_dimension/head_count)

        # head size = how many parallel 'scaled dot product attention' calculations to perform on each input
        self.dropout = nn.Dropout(self.dropout_rate)
        self.q_linear = nn.Linear(model_dimension, model_dimension)     # implement a linear system of equations
        self.v_linear = nn.Linear(model_dimension, model_dimension)     # y = Ax^T + B
        self.k_linear = nn.Linear(model_dimension, model_dimension)
        self.output = nn.Linear(model_dimension, model_dimension)
        xavier_init(self.k_linear)
        xavier_init(self.q_linear)
        xavier_init(self.v_linear)
        xavier_init(self.output)

    def forward(self, q, k, v, mask):   # with q, k, v as input tensors, for query, key, value
        batch_size = q.size(0)
        # batch_size in dim=0, head_count in dim=2, self.d_k in dim=3
        # -1 in dim =1 represents the length of the input sentence. this will be inferred automatically such that the
        # tensor k can be properly built from the output of the linear functions

        k = self.k_linear(k).view(batch_size, -1, self.head_count, self.d_k)
        q = self.q_linear(q).view(batch_size, -1, self.head_count, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.head_count, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # now all k, v, q have dims:
        # 0 = batch_size, 1 = head_count, 2 = (length), 3 = d_k

        # calculate attention
        # attention = Softmax((QK)/sqrt(d_k))V

        # scale q:
        q.mul_(1 / (math.sqrt(self.d_k)))
        # transpose K
        k = k.transpose(2, 3)   # dims (batch_size, head_count, d_k, length_k)
        product = torch.matmul(q, k)
        # prodcut has dim: (batch_size, head_count, length_k, length_q)
        mask.unsqueeze(1)
        product = product.masked_fill(mask == 0, -1e9)  # mask vals accordingly

        smax_result = softmax(product, dim=3)

        if self.dropout_rate is not None:
            smax_result = self.dropout(smax_result)

        attn_result = torch.matmul(smax_result,v)      # attn_result with dims (batch,head_c, length_q, d_k )
        # unsure about the last dimension. these weights will all be concatenated to form the weight matrix which is the
        # output of the entire multihead attention layer

        # concatenate all results
        concat = attn_result.transpose(1, 2).contiguous()
        concat = concat.view(batch_size, -1, self.model_dimension)
        out = self.output(concat)

        return out


class FeedForward(nn.Module):
    def __init__(self,
                 model_dimension,
                 inner_layer_dff=2048,
                 dropout_rate=0.1):
        super().__init__()
        # create simple NN with input, hidden w/ relu and output layers
        self.input = nn.Linear(model_dimension, inner_layer_dff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(model_dimension, inner_layer_dff)

        xavier_init(self.input)
        xavier_init(self.output)

    def forward(self, t):
        t = self.input(t)
        t = self.relu(t)
        t = self.dropout(t)
        t = self.output(t)
        return t


'''
def mask_target(self, target_sentence):

    initial_mask = self.mask_input(target_sentence)
    # find length of sentence

    sentence_len = np.shape(target_sentence)[1] - 1
    while sentence_len >= 0:
        if initial_mask[0, sentence_len, 0] == 0:
            break
        else:
            sentence_len -= 1
    # Prevent leftward information flow in self-attention.
    # only allow to self-attend to the remaining parts of the sentence
    target_mask = self.create_trg_self_mask(np.shape(target_sentence)[1])
    print(f'target mask is {target_mask}')
    initial_mask = torch.Tensor(initial_mask)
    print(initial_mask)
'''
'''
    def second_attempt_mask_input(self, input_sentence):
        # create a mask to be used when inputting into the attention layer, with 0's in positions of the '<PAD>' string
        input_padding, input_mask = '<pad>', np.ones_like(input_sentence)
        useless_required_counter = Counter()
        vocab_obj = torchtext.vocab.Vocab(counter=useless_required_counter)
        # ^ this obj is not needed beyond its function to convert string to int
        padding_encoded = vocab_obj.stoi[input_padding]
        mask = (input_sentence == padding_encoded).unsqueeze(-2)
        print(mask)
'''
