import torch.nn as nn
# this is suffering cos i cant seem to install torchtext properly and have it as folder instead
from general import PosEncoder, MultiHeadAttention, FeedForward
from general import copy_modules


class Encoder(nn.Module):
    def __init__(self,
                 num_of_words=None,
                 model_dimension=None,
                 max_seq_len=None,
                 head_count=8,  # default values for head count
                 N=6  # default value for N number of layers
                 ):
        super().__init__()
        self.N = N
        self.embedding = nn.Embedding(num_embeddings=num_of_words, embedding_dim=model_dimension)
        self.positional_encoder = PosEncoder(max_seq_len=max_seq_len, model_dimension=model_dimension)
        self.N_modulelist = copy_modules(EncoderLayer(model_dimension, head_count), N)  # build encoder using N layers
        self.norm_final = nn.LayerNorm(model_dimension)

    def forward(self, input_seq, source_mask):
        x = self.embedding(input_seq)
        x = self.positional_encoder(x)
        for i in range(self.N):
            x = self.layers[i](x, source_mask)
        return self.norm_final(x)


class EncoderLayer(nn.Module):
    def __init__(self,
                 model_dimension,
                 head_count,
                 dropout_rate=0.1):
        super().__init__()

        # encoder attention: part a
        self.attention_norm = nn.LayerNorm(model_dimension, eps=1e-6)
        self.attention = MultiHeadAttention(model_dimension, head_count, dropout_rate)
        self.attention_dropout = nn.Dropout(dropout_rate)

        # encoder FFN: part f
        self.ffn_norm = nn.LayerNorm(model_dimension, eps=1e-6)
        self.ffn = FeedForward(model_dimension, head_count, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        a = self.attention_norm(x)
        a = self.attention(a, a, a, mask)
        a = self.attention_dropout(a)
        residual_state = x + a

        f = self.ffn_norm(residual_state)
        f = self.ffn(f)
        f = self.ffn_dropout(f)
        end_state = residual_state + f

        return end_state


