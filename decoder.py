import torch.nn as nn
from general import FeedForward, MultiHeadAttention, PosEncoder
from general import copy_modules


class Decoder(nn.Module):
    def __init__(self,
                 num_of_words=None,
                 model_dimension=None,
                 max_seq_len=None,
                 head_count=8,  # default values for head count
                 N=6          # default value for N number of layers
                 ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_of_words, embedding_dim=model_dimension)
        self.positional_encoder = PosEncoder(max_seq_len=max_seq_len, model_dimension=model_dimension)
        self.N_modulelist = copy_modules(DecoderLayer(model_dimension, head_count), N)  # build decoder using N layers
        self.norm_final = nn.LayerNorm(model_dimension)

    def forward(self, target_seq, target_mask, encoder_outputs, source_mask):
        x = self.embedding(target_seq)
        x = self.positional_encoder(x)
        for i in range(self.N):
            x = self.N_modulelist[i](x, encoder_outputs, source_mask, target_mask)
        return self.norm_final(x)


class DecoderLayer(nn.Module):
    def __init__(self,
                 model_dimension,
                 head_count,
                 dropout_rate=0.1):
        super().__init__()

        # decoder attention: part a_i
        self.a_i_attention_norm = nn.LayerNorm(model_dimension, eps=1e-6)
        self.a_i_attention = MultiHeadAttention(model_dimension, head_count, dropout_rate)
        self.a_i_attention_dropout = nn.Dropout(dropout_rate)

        # decoder attention: combined input from encoder part a_ii
        self.a_ii_attention_norm = nn.LayerNorm(model_dimension, eps=1e-6)
        self.a_ii_attention = MultiHeadAttention(model_dimension, head_count, dropout_rate)
        self.a_ii_attention_dropout = nn.Dropout(dropout_rate)

        # encoder FFN: part f
        self.ffn_norm = nn.LayerNorm(model_dimension, eps=1e-6)
        self.ffn = FeedForward(model_dimension, head_count, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, source_mask, target_mask):
        a_i = self.a_i_attention_norm(x)
        a_i = self.a_i_attention(a_i, a_i, a_i, target_mask)
        a_i = self.a_i_attention_dropout(a_i)
        x = x + a_i

        a_ii = self.a_ii_attention_norm(x)
        a_ii = self.a_ii_enc_dec_attention(a_ii, enc_output, enc_output, source_mask)
        a_ii = self.a_ii_enc_dec_attention_dropout(a_ii)
        x = x + a_ii

        f = self.ffn_norm(x)
        f = self.ffn(f)
        f = self.ffn_dropout(f)
        x = x + f
        return x
