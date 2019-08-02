from encoder import Encoder
from decoder import Decoder
from torch.functional import log_softmax
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self,
                 src_number_of_words,
                 src_max_sequence_len,
                 target_number_of_words,
                 target_max_sequence_len,
                 model_dimension=512,
                 N_layers=6,
                 head_count=8,
                 ):
        super().__init__()
        assert not model_dimension % head_count
        self.model_dimension = model_dimension
        self.N_layers = N_layers
        self.head_count = head_count
        # ensure that the model dimension and head_count are compatible when doing matrix transposes etc
        # this is not an issue with the above config(model_dim%h ==0)
        self.encoder_unit = Encoder(
                                    num_of_words=src_number_of_words,
                                    model_dimension=model_dimension,
                                    max_seq_len=src_max_sequence_len,
                                    head_count=head_count,
                                    N=N_layers
                                    )
        self.decoder_unit = Decoder(
                                    num_of_words=target_number_of_words,
                                    model_dimension=model_dimension,
                                    max_seq_len=target_max_sequence_len,
                                    head_count=head_count,
                                    N=N_layers
                                    )
        self.final_linear = nn.Linear(model_dimension, target_number_of_words)

    def forward(self, source, source_mask, target, target_mask):      # where source and target are the to encode/decode
        encoder_out = self.encoder_unit(
                                        input_seq=source,
                                        source_mask=source_mask
                                        )

        decoder_out = self.decoder_unit(
                                        target_seq=target,
                                        target_mask=target_mask,
                                        encoder_outputs=encoder_out,
                                        source_mask=source_mask
                                        )
        linear_out = self.final_linear(decoder_out)
        # possibly look into AdaptiveLogSoftmaxWithLoss pytorch?
        soft_out = log_softmax(linear_out, dim=-1)  # apply to rightmost dimension
        return soft_out
