import torch 
from torch import nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

from utils.model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask



class PositionalEncodingSinCos(nn.Module):
    """
    Positional encoding: sinusoidal: vanilla positional encoding from 
    attention is all you need by vaswani et al. 
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PositionalEncodingLUT(nn.Module):
    """
    Positional encoding: simple positional encoding from deepcad by xu et al.
    """

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)

        self.pos_embed = nn.Embedding(max_len, d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)
    

class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, config, seq_len, use_group=False, group_len=None):
        super().__init__()

        self.command_embed = nn.Embedding(config.n_commands, config.d_model)

        args_dim = config.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * config.n_args, config.d_model)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = config.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, config.d_model)

    def forward(self, commands, args, groups=None):
        S, N = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        return src
    

class ConstEmbedding(nn.Module):
    """
    Learned constant embedding:
    This is mainly for the decoder, 
    usage: 
        the constant embedding helps to project the latent vector to the sequence of 
        latent vectors that can be processed by the transformer decoder 
    """
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.seq_len = config.max_total_len

        self.PE = PositionalEncodingLUT(config.d_model, max_len=self.seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src
    

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        seq_len = config.max_total_len
        self.use_group = config.use_group_emb
        self.embedding = CADEmbedding(config, seq_len, use_group=self.use_group)

        encoder_layer = TransformerEncoderLayer(config.d_model, config.n_enc_heads, config.dim_feedforward, config.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, config.n_enc_layers, enable_nested_tensor=False)
        self.pos_encoding = PositionalEncodingLUT(config.d_model, max_len=seq_len+2)

    def forward(self, commands, args):
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)

        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        # embedding 

        src = self.embedding(commands, args, group_mask)

        # positional encoding 

        src = self.pos_encoding(src)

        # transformer encoder

        memory = self.transformer_encoder(src, src_key_padding_mask=key_padding_mask)

        # average pooling 
        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z
    
class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits
    

class Bottleneck(nn.Module):
    def __init__(self, config):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(config.d_model, config.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        seq_len = config.max_total_len
        self.use_group = config.use_group_emb
        self.const_embedding = ConstEmbedding(config)


        decoder_layer = TransformerDecoderLayer(config.d_model, config.n_dec_heads, config.dim_feedforward, config.dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=config.n_dec_layers)
        
        args_dim = config.args_dim + 1
        self.fcn = FCN(config.d_model, config.n_commands, config.n_args, args_dim)

    def forward(self, latent):
        # constant embedding
        src = self.const_embedding(latent)  # (S, N, d_z)
        latent = latent.repeat(src.size(0), 1, 1)  # (1, N, d_z) --> (S, N, d_z)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.config.device)

        out = self.transformer_decoder(tgt=src, memory=latent, tgt_mask=causal_mask, memory_mask=causal_mask)

        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits)
        return out_logits
    

class VanillaCADTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.max_total_len = config.max_total_len

        self.args_dim = config.args_dim + 1

        self.encoder = Encoder(config)

        self.bottleneck = Bottleneck(config)

        self.decoder = Decoder(config)

    def forward(self, commands_enc, args_enc,
                z=None, return_tgt=True, encode_mode=False):
       
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)  # Possibly None, None
        if z is None:
            z = self.encoder(commands_enc_, args_enc_)
            z = self.bottleneck(z)
        else:
            z = _make_seq_first(z)

        if encode_mode: return _make_batch_first(z)

        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)

        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }

        if return_tgt:
            # ---------- IMPORTANT ----------- 
            # autoregressive
            res["tgt_commands"] = commands_enc
            res["tgt_args"] = args_enc

        return res