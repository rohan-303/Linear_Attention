import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, use_Linear: bool):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.use_Linear = use_Linear

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def elu_feature_map(x):
        return torch.nn.functional.elu(x) + 1

    @staticmethod
    def attention(query, key, value, mask, dropout, Linear_Attention=False):
        d_k = query.size(-1)
        if Linear_Attention:
            Q = MultiHeadAttentionBlock.elu_feature_map(query)
            K = MultiHeadAttentionBlock.elu_feature_map(key)
            KV = torch.einsum('bhnd,bhnm->bhdm', K, value)
            Z = 1 / (torch.einsum('bhnd,bhd->bhn', Q, K.sum(dim=2)) + 1e-8)
            output = torch.einsum('bhnd,bhdm,bhn->bhnm', Q, KV, Z)
            approx_scores = torch.einsum('bhnd,bhmd->bhnm', Q, K)
            attn_weights = approx_scores / (approx_scores.sum(dim=-1, keepdim=True) + 1e-8)
            return output, attn_weights
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attn_weights = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        return torch.matmul(attn_weights, value), attn_weights


    def forward(self, q, k, v, mask, return_attention=False):
        B = q.size(0)
        L_q = q.size(1)
        L_k = k.size(1)

        query = self.w_q(q).view(B, L_q, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(B, L_k, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(B, L_k, self.h, self.d_k).transpose(1, 2)

        output, attn_weights = self.attention(query, key, value, mask, self.dropout, self.use_Linear)
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        output = self.w_o(output)

        if return_attention:
            return output, attn_weights
        return output

class EncoderBlock(nn.Module):
    def __init__(self, features, self_attn, ff, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.res_conns = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.res_conns[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.res_conns[1](x, self.ff)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, features, self_attn, cross_attn, ff, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff
        self.res_conns = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask, return_attention=False):
        x = self.res_conns[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        if return_attention:
            x2, attn = self.cross_attn(x, encoder_output, encoder_output, src_mask, return_attention=True)
            x = self.res_conns[1](x, lambda x: x2)
        else:
            x = self.res_conns[1](x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        x = self.res_conns[2](x, self.ff)
        return (x, attn) if return_attention else x

class Encoder(nn.Module):
    def __init__(self, layers, norm):
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layers, norm):
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, x, encoder_output, src_mask, tgt_mask, return_attention=False):
        attn_weights = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, encoder_output, src_mask, tgt_mask, return_attention=True)
                attn_weights.append(attn)
            else:
                x = layer(x, encoder_output, src_mask, tgt_mask)
        return (self.norm(x), attn_weights) if return_attention else self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(self, src, mask):
        return self.encoder(self.src_pos(self.src_embed(src)), mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask, return_attention=False):
        x = self.tgt_pos(self.tgt_embed(tgt))
        return self.decoder(x, encoder_output, src_mask, tgt_mask, return_attention=return_attention)

    def project(self, x):
        return self.proj(x)

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff, use_Linear=False):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = nn.ModuleList([
        EncoderBlock(d_model, MultiHeadAttentionBlock(d_model, h, dropout, use_Linear), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(N)
    ])

    decoder_blocks = nn.ModuleList([
        DecoderBlock(d_model,
                     MultiHeadAttentionBlock(d_model, h, dropout, use_Linear),
                     MultiHeadAttentionBlock(d_model, h, dropout, use_Linear),
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout) for _ in range(N)
    ])

    encoder = Encoder(encoder_blocks, LayerNormalization(d_model))
    decoder = Decoder(decoder_blocks, LayerNormalization(d_model))
    projection = ProjectionLayer(d_model, tgt_vocab_size)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
