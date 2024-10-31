import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import parameters as params


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # print(self.hidden_dim)
        x = self.net(x)
        # print(x.shape)

        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)

        x_dots = attn[:, :, 0, 0]
        # print(x_dots.shape)
        if params.head_attention_mode == 'avg':
            dots_return = x_dots.mean(dim=1)
        elif params.head_attention_mode == 'sum':
            dots_return = x_dots.sum(dim=1)
        else:
            dots_return, _ = x_dots.max(dim=1)
        # print(dots_mean.shape)

  
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), dots_return.unsqueeze(dim=0)

class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        self.att1 = PreNorm(params.encoder_dim[0], Attention(params.encoder_dim[0], heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff1 = PreNorm(params.encoder_dim[0], FeedForward(params.encoder_dim[0], params.encoder_dim[0], dropout = dropout))

        self.att2 = PreNorm(params.encoder_dim[0], Attention(params.encoder_dim[0], heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff2 = PreNorm(params.encoder_dim[0], FeedForward(params.encoder_dim[0], params.encoder_dim[0], dropout = dropout))
    
        self.att3 = PreNorm(params.encoder_dim[0], Attention(params.encoder_dim[0], heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff3 = PreNorm(params.encoder_dim[0], FeedForward(params.encoder_dim[0], params.encoder_dim[1], dropout = dropout))

        self.att4 = PreNorm(params.encoder_dim[1], Attention(params.encoder_dim[1], heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff4 = PreNorm(params.encoder_dim[1], FeedForward(params.encoder_dim[1], params.encoder_dim[1], dropout = dropout))

        self.att5 = PreNorm(params.encoder_dim[1], Attention(params.encoder_dim[1], heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff5 = PreNorm(params.encoder_dim[1], FeedForward(params.encoder_dim[1], params.encoder_dim[1], dropout = dropout))
        
        self.att6 = PreNorm(params.encoder_dim[1], Attention(params.encoder_dim[1], heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff6 = PreNorm(params.encoder_dim[1], FeedForward(params.encoder_dim[1], params.encoder_dim[2], dropout = dropout))

    def forward(self, x):
        att_tensor = torch.tensor([])
        z, att = self.att1(x)
        x = z + x
        x = self.ff1(x) + x
        att_tensor = att
        # print(att_tensor.shape)

        z, att = self.att2(x)
        x = z + x
        x = self.ff2(x) + x
        att_tensor = torch.cat((att_tensor, att), dim=0)

        z, att = self.att3(x)
        x = z + x
        x = self.ff3(x)
        att_tensor = torch.cat((att_tensor, att), dim=0)

        z, att = self.att4(x)
        x = z + x
        x = self.ff4(x) + x
        att_tensor = torch.cat((att_tensor, att), dim=0)
    
        z, att = self.att5(x)
        x = z + x
        x = self.ff5(x) + x
        att_tensor = torch.cat((att_tensor, att), dim=0)

        z, att = self.att6(x)
        x = z + x
        x = self.ff6(x)
        att_tensor = torch.cat((att_tensor, att), dim=0)

        # print(att_tensor.shape)

        if params.layer_attention_mode == 'avg':
            att_return = att_tensor.mean(dim=0)
        elif params.layer_attention_mode == 'sum':
            att_return = att_tensor.sum(dim=0)
        else:
            att_return, _ = att_tensor.max(dim=0)

        return x, att_return

class VATMAN(nn.Module):
    def __init__(self, dim, heads, dim_head = 64, dropout = 0.):
        super().__init__()

        self.ln = nn.LayerNorm(dim)
        self.transformer = Transformer(dim, heads, dim_head, dropout)
 
        # self.pool = pool
        self.to_latent = nn.Identity()

        self.decoder = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(params.decoder_dim[0], params.decoder_dim[1]),
            nn.LayerNorm(params.decoder_dim[1]),
            nn.GELU(),

            nn.Linear(params.decoder_dim[1], params.decoder_dim[2]),
            nn.LayerNorm(params.decoder_dim[2]),
            nn.GELU(),

            nn.Linear(params.decoder_dim[2], params.decoder_dim[2])
        )

    def forward(self, embedding):

        x = self.ln(embedding)
        b, n, _ = x.shape


        x, att = self.transformer(x)


        x = self.to_latent(x)
        # print(x.shape)
        x = self.decoder(x)

        return x, att




        

        
        


