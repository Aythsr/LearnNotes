import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPT2config:
    block_size: int = 256 # 通道数量
    vocab_size: int = 64 # 词表大小
    n_layer: int = 6
    n_head: int = 6
    n_emb: int = 786 # 嵌入向量

class MLP(nn.Module):
    def __init__(self, config: GPT2config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, config.n_emb * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_porj = nn.Linear(config.n_emb * 4, config.n_emb)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_porj(x)
        return x

class CausaSelfAttention(nn.Module): # 注意力机制
    def __init__(self, config: GPT2config) -> None:
        super().__init__()
        assert config.n_emb % config.n_head == 0 # 检查是否能被分为多头
        
        # 将emb值，拆分为 Q K V 三个向量
        self.c_attn = nn.Linear(config.n_emb, config.n_emb * 3) 
        
        
        # 线性混合输出
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
         
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        
        self.register_buffer("tmp", torch.tril(
            torch.ones(config.block_size, config.block_size).bool()
        ))

    def forward(self, x: torch.Tensor):
        
        B, T, C = x.shape
        
        qkv = self.c_attn(x) # 分割出q k v
        
        q, k, v = torch.split(qkv, 3, dim=-1)
        
        q = q.view(B, T, self.n_head, self.n_emb // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_emb // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_emb // self.n_head).transpose(1, 2)
        
        attn = (q @ k.transpose(-1, -2)) * (1.0 * math.sqrt(k.size(-1)))
        
        attn = attn.masked_fill(self.tmp[:, :] == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        y = attn @ v
        
        y = y.view(1, 2).contiguous().view(B, T, C)
        
        y = self.c_proj(y)
        
        return y

class Block(nn.Module):
    def __init__(self, config: GPT2config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CausaSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x # 残差连接
        x = self.mlp(self.ln_2(x)) + x 
        return x
        

class GPT2(nn.Module):
    def __init__(self, config: GPT2config) -> None:
        super().__init__()
        self.config = config
        
        # transformer 的解码层
        self.transformer = nn.ModuleDict(dict(
            # 嵌入层 emb
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            # 位置编码
            wpe = nn.Embedding(config.block_size, config.n_emb),
            # 注意力层
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 归一化层
            ln_f = nn.LayerNorm(config.n_emb)
        ))
        
        # 映射回词表
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size)
        
    def forward(self, x):
        
        return x

print(GPT2config())