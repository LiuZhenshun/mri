import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None, shift_t=0):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    tensor = pool(tensor)
    if shift_t > 0:
        tensor = tensor[:, :, :-shift_t]

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


class MemAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, keep_max_len=3, mem=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mem = mem

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if mem:
            self.cached_k = []
            self.cached_v = []
            self.cached_video_names = []
            self.keep_max_len = keep_max_len
            self.compress_k = nn.Conv3d(head_dim, head_dim, kernel_size=(2,2,2), stride=(2,2,2), groups=head_dim, bias=False)
            self.compress_v = nn.Conv3d(head_dim, head_dim, kernel_size=(2,2,2), stride=(2,2,2), groups=head_dim, bias=False)
            self.norm_compress_k = nn.LayerNorm(head_dim)
            self.norm_compress_v = nn.LayerNorm(head_dim)

    def mask_memory(self):
        if (self.cached_video_names[-1] != self.cached_video_names[-2]).any():
            self.cached_k, self.cached_v = ([self.cached_k[-1]], [self.cached_v[-1]])
            self.cached_video_names = [self.cached_video_names[-1]]

    def memory_management(self, k, v, video_names, shape):
        # Cache memory
        self.cached_k.append(k.detach())
        self.cached_v.append(v.detach())
        self.cached_video_names.append(video_names)

        if len(self.cached_k) > 1:
            # Mask memory from different video clips
            self.mask_memory()
        
        if len(self.cached_k) > 1:
            # Compress memory   
            self.cached_k[-2], mem_shape = attention_pool(self.cached_k[-2], self.compress_k, shape,
                                has_cls_embed=True, norm=self.norm_compress_k)
            self.cached_v[-2], mem_shape = attention_pool(self.cached_v[-2], self.compress_v, shape,
                                has_cls_embed=True, norm=self.norm_compress_v)
            self.cached_k[-2] = self.cached_k[-2].detach()
            self.cached_v[-2] = self.cached_v[-2].detach()
        
        if len(self.cached_k) > 1:
            # Concate memory
            k = torch.cat(self.cached_k[:-1] + [k], dim=2)
            v = torch.cat(self.cached_v[:-1] + [v], dim=2)

        # Keep at most (self.keep_max_len) memory.
        if len(self.cached_k) == self.keep_max_len:
            self.cached_k, self.cached_v = (self.cached_k[1:], self.cached_v[1:])
            self.cached_video_names = self.cached_video_names[1:]

        return k, v

    def forward(self, x, video_names, shape):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
       
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        
        if self.mem:
            k, v = self.memory_management(k, v, video_names, shape)

        x = scaled_dot_product_attention(q,k,v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class AdaMemAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, keep_max_len=2, kv_length=50, mem=True, 
            compress=False, TRF_Record=False, mem_bank=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.keep_max_len = keep_max_len
        self.kv_length = kv_length
        self.mem = mem
        self.compress = compress
        self.TRF_Record = TRF_Record
        self.mem_bank = mem_bank
        self.scale = qk_scale or head_dim ** -0.5
        
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if mem:
            self.cached_k = []
            self.cached_v = []
            self.bank_k = None
            self.bank_v = None
            self.cached_video_names = []
            if compress:
                self.compress_k = nn.Conv3d(head_dim, head_dim, kernel_size=(2,2,2), stride=(2,2,2), groups=head_dim, bias=False)
                self.compress_v = nn.Conv3d(head_dim, head_dim, kernel_size=(2,2,2), stride=(2,2,2), groups=head_dim, bias=False)
                self.norm_compress_k = nn.LayerNorm(head_dim)
                self.norm_compress_v = nn.LayerNorm(head_dim)
            if TRF_Record:
                self.TRF_Recorder = None

    def mask_memory(self):
        if (self.cached_video_names[-1] != self.cached_video_names[-2]).any():
            self.cached_k, self.cached_v = ([self.cached_k[-1]], [self.cached_v[-1]])
            self.bank_k = None
            self.bank_v = None
            self.cached_video_names = [self.cached_video_names[-1]]
            
            if self.TRF_Record:
                if self.TRF_Recorder is not None:
                    self.TRF_Recorder.zero_()
    
    def kv_selection(self, q, cached_k, cached_v, kv_length):
        B, H, N, D = q.shape
        
        atten_score = (q[:,:,:1] @ cached_k.transpose(-2,-1)).reshape(B, H, -1)
        _, indices = torch.topk(atten_score, kv_length, dim=2)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)
        
        return torch.gather(cached_k, 2, indices), torch.gather(cached_v, 2, indices), atten_score

    def long_term_context(self, q, k, v, video_names, shape):
        # Cache memory
        self.cached_k.append(k.detach())
        self.cached_v.append(v.detach())
        self.cached_video_names.append(video_names)

        if len(self.cached_k) > 1:
            # Mask memory from different video clips
            self.mask_memory()
        
        if (len(self.cached_k) > 1) and self.compress:
            # Compress memory   
            self.cached_k[-2], mem_shape = attention_pool(self.cached_k[-2], self.compress_k, shape,
                                has_cls_embed=True, norm=self.norm_compress_k)
            self.cached_v[-2], mem_shape = attention_pool(self.cached_v[-2], self.compress_v, shape,
                                has_cls_embed=True, norm=self.norm_compress_v)
            self.cached_k[-2] = self.cached_k[-2].detach()
            self.cached_v[-2] = self.cached_v[-2].detach()

        # Keep at most (self.keep_max_len) memory.
        if len(self.cached_k) == self.keep_max_len:
            if self.bank_k is None:
                self.bank_k, self.bank_v, _ = self.kv_selection(q, self.cached_k[0], self.cached_v[0], int(self.kv_length))
            else:
                bank_k_0, bank_v_0, atten_score_b = self.kv_selection(q, self.bank_k, self.bank_v, int((self.kv_length)*0.3))
                bank_k_1, bank_v_1, _ = self.kv_selection(q, self.cached_k[0], self.cached_v[0], int((self.kv_length)*0.7))
                self.bank_k = torch.cat([bank_k_0] + [bank_k_1], dim=2)
                self.bank_v = torch.cat([bank_v_0] + [bank_v_1], dim=2)
                if self.TRF_Record:
                    if self.TRF_Recorder is None:
                        self.TRF_Recorder = torch.zeros_like(atten_score_b)
                        _, indices = torch.topk(atten_score_b, int(self.kv_length*0.3), dim=2)
                        selected_times = self.TRF_Recorder.gather(2, indices) + 1
                        self.TRF_Recorder[:,:,:int(self.kv_length*0.3)] = selected_times
                        self.TRF_Recorder[:,:,int(self.kv_length*0.3):] = 0
                    else:
                        _, indices = torch.topk(atten_score_b, int(self.kv_length*0.3), dim=2)
                        selected_times = self.TRF_Recorder.gather(2, indices) + 1
                        self.TRF_Recorder[:,:,:int(self.kv_length*0.3)] = selected_times
                        self.TRF_Recorder[:,:,int(self.kv_length*0.3):] = 0
            
            self.cached_k, self.cached_v = (self.cached_k[1:], self.cached_v[1:])
            self.cached_video_names = self.cached_video_names[1:]

        if len(self.cached_k) > 1:
            # kv selection
            selected_k_list = []
            selected_v_list = []
            for (cached_k, cached_v) in zip(self.cached_k[:-1], self.cached_v[:-1]):
                selected_k, selected_v, _ = self.kv_selection(q, cached_k, cached_v, self.kv_length)
                selected_k_list.append(selected_k)
                selected_v_list.append(selected_v)
            k = torch.cat(selected_k_list + [k], dim=2)
            v = torch.cat(selected_v_list + [v], dim=2)
        
        if (self.bank_k is not None) and self.mem_bank:
            k = torch.cat([self.bank_k] + [k], dim=2)
            v = torch.cat([self.bank_v] + [v], dim=2)
            
        return q, k, v

    def forward(self, x, video_names, shape):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # QKV B H N D 
        q,k,v = qkv[0],qkv[1],qkv[2]
        
        if self.mem:
            q,k,v = self.long_term_context(q, k, v, video_names, shape)
        
        x = scaled_dot_product_attention(q,k,v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, *args):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        
        x = scaled_dot_product_attention(q,k,v).transpose(1, 2).reshape(B, N, -1)
 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def atten_layer_factory(name, **kwargs):
    args = kwargs["args"]
    kwargs.pop('args', None)
    if 'AdaMemAttention' in name:
        return AdaMemAttention(keep_max_len=args.keep_max_len, kv_length=args.kv_length, 
                               compress=args.compress, TRF_Record=args.TRF_Record, mem_bank=args.mem_bank, **kwargs)
    elif 'MemAttention' in name: 
        return MemAttention(keep_max_len=args.keep_max_len, **kwargs)
    elif 'Attention' in name:
        kwargs.pop('mem', None)
        return Attention(**kwargs)


if __name__ == "__main__":
    q_size = 14
    kv_size = 14
    head_dim = 64
    rel_sp_dim = 2 * max(q_size, kv_size) - 1
    rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
    rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
    Rh, Rw = get_spatial_embeddings((8,14,14), (8,14,14), rel_pos_h, rel_pos_w)