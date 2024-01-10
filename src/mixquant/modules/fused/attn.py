import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from mixquant.modules.fused.cache import WindowedCache
from mixquant.utils.fused_utils import get_attention_shapes



from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

try:
    import ft_inference_engine
    FT_INSTALLED = True
except:
    FT_INSTALLED = False

class RoPE(nn.Module):
    def __init__(self, hidden_size, n_heads, max_seq_len, device):
        super(RoPE, self).__init__()
        
        self.freqs_cis = nn.Parameter(
            self.precompute_freqs_cis(hidden_size // n_heads, max_seq_len * 2).to(device),
            requires_grad=False
        )

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int, seqlen: int):
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
        )
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
        )
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_).to(xq_.device)
        
        xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-2, -1).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-2, -1).flatten(3)
        
        return xq_out.type_as(xq), xk_out.type_as(xk)

class ALiBi(nn.Module):
    def __init__(self, n_heads, max_seq_len, device, alibi_bias_max=8):
        super(ALiBi, self).__init__()
        
        # Initialize ALiBi slopes and bias
        slopes, bias = self.build_alibi_bias(n_heads, max_seq_len, alibi_bias_max=alibi_bias_max)
        self.slopes = nn.Parameter(slopes.float().to(device), requires_grad=False)
        self.bias = nn.Parameter(bias.float().to(device), requires_grad=False)

    @staticmethod
    def gen_slopes(n_heads, alibi_bias_max=8):
        _n_heads = 2 ** math.ceil(math.log2(n_heads))
        m = torch.arange(1, _n_heads + 1, dtype=torch.float32)
        m = m.mul(alibi_bias_max / _n_heads)
        slopes = 1.0 / torch.pow(2, m)
        
        if _n_heads != n_heads:
            slopes = torch.cat([slopes[1::2], slopes[::2]])[:n_heads]
            
        return slopes.view(1, n_heads, 1, 1)

    @staticmethod
    def build_alibi_bias(n_heads, seq_len, alibi_bias_max=8, dtype=torch.float32):
        alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32).view(1, 1, 1, seq_len)
        slopes = ALiBi.gen_slopes(n_heads, alibi_bias_max)
        alibi_bias = alibi_bias * slopes
        slopes = slopes.squeeze(0).squeeze(-1).squeeze(-1)
        return slopes.to(dtype=dtype), alibi_bias.to(dtype=dtype)
    
    def forward(self, scores, seqlen):
        scores += self.bias[..., :seqlen]
        return scores

class QuantAttentionFused_(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads, qkv_layer, o_proj, dev, max_seq_len, 
                       use_alibi=False, attention_shapes=None,MixGemmCache=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_kv_groups = n_heads // n_kv_heads if n_kv_heads != 0 else 0
        self.head_dim = self.hidden_size // n_heads
        self.qkv_proj = qkv_layer
        self.o_proj = o_proj
        self.start_pos = 0
        self.use_alibi = use_alibi
        self.cache_batch_size = int(os.getenv("AWQ_BATCH_SIZE", "1"))
        self.max_seq_len = max_seq_len
        self.MixGemmCache = MixGemmCache

        # attention shapes for self attention
        self.attention_shapes = get_attention_shapes(
            attention_shapes, max_seq_len, self.cache_batch_size, n_heads, n_kv_heads, self.head_dim
        )
        # cache store that rolls cache
        self.cache = WindowedCache(
            self.attention_shapes["cache_v"], self.attention_shapes["cache_k"], dev
        )

        if use_alibi:
            self.alibi = ALiBi(n_heads, max_seq_len, dev)
            self.rotary_dim = 0
            self.is_neox = False
        else:
            self.alibi = None
            self.rope = RoPE(hidden_size, n_heads, max_seq_len, dev)
            self.rotary_dim = self.head_dim
            self.is_neox = True
    
    def forward(self, hidden_states:torch.Tensor, attention_mask=None, *args, **kwargs):

        bsz, seqlen, _ = hidden_states.shape
        if bsz != self.cache_batch_size:
            raise RuntimeError(
                f"Batch size is incorrectly set - input batch size {bsz}, kv-cache batch size {self.cache_batch_size}. "
                f"Use: AutoAWQForCausalLM.from_quantized(batch_size={bsz})"
            )

        if self.start_pos > self.max_seq_len or self.start_pos + seqlen > self.max_seq_len:
            excess_length = self.start_pos + seqlen - self.max_seq_len
            self.start_pos = self.cache.roll_kv(excess_length, self.start_pos)
            
        xqkv = self.qkv_proj(hidden_states,  self.MixGemmCache)
        xqkv = xqkv.view((bsz, seqlen) + self.attention_shapes["xqkv_view"])
        
        xq = self.attention_shapes["xq_slice"](xqkv)
        xk = self.attention_shapes["xk_slice"](xqkv)
        xv = self.attention_shapes["xv_slice"](xqkv)

        if seqlen > 1 or not FT_INSTALLED:
            xq = xq.view((bsz, seqlen) + self.attention_shapes["xq_view"])
            xk = xk.view((bsz, seqlen) + self.attention_shapes["xk_view"])
            xv = xv.view((bsz, seqlen) + self.attention_shapes["xv_view"])

            if not self.use_alibi:
                xq, xk = self.rope.forward(xq, xk, self.start_pos, seqlen)

            self.cache.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape((bsz, seqlen) + self.attention_shapes["xk_reshape"])
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )
            
            self.cache.update_kv(values_store, keys_store, bsz, self.start_pos, seqlen)

            if seqlen == 1:
                xv, xk = self.cache.get_kv(bsz, self.start_pos, seqlen, self.head_dim)
            
            keys = xk
            values = xv

            if self.n_kv_groups != 0:
                keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_kv_groups)
                values = torch.repeat_interleave(values, dim=2, repeats=self.n_kv_groups)
            
            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

            if self.use_alibi:
                scores = self.alibi.forward(scores, seqlen)

            # When seqlen is 1, there is nothing else to attend to
            if attention_mask is not None and seqlen > 1:
                scores = scores + attention_mask  # (bs, n_local_heads, slen, cache_len + slen)
                
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            attention_weight = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:

            #import inspect
            #print(inspect.getfile(ft_inference_engine))
            xq = xq.view((bsz,) + self.attention_shapes["single_xq_view"])
            xk = xk.view((bsz,) + self.attention_shapes["single_xk_view"])
            xv = xv.view((bsz,) + self.attention_shapes["single_xv_view"])

            alibi_slopes = self.alibi.slopes if self.alibi is not None else None
            attention_weight = ft_inference_engine.single_query_attention(
                xq, # query
                xk, # key
                xv, # value
                self.cache.k, # key cache
                self.cache.v, # value cache
                None, # length per sample
                alibi_slopes, # alibi slopes
                self.start_pos, # timestep
                self.rotary_dim, # rotary embedding dimension
                10000, # rotary embedding base
                self.is_neox, # is neox
            )
            attention_weight = attention_weight.reshape(bsz, 1, -1)
        
        attn_output = self.o_proj(attention_weight,self.MixGemmCache,True)
        self.start_pos += seqlen

        # past_key_value is replaced with cache_v, cache_k, returning empty data
        past_key_value = [torch.Tensor([ [ [[0]], [[0]], [[0]] ] ])]
        return attn_output, attention_weight, past_key_value
    

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
        self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(torch.float32).to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(torch.float32).to(x.device)
        elif self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)  
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_, sin_, position_ids):
    cos = cos_.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin_.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)



def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class QuantAttentionFused(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self,  hidden_size, n_heads, n_kv_heads, qkv_layer, o_proj, dev, max_seq_len, 
                       use_alibi=False, attention_shapes=None,MixGemmCache=None):
        super().__init__()

        # print("--hidden size is")
        # print(hidden_size)
        # print('qkv size is')

        # print(qkv_layer.in_features)
        # print(qkv_layer.out_features)


        self.hidden_size = hidden_size
        self.num_heads = n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_seq_len
        self.cache_batch_size = int(os.getenv("AWQ_BATCH_SIZE", "1"))
        self.attention_shapes = get_attention_shapes(
            attention_shapes, max_seq_len, self.cache_batch_size, n_heads, n_kv_heads, self.head_dim
        )
        # print("atten shape is ")
        # print(self.attention_shapes)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.W_pack = qkv_layer
        self.o_proj = o_proj
        self.MixGemmCache = MixGemmCache
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            padding_mask = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
    def _flash_attention_forward(
        self, query_states, key_states, value_states, padding_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True
            )

        return attn_output
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions: bool = False,
            use_cache: bool = False,
            padding_mask = None,
            *args, **kwargs
    ) :
        bsz, q_len, _ = hidden_states.size()
        assert self.cache_batch_size == bsz
        proj = self.W_pack(hidden_states, self.MixGemmCache, False)

        
        xqkv = proj.view((bsz, q_len) + self.attention_shapes["xqkv_view"])
 

        xq = self.attention_shapes["xq_slice"](xqkv)
        xk = self.attention_shapes["xk_slice"](xqkv)
        xv = self.attention_shapes["xv_slice"](xqkv)
        query_states = xq.transpose(1, 2)
        key_states = xk.transpose(1, 2)
        value_states = xv.transpose(1, 2)



        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        dropout_rate = 0.0
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, padding_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output, self.MixGemmCache, True)
        # print(attn_output)
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        #     attn_output2 = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = attention_mask)
        # attn_output2 = attn_output2.transpose(1, 2)
        # attn_output2 = attn_output2.reshape(bsz, q_len, self.hidden_size)
        # attn_output2 = self.o_proj(attn_output, self.MixGemmCache, True)

        # print(attn_output2)
        # exit(0)


        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    


# 百川 13B
class QuantAttentionFused_(torch.nn.Module):
    def __init__(self,  hidden_size, n_heads, n_kv_heads, qkv_layer, o_proj, dev, max_seq_len, 
                       use_alibi=False, attention_shapes=None,MixGemmCache=None):
        super().__init__()
 
        self.hidden_size = hidden_size
        self.num_heads = n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_seq_len

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = qkv_layer
        self.o_proj = o_proj

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value   = None,
        output_attentions  = False,
        use_cache = False,
        *args, **kwargs
    ) :
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = (
            proj.unflatten(-1, (3, self.hidden_size))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value