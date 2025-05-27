import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
from typing import Dict, List, Callable, Any

import transformers
from transformers.models.llama.configuration_llama import LlamaConfig as LlamaConfigOriginal # Keep original for reference if needed
# Qwen2 specific imports
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention, Qwen2MLP, Qwen2RMSNorm

# Qwen2VL specific imports
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig # Assuming Qwen2VLViTConfig might be Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLViTEncoderLayer, Qwen2VLViTAttention, Qwen2VLViTMLP # Assuming these class names

from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast
# Remove direct CLIP and old LLaMA layer imports if Qwen2 versions are comprehensive
# from transformers.models.clip.modeling_clip import CLIPAttention, CLIPConfig, CLIPEncoderLayer, CLIPMLP

# Removing EVA ViT specific parts as they are not relevant for Qwen2VL
# from .eva_vit import Attention as EvaAttention
# import pink.model.eva_vit

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch.cuda.amp import autocast


class AdapterLayer(nn.Module):
    def __init__(
        self, 
        in_features,
        hidden_dim=8, 
        scale=1,
        dropout=0.1,
        non_linear=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.in_features = in_features
        self.tune_adapter_a = nn.Linear(self.in_features, hidden_dim, bias=True)
        self.tune_adapter_b = nn.Linear(hidden_dim, self.in_features, bias=True)
        self.dropout = nn.Dropout(dropout)

        if non_linear:
            self.activate = nn.SiLU()
        else:
            self.activate = nn.Identity()

    def train(self, mode: bool = True):
        self.tune_adapter_a.train(mode)
        self.tune_adapter_b.train(mode)
        self.dropout.train(mode)

    def forward(self, x):
        previous_dtype = x.dtype
        weight_dtype = self.tune_adapter_a.weight.data.dtype
        down_x = self.tune_adapter_a(x.to(weight_dtype))
        down_x = self.activate(down_x)
        up_x = self.tune_adapter_b(self.dropout(down_x))
        result = up_x.to(previous_dtype) + x
        return result


def mark_only_adapter_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias: 
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if 'adapter_' not in n:
            p.requires_grad = False
        else:
            p.data = p.data.float()
            p.requires_grad = True

    # depending on the `bias` value unfreeze bias weights
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError

@dataclass
class AdapterConfig:
    hidden_dim: int = 8
    scale: float = 1.0
    dropout: float = 0.1
    adapter_attn: bool = True
    adapter_mlp: bool = True
    non_linear: bool = False


# Adapted for Qwen2VL Vision Encoder Layer
class Qwen2VLViTAdapterEncoderLayer(nn.Module):
    adapter_config = None
    def __init__(self, config: Qwen2VLVisionConfig): # Changed from CLIPConfig
        super().__init__()
        self.embed_dim = config.hidden_size
        # IMPORTANT: Qwen2VLViTEncoderLayer uses Qwen2VLViTAttention and Qwen2VLViTMLP.
        # These need to be compatible with the original CLIPAttention/MLP structure if we reuse AdapterLayer logic directly
        # or the AdapterLayer needs to be made more generic.
        # For now, assuming Qwen2VLViTAttention and Qwen2VLViTMLP are constructor arguments to the original layer.
        # This part requires careful verification against transformers source.
        # If Qwen2VLViTEncoderLayer directly instantiates these, this adapter needs to replicate that.
        self.self_attn = Qwen2VLViTAttention(config) # Changed from CLIPAttention
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) # Common
        self.mlp = Qwen2VLViTMLP(config) # Changed from CLIPMLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) # Common

        if self.adapter_config.adapter_attn:
            self.adapter_attn = AdapterLayer(
                self.embed_dim,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )
        if self.adapter_config.adapter_mlp:
            self.adapter_mlp = AdapterLayer(
                self.embed_dim,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        # Qwen2VLViTEncoderLayer forward signature might differ from CLIPEncoderLayer.
        # Specifically, `causal_attention_mask` might not be used.
        # The `forward` signature should match the original Qwen2VLViTEncoderLayer's forward signature.
        # For now, keeping it similar to CLIP's for structure.
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        # Potentially other args like `head_mask`, `patch_embedding_indices` etc. if present in original
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        if hasattr(self, "adapter_attn"):
            assert self.adapter_config.adapter_attn
            hidden_states = self.adapter_attn(hidden_states)

        # The call to self.self_attn must match how the original Qwen2VLViTEncoderLayer calls its attention.
        # It might not take `causal_attention_mask`.
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0] # Assuming attention output is a tuple with hidden_states as first element
        attn_weights = attn_outputs[1] if output_attentions and len(attn_outputs) > 1 else None

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        if hasattr(self, "adapter_mlp"):
            assert self.adapter_config.adapter_mlp
            hidden_states = self.adapter_mlp(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Removed EvaAdapterAttention and related eva_adapter context manager

# Qwen2 specific attention classes
QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}


# Adapted for Qwen2 Decoder Layer
class Qwen2AdapterDecoderLayer(nn.Module):
    adapter_config = None
    def __init__(self, config: Qwen2Config, layer_idx: int): # Changed from LlamaConfig
        super().__init__()
        self.hidden_size = config.hidden_size
        # Qwen2 uses Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config) # Changed from LlamaMLP
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # Changed from LlamaRMSNorm
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # Changed from LlamaRMSNorm
        if self.adapter_config.adapter_attn:
            self.adapter_attn = AdapterLayer(
                self.hidden_size,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                self.adapter_config.non_linear,
            )
        if self.adapter_config.adapter_mlp:
            self.adapter_mlp = AdapterLayer( # This AdapterLayer might need adjustment if Qwen2MLP has a different structure
                self.hidden_size,
                self.adapter_config.hidden_dim,
                self.adapter_config.scale,
                self.adapter_config.dropout,
                # non_linear for MLP adapter is typically False or handled by MLP's activation
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.adapter_config.adapter_attn:
            hidden_states = self.adapter_attn(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.adapter_config.adapter_mlp:
            hidden_states = self.adapter_mlp(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@contextmanager
def adapter(hidden_dim, scale, dropout, enabled: bool = True, non_linear=False, attn=True, mlp=False):
    if not enabled:
        yield
        return

    Qwen2AdapterDecoderLayer.adapter_config = AdapterConfig(hidden_dim=hidden_dim, scale=scale, dropout=dropout, non_linear=non_linear, adapter_attn=attn, adapter_mlp=mlp)
    original_layer = Qwen2DecoderLayer # Target Qwen2 specific layer
    
    # Path to Qwen2DecoderLayer in transformers library
    # Ensure this path is correct based on your transformers installation structure
    # It's typically transformers.models.<model_name>.modeling_<model_name>
    target_module_path = "transformers.models.qwen2.modeling_qwen2"
    
    try:
        module = __import__(target_module_path, fromlist=['Qwen2DecoderLayer'])
        original_layer_ref = getattr(module, 'Qwen2DecoderLayer')
        setattr(module, 'Qwen2DecoderLayer', Qwen2AdapterDecoderLayer)
        yield
    finally:
        if 'module' in locals() and 'original_layer_ref' in locals(): # Ensure module was imported
            setattr(module, 'Qwen2DecoderLayer', original_layer_ref) # Restore original
        elif enabled: # Only print warning if adapter was supposed to be enabled
            print(f"Warning: Could not patch Qwen2DecoderLayer in {target_module_path}. Adapter not applied for LLM.")
            yield # Still yield to not break context management


@contextmanager
def visual_adapter(hidden_dim, scale, dropout, attn=True, mlp=False, enabled: bool = True, non_linear=False):
    if not enabled:
        yield
        return

    Qwen2VLViTAdapterEncoderLayer.adapter_config = AdapterConfig(hidden_dim=hidden_dim, scale=scale, dropout=dropout, adapter_attn=attn, adapter_mlp=mlp, non_linear=non_linear) # non_linear for vision can also be True
    
    # Path to Qwen2VLViTEncoderLayer in transformers library
    target_module_path = "transformers.models.qwen2_vl.modeling_qwen2_vl"

    try:
        module = __import__(target_module_path, fromlist=['Qwen2VLViTEncoderLayer'])
        original_layer_ref = getattr(module, 'Qwen2VLViTEncoderLayer')
        setattr(module, 'Qwen2VLViTEncoderLayer', Qwen2VLViTAdapterEncoderLayer)
        yield
    finally:
        if 'module' in locals() and 'original_layer_ref' in locals(): # Ensure module was imported
            setattr(module, 'Qwen2VLViTEncoderLayer', original_layer_ref) # Restore original
        elif enabled: # Only print warning if adapter was supposed to be enabled
            print(f"Warning: Could not patch Qwen2VLViTEncoderLayer in {target_module_path}. Adapter not applied for Vision.")
            yield # Still yield to not break context management

# Removed eva_adapter context manager
