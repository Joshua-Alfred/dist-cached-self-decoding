from typing import Optional, List, Union, Tuple

from transformers.generation.utils import (
        GenerateNonBeamOutput, 
        LogitsProcessorList, 
        StoppingCriteriaList, 
        GenerateDecoderOnlyOutput, 
        GenerateEncoderDecoderOutput
)
from transformers.generation.stopping_criteria import validate_stopping_criteria

from transformers.generation import GenerateDecoderOnlyOutput
import torch

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import torch.distributed as dist

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaAttention,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)


from transformers.cache_utils import Cache, DynamicCache, StaticCache


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(self, x):
        _, _, hsz = x.shape
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_weights = self.gate_proj.weight.split(slice, dim=0)
            up_proj_weights = self.up_proj.weight.split(slice, dim=0)
            down_proj_weights = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_weights[i], None) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_weights[i], None) for i in range(self.config.pretraining_tp)], dim=-1
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_weights[i], None) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            gate_proj = F.linear(x, self.gate_proj.weight[:, :hsz], None)
            up_proj = F.linear(x, self.up_proj.weight[:, :hsz], None)
            down_proj = F.linear(self.act_fn(gate_proj) * up_proj, self.down_proj.weight[:hsz], None)

        return down_proj



class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        skip_layer = kwargs.pop("skip_layer", False)

        if skip_layer:
            hidden_states = residual
            present_key_value = None
        else:
            hidden_states = self.input_layernorm(hidden_states)

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

        if skip_layer:
            hidden_states = residual
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # register a causal mask to separate causal and padding mask creation. Merging happends in the attention class
        causal_mask = torch.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1)
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        skip_layers: Optional[List[int]] = None,
        draft_hidden_states: Optional[torch.FloatTensor] = None,
        verify_start_index: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if draft_hidden_states is not None:
            start_idx = verify_start_index
            hidden_states = draft_hidden_states[verify_start_index]

            # Fix hidden states from draft output
            if output_hidden_states: 
                for i in range(0, verify_start_index):
                    all_hidden_states += (draft_hidden_states[i],)
        else:
            start_idx = 0
            hidden_states = inputs_embeds

        for i in range(start_idx, len(self.layers)):
            decoder_layer = self.layers[i]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if skip_layers and i in skip_layers:
                skip_layer = True
            else:
                skip_layer = False

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                skip_layer=skip_layer,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full((2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), fill_value=1)
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        if hasattr(self, "causal_mask"):  # we use the current dtype to avoid any overflows
            causal_mask = (
                self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * torch.finfo(dtype).min
            )
        else:
            mask = torch.full(
                (self.config.max_position_embeddings, self.config.max_position_embeddings),
                fill_value=torch.finfo(dtype).min,
            )
            causal_mask = torch.triu(mask, diagonal=1)

        causal_mask = causal_mask.to(dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask, torch.finfo(dtype).min
            )

        if self.config._attn_implementation == "sdpa":
            is_tracing = torch.jit.is_tracing() or isinstance(input_tensor, torch.fx.Proxy)
            if not is_tracing and attention_mask is not None and torch.any(attention_mask != 1):
                causal_mask = causal_mask.mul(~torch.all(causal_mask == causal_mask.min(), dim=-1)[..., None]).to(
                    dtype
                )

        return causal_mask

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _validate_model_kwargs(self, model_kwargs):
        # Skip validation for custom parameters
        if "draft_skip_layers" in model_kwargs:
            model_kwargs.pop("draft_skip_layers")
        if "num_candidate_tokens" in model_kwargs:
            model_kwargs.pop("num_candidate_tokens")

        super()._validate_model_kwargs(model_kwargs)

    def generate(self, *args, **kwargs):
        # Check if the custom generation strategy should be used
        if "draft_skip_layers" in kwargs:
            return self._generate_self_speculative_decoding(*args, **kwargs)
        else:
            return super().generate(*args, **kwargs)

    def _sample(self, logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):
        if return_probs:

            all_probs = logits.softmax(-1)
            if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
                _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
                output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
                probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
            else:
                probs, output_ids = torch.max(all_probs, dim=-1)
                
            return output_ids, probs

        else:

            if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
                _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
                output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            else:
                output_ids = torch.argmax(logits, dim=-1)
                
            return output_ids

    def _generate_self_speculative_decoding(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        draft_skip_layers = model_kwargs.pop("draft_skip_layers", None)
        num_candidate_tokens = model_kwargs.pop("num_candidate_tokens", 5)

        # (brad) Edit: I moved this up here...
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        step = 0
        step_draft = 0
        step_verify = 0
        max_new_tokens=max_length
        max_step_draft=num_candidate_tokens
        th_stop_draft=0.3
        auto_th_stop_draft=True
        auto_parameters=[1,0.5,0.9,1e-2,0.9]
        do_sample=False
        do_sample_draft=False
        top_k=0
        top_p=0.85
        temperature=0.2
        
        current_input_ids = input_ids
        generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=self.model.device)
        draft_generate_ids = torch.empty([input_ids.size(0), max_step_draft+1], dtype=torch.long, device=self.model.device)
        past_key_values = None

        n_matched = 0
        n_drafted = 0
        tmp_n_matched = 0
        tmp_n_drafted = 0
        tmp_matchness = 0
        n_layers = len(self.model.layers) + 1
        draft_hidden_states = torch.empty(
            [n_layers, input_ids.size(0), max_step_draft, self.config.hidden_size],
            dtype=torch.bfloat16,
            device=self.model.device
        )
        with torch.no_grad():
            while True:
                if step >= max_new_tokens:
                    break

                if step == 0:

                    # TODO sending to central hub and receive results

                    output_ids, past_key_values = self._first_token_run(current_input_ids, past_key_values)
                    generate_ids[:, step] = output_ids
                    current_input_ids = output_ids
                    past_key_values = past_key_values
                    step += 1

                else:
                       
                    draft_current_input_ids = current_input_ids
                    draft_past_key_values = past_key_values
                    draft_generate_ids[:, 0] = current_input_ids
                    for step_draft in range(max_step_draft):
                        draft_output = self(
                            input_ids=draft_current_input_ids,
                            past_key_values=draft_past_key_values,
                            return_dict=True,
                            use_cache=True,
                            output_hidden_states=True,
                            skip_layers=draft_skip_layers,
                        )

                        
                        draft_output_ids = torch.argmax(draft_output['logits'], dim=-1)
                        draft_output_ids, draft_output_probs = self._sample(
                            draft_output['logits'],
                            return_probs=True,
                            do_sample=do_sample_draft,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature
                        )
                        draft_generate_ids[:, step_draft+1] = draft_output_ids
                        draft_current_input_ids = draft_output_ids
                        draft_past_key_values = draft_output['past_key_values']
                        for i in range(n_layers):
                            draft_hidden_states[i, :, step_draft] = draft_output['hidden_states'][i]

                        # if draft_output_probs.item() < th_stop_draft or step + step_draft + 2 >= max_new_tokens:
                        if step + step_draft + 1 >= max_new_tokens:
                            break
                    
                    drafted_n_tokens = step_draft + 1
                    drafted_input_ids = draft_generate_ids[:, :drafted_n_tokens]

                    if draft_skip_layers and len(draft_skip_layers) > 0:
                        verify_start_index = draft_skip_layers[0]
                    else:
                        verify_start_index = 0

                    # TODO sending to central hub and receive results 

                    step, step_verify, generate_ids, current_input_ids, past_key_values, tmp_matchness, n_matched, n_drafted, th_stop_draft = self._backbone_verifier(
                                                                                                    drafted_input_ids=drafted_input_ids,
                                                                                                    current_input_ids=current_input_ids,
                                                                                                    past_key_values=past_key_values,
                                                                                                    draft_hidden_states=draft_hidden_states,
                                                                                                    verify_start_index=verify_start_index,
                                                                                                    do_sample=do_sample,
                                                                                                    draft_past_key_values=draft_past_key_values,
                                                                                                    generate_ids=generate_ids,
                                                                                                    tmp_matchness=tmp_matchness,
                                                                                                    n_matched=n_matched,
                                                                                                    n_drafted=n_drafted,
                                                                                                    th_stop_draft=th_stop_draft,
                                                                                                    step_verify=step_verify,
                                                                                                    drafted_n_tokens=drafted_n_tokens,
                                                                                                    step=step)

                    
        step = min(step, max_new_tokens)
        generate_ids = generate_ids[:, :step]

        return {
            'sequences': generate_ids,
            'matchness': n_matched/n_drafted,
            'num_drafted_tokens': n_drafted,
            'th_stop_draft': th_stop_draft,
        }       
                    

    def _first_token_run(self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[List[torch.Tensor]] = None
            ) -> Tuple[torch.LongTensor, Optional[List[torch.Tensor]]]:
        

        output = self(
            input_ids=input_ids,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True
            )
    
        logits = output['logits'][:, -1:]
        output_ids = self._sample(
            logits,
            do_sample=False,
            top_k=0,
            top_p=0.85,
            temperature=0.2
            )   

        return output_ids, output['past_key_values']

    def _backbone_verifier(self,
            drafted_input_ids: torch.LongTensor,
            current_input_ids: torch.LongTensor,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            draft_hidden_states: Optional[torch.FloatTensor] = None,
            verify_start_index: Optional[int] = None,
            do_sample: Optional[bool] = None,
            draft_past_key_values: Optional[List[torch.FloatTensor]] = None,
            generate_ids: Optional[List[torch.FloatTensor]] = None,
            tmp_matchness: Optional[int] = None,
            n_matched: Optional[int] = None,
            n_drafted: Optional[int] = None,
            th_stop_draft: Optional[int] = None,
            step_verify: Optional[int] = None,
            drafted_n_tokens: Optional[int] = None,
            step: Optional[int] = None) -> Tuple[int, int, Optional[List[torch.FloatTensor]], torch.LongTensor, Optional[List[torch.FloatTensor]], int, int, int, float]: 
                     
        output = self(
            input_ids=drafted_input_ids,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True,
            draft_hidden_states=draft_hidden_states[:, :, :drafted_n_tokens],
            verify_start_index=verify_start_index,
            )
        logits = output['logits']
        output_ids = self._sample(
            logits,
            do_sample=do_sample,
            top_k=0,
            top_p=0.85,
            temperature=0.2
            )

        past_key_values = output['past_key_values']
        

        # (brad) use draft_past_key_values up to verify_start_index
        tmp_past_key_values = past_key_values
        past_key_values = []
        for i in range(verify_start_index):
            past_key_values.append(draft_past_key_values[i])
        for i in range(verify_start_index, len(tmp_past_key_values)):
            past_key_values.append(tmp_past_key_values[i])

        past_key_values = tuple(past_key_values)

        max_matched = ((output_ids[:, :-1] != drafted_input_ids[:, 1:]).cumsum(-1) == 0).sum(-1).item() + 1
        max_of_max_matched = output_ids.size(1)

        # trim kv-cache up to max_matched
        if max_of_max_matched != max_matched:
            output_ids = output_ids[:, :max_matched]
                        
            past_key_values = [
                (k[:, :, :-(max_of_max_matched - max_matched)], v[:, :, :-(max_of_max_matched - max_matched)]) for k, v in past_key_values
            ]

        generate_ids[:, step:step+output_ids.size(1)] = output_ids
        current_input_ids = output_ids[:, -1:]

        step += output_ids.size(1)

        # remove one generated by the base model
        n_matched += max_matched
        n_drafted += drafted_n_tokens
        # tmp_n_matched += max_matched
        # tmp_n_drafted += drafted_n_tokens
        step_verify += 1
        auto_th_stop_draft=True
        auto_parameters=[1,0.5,0.9,1e-2,0.9]

        if auto_th_stop_draft and step_verify % auto_parameters[0] == 0:
            tmp_matchness = auto_parameters[1]*(tmp_matchness) + (1-auto_parameters[1])*((max_matched - 1)/drafted_n_tokens)
            if tmp_matchness<auto_parameters[2]:
                new_th_stop_draft = th_stop_draft+auto_parameters[3]
            else:
                if drafted_n_tokens==max_step_draft:
                    new_th_stop_draft = th_stop_draft
                else:
                    new_th_stop_draft = th_stop_draft-auto_parameters[3]
            th_stop_draft = auto_parameters[4] * th_stop_draft + (1-auto_parameters[4]) * new_th_stop_draft
            print(f"Drafted: {drafted_n_tokens}, Matched: {max_matched}, Matchness: {tmp_matchness:.2f}, Th_stop_draft: {th_stop_draft:.2f}")
        
        return step, step_verify, generate_ids, current_input_ids, past_key_values, tmp_matchness, n_matched, n_drafted, th_stop_draft 


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        skip_layers: Optional[List[int]] = None,
        draft_hidden_states: Optional[torch.FloatTensor] = None,
        verify_start_index: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            skip_layers=skip_layers,
            draft_hidden_states=draft_hidden_states,
            verify_start_index=verify_start_index,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # logits = self.lm_head(hidden_states)
            _, _, hsz = hidden_states.shape
            logits = F.linear(hidden_states, self.lm_head.weight[:, :hsz])
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if past_key_value := getattr(self.model.layers[0].self_attn, "past_key_value", None):
            # generation with static cache
            past_length = past_key_value.get_seq_length()
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = kwargs.get("cache_position", None)
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + position_ids.shape[-1], device=position_ids.device
            )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


