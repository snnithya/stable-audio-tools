import typing as tp
import math
import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .blocks import FourierFeatures
from .transformer import ContinuousTransformer

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        input_add_dims=[],
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        postpend=False,
        split_qkv=False,
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type

        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        self.diffusion_objective = diffusion_objective

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if len(input_add_dims) > 0:
            # Input add conditioning, module dict
            # self.to_input_add_embed = nn.ModuleDict()
            # for id, dim in input_add_dims.items():
            #     self.to_input_add_embed[id] = nn.Linear(dim, embed_dim, bias=False)
            # self.input_add_cond_cache = None
            # convert to just a single concatenated linear layer
            # input_add_dims is now an ordered list of tuples (id, dim)
            self.input_add_dims = input_add_dims
            total_input_add_dim = sum([dim for id, dim in input_add_dims])
            self.to_input_add_embed = nn.Linear(total_input_add_dim, embed_dim, bias=False)

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size
        self.postpend = postpend

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

        if split_qkv:
            for block in self.transformer.layers:
                block.self_attn._split_qkv_projections_for_cache()

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        input_add_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        exit_layer_ix=None,
        enc_enc_mask=None,
        use_kv_cache=False,
        kv_cache=None,
        postpend=None,
        prefill=False,
        **kwargs):

        postpend = self.postpend if postpend is None else postpend

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

            prepend_length = prepend_cond.shape[1]

        add_emb = self.to_input_add_embed(input_add_cond.transpose(1, 2)) if input_add_cond is not None else None
        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(1).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        if enc_enc_mask is not None:
            x = x * enc_enc_mask

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        # Prefill: if KV cache exists but not yet initialized, run encoder-only pass to populate it.
        # This lets all N denoising steps use fast decoder-only flash attention instead of flex attention.
        if use_kv_cache and kv_cache is not None and not kv_cache.get('initialized', False) and prefill:
            enc_seq_len_prefill = 208
            kv_cache['encoder_seq_len'] = 208 #TODO hardcoded for now, need to figure out a good way to determine this dynamically based on the enc_enc_mask or input length but not trigger graph recompilation
            if enc_seq_len_prefill is not None and enc_seq_len_prefill > 0:
                x_enc = x[:, :enc_seq_len_prefill]
                add_emb_enc = add_emb[:, :enc_seq_len_prefill] if add_emb is not None else None
                # Exclude flex-attention block mask — use standard flash attention for prefill
                if kwargs.get('self_attention_block_mask', None) is not None:
                    kwargs.pop('self_attention_block_mask') # this should turn off flex attention
                enc_out = self.transformer(
                    x_enc,
                    # prepend_embeds=prepend_inputs,
                    context=cross_attn_cond,
                    return_info=False,
                    input_add_emb=add_emb_enc,
                    enc_enc_mask=None,
                    use_kv_cache=True,
                    kv_cache=kv_cache,
                    postpend=postpend,
                    **extra_args,
                    **kwargs,
                )
                # Run encoder output through the same post-processing pipeline, then cache it
                enc_out = rearrange(enc_out, "b t c -> b c t")
                # if not postpend:
                #     enc_out = enc_out[:, :, prepend_length:]
                # else:
                #     enc_out = enc_out[:, :, :-prepend_length] if prepend_length > 0 else rearrange(enc_out, "b t c -> b c t")
                if self.patch_size > 1:
                    enc_out = rearrange(enc_out, "b (c p) t -> b c (t p)", p=self.patch_size)
                enc_out = self.postprocess_conv(enc_out) + enc_out
                kv_cache['encoder_output'] = enc_out.detach()
                kv_cache['initialized'] = True

        # When KV cache is initialized (by prefill above or previous call), only pass decoder portion through the network
        cache_is_initialized = use_kv_cache and kv_cache is not None and kv_cache.get('initialized', False)
        if cache_is_initialized:
            encoder_seq_len = kv_cache['encoder_seq_len']

            # Slice x to decoder only
            rotary_seq_len = x.shape[1] + prepend_length if prepend_inputs is not None else x.shape[1]
            kwargs['rotary_seq_len'] = rotary_seq_len
            x = x[:, encoder_seq_len:]

            # Truncate input_add_emb to decoder portion (should be zeros there anyway)
            if add_emb is not None:
                add_emb = add_emb[:, encoder_seq_len:]

            # Use standard bidirectional attention for decoder (no custom mask needed)
            enc_enc_mask = None

            # remove mask kwargs
            if kwargs.get('self_attention_block_mask', None) is not None:
                kwargs.pop('self_attention_block_mask') # this should turn off flex attention

        if self.transformer_type == "continuous_transformer":
            # Masks not currently implemented for continuous transformer
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, return_info=return_info, exit_layer_ix=exit_layer_ix, input_add_emb=add_emb, enc_enc_mask=enc_enc_mask, use_kv_cache=use_kv_cache, kv_cache=kv_cache, postpend=postpend, **extra_args, **kwargs)

            if return_info:
                output, info = output

            # Avoid postprocessing on early exit
            if exit_layer_ix is not None:
                if return_info:
                    return output, info
                else:
                    return output



        if not postpend:
            output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]
        else:
            output = rearrange(output, "b t c -> b c t")[:,:,:-prepend_length] if prepend_length > 0 else rearrange(output, "b t c -> b c t")

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        # Cache encoder output on first pass, or restore it on subsequent passes
        if use_kv_cache and kv_cache is not None:
            if not kv_cache.get('initialized', False):
                # First pass: cache encoder portion of output
                encoder_seq_len = kv_cache.get('encoder_seq_len')
                if encoder_seq_len is not None and encoder_seq_len > 0:
                    kv_cache['encoder_output'] = output[..., :encoder_seq_len].detach()
                kv_cache['initialized'] = True
            elif 'encoder_output' in kv_cache:
                # Subsequent passes: prepend cached encoder output to decoder output
                kv_cache['initialized'] = True
                encoder_output = kv_cache['encoder_output']
                output = torch.cat([encoder_output, output], dim=-1)

        if return_info:
            return output, info
        return output

    def apg_project(self, v0, v1):
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        input_add_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_norm_threshold=0.0,
        cfg_interval = (0, 1),
        scale_phi=0.0,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        enc_enc_mask=None,
        use_kv_cache=False,
        kv_cache=None,
        **kwargs):


        model_dtype = next(self.parameters()).dtype
        
        x = x.to(model_dtype)

        t = t.to(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        if input_add_cond is not None:
            # Convert input_add_cond to the model dtype
            input_add_cond = input_add_cond.to(model_dtype)

        # Early exit bypasses CFG processing
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                input_add_cond=input_add_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                enc_enc_mask=enc_enc_mask,
                use_kv_cache=use_kv_cache,
                kv_cache=kv_cache,
                **kwargs
            )

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:

            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)


            if input_add_cond is not None:
                # get dims, apply dropout to each individual conditioning
                # input_add_cond is a concat of multiple conditionings into 1 tensor along channel dim
                # self.input_add_dims is ordered list of tuples (id, dim)
                total_dim = input_add_cond.shape[1]
                start_idx = 0
                for id, dim in self.input_add_dims:
                    end_idx = start_idx + dim
                    null_embed = torch.zeros_like(input_add_cond[:, start_idx:end_idx, :], device=input_add_cond.device)
                    dropout_mask = torch.bernoulli(torch.full((input_add_cond.shape[0], 1, 1), cfg_dropout_prob, device=input_add_cond.device)).to(torch.bool)
                    input_add_cond[:, start_idx:end_idx, :] = torch.where(dropout_mask, null_embed, input_add_cond[:, start_idx:end_idx, :])
                    start_idx = end_idx


        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            if input_add_cond is not None:
                batch_input_add_cond = torch.cat([input_add_cond, input_add_cond], dim=0)
            else:
                batch_input_add_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:

                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                    
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
         

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None

            if enc_enc_mask is not None:
                batch_enc_enc_mask = torch.cat([enc_enc_mask, enc_enc_mask], dim=0)
            else:
                batch_enc_enc_mask = None
            
            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                mask = batch_masks,
                input_concat_cond=batch_input_concat_cond,
                input_add_cond=batch_input_add_cond,
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                return_info = return_info,
                enc_enc_mask = batch_enc_enc_mask,
                use_kv_cache=use_kv_cache,
                kv_cache=kv_cache,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale
                
            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                output = cfg_output
                
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                input_add_cond=input_add_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                enc_enc_mask=enc_enc_mask,
                use_kv_cache=use_kv_cache,
                kv_cache=kv_cache,
                **kwargs
            )