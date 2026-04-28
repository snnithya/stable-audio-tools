import numpy as np
import torch 
import typing as tp
import math 
from torchaudio import transforms as T
from torch.nn.functional import interpolate

from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf
from ..data.utils import PadCrop
from torch.nn.attention.flex_attention import create_block_mask, or_masks
from stable_audio_tools.inference.utils import prepare_audio
from stable_audio_tools.inference.sampling import sample_discrete_euler
import copy

def generate_diffusion_uncond(
        model,
        steps: int = 250,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None

    # Inpainting mask
    
    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level
        mask = None 
    else:
        mask = None

    # Now the generative AI part:

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, mask, steps, **sampler_kwargs, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:
        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, device=device)

    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        use_kv_cache: bool = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    # Initialize KV cache if enabled
    kv_cache = None
    if use_kv_cache:
        kv_cache = {
            'initialized': False,
        }
        conditioning_inputs['use_kv_cache'] = True
        conditioning_inputs['kv_cache'] = kv_cache

    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled

def generate_diffusion_cond_inpaint(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask = None,
        return_latents = False,
        use_kv_cache: bool = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion inpainting model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        inpaint_mask: A mask to use for inpainting. Shape should be [batch_size, sample_size]
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
    
    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.float()

    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=init_audio.shape[-1], mode='nearest').squeeze(1)

        init_audio = init_audio.repeat(batch_size, 1, 1)

    if inpaint_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        inpaint_sr, inpaint_audio = inpaint_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        inpaint_audio = prepare_audio(inpaint_audio, in_sr=inpaint_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            inpaint_audio = model.pretransform.encode(inpaint_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=inpaint_audio.shape[-1], mode='nearest').squeeze(1)

        inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
    else:
       
        if inpaint_mask is not None:
            # interpolate inpaint mask to the sample size
            inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode='nearest').squeeze(1)

    if inpaint_mask is None:
        mask = torch.zeros((batch_size, 1, sample_size), device=device)  
    else:
        mask = inpaint_mask.unsqueeze(1)

    # Inpainting mask
    mask = mask.to(device)

    if inpaint_audio is not None:
        inpaint_input = inpaint_audio * mask.expand_as(inpaint_audio)
    else:
        inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    
    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    # Initialize KV cache if enabled
    kv_cache = None
    if use_kv_cache:
        kv_cache = {
            'initialized': False,
        }
        conditioning_inputs['use_kv_cache'] = True
        conditioning_inputs['kv_cache'] = kv_cache

    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled

def generate_diffusion_cond_blockar(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        return_latents = False,
        ar_style: str = 'outpaint',
        block_size: int = 98304,
        generation_length: int = 2097152,
        silence_dir: str = '/home/zachary/code/stable-audio-tools/notebooks/',
        use_kv_cache: bool = False,
        enc_enc: bool = False,
        enc_enc_attention_pattern: tp.Optional[str] = None,
        speedtest: bool = False,
        **sampler_kwargs
    ) -> torch.Tensor: 
    '''
    The idea here is to do block-wise autoregressive generation, where we generate a block of audio at a time
    TODO: This currently will only support the 'outpaint' style, where we generate by:
    1) Initialize the mask to condition on the latent sample_size - block_size samples, with the masked_input set to the initial audio (if it exists) or zeros. if init_audio is provided, we'll condition on the last sample_size - block_size samples of it
    2) Generate the first block of size block_size (i.e. the rightmost block_size latent samples)
    3) Update the masked input to include the newly generated audio by sliding it over to the left by block_size samples
    4) Repeat steps 2-3 until we've generated generation_length samples
    The interior sampling loop should be similar to generate_diffusion_cond
    '''
    assert ar_style == 'outpaint', "Only 'outpaint' ar_style is currently supported"

    generated_audio = []
    total_generated = 0

    audio_sample_size = sample_size

    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        block_size = block_size // model.pretransform.downsampling_ratio
        generation_length = generation_length // model.pretransform.downsampling_ratio
    
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}


    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio.to(next(model.pretransform.parameters()).dtype))
            
        init_audio = init_audio.repeat(batch_size, 1, 1)

    mask = torch.zeros((batch_size, 1, sample_size), device=device)
    print(f"Block size: {block_size}, Sample size: {sample_size}")
    mask[:, :, :sample_size - block_size] = 1.0  # condition on the leftmost sample_size - block_size samples

    if init_audio is not None:
        # truncate init_audio to the first sample_size - block_size samples
        init_audio = init_audio[:, :, : (sample_size - block_size)]
        # add initial to generated_audio
        generated_audio.append(init_audio.detach())
        # pad init_audio to sample_size with zeros on the right
        init_audio = torch.cat([init_audio, torch.zeros((batch_size, model.io_channels, block_size), device=device)], dim=2)
        inpaint_input = init_audio
    else:
        # try to load in mean_silence.pt and scale_silence.pt from silence_dir to use as the initial audio
        try:
            silence_mean = torch.load(silence_dir + 'mean_silence.pt').to(device)
            silence_scale = torch.load(silence_dir + 'scale_silence.pt').to(device)
            # truncate or extend to sample_size
            if silence_mean.shape[2] > sample_size:
                silence_mean = silence_mean[:, :, :sample_size]
                silence_scale = silence_scale[:, :, :sample_size]
            elif silence_mean.shape[2] < sample_size:
                # repeat
                repeat_factor = (sample_size + silence_mean.shape[2] - 1) // silence_mean.shape[2]
                silence_mean = silence_mean.repeat(1, 1, repeat_factor)[:, :, :sample_size]
                silence_scale = silence_scale.repeat(1, 1, repeat_factor)[:, :, :sample_size]
            inpaint_input = silence_mean + torch.randn((batch_size, model.io_channels, sample_size), device=device) * silence_scale
            print("Loaded silence mean and scale for initial inpaint input")
        except Exception as e:
            print(f"Could not load silence mean and scale for initial inpaint input: {e}")
            inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    if enc_enc:
        print("Using enc-enc attention with block size", block_size)
        sampler_kwargs["enc_enc_mask"] = (1 - mask)
        assert torch.all(sampler_kwargs["enc_enc_mask"][..., :sample_size - block_size] == 0), f"enc-enc mask should be 0 for the first sample_size - block_size samples, but got {sampler_kwargs['enc_enc_mask'][..., :sample_size+1 - block_size]}"
        if enc_enc_attention_pattern is not None:
            fixed_mask_size = sample_size - block_size
            seq_len = sample_size
            assert fixed_mask_size is not None, "fixed_mask_size must be specified in inpaint_mask_kwargs when using enc_enc"
            print(f"Creating enc-enc self attention block mask with pattern {enc_enc_attention_pattern} and fixed mask size {fixed_mask_size} for sequence length {seq_len}")
            match enc_enc_attention_pattern:
                case "enc-dec":
                    # if "postpend" in sampler_kwargs and sampler_kwargs["postpend"]:
                    #     # we're moving the prepend cond to the end of the sequnce, so we need to roll the sequence by 1
                    #     # basically like a modulo operation, the last position should be like the whole prefix
                    #     def prefix_mask(b, h, q_idx, kv_idx):
                    #         return (kv_idx + 1) % (seq_len+1) < fixed_mask_size

                    #     def postfix_mask(b, h, q_idx, kv_idx):
                    #         return (q_idx + 1) % (seq_len+1) >= fixed_mask_size
                    # else:
                    def prefix_mask(b, h, q_idx, kv_idx):
                        return kv_idx < fixed_mask_size
                    
                    def postfix_mask(b, h, q_idx, kv_idx):
                        return q_idx >= fixed_mask_size
                case "block-causal":
                    block_size = seq_len - fixed_mask_size # TODO: all this +1 bullshit is because of preprend conditioning the timestep, so all sequences are actually +1 longer
                    # make prefix mask block causal
                    def prefix_mask(b, h, q_idx, kv_idx):
                        mod_idx = fixed_mask_size % block_size
                        block_idx = (kv_idx - mod_idx) // block_size
                        q_block_idx = (q_idx - mod_idx) // block_size
                        return q_block_idx >= (block_idx)
                    def postfix_mask(b, h, q_idx, kv_idx):
                        return q_idx >= fixed_mask_size
            
            mask_mod = or_masks(prefix_mask, postfix_mask) 
            # create the mask now
            
            
            sampler_kwargs["self_attention_block_mask"] = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len+1, KV_LEN=seq_len+1, device=noise.device, _compile=True)
            print('Created self attention block mask:')
            print(sampler_kwargs["self_attention_block_mask"].to_string())

    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)

    model_dtype = next(model.model.parameters()).dtype
    while total_generated < generation_length:
        noise = noise.type(model_dtype)
        conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

        # Initialize KV cache if enabled (reset for each block)
        kv_cache = None
        if use_kv_cache:
            # Clear per-module caches from previous block
            model.model.model.transformer.clear_kv_cache()
            kv_cache = {
                'initialized': False,
            }
            conditioning_inputs['use_kv_cache'] = True
            conditioning_inputs['kv_cache'] = kv_cache

        # k-diffusion denoising process go!
        diff_objective = model.diffusion_objective

        if diff_objective == "v":    
            # k-diffusion denoising process go!
            sampled = sample_k(model.model, noise, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device, inpaint_masked_input=inpaint_input, inpaint_mask=mask)
        elif diff_objective in ["rectified_flow", "rf_denoiser"]:

            if "sigma_min" in sampler_kwargs:
                del sampler_kwargs["sigma_min"]

            if "rho" in sampler_kwargs:
                del sampler_kwargs["rho"]

            if speedtest:
                n_warmup = 10
                n_iters = 100
                # warmup
                for _ in range(n_warmup):
                    _ = sample_rf(model.model, noise, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device, inpaint_masked_input=inpaint_input, inpaint_mask=mask)
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                for _ in range(n_iters):
                    _ = sample_rf(model.model, noise, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device, inpaint_masked_input=inpaint_input, inpaint_mask=mask)
                t1.record()
                torch.cuda.synchronize()
                print(f"Average inference time per block: {t0.elapsed_time(t1) / n_iters} ms")
                return

            else:
                sampled = sample_rf(model.model, noise, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device, inpaint_masked_input=inpaint_input, inpaint_mask=mask)
        
       # Get the last block_size samples from sampled
        generated_block = sampled[:, :, -block_size:]
        generated_audio.append(generated_block.detach())
        total_generated += block_size
        if total_generated >= generation_length:
            del sampled
            del conditioning_tensors
            del conditioning_inputs
            torch.cuda.empty_cache()
            break
        # update inpaint_input
        inpaint_input = inpaint_input.detach()
        inpaint_input[..., -block_size:] = generated_block
        # slide inpaint_input to the left by block_size samples
        inpaint_input = torch.cat([inpaint_input[:, :, block_size:], torch.zeros_like(inpaint_input[:, :, :block_size])], dim=2)
        # mask stays the same
        conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
        if negative_conditioning_tensors:
            negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
            negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
        noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
        del sampled

    generated_audio = torch.cat(generated_audio, dim=2)
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        generated_audio = generated_audio.to(next(model.pretransform.parameters()).dtype)
        generated_audio = model.pretransform.decode(generated_audio)

    # Return audio
    return generated_audio
        


# builds a softmask given the parameters
# returns array of values 0 to 1, size sample_size, where 0 means noise / fresh generation, 1 means keep the input audio, 
# and anything between is a mixture of old/new
# ideally 0.5 is half/half mixture but i haven't figured this out yet
def build_mask(sample_size, mask_args):
    maskstart = math.floor(mask_args["maskstart"]/100.0 * sample_size)
    maskend = math.ceil(mask_args["maskend"]/100.0 * sample_size)
    softnessL = round(mask_args["softnessL"]/100.0 * sample_size)
    softnessR = round(mask_args["softnessR"]/100.0 * sample_size)
    marination = mask_args["marination"]
    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softnessL*2, periodic=False)[:softnessL]
    hannR = torch.hann_window(softnessR*2, periodic=False)[softnessR:]
    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softnessL] = hannL
    mask[maskend-softnessR:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final rounds
    if marination > 0:        
        mask = mask * (1-marination) 
    #print(mask)
    return mask

def generate_diffusion_flowedit(
        model,
        steps: int = 250,
        src_cfg_scale=6,
        tar_cfg_scale=6,
        src_conditioning_tensors: tp.Optional[dict] = None,
        tar_conditioning_tensors: tp.Optional[dict] = None,
        n_avg: int = 1,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert src_conditioning_tensors is not None, "Must provide src_conditioning_tensors"
    assert tar_conditioning_tensors is not None, "Must provide tar_conditioning_tensors"
    src_conditioning_inputs = model.get_conditioning_inputs(src_conditioning_tensors)
    for k, v in src_conditioning_inputs.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
    tar_conditioning_inputs = model.get_conditioning_inputs(tar_conditioning_tensors)
    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    src_conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in src_conditioning_inputs.items()}
    tar_conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in tar_conditioning_inputs.items()}

    z_t_fe = init_audio.clone()
    init_audio = init_audio.unsqueeze(0).repeat(n_avg, 1, 1, 1) # (n_avg, batch_size, channels, length)??
    z_t_fe = z_t_fe.unsqueeze(0).repeat(n_avg, 1, 1, 1) # (n_avg, batch_size, channels, length)??
    print('z_t_fe.shape', z_t_fe.shape)

    for i in np.linspace(1, 0, steps+1)[:-1]:
        print('i', i)
        t = torch.Tensor([i]).repeat(n_avg).to(device)
        noise = torch.randn_like(z_t_fe).to(device)
        z_t_src = (1 - i) * init_audio + i * noise
        print('z_t_src.shape', z_t_src.shape, z_t_src.device)
   
        z_t_tar = z_t_fe + z_t_src - init_audio
        # z_t_tar = z_t_tar.view(-1, *z_t_tar.shape[2:]) # (n_avg * batch_size, channels, length)
        print('z_t_tar.shape', z_t_tar.shape, z_t_tar.device)

        z_t_src = z_t_src.view(-1, *z_t_src.shape[2:]) # (n_avg * batch_size, channels, length)
        z_t_tar = z_t_tar.view(-1, *z_t_tar.shape[2:]) # (n_avg * batch_size, channels, length)

        v_tar = model.model(z_t_tar, t, **tar_conditioning_inputs, cfg_scale=tar_cfg_scale)
        v_src = model.model(z_t_src, t, **src_conditioning_inputs, cfg_scale=src_cfg_scale)
        v_delta = v_tar - v_src

        v_delta = v_delta.view(n_avg, batch_size, *v_delta.shape[1:])
        v_delta = v_delta.mean(0, keepdim=True) # (1, batch_size, channels, length)
        print('v_delta.shape', v_delta.shape)
        z_t_fe = z_t_fe - v_delta/steps
        print('z_t_fe.shape', z_t_fe.shape)
    
    z_t_fe = z_t_fe[0]  # (batch_size, channels, length), all elements along first dim should be the same

    del noise
    del src_conditioning_tensors
    del tar_conditioning_tensors
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = z_t_fe.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled

def generate_diffusion_inversion(
        model,
        steps: int = 250,
        src_cfg_scale=6,
        tar_cfg_scale=6,
        src_conditioning_tensors: tp.Optional[dict] = None,
        tar_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert src_conditioning_tensors is not None, "Must provide src_conditioning_tensors"
    assert tar_conditioning_tensors is not None, "Must provide tar_conditioning_tensors"
    src_conditioning_inputs = model.get_conditioning_inputs(src_conditioning_tensors)
    tar_conditioning_inputs = model.get_conditioning_inputs(tar_conditioning_tensors)
    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
        print(init_audio.shape)
        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    src_conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in src_conditioning_inputs.items()}
    tar_conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in tar_conditioning_inputs.items()}

    ts = torch.linspace(0, 1, steps+1).to(device)
    inverse = sample_discrete_euler(model.model, init_audio, steps=steps, sigmas=ts[:-1],  **src_conditioning_inputs, cfg_scale=src_cfg_scale)
    print(inverse.shape)
    sampled = sample_discrete_euler(model.model, inverse, steps=steps, sigmas=ts.flip(0)[:-1], **tar_conditioning_inputs, cfg_scale=tar_cfg_scale)
    print(sampled.shape)

    del src_conditioning_tensors
    del tar_conditioning_tensors
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled

def generate_diffusion_latent_flowedit(
        model,
        src_inv_cfg_scale=6,
        tar_inv_cfg_scale=6,
        src_lfe_cfg_scale=6,
        tar_lfe_cfg_scale=6,
        src_conditioning_tensors: tp.Optional[dict] = None,
        tar_conditioning_tensors: tp.Optional[dict] = None,
        n_avg: int = 1,
        batch_size: int = 1,
        sample_size: int = 2097152,
        # sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        deterministic_inverse = False,
        noise_amt = 1.0,
        inv_steps = 10,
        lfe_steps = 10,
        return_intermediate_latents = False,
        intermediate_latents_interval = 5,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    # print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert src_conditioning_tensors is not None, "Must provide src_conditioning_tensors"
    assert tar_conditioning_tensors is not None, "Must provide tar_conditioning_tensors"
    src_conditioning_inputs = model.get_conditioning_inputs(src_conditioning_tensors)
    tar_conditioning_inputs = model.get_conditioning_inputs(tar_conditioning_tensors)
    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    src_conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in src_conditioning_inputs.items()}
    src_conditioning_inputs_lfe = {k: v.expand(n_avg, *v.shape[1:]).type(model_dtype) if v is not None else v for k, v in src_conditioning_inputs.items()} # allowing batch processing of lfe
    tar_conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in tar_conditioning_inputs.items()}
    tar_conditioning_inputs_lfe = {k: v.expand(n_avg, *v.shape[1:]).type(model_dtype) if v is not None else v for k, v in tar_conditioning_inputs.items()} # allowing batch processing of latent flowedit
    intermediate_latents = []
    if inv_steps > 0 and noise_amt > 0:
        ts = torch.linspace(0, noise_amt, inv_steps+1).to(device)
    else:
        ts = None
        
    # inverse
    if ts is not None:
        # print('in inverse')
        if deterministic_inverse:
            # print('deterministic inverse')
            z_t_lfe = sample_discrete_euler(model.model, init_audio, steps=inv_steps, sigmas=ts[:-1], **src_conditioning_inputs, cfg_scale=src_inv_cfg_scale)
        else:
            # print('stochastic inverse')
            z_t_lfe = (1 - noise_amt) * init_audio + noise_amt * noise
    else:
        z_t_lfe = init_audio
    
    if lfe_steps > 0 and noise_amt < 1:
        # print('in latent flowedit')
        # latent flowedit
        inv = z_t_lfe.clone().unsqueeze(0)
        z_t_lfe = z_t_lfe.unsqueeze(0).repeat(n_avg, 1, 1, 1) # (n_avg, batch_size, channels, length)??
        # print('z_t_lfe.shape', z_t_lfe.shape)
        
        for ind, i in enumerate(np.linspace(1, 0, lfe_steps+1)[:-1]):
            t = torch.Tensor([i]).repeat(n_avg).to(device)
            noise = torch.randn_like(z_t_lfe).to(device)
            z_t_src = (1 - i) * init_audio + i * noise
    
            z_t_tar = (z_t_lfe - inv) / (1 - noise_amt) + z_t_src
            # z_t_tar = z_t_tar.view(-1, *z_t_tar.shape[2:]) # (n_avg * batch_size, channels, length)

            z_t_src = z_t_src.view(-1, *z_t_src.shape[2:]) # (n_avg * batch_size, channels, length)
            z_t_tar = z_t_tar.view(-1, *z_t_tar.shape[2:]) # (n_avg * batch_size, channels, length)

            v_tar = model.model(z_t_tar, t, **tar_conditioning_inputs_lfe, cfg_scale=tar_lfe_cfg_scale)
            v_src = model.model(z_t_src, t, **src_conditioning_inputs_lfe, cfg_scale=src_lfe_cfg_scale)
            v_delta = v_tar - v_src

            v_delta = v_delta.view(n_avg, batch_size, *v_delta.shape[1:])
            v_delta = v_delta.mean(0, keepdim=True) # (1, batch_size, channels, length)
            
            z_t_lfe = z_t_lfe - v_delta/lfe_steps* (1 - noise_amt)
            if return_intermediate_latents and ind % intermediate_latents_interval == 0:
                intermediate_latents.append(z_t_lfe[0].clone())
        z_t_lfe = z_t_lfe[0]
    # decode
    if ts is not None:
        # print('in decode')
        sampled = sample_discrete_euler(model.model, z_t_lfe, steps=inv_steps, sigmas=ts.flip(0)[:-1], **tar_conditioning_inputs, cfg_scale=tar_inv_cfg_scale)
    else:
        sampled = z_t_lfe
    intermediate_sampled = []
    if return_intermediate_latents:
        ts = torch.linspace(0, 1, lfe_steps+1).to(device)
        for ind, intermediate_latent in enumerate(intermediate_latents):
            t_val = ts[ind]
            mixed_cond = copy.deepcopy(src_conditioning_tensors)
            mixed_cond['prompt'] = (
                src_conditioning_tensors['prompt'][0] * (1-t_val) + tar_conditioning_tensors['prompt'][0] * (t_val),
                src_conditioning_tensors['prompt'][1] & tar_conditioning_tensors['prompt'][1]
            )
            lfe_mixed_cond = model.get_conditioning_inputs(mixed_cond)
            intermediate_latent_val = sample_discrete_euler(model.model, intermediate_latent, steps=inv_steps, sigmas=ts.flip(0)[:-1], **lfe_mixed_cond, cfg_scale=tar_inv_cfg_scale)
            if model.pretransform is not None:
                intermediate_latent_val = intermediate_latent_val.to(next(model.pretransform.parameters()).dtype)
                intermediate_latent_val = model.pretransform.decode(intermediate_latent_val)
            intermediate_sampled.append(intermediate_latent_val)
    del noise
    del src_conditioning_tensors
    del tar_conditioning_tensors
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled, intermediate_sampled