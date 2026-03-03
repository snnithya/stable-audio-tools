import gc
import pytorch_lightning as pl
import random
import torch
import typing as tp

from ema_pytorch import EMA
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, or_masks
import torch.distributed as dist

from ..inference.sampling import truncated_logistic_normal_rescaled
from ..models.diffusion import ConditionedDiffusionModelWrapper
from ..models.inpainting import random_inpaint_mask
from .utils import create_optimizer_from_config, create_scheduler_from_config

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def euler_step(x_t, v_t, t, s):
    return x_t + (s - t)[:, None, None] * v_t

@torch.no_grad()
def sample_flow_dpmpp_w_intermediates(model, x, sigmas=None, steps=None, callback=None, dist_shift=None, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models. Return output at each step."""

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        # Create the noise schedule
        t = torch.linspace(1, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])
    
    else:
        t = sigmas

    old_denoised = None

    log_snr = lambda t: ((1-t) / t).log()
    inters_x = []
    inters_t = []

    for i in range(len(t) - 1):
        inters_x.append(x)
        inters_t.append(t[i])
        denoised = x - t[i] * model(x, t[i] * ts, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': t[i], 'sigma_hat': t[i], 'denoised': denoised})
        t_curr, t_next = t[i], t[i + 1]
        alpha_t = 1-t_next
        h = log_snr(t_next) - log_snr(t_curr)
        if old_denoised is None or t_next == 0:
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised
        else:
            h_last = log_snr(t_curr) - log_snr(t[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised_d
        old_denoised = denoised
    target = x.detach()
    inters_x = torch.stack(inters_x).detach() # steps x B x C x T
    inters_t = torch.stack(inters_t).unsqueeze(-1).detach() # steps x 1

    return {'target': target, 'x': inters_x, 't': inters_t}

class ARCTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for ARC post-training on a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            discriminator: ConditionedDiffusionModelWrapper,
            arc_config: dict,
            optimizer_configs: dict,
            teacher_model: ConditionedDiffusionModelWrapper = None,
            use_ema: bool = True,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.0,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal"] = "uniform",
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            clip_grad_norm: float = 0.0,
            trim_config = None,
            inpainting_config = None
    ):
        super().__init__()

        self.automatic_optimization = False

        self.diffusion = model

        self.teacher_model = teacher_model

        if self.teacher_model is not None:
            self.teacher_model.eval().requires_grad_(False)

        self.discriminator = discriminator

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.ode_warmup_config = arc_config.get('ode_warmup', None)

        if self.ode_warmup_config is not None:

            self.ode_warmup_steps = self.ode_warmup_config.get('warmup_steps', 0)
            self.ode_refresh_rate = self.ode_warmup_config.get('refresh_rate', 1)
            self.ode_n_sampling_steps = self.ode_warmup_config.get('sampling_steps', 20)
            self.ode_warmup_cfg = self.ode_warmup_config.get('cfg', 4.0)
        else:
            self.ode_warmup_steps = 0

        sampling_config = arc_config.get('sampling', None)

        self.diff_states = []

        self.noise_dist_config = arc_config.get('noise_dist', {})
        self.gen_noise_dist = self.build_noise_dist('generator')
        self.dis_noise_dist = self.build_noise_dist('discriminator')

        self.discriminator_config = arc_config.get('discriminator', {})

        self.discriminator_dit_layer = self.discriminator_config.get('dit_hidden_layer', None)

        self.do_contrastive_disc = self.discriminator_config.get('contrastive', False)
        
        self.include_grad_penalties = self.discriminator_config.get('include_grad_penalties', False)

        assert self.discriminator_dit_layer is not None, "Must specify discriminator dit_hidden_layer in ARC config"

        discriminator_type = self.discriminator_config.get('type', 'convnext')
        discriminator_model_config = self.discriminator_config.get('config', {})

        if discriminator_type == 'convnext':
            from ..models.arc import ConvNeXtDiscriminator
            self.discriminator_head = ConvNeXtDiscriminator(in_channels = self.discriminator.model.model.transformer.dim, latent_dim=1, **discriminator_model_config)
        elif discriminator_type == 'conv':
            from ..models.arc import ConvDiscriminator
            self.discriminator_head = ConvDiscriminator(channels = self.discriminator.model.model.transformer.dim, **discriminator_model_config)

        self.gen_gan_weight = self.discriminator_config.get('weights', {}).get('generator', 1.0)
        self.dis_gan_weight = self.discriminator_config.get('weights', {}).get('discriminator', 1.0)

        if self.do_contrastive_disc:
            self.contrastive_loss_weight = self.discriminator_config.get('weights', {}).get('contrastive', 1.0)

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler     

        self.diffusion_objective = model.diffusion_objective

        assert optimizer_configs is not None, "Must specify optimizer_configs in training config"

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        self.model_last_layer = self.diffusion.model.model.transformer.project_out.weight

        self.clip_grad_norm = clip_grad_norm

        self.trim_config = trim_config

        if self.trim_config is not None:
            self.trim_prob = self.trim_config.get("trim_prob", 0.0)
            self.trim_type = self.trim_config.get("type", "random_item")

        self.inpainting_config = inpainting_config

        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})
            self.ode_inpaint_mask = None
            self.ode_inpaint_masked_input = None

        # Validation

        self.validation_timesteps = validation_timesteps

        self.validation_step_outputs = {}

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        disc_opt_config = self.optimizer_configs['discriminator']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())
        opt_disc = create_optimizer_from_config(disc_opt_config['optimizer'], list(self.discriminator_head.parameters()) + list(self.discriminator.parameters()))
        if "scheduler" in diffusion_opt_config and "scheduler" in disc_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            sched_disc = create_scheduler_from_config(disc_opt_config['scheduler'], opt_disc)
            sched_disc_config = {
                "scheduler": sched_disc,
                "interval": "step"
            }
            return [opt_diff, opt_disc], [sched_diff_config, sched_disc_config]

        return [opt_diff, opt_disc]

    def ode_warmup_step(self, diffusion_input, metadata, padding_masks):
        refresh_diff_states = self.global_step % self.ode_refresh_rate == 0

        if refresh_diff_states:
            start_noise = torch.randn_like(diffusion_input)
            teacher_conditioning = self.teacher_model.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                # Create a mask of random length for a random slice of the input
                inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, **self.inpaint_mask_kwargs)

                teacher_conditioning['inpaint_mask'] = [inpaint_mask]
                teacher_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                self.ode_inpaint_mask = inpaint_mask
                self.ode_inpaint_masked_input = inpaint_masked_input

            logsnr = torch.linspace(-6, 2, self.ode_n_sampling_steps + 1)
            t = torch.sigmoid(-logsnr)
            t[0] = 1
            t[-1] = 0

            self.ode_metadata = metadata
            self.diff_states = sample_flow_dpmpp_w_intermediates(self.teacher_model, start_noise, sigmas=t, cond=teacher_conditioning, cfg_scale=self.ode_warmup_cfg, dist_shift=self.teacher_model.dist_shift, batch_cfg=True)

        conditioning = self.diffusion.conditioner(self.ode_metadata, self.device)

        if self.inpainting_config is not None:
            conditioning['inpaint_mask'] = [self.ode_inpaint_mask]
            conditioning['inpaint_masked_input'] = [self.ode_inpaint_masked_input]

        ixs = torch.randint(0, self.ode_n_sampling_steps, (diffusion_input.shape[0],))
        t = self.diff_states['t'].clone().detach()[ixs].squeeze(-1)
        x_t = self.diff_states['x'].clone().detach()[ixs.flatten(), torch.arange(diffusion_input.shape[0])].squeeze(0)

        t = t.to(self.device).detach().clone().requires_grad_(False)
        x_t = x_t.to(self.device).detach().clone().requires_grad_(False)

        v_t_student = self.diffusion(x_t, t, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob)
        denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))
        ode_mse_loss = F.mse_loss(denoised_student, self.diff_states['target'])

        return ode_mse_loss

    def calculate_disc_loss(self, real_scores, fake_scores):
    
        # Relativistic discriminator loss
        diff = real_scores - fake_scores
        loss_dis = F.softplus(-diff).mean() * self.dis_gan_weight

        return loss_dis

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        opt_gen, opt_disc = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers
        else:
            sched_gen = sched_disc = None

        log_dict = {}

        if self.global_step < self.ode_warmup_steps:
            ode_mse_loss = self.ode_warmup_step(diffusion_input, metadata, padding_masks)

            opt_gen.zero_grad()
            self.manual_backward(ode_mse_loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict = {'train/ode_mse_loss': ode_mse_loss.detach()}
            self.log_dict(log_dict, prog_bar=True, on_step=True)

            if self.diffusion_ema is not None:
                self.diffusion_ema.update()

            return ode_mse_loss

        # Start trimming after ODE warmup to avoid sequence length issues
        if self.trim_config is not None:
            if random.random() < self.trim_prob:
                
                data_lengths = (torch.sum(padding_masks, dim=1) - 1).tolist()
                
                if self.trim_type == "random_item":
                    trim_length = max(random.choice(data_lengths), 128)

                diffusion_input = diffusion_input[:, :, :trim_length]

        conditioning = self.diffusion.conditioner(metadata, self.device)

        if self.inpainting_config is not None:

            # Create a mask of random length for a random slice of the input
            inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, **self.inpaint_mask_kwargs)

            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]

        t = self.gen_noise_dist(reals.shape[0])  

        gen_noise = torch.randn_like(diffusion_input)
        x_t = diffusion_input * (1-t)[:, None, None] + gen_noise * t[:, None, None]       
      
        train_gen = (self.global_step % 2) == 0

        if train_gen or self.global_step < self.ode_warmup_steps:
            v_t_student = checkpoint(self.diffusion, x_t, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob)
        else:
            x_t = x_t.requires_grad_(False)
            with torch.no_grad():
                v_t_student = checkpoint(self.diffusion, x_t, t, cond=conditioning).detach()

        denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))

        if train_gen:
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

            if self.diffusion_ema is not None:
                self.diffusion_ema.update()

            # Get discriminator scores for adversarial loss
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            x_t_gan = denoised_student * (1-t_gan)[:, None, None] + noise * t_gan[:, None, None]

            fake_conditioning = self.discriminator.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                fake_conditioning['inpaint_mask'] = [inpaint_mask]
                fake_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            v_t_gan_hidden_states = self.discriminator(x_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, use_checkpointing=True, exit_layer_ix=self.discriminator_dit_layer)

            disc_scores = self.discriminator_head(v_t_gan_hidden_states.transpose(1, 2))

            log_dict['gen_disc_scores_mean'] = disc_scores.mean().detach()
            log_dict['gen_disc_scores_std'] = disc_scores.std().detach()

            # Relativistic discriminator loss
            x_t_gan_real = diffusion_input * (1-t_gan)[:, None, None] + noise * t_gan[:, None, None]
            v_t_gan_real_hidden_states = self.discriminator(x_t_gan_real, t_gan, cond=fake_conditioning, cfg_scale=1.0, use_checkpointing=True, exit_layer_ix=self.discriminator_dit_layer)
            disc_scores_real = self.discriminator_head(v_t_gan_real_hidden_states.transpose(1, 2))

            diff = disc_scores_real - disc_scores

            loss_adv = F.softplus(diff).mean() * self.gen_gan_weight

            loss = loss_adv

            log_dict['train/adv_loss'] = loss_adv.detach()

            opt_gen.zero_grad()

            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict['train/gen_loss'] = loss.detach()
        else:
            denoised_student = denoised_student.detach().requires_grad_(True)
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            reals_t_gan = diffusion_input * (1-t_gan)[..., None, None] + noise * t_gan[..., None, None]
            denoised_t_gan = denoised_student * (1-t_gan)[..., None, None] + noise * t_gan[..., None, None]
            
            reals_t_gan = reals_t_gan.detach().requires_grad_(True)
            denoised_t_gan = denoised_t_gan.detach().requires_grad_(True)
            
            fake_conditioning = self.discriminator.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                fake_conditioning['inpaint_mask'] = [inpaint_mask]
                fake_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            reals_gan_hidden_states = checkpoint(self.discriminator, reals_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)
            denoised_gan_hidden_states = checkpoint(self.discriminator, denoised_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)

            disc_scores_reals = checkpoint(self.discriminator_head, reals_gan_hidden_states)
            disc_scores_denoised = checkpoint(self.discriminator_head, denoised_gan_hidden_states)

            if self.include_grad_penalties:
        
                r1_approx_variance = 0.05

                noised_reals_t_gan = reals_t_gan + r1_approx_variance * torch.randn_like(reals_t_gan)
                noised_denoised_t_gan = denoised_t_gan + r1_approx_variance * torch.randn_like(denoised_t_gan)
                noised_reals_gan_hidden_states = checkpoint(self.discriminator, noised_reals_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)
                noised_denoised_gan_hidden_states = checkpoint(self.discriminator, noised_denoised_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)
                disc_scores_noised_reals = checkpoint(self.discriminator_head, noised_reals_gan_hidden_states)
                disc_scores_noised_denoised = checkpoint(self.discriminator_head, noised_denoised_gan_hidden_states)
                r1_diff = disc_scores_noised_reals - disc_scores_reals
                r2_diff = disc_scores_noised_denoised - disc_scores_denoised

                r1_penalty = torch.sum(r1_diff ** 2, dim=[1, 2])
                r2_penalty = torch.sum(r2_diff ** 2, dim=[1, 2])

                log_dict['r1_penalty'] = r1_penalty.mean().detach()
                log_dict['r2_penalty'] = r2_penalty.mean().detach()

                grad_penalty_loss = ((r1_penalty.mean() + r2_penalty.mean())/2)

                log_dict['train/grad_penalty_loss'] = grad_penalty_loss.detach()
            else:
                grad_penalty_loss = torch.tensor(0.0, device=self.device)

           

            log_dict['disc_real_scores_mean'] = disc_scores_reals.mean().detach()
            log_dict['disc_fake_scores_mean'] = disc_scores_denoised.mean().detach()
            log_dict['disc_real_scores_std'] = disc_scores_reals.std().detach()
            log_dict['disc_fake_scores_std'] = disc_scores_denoised.std().detach()

            loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_denoised)

            if self.do_contrastive_disc:

                rolled_metadata = []

                for i in range(reals.shape[0]):
                    rolled_keys = ["prompt"]
                    rolled_metadata.append(metadata[i])
                    for rolled_key in rolled_keys:
                        rolled_metadata[i][rolled_key] = metadata[(i + 1) % reals.shape[0]][rolled_key]

                rolled_conditioning = self.discriminator.conditioner(rolled_metadata, self.device)

                if self.inpainting_config is not None:
                    # Hold inpainting conditioning constant during contrastive conditioning
                    rolled_conditioning['inpaint_mask'] = [inpaint_mask]
                    rolled_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                rolled_reals_gan_hidden_states = checkpoint(self.discriminator, reals_t_gan, t_gan, cond=rolled_conditioning, cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer).transpose(1,2)

                disc_scores_rolled_reals = checkpoint(self.discriminator_head, rolled_reals_gan_hidden_states)

                contrastive_loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_rolled_reals) * self.contrastive_loss_weight

                log_dict['train/contrastive_loss_dis'] = contrastive_loss_dis.detach()
            else:
                contrastive_loss_dis = 0

            loss = loss_dis + contrastive_loss_dis + grad_penalty_loss

            log_dict['train/dis_loss'] = loss_dis.detach()

            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']
            log_dict['train/discriminator_loss'] = loss.detach()
            opt_disc.zero_grad()

            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(list(self.discriminator_head.parameters()) + list(self.discriminator.parameters()), self.clip_grad_norm)
            opt_disc.step()

            if sched_disc is not None:
                sched_disc.step()

        log_dict['train/std_data']: diffusion_input.std()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        
        return loss

    def build_noise_dist(self, key):
        dist = self.noise_dist_config.get(key, 'uniform')
        match dist:
            case 'uniform':
                return lambda b: self.rng.draw(b)[:, 0].to(self.device)
            case 'logit_normal':
                return lambda b: torch.sigmoid(torch.randn(b, device=self.device))
            case 'trunc_logit_normal':
                return lambda b: 1 - truncated_logistic_normal_rescaled(b).to(self.device)
            case 'one_shot':
                return lambda b: torch.ones(b, device=self.device)
            case 'denoised':
                return lambda b: torch.zeros(b, device=self.device)
            case 'logsnr_uniform':
                
                min_logsnr = -6
                max_logsnr = 2

                return lambda b: torch.sigmoid(-(torch.rand(b, device=self.device) * (max_logsnr - min_logsnr) + min_logsnr))
            case _:
                raise ValueError(f"Invalid noise distribution: {dist}")


class SelfForcingARCTrainingWrapper(ARCTrainingWrapper):
    '''
    ARC post-training with self-forcing: generates multi-chunk autoregressive rollouts
    using ping-pong sampling with randomized gradient attachment points, then scores
    the full rollout against real data using the ARC discriminator.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            discriminator: ConditionedDiffusionModelWrapper,
            arc_config: dict,
            optimizer_configs: dict,
            teacher_model: ConditionedDiffusionModelWrapper = None,
            use_ema: bool = True,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.0,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal"] = "uniform",
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            clip_grad_norm: float = 0.0,
            trim_config = None,
            inpainting_config = None,
            enc_enc = False
    ):
        super().__init__(
            model=model,
            discriminator=discriminator,
            arc_config=arc_config,
            optimizer_configs=optimizer_configs,
            teacher_model=teacher_model,
            use_ema=use_ema,
            pre_encoded=pre_encoded,
            cfg_dropout_prob=cfg_dropout_prob,
            timestep_sampler=timestep_sampler,
            validation_timesteps=validation_timesteps,
            clip_grad_norm=clip_grad_norm,
            trim_config=trim_config,
            inpainting_config=inpainting_config
        )

        # Self-forcing config
        self_forcing_config = arc_config.get('self_forcing', {})
        self.max_sampling_steps = self_forcing_config.get('max_sampling_steps', 8)
        self.n_rollout_chunks = self_forcing_config.get('n_rollout_chunks', 4)
        self.chunk_size = self_forcing_config.get('chunk_size', 48)  # in latent frames
        self.context_size = self_forcing_config.get('context_size', 208)  # in latent frames
        self.use_kv_cache = self_forcing_config.get('use_kv_cache', False)
        self.window_size = self.context_size + self.chunk_size
        self.enc_enc = enc_enc

        # Outpainting dropout probs for initial context
        self.outpainting_dropout_probs = self_forcing_config.get('outpainting_dropout_probs', None)
        self.use_silence_shift = self_forcing_config.get('use_silence_shift', True)

        # Build noise schedule once: logsnr linspace [-6, 2] -> t = sigmoid(-logsnr)
        logsnr = torch.linspace(-6, 2, self.max_sampling_steps + 1)
        t_schedule = torch.sigmoid(-logsnr)
        t_schedule[0] = 1.0
        t_schedule[-1] = 0.0
        self.register_buffer('t_schedule', t_schedule)

        # Lazy-init FlexAttention block mask
        self.self_attention_block_mask = None

        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})
            print(f"Inpainting mask kwargs: {self.inpaint_mask_kwargs}")
            if "outpainting_dropout_probs" in self.inpaint_mask_kwargs:
                # load in silence tensors
                self.silence_mean = torch.load("/home/zachary/code/stable-audio-tools/notebooks/mean_silence.pt", map_location="cpu")
                self.silence_scale = torch.load("/home/zachary/code/stable-audio-tools/notebooks/scale_silence.pt", map_location="cpu")
                self.silence_mean = self.silence_mean.to(self.device)
                self.silence_scale = self.silence_scale.to(self.device)
            else:
                self.silence_mean = None
                self.silence_scale = None


    def _get_block_mask(self, device):
        """Build the enc-dec FlexAttention block mask (lazy, cached)."""
        if self.self_attention_block_mask is not None:
            return self.self_attention_block_mask

        fixed_mask_size = self.context_size
        seq_len = self.window_size

        def prefix_mask(b, h, q_idx, kv_idx):
            return kv_idx < fixed_mask_size

        def postfix_mask(b, h, q_idx, kv_idx):
            return q_idx >= fixed_mask_size

        mask_mod = or_masks(prefix_mask, postfix_mask)
        # +1 for the prepended/postpended timestep embedding token
        self.self_attention_block_mask = create_block_mask(
            mask_mod, B=None, H=None,
            Q_LEN=seq_len + 1, KV_LEN=seq_len + 1,
            device=device, _compile=True
        )
        print("Created self-attention block mask for enc-dec attention with context size", self.context_size)
        print(self.self_attention_block_mask.to_string())
        return self.self_attention_block_mask

    def _apply_silence_dropout(self, diffusion_input):
        """
        Apply silence dropout to the context region of diffusion_input.

        When use_silence_shift=True (default): prepends silence and shifts the
        entire sequence right, truncating from the end. This ensures audio after
        the silence context is the true start of the song, not mid-song audio.

        When use_silence_shift=False: replaces the start of the context region
        with silence in-place (naive mode), leaving the rest of the sequence
        at its original position.

        Args:
            diffusion_input: (B, C, total_len) full latent sequence

        Returns:
            result: (B, C, total_len) modified sequence
        """
        if self.outpainting_dropout_probs is None:
            return diffusion_input.clone()

        B, C, total_len = diffusion_input.shape
        device = diffusion_input.device

        p_uncond = self.outpainting_dropout_probs.get('p_uncond', 0.0)
        p_partial = self.outpainting_dropout_probs.get('p_partial', 0.0)

        result = diffusion_input.clone()

        # Ensure silence tensors are on the right device
        if self.silence_mean is not None and self.silence_mean.device != device:
            self.silence_mean = self.silence_mean.to(device)
            self.silence_scale = self.silence_scale.to(device)

        for i in range(B):
            rand_val = random.random()
            if rand_val < p_uncond:
                drop_length = self.context_size
            elif rand_val < p_uncond + p_partial and self.context_size > self.chunk_size:
                num_chunks = self.context_size // self.chunk_size
                chunks_to_drop = random.randint(1, num_chunks)
                drop_length = chunks_to_drop * self.chunk_size
            else:
                drop_length = 0

            if drop_length > 0:
                # Sample silence latents
                if self.silence_mean is not None and self.silence_scale is not None:
                    silence = self.silence_mean[..., :drop_length] + self.silence_scale[..., :drop_length] * torch.randn_like(result[i, :, :drop_length])
                    silence = silence[0]  # squeeze leading dim from silence tensors
                else:
                    silence = torch.zeros_like(result[i, :, :drop_length])

                if self.use_silence_shift:
                    # Shift right: [silence | original_from_0...(total-drop_length)]
                    result[i] = torch.cat([silence, diffusion_input[i, :, :total_len - drop_length]], dim=-1)
                else:
                    # Naive: replace the start of the sequence with silence in-place
                    result[i, :, :drop_length] = silence

        return result

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def generate_rollout(self, diffusion_input, conditioning, padding_masks):
        """
        Generate a multi-chunk autoregressive rollout using ping-pong sampling
        with self-forcing gradient flow.

        Args:
            diffusion_input: Full latent sequence (B, C, context_size + chunk_size * n_rollout_chunks)
            conditioning: Dict of conditioning tensors (text, etc.)
            padding_masks: (B, seq_len) padding masks

        Returns:
            full_student: (B, C, context_size + chunk_size * n_rollout_chunks) — context + generated rollout
            shifted_input: (B, C, same) — the shifted real sequence (for discriminator)
        """
        B, C, _ = diffusion_input.shape
        device = diffusion_input.device

        # Apply silence dropout via prepend-and-shift
        shifted_input = self._apply_silence_dropout(diffusion_input)

        # Extract context from the shifted input
        context = shifted_input[:, :, :self.context_size].clone()

        # Initialize inpaint_input: [context | zeros]
        inpaint_input = torch.zeros(B, C, self.window_size, device=device, dtype=diffusion_input.dtype)
        inpaint_input[:, :, :self.context_size] = context

        # Inpainting mask: 1 = keep (context), 0 = generate (new chunk)
        mask = torch.zeros(B, 1, self.window_size, device=device, dtype=diffusion_input.dtype)
        mask[:, :, :self.context_size] = 1.0

        # enc_enc_mask marks the decoder region (inverted mask)
        enc_enc_mask = 1.0 - mask

        # Get FlexAttention block mask
        block_mask = self._get_block_mask(device)

        rollout_chunks = []

        # pregenerate all k's to make it sync across gpus
        k_list = self.generate_and_sync_list(self.n_rollout_chunks, self.max_sampling_steps, device)

        for chunk_idx in range(self.n_rollout_chunks):
            # Sample number of steps for this chunk
            k = k_list[chunk_idx]

            # Fresh noise for the full window
            x = torch.randn(B, C, self.window_size, device=device, dtype=diffusion_input.dtype)

            # Update conditioning with current inpaint input/mask
            conditioning['inpaint_mask'] = [mask]
            conditioning['inpaint_masked_input'] = [inpaint_input]

            if self.use_kv_cache:
                self.diffusion.model.model.transformer.clear_kv_cache()
                kv_cache = {
                    'initialized': False,
                }

            # Timestep as batch tensor
            ts = torch.ones(B, device=device, dtype=diffusion_input.dtype)

            for step_i in range(k):
                t_curr = self.t_schedule[step_i]
                t_next = self.t_schedule[step_i + 1]

                # Apply inpainting: noise the context to match current noise level
                noised_inpaint = inpaint_input * (1 - t_curr) + torch.randn_like(x) * t_curr
                x = x * (1 - mask) + noised_inpaint * mask

                t_batch = t_curr * ts

                if step_i < k - 1:
                    # No-grad steps
                    with torch.no_grad():
                        v = self.diffusion(
                            x, t_batch, cond=conditioning,
                            cfg_dropout_prob=self.cfg_dropout_prob,
                            enc_enc_mask=enc_enc_mask,
                            self_attention_block_mask=block_mask if not self.use_kv_cache else None,
                            kv_cache=kv_cache if self.use_kv_cache else None,
                            use_kv_cache=self.use_kv_cache,
                            prefill=self.use_kv_cache 
                        )
                        denoised = x - t_curr * v
                        # Ping-pong: re-noise to next level
                        x = (1 - t_next) * denoised + t_next * torch.randn_like(x)
                else:
                    # Gradient step: detach input, run with gradients
                    v = self.diffusion(
                        x, t_batch, cond=conditioning,
                        cfg_dropout_prob=self.cfg_dropout_prob,
                        enc_enc_mask=enc_enc_mask,
                        self_attention_block_mask=block_mask if not self.use_kv_cache else None,
                        kv_cache=kv_cache if self.use_kv_cache else None,
                        use_kv_cache=self.use_kv_cache,
                        prefill=self.use_kv_cache 
                    )
                    denoised = x - t_curr * v
                    x = denoised  # final clean output

            # Apply final inpainting (clean context in the encoder region)
            x = x * (1 - mask) + inpaint_input * mask

            # Extract generated chunk (the decoder/rightmost region)
            chunk = x[:, :, -self.chunk_size:]
            rollout_chunks.append(chunk)

            # Slide window for next chunk: detach for the next chunk's context
            # New inpaint_input: shift left by chunk_size, append zeros for next generation region
            inpaint_input = inpaint_input.detach().clone()
            inpaint_input[:, :, -self.chunk_size:] = chunk.detach()
            inpaint_input = torch.cat([
                inpaint_input[:, :, self.chunk_size:],
                torch.zeros(B, C, self.chunk_size, device=device, dtype=diffusion_input.dtype)
            ], dim=2)

        # Clear KV cache after rollout to free GPU memory
        if self.use_kv_cache:
            self.diffusion.model.model.transformer.clear_kv_cache()

        # Concatenate context + generated chunks for the full student sequence
        full_student = torch.cat([context, torch.cat(rollout_chunks, dim=2)], dim=2)
        return full_student, shifted_input

    @torch.no_grad()
    def generate_demo_rollout(self, diffusion_input, conditioning, demo_steps=None, n_chunks=None):
        """
        Generate a multi-chunk autoregressive rollout for demo/inference.
        Unlike generate_rollout, this runs fully no-grad with all steps per chunk
        and no self-forcing randomness.

        Args:
            diffusion_input: (B, C, L) latent input — only the first context_size frames are used as initial context
            conditioning: Dict of conditioning tensors (from conditioner)
            demo_steps: Number of sampling steps per chunk (defaults to max_sampling_steps)
            n_chunks: Number of chunks to generate (defaults to n_rollout_chunks)

        Returns:
            context: (B, C, context_size) — the initial context used
            rollout: (B, C, chunk_size * n_chunks) — concatenated generated chunks
        """
        if demo_steps is None:
            demo_steps = self.max_sampling_steps
        if n_chunks is None:
            n_chunks = self.n_rollout_chunks

        B, C, _ = diffusion_input.shape
        device = diffusion_input.device

        # Build the t_schedule for the requested number of steps
        logsnr = torch.linspace(-6, 2, demo_steps + 1, device=device)
        t_schedule = torch.sigmoid(-logsnr)
        t_schedule[0] = 1.0
        t_schedule[-1] = 0.0

        # Apply silence dropout via prepend-and-shift (same as training)
        shifted_input = self._apply_silence_dropout(diffusion_input)

        # Extract context from the shifted input
        context = shifted_input[:, :, :self.context_size].clone()

        # Initialize inpaint_input: [context | zeros]
        inpaint_input = torch.zeros(B, C, self.window_size, device=device, dtype=diffusion_input.dtype)
        inpaint_input[:, :, :self.context_size] = context

        # Inpainting mask: 1 = keep (context), 0 = generate
        mask = torch.zeros(B, 1, self.window_size, device=device, dtype=diffusion_input.dtype)
        mask[:, :, :self.context_size] = 1.0

        enc_enc_mask = 1.0 - mask
        block_mask = self._get_block_mask(device)

        rollout_chunks = []

        model = self.diffusion_ema.ema_model if self.diffusion_ema is not None else self.diffusion.model

        for chunk_idx in range(n_chunks):
            x = torch.randn(B, C, self.window_size, device=device, dtype=diffusion_input.dtype)

            conditioning['inpaint_mask'] = [mask]
            conditioning['inpaint_masked_input'] = [inpaint_input]

            cond_inputs = self.diffusion.get_conditioning_inputs(conditioning)

            if self.use_kv_cache:
                self.diffusion.model.model.transformer.clear_kv_cache()
                kv_cache = {'initialized': False}

            ts = torch.ones(B, device=device, dtype=diffusion_input.dtype)

            for step_i in range(demo_steps):
                t_curr = t_schedule[step_i]
                t_next = t_schedule[step_i + 1]

                # Apply inpainting: noise the context to match current noise level
                noised_inpaint = inpaint_input * (1 - t_curr) + torch.randn_like(x) * t_curr
                x = x * (1 - mask) + noised_inpaint * mask

                t_batch = t_curr * ts

                v = model(
                    x, t_batch,
                    **cond_inputs,
                    enc_enc_mask=enc_enc_mask,
                    self_attention_block_mask=block_mask if not self.use_kv_cache else None,
                    kv_cache=kv_cache if self.use_kv_cache else None,
                    use_kv_cache=self.use_kv_cache,
                    prefill=self.use_kv_cache
                )
                denoised = x - t_curr * v

                if step_i < demo_steps - 1:
                    # Ping-pong: re-noise to next level
                    x = (1 - t_next) * denoised + t_next * torch.randn_like(x)
                else:
                    x = denoised

            # Apply final inpainting
            x = x * (1 - mask) + inpaint_input * mask

            chunk = x[:, :, -self.chunk_size:]
            rollout_chunks.append(chunk)

            # Slide window
            inpaint_input[:, :, -self.chunk_size:] = chunk
            inpaint_input = torch.cat([
                inpaint_input[:, :, self.chunk_size:],
                torch.zeros(B, C, self.chunk_size, device=device, dtype=diffusion_input.dtype)
            ], dim=2)

        # Clear KV cache after rollout to free GPU memory
        if self.use_kv_cache:
            self.diffusion.model.model.transformer.clear_kv_cache()

        rollout = torch.cat(rollout_chunks, dim=2)
        return context, rollout

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        padding_masks = torch.stack(
            [md["padding_mask"][0] for md in metadata], dim=0
        ).to(self.device)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    padding_masks = F.interpolate(
                        padding_masks.unsqueeze(1).float(),
                        size=diffusion_input.shape[2], mode="nearest"
                    ).squeeze(1).bool()
            else:
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        opt_gen, opt_disc = self.optimizers()

        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers
        else:
            sched_gen = sched_disc = None

        log_dict = {}

        conditioning = self.diffusion.conditioner(metadata, self.device)

        train_gen = (self.global_step % 2) == 0

        # Generate rollout (the "denoised_student") and get shifted real for discriminator
        if train_gen:
            denoised_student, real_rollout = self.generate_rollout(diffusion_input, conditioning, padding_masks)
        else:
            diffusion_input.requires_grad_(False)
            with torch.no_grad():
                denoised_student, real_rollout = self.generate_rollout(diffusion_input, conditioning, padding_masks)
                denoised_student = denoised_student.detach()

        # Clear inpainting tensors added to conditioning dict during rollout to free GPU memory
        conditioning.pop('inpaint_mask', None)
        conditioning.pop('inpaint_masked_input', None)
        del conditioning

        if train_gen:
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

            if self.diffusion_ema is not None:
                self.diffusion_ema.update()

            # Discriminator scoring for adversarial loss
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            x_t_gan = denoised_student * (1 - t_gan)[:, None, None] + noise * t_gan[:, None, None]

            disc_conditioning = self.discriminator.conditioner(metadata, self.device)
        # with torch.compiler.disable():
            v_t_gan_hidden_states = self.discriminator(
                x_t_gan, t_gan, cond=disc_conditioning,
                cfg_scale=1.0, use_checkpointing=False,
                exit_layer_ix=self.discriminator_dit_layer
            )
            # truncate to the rollout region for scoring
            v_t_gan_hidden_states = v_t_gan_hidden_states[:, self.context_size:, :]
            disc_scores = self.discriminator_head(v_t_gan_hidden_states.transpose(1, 2))

            log_dict['gen_disc_scores_mean'] = disc_scores.mean().detach()
            log_dict['gen_disc_scores_std'] = disc_scores.std().detach()

            # Relativistic: also score real
            x_t_gan_real = real_rollout * (1 - t_gan)[:, None, None] + noise * t_gan[:, None, None]
            v_t_gan_real_hidden_states = self.discriminator(
                x_t_gan_real, t_gan, cond=disc_conditioning,
                cfg_scale=1.0, use_checkpointing=False,
                exit_layer_ix=self.discriminator_dit_layer
            )
            v_t_gan_real_hidden_states = v_t_gan_real_hidden_states[:, self.context_size:, :]
            disc_scores_real = self.discriminator_head(v_t_gan_real_hidden_states.transpose(1, 2))

            diff = disc_scores_real - disc_scores
            loss_adv = F.softplus(diff).mean() * self.gen_gan_weight

            loss = loss_adv
            log_dict['train/adv_loss'] = loss_adv.detach()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict['train/gen_loss'] = loss.detach()

        else:
            # Discriminator training step
            denoised_student = denoised_student.detach()
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            reals_t_gan = real_rollout * (1 - t_gan)[..., None, None] + noise * t_gan[..., None, None]
            denoised_t_gan = denoised_student * (1 - t_gan)[..., None, None] + noise * t_gan[..., None, None]

            reals_t_gan = reals_t_gan.detach()
            denoised_t_gan = denoised_t_gan.detach()

            disc_conditioning = self.discriminator.conditioner(metadata, self.device)
            # with torch.compiler.disable():
            reals_gan_hidden_states = self.discriminator(
                reals_t_gan, t_gan, cond=disc_conditioning,
                cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer, use_checkpointing=False,
            ).transpose(1, 2)
            reals_gan_hidden_states = reals_gan_hidden_states[:, :,  self.context_size:]
            denoised_gan_hidden_states = self.discriminator(
                denoised_t_gan, t_gan, cond=disc_conditioning,
                cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer, use_checkpointing=False,
            ).transpose(1, 2)
            denoised_gan_hidden_states = denoised_gan_hidden_states[:, :,  self.context_size:]
            disc_scores_reals = checkpoint(self.discriminator_head, reals_gan_hidden_states)
            disc_scores_denoised = checkpoint(self.discriminator_head, denoised_gan_hidden_states)

            if self.include_grad_penalties:
                r1_approx_variance = 0.05
                noised_reals_t_gan = reals_t_gan + r1_approx_variance * torch.randn_like(reals_t_gan)
                noised_denoised_t_gan = denoised_t_gan + r1_approx_variance * torch.randn_like(denoised_t_gan)
                noised_reals_gan_hidden_states = self.discriminator(
                    noised_reals_t_gan, t_gan, cond=disc_conditioning,
                    cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer, use_checkpointing=False
                ).transpose(1, 2)
                noised_denoised_gan_hidden_states = self.discriminator(
                    noised_denoised_t_gan, t_gan, cond=disc_conditioning,
                    cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer, use_checkpointing=False
                ).transpose(1, 2)
                disc_scores_noised_reals = checkpoint(self.discriminator_head, noised_reals_gan_hidden_states)
                disc_scores_noised_denoised = checkpoint(self.discriminator_head, noised_denoised_gan_hidden_states)
                r1_diff = disc_scores_noised_reals - disc_scores_reals
                r2_diff = disc_scores_noised_denoised - disc_scores_denoised
                r1_penalty = torch.sum(r1_diff ** 2, dim=[1, 2])
                r2_penalty = torch.sum(r2_diff ** 2, dim=[1, 2])
                log_dict['r1_penalty'] = r1_penalty.mean().detach()
                log_dict['r2_penalty'] = r2_penalty.mean().detach()
                grad_penalty_loss = (r1_penalty.mean() + r2_penalty.mean()) / 2
                log_dict['train/grad_penalty_loss'] = grad_penalty_loss.detach()
            else:
                grad_penalty_loss = torch.tensor(0.0, device=self.device)

            log_dict['disc_real_scores_mean'] = disc_scores_reals.mean().detach()
            log_dict['disc_fake_scores_mean'] = disc_scores_denoised.mean().detach()
            log_dict['disc_real_scores_std'] = disc_scores_reals.std().detach()
            log_dict['disc_fake_scores_std'] = disc_scores_denoised.std().detach()

            loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_denoised)

            if self.do_contrastive_disc:
                rolled_metadata = []
                for i in range(reals.shape[0]):
                    rolled_keys = ["prompt"]
                    rolled_metadata.append(metadata[i])
                    for rolled_key in rolled_keys:
                        rolled_metadata[i][rolled_key] = metadata[(i + 1) % reals.shape[0]][rolled_key]

                rolled_conditioning = self.discriminator.conditioner(rolled_metadata, self.device)
                # with torch.compiler.disable():
                rolled_reals_gan_hidden_states = checkpoint(
                    self.discriminator, reals_t_gan, t_gan, cond=rolled_conditioning,
                    cfg_scale=1.0, exit_layer_ix=self.discriminator_dit_layer
                ).transpose(1, 2)
                rolled_reals_gan_hidden_states = rolled_reals_gan_hidden_states[:, :, self.context_size:]
                disc_scores_rolled_reals = checkpoint(self.discriminator_head, rolled_reals_gan_hidden_states)
                contrastive_loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_rolled_reals) * self.contrastive_loss_weight
                log_dict['train/contrastive_loss_dis'] = contrastive_loss_dis.detach()
            else:
                contrastive_loss_dis = 0

            loss = loss_dis + contrastive_loss_dis + grad_penalty_loss

            log_dict['train/dis_loss'] = loss_dis.detach()
            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']
            log_dict['train/discriminator_loss'] = loss.detach()

            opt_disc.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.discriminator_head.parameters()) + list(self.discriminator.parameters()),
                    self.clip_grad_norm
                )
            opt_disc.step()

            if sched_disc is not None:
                sched_disc.step()

        log_dict['train/std_data'] = diffusion_input.std()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        # # Periodic cleanup — critical for non-rank-0 GPUs which never run the
        # # demo callback (where gc.collect + empty_cache normally happen).
        # if self.global_step % 50 == 0:
        #     gc.collect()
        #     torch.cuda.empty_cache()

        # # Detach to drop the autograd graph — PL holds the return value as
        # # `outputs` until the next batch, keeping the entire graph alive.
        return loss.detach()