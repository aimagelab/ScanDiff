from src.model.components.gaussian_diffusion import LossType, ModelMeanType, ModelVarType, _extract_into_tensor, mean_flat
from src.model.components.respace import SpacedDiffusion
import torch as th

class CustomSpacedDiffusion(SpacedDiffusion):
    def __init__(self, lambda_x, lambda_y, lambda_t, one_step_noise, compute_T_loss, use_timesteps, **kwargs):
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_t = lambda_t
        self.one_step_noise = one_step_noise
        self.compute_T_loss = compute_T_loss
        
        super().__init__(use_timesteps, **kwargs)

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        model = self._wrap_model(model)
        
        if model_kwargs is None:
            model_kwargs = {}
            
        padding_mask = model_kwargs.pop('padding_mask').to(x_start.device)
        force_initial_central_fixation = model_kwargs.pop('force_initial_central_fixation')
        
        gt_scanpath = x_start
        
        x_start_mean = model.model.get_embeds(x_start)
        
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                            th.tensor([0]).to(x_start_mean.device),
                            x_start_mean.shape)
        
        if self.one_step_noise:
            x_start = self._get_x_start(x_start_mean, std) # this is the starting x_0
        else:
            x_start = x_start_mean
            
        if noise is None:
            noise = th.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise=noise)
        
        if force_initial_central_fixation:
            x_t[:, 0] = x_start[:, 0]
        

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                model_output, model_var_values = th.split(model_output, model_output.shape[-1] // 2, dim=2)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            
            
            # Loss 1: Mean Squared Error (MSE) (L_{VLB})
            if force_initial_central_fixation:
                terms["mse"] = mean_flat((target[:,1:] - model_output[:,1:]) ** 2) # this is the L_simple loss
            else:
                terms["mse"] = mean_flat((target - model_output) ** 2) # this is the L_simple loss
            
            model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart']
            
            if self.one_step_noise:
                t0_mask = (t == 0) # mask that says true for every instance where no noise was received, i.e. t=0
                t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)

                terms['mse'] = th.where(t0_mask, t0_loss, terms['mse'])

            if self.compute_T_loss:
                out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
                tT_loss = mean_flat(out_mean ** 2)
            
            get_fixations = model.model.get_coords_and_time
            
            if force_initial_central_fixation:
                terms['reg_loss'], terms['t_loss'] = self.reconstruction_loss(x_start[:,1:], padding_mask[:,1:], get_fixations, gt_scanpath[:,1:])
                terms['token_loss'] = self.token_validity_loss(x_start[:,1:], model.model.token_validity_predictor, padding_mask[:,1:])
            else:
                terms['reg_loss'], terms['t_loss'] = self.reconstruction_loss(x_start, padding_mask, get_fixations, gt_scanpath)
                terms['token_loss'] = self.token_validity_loss(x_start, model.model.token_validity_predictor, padding_mask)
            
        
        
        if self.compute_T_loss:
            terms["loss"] = terms['mse'] + tT_loss + terms['reg_loss'] + terms['t_loss'] + terms['token_loss']
        else:
            terms["loss"] = terms['mse'] + terms['reg_loss'] + terms['t_loss'] + terms['token_loss']
        
        return terms
    
    def reconstruction_loss(self, x_t, padding_mask, get_fixations, ground_truth_scanpath):
        reconstruction = get_fixations(x_t)
        loss_fn_x = th.nn.L1Loss(reduction='none')
        loss_fn_y = th.nn.L1Loss(reduction='none')
        loss_fn_t = th.nn.L1Loss(reduction='none')
        
        masked_reconstruction = reconstruction * padding_mask.unsqueeze(-1)
        masked_ground_truth = ground_truth_scanpath * padding_mask.unsqueeze(-1)
        
        x_loss = loss_fn_x(masked_reconstruction[:,:,0], masked_ground_truth[:,:,0]).sum(dim=1)/padding_mask.sum(dim=1)
        y_loss = loss_fn_y(masked_reconstruction[:,:,1], masked_ground_truth[:,:,1]).sum(dim=1)/padding_mask.sum(dim=1)
        reg_loss = x_loss * self.lambda_x + y_loss * self.lambda_y
        
        t_loss = (loss_fn_t(masked_reconstruction[:,:,2], masked_ground_truth[:,:,2]).sum(dim=1)/padding_mask.sum(dim=1))
        t_loss *= self.lambda_t
        
        return reg_loss, t_loss