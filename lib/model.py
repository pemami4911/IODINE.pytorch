import torch
import torch.nn as nn
import math
from sacred import Ingredient
from lib.utils import init_weights, _softplus_to_std, mvn, std_mvn
from lib.utils import gmm_loglikelihood
import numpy as np

net = Ingredient('Net')


@net.config
def cfg():
    input_size = [3,64,64] # [C, H, W]
    z_size = 64
    K = 4
    inference_iters = 4
    log_scale = math.log(0.10)  # log base e
    refinenet_channels_in = 16
    lstm_dim = 128
    conv_channels = 32
    kl_beta = 1
    geco_warm_start = 1000


class RefinementNetwork(nn.Module):
    @net.capture
    def __init__(self, z_size, input_size, refinenet_channels_in, conv_channels, lstm_dim):
        super(RefinementNetwork, self).__init__()
        self.input_size = input_size
        self.z_size = z_size

        self.conv = nn.Sequential(
            nn.Conv2d(refinenet_channels_in, conv_channels, 5, 1, 2),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 5, 1, 2),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 5, 1, 2),
            nn.ELU(True),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear((input_size[1]//4)*(input_size[1]//4)*conv_channels, lstm_dim),
            nn.ELU(True)
        )

        self.input_proj = nn.Sequential(
                nn.Linear(lstm_dim + 4*self.z_size, lstm_dim),
                nn.ELU(True)
            )
        self.lstm = nn.LSTM(lstm_dim, lstm_dim)
        self.loc = nn.Linear(lstm_dim, z_size)
        self.softplus = nn.Linear(lstm_dim, z_size)

    def forward(self, img_inputs, vec_inputs, h, c):
        """
        img_inputs: [N * K, C, H, W]
        vec_inputs: [N * K, 4*z_size]
        """
        x = self.conv(img_inputs)
        # concat with \lambda and \nabla \lambda
        x = torch.cat([x, vec_inputs], 1)
        x = self.input_proj(x)
        x = x.unsqueeze(0) # seq dim
        self.lstm.flatten_parameters()
        out, (h,c) = self.lstm(x, (h,c))
        out = out.squeeze(0)
        loc = self.loc(out)
        softplus = self.softplus(out)
        lamda = torch.cat([loc, softplus], 1)
        return lamda, (h,c)


class SpatialBroadcastDecoder(nn.Module):
    """
    Decodes the individual Gaussian image componenets
    into RGB and mask. This is the architecture used for the 
    Multi-dSprites experiment but I haven't seen any issues 
    with re-using it for CLEVR. In their paper they slightly 
    modify it (e.g., uses 3x3 conv instead of 5x5).
    """
    @net.capture
    def __init__(self, input_size, z_size, conv_channels):
        super(SpatialBroadcastDecoder, self).__init__()
        self.h, self.w = input_size[1], input_size[2]
        self.decode = nn.Sequential(
            nn.Conv2d(z_size+2, conv_channels, 5, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 5, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 5, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 5, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, 4, 1, 1)
        )


    @staticmethod
    def spatial_broadcast(z, h, w):
        """
        source: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py
        """
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, z):
        z_sb = SpatialBroadcastDecoder.spatial_broadcast(z, self.h + 8, self.w + 8)
        out = self.decode(z_sb) # [batch_size * K, output_size, h, w]
        return torch.sigmoid(out[:,:3]), out[:,3]


class IODINE(nn.Module):
    @net.capture
    def __init__(self, z_size, input_size, K, inference_iters, batch_size, log_scale, kl_beta, lstm_dim, geco_warm_start):
        super(IODINE, self).__init__()

        self.z_size = z_size
        self.input_size = input_size
        self.K = K
        self.inference_iters = inference_iters
        self.batch_size = batch_size
        self.kl_beta = kl_beta
        self.gmm_log_scale = log_scale * torch.ones(K)
        self.gmm_log_scale = self.gmm_log_scale.view(1, K, 1, 1, 1)
        self.geco_warm_start = geco_warm_start

        self.image_decoder = SpatialBroadcastDecoder()
        self.refine_net = RefinementNetwork()

        init_weights(self.image_decoder, 'xavier')
        init_weights(self.refine_net, 'xavier')

        # learnable initial posterior distribution
        # loc = 0, variance = 1
        self.lamda_0 = nn.Parameter(torch.cat([torch.zeros(1,self.z_size),torch.ones(1,self.z_size)],1))

        # layernorms for iterative inference input
        n = self.input_size[1]
        self.layer_norms = torch.nn.ModuleList([
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((3,n,n), elementwise_affine=False),
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((self.z_size,), elementwise_affine=False), # layer_norm_mean
                nn.LayerNorm((self.z_size,), elementwise_affine=False)  # layer_norm_log_scale
            ])

        self.h_0, self.c_0 = (torch.zeros(1, self.batch_size*self.K, lstm_dim),
                    torch.zeros(1, self.batch_size*self.K, lstm_dim))
        self.geco_C_ema = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.geco_beta = nn.Parameter(torch.tensor(0.55), requires_grad=False)


    @staticmethod
    def refinenet_inputs(image, means, masks, mask_logits, log_p_k, normal_ll, lamda, loss, layer_norms, eval_mode):
        N, K, C, H, W = image.shape
        # non-gradient inputs
        # 1. image [N, K, C, H, W]
        # 2. means [N, K, C, H, W]
        # 3. masks  [N, K, 1, H, W] (log probs)
        # 4. mask logits [N, K, 1, H, W]
        # 5. mask posterior [N, K, 1, H, W]
        normal_ll = torch.sum(normal_ll, dim=2)
        mask_posterior = (normal_ll - torch.logsumexp(normal_ll, dim=1).unsqueeze(1)).unsqueeze(2) # logscale
        # 6. pixelwise likelihood [N, K, 1, H, W]
        log_p_k = torch.logsumexp(log_p_k, dim=[1,2])
        log_p_k = log_p_k.view(-1, 1, 1, H, W).repeat(1, K, 1, 1, 1)
        px_l = log_p_k  # log scale
        #px_l = log_p_k.exp() # not log scale
        # 7. LOO likelihood
        #loo_px_l = torch.log(1e-6 + (px_l.exp()+1e-6 - (masks + normal_ll.unsqueeze(2).exp())+1e-6)) # [N,K,1,H,W]

        # 8. Coordinate channels
        x = torch.linspace(-1, 1, W, device='cuda')
        y = torch.linspace(-1, 1, H, device='cuda')
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, k, 1, h, w)
        x_mesh = x_b.expand(N, K, 1, -1, -1)
        y_mesh = y_b.expand(N, K, 1, -1, -1)

        # 9. \partial L / \partial means
        # [N, K, C, H, W]
        # 10. \partial L/ \partial masks
        # [N, K, 1, H, W]
        # 11. \partial L/ \partial lamda
        # [N*K, 2 * self.z_size]
        d_means, d_masks, d_lamda = \
                torch.autograd.grad(loss, [means, masks, lamda], create_graph=not eval_mode,
                        retain_graph=not eval_mode, only_inputs=True)

        d_loc_z, d_sp_z = d_lamda.chunk(2, dim=1)
        d_loc_z, d_sp_z = d_loc_z.contiguous(), d_sp_z.contiguous()

        # apply layernorms
        px_l = layer_norms[0](px_l).detach()
        #loo_px_l = layer_norms[1](loo_px_l).detach()
        d_means = layer_norms[2](d_means).detach()
        d_masks = layer_norms[3](d_masks).detach()
        d_loc_z = layer_norms[4](d_loc_z).detach()
        d_sp_z = layer_norms[5](d_sp_z).detach()

        # concat image-size and vector inputs
        image_inputs = torch.cat([
            image, means, masks, mask_logits, mask_posterior, px_l,
            d_means, d_masks, x_mesh, y_mesh], 2)
        vec_inputs = torch.cat([
            lamda, d_loc_z, d_sp_z], 1)

        return image_inputs.view(N * K, -1, H, W), vec_inputs


    def forward(self, x, geco, step):
        """
        Evaluates the model as a whole, encodes and decodes
        and runs inference for T steps
        """
        C, H, W = self.input_size[0], self.input_size[1], self.input_size[2]

        # expand lambda_0
        lamda = self.lamda_0.repeat(self.batch_size*self.K,1) # [N*K, 2*z_size]
        p_z = std_mvn(shape=[self.batch_size * self.K, self.z_size], device=x.device)

        total_loss = 0.
        losses = []
        x_means = []
        masks = []
        h, c = self.h_0, self.c_0
        h = h.to(x.device)
        c = c.to(x.device)

        for i in range(self.inference_iters):
            # sample initial posterior
            loc_z, sp_z = lamda.chunk(2, dim=1)
            loc_z, sp_z = loc_z.contiguous(), sp_z.contiguous()
            q_z = mvn(loc_z, sp_z)
            z = q_z.rsample()

            # Get means and masks
            x_loc, mask_logits = self.image_decoder(z)  #[N*K, C, H, W]
            x_loc = x_loc.view(self.batch_size, self.K, C, H, W)

            # softmax across slots
            mask_logits = mask_logits.view(self.batch_size, self.K, 1, H, W)
            mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)

            # NLL [batch_size, 1, H, W]
            log_var = (2 * self.gmm_log_scale).to(x.device)
            nll, ll_outs = gmm_loglikelihood(x, x_loc, log_var, mask_logprobs)

            # KL div
            kl_div = torch.distributions.kl.kl_divergence(q_z, p_z)
            kl_div = kl_div.view(self.batch_size, self.K).sum(1)
            #loss = nll + self.kl_beta * kl_div
            #loss = torch.mean(loss)
            if self.kl_beta == 0. or self.geco_warm_start > step or geco is None:
                loss = torch.mean(nll + self.kl_beta * kl_div)
            else:
                loss = self.kl_beta * torch.mean(kl_div) - geco.constraint(self.geco_C_ema, self.geco_beta, torch.mean(nll))
            scaled_loss = ((i+1.)/self.inference_iters) * loss
            losses += [scaled_loss]
            total_loss += scaled_loss

            x_means += [x_loc]
            masks += [mask_logprobs]

            # Refinement
            if i == self.inference_iters-1:
                # after T refinement steps, just output final loss
                continue

            # compute refine inputs
            x_ = x.repeat(self.K, 1, 1, 1).view(self.batch_size, self.K, C, H, W)

            img_inps, vec_inps = IODINE.refinenet_inputs(x_, x_loc, mask_logprobs,
                    mask_logits, ll_outs['log_p_k'], ll_outs['normal_ll'], lamda, loss, self.layer_norms, not self.training)

            delta, (h,c) = self.refine_net(img_inps, vec_inps, h, c)
            lamda = lamda + delta
        
        return {
            'total_loss': total_loss,
            'nll': torch.mean(nll),
            'kl': torch.mean(kl_div),
            'x_means': x_means,
            'masks': masks,
            'z': z
        }
