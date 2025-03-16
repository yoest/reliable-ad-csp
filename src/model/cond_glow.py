import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        safe_scale = torch.clamp(self.scale, min=1e-6)  # Avoid division by zero
        log_abs = torch.log(torch.abs(safe_scale))
        logdet = height * width * torch.sum(log_abs)

        # Check for NaN or inf values
        if torch.isnan(logdet).any() or torch.isinf(logdet).any():
            print(f"Warning: NaN or Inf detected in logdet. Resetting scale.")
            self.scale.data.fill_(1.0)  # Reset scale
            logdet = 0

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        # Check for NaN or inf values
        if torch.isnan(logdet).any() or torch.isinf(logdet).any():
            print(f"Warning: NaN or Inf detected in logdet: min={logdet.min()}, max={logdet.max()}")

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(torch.clamp(self.w_s, min=-20, max=20))

        # Check for NaN or inf values
        if torch.isnan(logdet).any() or torch.isinf(logdet).any():
            print(f"Warning: NaN or Inf detected in logdet: min={logdet.min()}, max={logdet.max()}")

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.scale.data.zero_()

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        # Check for NaN or inf values
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"Warning: NaN or Inf detected in out: min={out.min()}, max={out.max()}")

        return out
    

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, in_channel, condition_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine
        self.condition_net = nn.Sequential(
            nn.Conv2d(condition_channel, in_channel, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channel, in_channel // 2, 1),  # Bottleneck layer
        )

        self.net = nn.Sequential(
            nn.Conv2d(in_channel, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

    def forward(self, input, condition):
        in_a, in_b = input.chunk(2, 1)

        # Process condition
        condition_processed = self.condition_net(condition)
        condition_processed = F.interpolate(
            condition_processed, size=(in_a.shape[2], in_a.shape[3]), mode='bilinear', align_corners=False
        )

        # Concatenate output with condition
        concat = torch.cat([in_a, condition_processed], dim=1)

        if self.affine:
            log_s, t = self.net(concat).chunk(2, 1)
            log_s = torch.clamp(log_s, min=-10, max=10)  # Prevent extreme values
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(concat)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, condition):
        out_a, out_b = output.chunk(2, 1)

        # Process condition
        condition_processed = self.condition_net(condition)
        condition_processed = F.interpolate(
            condition_processed, size=(out_a.shape[2], out_a.shape[3]), mode='bilinear', align_corners=False
        )
        
        # Concatenate output with condition
        concat = torch.cat([out_a, condition_processed], dim=1)

        if self.affine:
            log_s, t = self.net(concat).chunk(2, 1)
            log_s = torch.clamp(log_s, min=-10, max=10)  # Prevent extreme values
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(concat)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


def gaussian_log_p(x, mean, log_sd):
    variance = torch.exp(2 * log_sd) + 1e-6  # Add epsilon for stability
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / variance


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, condition_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList()
        for _ in range(n_flow):
            self.flows.append(ConditionalFlow(squeeze_dim, condition_channel, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(squeeze_dim // 2, squeeze_dim)
        else:
            self.prior = ZeroConv2d(squeeze_dim, squeeze_dim * 2)

    def forward(self, input, condition):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out, condition)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, condition, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in reversed(self.flows):
            input = flow.reverse(input, condition)

        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(b_size, n_channel // 4, height * 2, width * 2)

        return unsqueezed
    

class ConditionalFlow(nn.Module):
    def __init__(self, in_channel, condition_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel) if conv_lu else InvConv2d(in_channel)
        self.coupling = ConditionalAffineCoupling(in_channel, condition_channel, affine=affine)

    def forward(self, input, condition):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out, condition)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output, condition):
        input = self.coupling.reverse(output, condition)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class ConditionalGlow(nn.Module):

    def __init__(self, in_channel, condition_channel, n_flow, n_block, img_shape, device, affine=True, conv_lu=True, temp=0.7):
        super().__init__()
        self.img_shape = img_shape
        self.device = device
        self.n_blocks = n_block
        self.n_channel = in_channel
        self.temp = temp

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, condition_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, condition_channel, n_flow, split=False, affine=affine))

    def forward(self, input, condition):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out, condition)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        nll = self.calc_loss(log_p_sum, logdet)
        return z_outs, nll

    def reverse(self, z_list, condition, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], condition, z_list[-1], reconstruct=reconstruct)
            else:
                input = block.reverse(input, condition, z_list[-(i + 1)], reconstruct=reconstruct)

        return input

    def calc_loss(self, log_p, logdet):
        C, H, W = self.img_shape
        n_pixel = W * H * C

        log_p_term = log_p / (torch.log1p(torch.tensor(2.0)) * n_pixel)
        logdet_term = logdet / (torch.log1p(torch.tensor(2.0)) * n_pixel)
        
        nll = -(log_p_term + logdet_term)
        return nll

    def calc_z_shapes(self, input_size):
        z_shapes = []
        n_channel = self.n_channel

        for i in range(self.n_blocks - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes

    def sample(self, condition, n_samples=1):
        # Get the shape of the input image from the model
        C, H, W = self.img_shape
        assert H == W, "H and W must be equal"

        # Generate random latent variables from the prior distribution
        z_sample = []
        z_shapes = self.calc_z_shapes(W)
        for z in z_shapes:
            z_new = torch.randn(n_samples, *z) * self.temp
            z_sample.append(z_new.to(self.device))

        # Reverse through the Glow model to generate images
        with torch.no_grad():
            samples = self.reverse(z_sample, condition)

        # Compute the NLL associated with the generated samples
        _, nll = self.forward(samples, condition)
        return samples, nll