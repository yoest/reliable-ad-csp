import model.cond_glow as cglow
import torch
import torch.nn as nn


class ModelWrapper(nn.Module):

    def forward(self, x, c=None) -> torch.Tensor:
        # Forward pass
        raise NotImplementedError
    
    def reverse(self, v, c=None) -> torch.Tensor:
        # Reverse pass
        raise NotImplementedError
    
    def reconstruct(self, x, c=None) -> torch.Tensor:
        # Reconstruct the input
        v = self.forward(x, c)
        return self.reverse(v, c)

    def sample(self, c=None, n_samples: int = 20) -> torch.Tensor:
        # Sample from the model
        raise NotImplementedError

    def get_latent(self, x, c=None) -> torch.Tensor:
        if self.repeat:
            x = x.repeat(1, self.num_channels, 1, 1)
        z_outs, _ = self.model(x, c)
        latent = torch.cat([v.view(v.shape[0], -1) for v in z_outs], dim=1)

        # Reshape to (batch_size, num_channels, height, width)
        if self.repeat:
            latent = latent.view(latent.shape[0], self.num_channels, *self.img_shape[1:])
            latent = latent[:, 0, :, :].flatten(1)
        return latent

    def init(self):
        # Initialize the model
        pass


class ConditionalGlowWrapper(ModelWrapper):

    def __init__(self, num_channels, num_flows, n_blocks, img_shape, device):
        super().__init__()
        self.num_channels = num_channels if num_channels > 1 else 2
        self.repeat = num_channels == 1
        self.num_blocks = n_blocks
        self.img_shape = (self.num_channels, *img_shape)

        self.model = cglow.ConditionalGlow(in_channel=self.num_channels, condition_channel=1, n_flow=num_flows, n_block=self.num_blocks, img_shape=img_shape, device=device)

    def forward(self, x, c=None) -> torch.Tensor:
        if self.repeat:
            x = x.repeat(1, self.num_channels, 1, 1)
        return self.model(x, c)

    def sample(self, c=None, n_samples: int = 20) -> torch.Tensor:
        samples, nll = self.model.sample(c, n_samples=n_samples)
        if self.repeat:
            samples = samples[:, 0, :, :]
        return samples, nll

    def reconstruct(self, x, c=None) -> torch.Tensor:
        latent, _ = self(x, c)
        reconstructed = self.model.reverse(latent, c, reconstruct=True)

        if self.repeat:
            reconstructed = reconstructed[:, 0, :, :]
        return reconstructed