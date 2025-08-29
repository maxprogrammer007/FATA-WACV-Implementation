import torch

class FATA_Augmenter:
    """Implements the FATA feature augmentation from Eq. 4 in the paper."""
    def __init__(self, ema_momentum=0.95, noise_std=1.0):
        self.ema_momentum = ema_momentum
        self.noise_std = noise_std
        # This will store the running average of the normalized std dev
        self.ema_normalized_std = None
        print("FATA Augmenter initialized.")

    def augment(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies FATA augmentation to a feature map z.
        z has shape (Batch, Channels, Height, Width)
        """
        if z.ndim != 4:
            raise ValueError("Input tensor must be 4-dimensional (B, C, H, W)")

        B, C, H, W = z.shape

        # 1. Sample random noise for alpha and beta
        alpha = torch.randn(B, C, 1, 1, device=z.device) * self.noise_std + 1.0
        beta = torch.randn(B, C, 1, 1, device=z.device) * self.noise_std + 1.0

        # 2. Calculate channel-wise mean and std dev
        mu_c = z.mean(dim=[2, 3], keepdim=True)
        sigma_c = z.std(dim=[2, 3], keepdim=True)

        # 3. Calculate normalized standard deviation (delta_sigma)
        # Using instance-wise max for stability as per common practice
        max_sigma_c_instance = torch.max(sigma_c.view(B, C), dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        delta_sigma = sigma_c / (max_sigma_c_instance + 1e-6)

        # 4. Update the Exponential Moving Average (EMA) of delta_sigma's mean
        # We average across the batch for a stable EMA value
        current_mean_delta = delta_sigma.mean(dim=0, keepdim=True) # Shape: (1, C, 1, 1)
        if self.ema_normalized_std is None:
            self.ema_normalized_std = current_mean_delta
        else:
            self.ema_normalized_std = (self.ema_momentum * self.ema_normalized_std.detach() +
                                       (1 - self.ema_momentum) * current_mean_delta)

        # 5. Apply the FATA formula
        noise_term = self.ema_normalized_std * (beta - alpha) * mu_c
        z_prime = alpha * z + noise_term

        return z_prime

