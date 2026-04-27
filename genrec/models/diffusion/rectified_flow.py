import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, List


class RectifiedFlow:
    """
    Rectified Flow implementation for generative modeling.

    This class implements the rectified flow approach, which models the transport
    from noise to data using straight paths, enabling faster sampling.
    """

    def __init__(self, num_timesteps: int = 1000):
        """
        Initialize the Rectified Flow model.

        Args:
            num_timesteps: Number of timesteps for the forward process
        """
        self.num_timesteps = num_timesteps

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward process: sample from the straight path between data and noise.

        Args:
            x_start: The original data [N x C x ...]
            t: Timestep values [N], normalized to [0, num_timesteps-1]
            noise: Optional noise to use, if None, will generate random noise

        Returns:
            x_t: Data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Normalize timesteps to [0, 1]
        t_norm = t.float() / (self.num_timesteps - 1)

        # Create straight path: x_t = (1-t) * x_0 + t * noise
        # Expand t_norm to match x_start dimensions
        while len(t_norm.shape) < len(x_start.shape):
            t_norm = t_norm.unsqueeze(-1)

        x_t = (1.0 - t_norm) * x_start + t_norm * noise
        return x_t

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for rectified flow.

        Args:
            model: The model to evaluate loss on
            x_start: The original data [N x C x ...]
            t: Timestep values [N]
            model_kwargs: Additional model arguments for conditioning
            noise: Optional noise to use

        Returns:
            Dictionary with loss values
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_start)

        # Sample from forward process
        x_t = self.q_sample(x_start, t, noise)

        # Compute target velocity (derivative of the path)
        # For straight path: velocity = noise - x_start
        target_velocity = noise - x_start

        # Predict velocity using the model
        predicted_velocity = model(x_t, t, **model_kwargs)

        # Keep one scalar per example so outer masks can drop padded targets.
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1)

        return {"loss": loss}

    def p_sample_loop(
        self,
        model,
        shape: Union[List[int], torch.Size],
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        num_sampling_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples using an ODE solver.

        Args:
            model: The model to use for generation
            shape: Shape of samples to generate
            noise: Optional initial noise
            model_kwargs: Additional model arguments for conditioning
            device: Device to generate on
            progress: Whether to show progress
            num_sampling_steps: Number of steps for sampling (can be different from training)

        Returns:
            Generated samples
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape).to(device)

        if num_sampling_steps is None:
            num_sampling_steps = self.num_timesteps

        # Create timesteps for sampling (reverse order)
        timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            num_sampling_steps,
            dtype=torch.long,
            device=device,
        )

        if progress:
            from tqdm.auto import tqdm

            timesteps = tqdm(timesteps)

        # Sample using Euler method
        for i, t in enumerate(timesteps[:-1]):
            t_next = timesteps[i + 1]
            dt = (t_next - t).float() / (self.num_timesteps - 1)

            # Predict velocity
            with torch.no_grad():
                t_tensor = torch.full(
                    (img.shape[0],), t, device=device, dtype=torch.long
                )
                velocity = model(img, t_tensor, **model_kwargs)

            # Euler step: x_{t+1} = x_t + dt * velocity
            img = img + dt * velocity

        return img

    def p_mean_variance(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict the velocity field at the given timestep.

        Args:
            model: The model to use
            x: Current state [N x C x ...]
            t: Timestep values [N]
            model_kwargs: Additional model arguments for conditioning

        Returns:
            Dictionary with predictions
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Predict velocity using the model
        velocity = model(x, t, **model_kwargs)

        # For compatibility with existing code
        t_norm = t.float() / (self.num_timesteps - 1)
        while len(t_norm.shape) < len(x.shape):
            t_norm = t_norm.unsqueeze(-1)

        # Predict x_0 using the velocity
        pred_xstart = x - t_norm * velocity

        return {
            "velocity": velocity,
            "pred_xstart": pred_xstart,
            "mean": x,  # For compatibility with existing code
            "variance": torch.zeros_like(x),  # For compatibility with existing code
        }

    def sample_ode(
        self,
        model,
        shape: Union[List[int], torch.Size],
        solver: str = "euler",
        num_steps: int = 50,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate samples using different ODE solvers.

        Args:
            model: The model to use for generation
            shape: Shape of samples to generate
            solver: ODE solver to use ("euler", "rk4")
            num_steps: Number of steps for the ODE solver
            model_kwargs: Additional model arguments for conditioning
            device: Device to generate on

        Returns:
            Generated samples
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Start from noise
        x = torch.randn(*shape).to(device)

        # Time points from 1 to 0 (reverse process)
        timesteps = torch.linspace(1, 0, num_steps, device=device)

        if solver == "euler":
            return self._euler_solver(model, x, timesteps, model_kwargs)
        elif solver == "midpoint":
            return self._midpoint_solver(model, x, timesteps, model_kwargs)
        elif solver == "rk4":
            return self._rk4_solver(model, x, timesteps, model_kwargs)
        else:
            raise ValueError(f"Unknown solver: {solver}")

    def _euler_solver(
        self,
        model,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Euler method for ODE solving."""
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # Convert to discrete timestep for model
            t_discrete = (t * (self.num_timesteps - 1)).long()
            t_tensor = torch.full((x.shape[0],), t_discrete, device=x.device)

            with torch.no_grad():
                velocity = model(x, t_tensor, **model_kwargs)

            # Euler step
            x = x + dt * velocity

        return x

    def _midpoint_solver(
        self,
        model,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Midpoint method (2nd order Runge-Kutta) for ODE solving.

        Args:
            model: The neural network model
            x: Initial state tensor
            timesteps: Time points from 1 to 0
            model_kwargs: Additional model arguments

        Returns:
            Final state tensor after ODE integration
        """
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # Convert to discrete timestep for model
            t_discrete = (t * (self.num_timesteps - 1)).long()
            t_tensor = torch.full((x.shape[0],), t_discrete, device=x.device)

            with torch.no_grad():
                # k1: slope at current point
                k1 = model(x, t_tensor, **model_kwargs)

                # Calculate midpoint
                t_mid_discrete = ((t + dt / 2) * (self.num_timesteps - 1)).long()
                t_mid_tensor = torch.full(
                    (x.shape[0],), t_mid_discrete, device=x.device
                )
                x_mid = x + dt / 2 * k1

                # k2: slope at midpoint
                k2 = model(x_mid, t_mid_tensor, **model_kwargs)

                # Update using midpoint slope
                x = x + dt * k2

        return x

    def _rk4_solver(
        self,
        model,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Runge-Kutta 4th order method for ODE solving."""
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # Convert to discrete timestep for model
            t_discrete = (t * (self.num_timesteps - 1)).long()
            t_tensor = torch.full((x.shape[0],), t_discrete, device=x.device)

            with torch.no_grad():
                # RK4 steps
                k1 = model(x, t_tensor, **model_kwargs)

                t2_discrete = ((t + dt / 2) * (self.num_timesteps - 1)).long()
                t2_tensor = torch.full((x.shape[0],), t2_discrete, device=x.device)
                k2 = model(x + dt / 2 * k1, t2_tensor, **model_kwargs)

                t3_discrete = ((t + dt / 2) * (self.num_timesteps - 1)).long()
                t3_tensor = torch.full((x.shape[0],), t3_discrete, device=x.device)
                k3 = model(x + dt / 2 * k2, t3_tensor, **model_kwargs)

                t4_discrete = ((t + dt) * (self.num_timesteps - 1)).long()
                t4_tensor = torch.full((x.shape[0],), t4_discrete, device=x.device)
                k4 = model(x + dt * k3, t4_tensor, **model_kwargs)

                # RK4 update
                x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x


def create_rectified_flow(num_timesteps: int = 1000) -> RectifiedFlow:
    """
    Create a RectifiedFlow instance.

    Args:
        num_timesteps: Number of timesteps for the forward process

    Returns:
        RectifiedFlow instance
    """
    return RectifiedFlow(num_timesteps=num_timesteps)
