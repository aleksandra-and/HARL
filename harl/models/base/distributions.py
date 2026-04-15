"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_init_method


class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """Modify standard PyTorch Normal."""

    def log_probs(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class SquashedNormal(FixedNormal):
    """FixedNormal with tanh squashing and linear rescaling to action space bounds.

    Samples are squashed via tanh and then scaled to [low, high].
    log_probs inverts the squash (atanh) and applies the Jacobian correction.
    """

    def __init__(self, loc, scale, act_scale, act_mean):
        super().__init__(loc, scale)
        self.act_scale = act_scale
        self.act_mean = act_mean

    def sample(self):
        u = torch.distributions.Normal.rsample(self)
        return torch.tanh(u) * self.act_scale + self.act_mean

    def mode(self):
        return torch.tanh(self.mean) * self.act_scale + self.act_mean

    def log_probs(self, actions):
        # Invert squash: u = atanh((a - mean) / scale)
        a_norm = (actions - self.act_mean) / self.act_scale
        u = torch.atanh(a_norm.clamp(-1 + 1e-6, 1 - 1e-6))
        # Gaussian log prob under pre-squash variable
        log_prob = torch.distributions.Normal.log_prob(self, u)
        # Jacobian correction: log|da/du| = log(act_scale * (1 - tanh(u)^2))
        log_prob -= torch.log(self.act_scale * (1 - torch.tanh(u).pow(2)) + 1e-6)
        return log_prob

    # entropy() uses the Gaussian entropy (pre-squash) as an approximation,
    # inherited from FixedNormal — standard practice for on-policy methods.


class Categorical(nn.Module):
    """A linear layer followed by a Categorical distribution."""

    def __init__(
        self, num_inputs, num_outputs, initialization_method="orthogonal_", gain=0.01
    ):
        super(Categorical, self).__init__()
        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """A linear layer followed by a Diagonal Gaussian distribution.

    If act_scale and act_mean are provided, outputs a SquashedNormal distribution
    that constrains actions to the declared action space via tanh squashing.
    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        initialization_method="orthogonal_",
        gain=0.01,
        args=None,
        act_scale=None,
        act_mean=None,
    ):
        super(DiagGaussian, self).__init__()

        init_method = get_init_method(initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        if args is not None:
            self.std_x_coef = args["std_x_coef"]
            self.std_y_coef = args["std_y_coef"]
        else:
            self.std_x_coef = 1.0
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        log_std = torch.ones(num_outputs) * self.std_x_coef
        self.log_std = torch.nn.Parameter(log_std)
        # Store action space bounds for tanh squashing (None = unbounded, legacy behaviour)
        self.act_scale = act_scale
        self.act_mean = act_mean

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        if self.act_scale is not None and self.act_mean is not None:
            act_scale = self.act_scale.to(x.device)
            act_mean = self.act_mean.to(x.device)
            return SquashedNormal(action_mean, action_std, act_scale, act_mean)
        return FixedNormal(action_mean, action_std)
