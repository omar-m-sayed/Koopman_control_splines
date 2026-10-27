import torch
import torch.nn as nn
import torch.nn.functional as F


class RationalQuadraticSpline1D(nn.Module):
    """
    1D monotone rational-quadratic spline (Durkan et al., NeurIPS 2019):

        s in [s_min, s_max]  ->  x in [x_min, x_max]

    where s is the slack state and x is the real state.

    Properties (by construction):
    - Strictly monotone  -> bijective
    - Exact closed-form inverse via quadratic formula  (no bisection)
    - Boundary-consistent: T(s_min)=x_min, T(s_max)=x_max
    - Diagonal: x(i) depends only on s(i)
    - Reduces to identity when all bins are uniform and derivatives = 1
    """

    def __init__(
            self,
            num_bins: int = 8,
            s_min: float = -1.0,
            s_max: float = 1.0,
            x_min: float = 0,
            x_max: float = 1.0,
            min_bin_width: float = 1e-3,
            min_bin_height: float = 1e-3,
            min_derivative: float = 1e-3,
    ):
        super().__init__()

        assert num_bins >= 1, "num_bins must be >= 1"
        assert s_max > s_min, "s_max must be > s_min"
        assert x_max > x_min, "x_max must be > x_min"
        assert min_bin_width * num_bins < 1.0, "min_bin_width too large"
        assert min_bin_height * num_bins < 1.0, "min_bin_height too large"

        self.num_bins = num_bins
        self.s_min = s_min  # input  (slack) domain
        self.s_max = s_max
        self.x_min = x_min  # output (state) codomain
        self.x_max = x_max
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Learnable parameters — initialised to identity spline
        self.width_logits = nn.Parameter(torch.randn(num_bins))
        self.height_logits = nn.Parameter(torch.randn(num_bins))
        # num_bins+1 interior derivatives (boundaries fixed to 1 for C1)
        self.derivative_logits = nn.Parameter(torch.randn(num_bins + 1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _params(self, device, dtype):
        """
        Compute knot positions and derivatives from raw parameters.
        Last bin is computed as remainder to guarantee exact boundary match.
        """
        K = self.num_bins

        # --- widths (input domain) ---
        w_raw = F.softmax(self.width_logits.to(device=device, dtype=dtype), dim=0)
        w_raw = self.min_bin_width + (1.0 - K * self.min_bin_width) * w_raw
        widths = w_raw * (self.s_max - self.s_min)

        # --- heights (output codomain) ---
        h_raw = F.softmax(self.height_logits.to(device=device, dtype=dtype), dim=0)
        h_raw = self.min_bin_height + (1.0 - K * self.min_bin_height) * h_raw
        heights = h_raw * (self.x_max - self.x_min)

        # Fix Bug 3: recompute last bin from remainder so knots land exactly
        # on boundaries without silent modification.
        widths = widths.clone()
        heights = heights.clone()
        widths[-1] = (self.s_max - self.s_min) - widths[:-1].sum()
        heights[-1] = (self.x_max - self.x_min) - heights[:-1].sum()

        # --- derivatives (must be positive) ---
        derivs = self.min_derivative + F.softplus(
            self.derivative_logits.to(device=device, dtype=dtype)
        )

        # --- knot positions ---
        s_knots = torch.cat([
            widths.new_tensor([self.s_min]),
            self.s_min + torch.cumsum(widths, dim=0),
        ])
        x_knots = torch.cat([
            heights.new_tensor([self.x_min]),
            self.x_min + torch.cumsum(heights, dim=0),
        ])

        # Hard-pin boundaries for floating-point safety
        s_knots = torch.cat([s_knots[:-1],
                             s_knots.new_tensor([self.s_max])])
        x_knots = torch.cat([x_knots[:-1],
                             x_knots.new_tensor([self.x_max])])

        return s_knots, x_knots, widths, heights, derivs

    @staticmethod
    def _find_bin(val, knots_inner):
        k = torch.bucketize(val, knots_inner, right=False)
        return torch.clamp(k, 0, knots_inner.numel())  # max valid bin = K-1

    def forward(self, s: torch.Tensor):
        shape = s.shape
        s = s.reshape(-1)
        device, dtype = s.device, s.dtype

        s = torch.clamp(s, self.s_min, self.s_max)
        s_knots, x_knots, widths, heights, derivs = self._params(device, dtype)

        k = self._find_bin(s, s_knots[1:-1])

        s0 = s_knots[k];
        x0 = x_knots[k]
        w = widths[k];
        h = heights[k]
        d0 = derivs[k];
        d1 = derivs[k + 1]

        delta = h / w
        theta = torch.clamp((s - s0) / w, 0.0, 1.0)
        omt = 1.0 - theta

        num = h * (delta * theta ** 2 + d0 * theta * omt)
        den = delta + (d0 + d1 - 2.0 * delta) * theta * omt
        x = x0 + num / den
        x = torch.clamp(x, self.x_min, self.x_max)

        # Derivative dx/ds
        # dnum = delta**2 * (d1 * theta**2 + 2.0 * delta * theta * omt + d0 * omt**2)
        # dxds = dnum / den**2
        # dxds = torch.clamp(dxds, min=torch.finfo(dtype).tiny)

        return x.reshape(shape)  # , torch.log(dxds).reshape(shape)

    def inverse(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(-1)
        device, dtype = x.device, x.dtype

        x = torch.clamp(x, self.x_min, self.x_max)
        s_knots, x_knots, widths, heights, derivs = self._params(device, dtype)

        k = self._find_bin(x, x_knots[1:-1])

        s0 = s_knots[k]
        x0 = x_knots[k]
        w = widths[k]
        h = heights[k]
        d0 = derivs[k]
        d1 = derivs[k + 1]

        delta = h / w
        xi = torch.clamp((x - x0) / h, 0.0, 1.0)  # normalised position

        # Solve quadratic: a*theta^2 + b*theta + c = 0
        a = h * (delta - d0) + xi * h * (d0 + d1 - 2.0 * delta)
        b = h * d0 - xi * h * (d0 + d1 - 2.0 * delta)
        c = -delta * xi * h

        # Fix Bug 5: safe linear fallback with denominator guard
        eps = 1e-14 if dtype == torch.float64 else 1e-7
        theta = torch.empty_like(x)

        linear_mask = a.abs() < eps
        nonlinear_mask = ~linear_mask

        # Linear case: b*theta + c = 0
        b_safe = torch.where(linear_mask & (b.abs() < eps),
                             torch.ones_like(b), b)
        theta[linear_mask] = (-c / b_safe)[linear_mask]

        # Quadratic case: standard formula, pick root in [0,1]
        if nonlinear_mask.any():
            a_n, b_n, c_n = a[nonlinear_mask], b[nonlinear_mask], c[nonlinear_mask]
            disc = torch.clamp(b_n ** 2 - 4.0 * a_n * c_n, min=0.0)
            sqrt_d = torch.sqrt(disc)
            root1 = (-b_n + sqrt_d) / (2.0 * a_n)
            root2 = (-b_n - sqrt_d) / (2.0 * a_n)
            valid1 = (root1 >= -1e-6) & (root1 <= 1.0 + 1e-6)
            theta[nonlinear_mask] = torch.where(valid1, root1, root2)

        theta = torch.clamp(theta, 0.0, 1.0)
        s = torch.clamp(s0 + theta * w, self.s_min, self.s_max)

        # Recompute derivative for log_det  (dx/ds at recovered theta)
        # omt  = 1.0 - theta
        # den  = delta + (d0 + d1 - 2.0 * delta) * theta * omt
        # dnum = delta**2 * (d1 * theta**2 + 2.0 * delta * theta * omt + d0 * omt**2)
        # dxds = torch.clamp(dnum / den**2, min=torch.finfo(dtype).tiny)

        return s.reshape(shape)  # , (-torch.log(dxds)).reshape(shape)


# ----------------------------------------------------------------------
# Diagonal multi-dimensional spline: one 1D spline per state dimension
# ----------------------------------------------------------------------

class DiagonalRQS(nn.Module):

    def __init__(
            self,
            n_x: int,
            num_bins: int = 8,
            s_bounds: list = None,
            x_bounds: list = None,
            min_bin_width: float = 1e-3,
            min_bin_height: float = 1e-3,
            min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.n_x = n_x

        if s_bounds is None:
            s_bounds = [(-1.0, 1.0)] * n_x
        if x_bounds is None:
            x_bounds = [(-1.0, 1.0)] * n_x

        assert len(s_bounds) == n_x and len(x_bounds) == n_x

        self.splines = nn.ModuleList([
            RationalQuadraticSpline1D(
                num_bins=num_bins,
                s_min=s_bounds[i][0],
                s_max=s_bounds[i][1],
                x_min=x_bounds[i][0],
                x_max=x_bounds[i][1],
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
            for i in range(n_x)
        ])

    def forward(self, s: torch.Tensor):

        assert s.shape[-1] == self.n_x
        xs, lds = [], []
        for i, spline in enumerate(self.splines):
            # x_i, ld_i = spline.forward(s[..., i])
            x_i = spline.forward(s[..., i])
            xs.append(x_i.unsqueeze(-1))
            # lds.append(ld_i)
        x = torch.cat(xs, dim=-1)
        #log_det = torch.stack(lds, dim=-1).sum(dim=-1)
        return x

    def inverse(self, x: torch.Tensor):

        assert x.shape[-1] == self.n_x
        ss, lds = [], []
        for i, spline in enumerate(self.splines):
            # s_i, ld_i = spline.inverse(x[..., i])
            s_i = spline.inverse(x[..., i])

            ss.append(s_i.unsqueeze(-1))
            # lds.append(ld_i)
        s = torch.cat(ss, dim=-1)
        # log_det = torch.stack(lds, dim=-1).sum(dim=-1)
        return s
