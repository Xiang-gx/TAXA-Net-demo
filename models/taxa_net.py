"""TAXA-Net for GHI forecasting with time-aligned cross-attention and a zero key–value option."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sinusoidal_pe(num_patches: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding [1, N, D]."""
    pe = torch.zeros(1, num_patches, d_model)
    pos = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features: int, affine: bool = False, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.affine_weight = self.affine_bias = None
        self._mean = self._std = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "normalize":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._std
            if self.affine_weight is not None:
                x = x * self.affine_weight + self.affine_bias
        elif mode == "denormalize":
            if self.affine_weight is not None:
                x = (x - self.affine_bias[:x.shape[-1]]) / (self.affine_weight[:x.shape[-1]] + self.eps)
            x = x * self._std[:, :, :x.shape[-1]] + self._mean[:, :, :x.shape[-1]]
        return x


class TemporalEncoderSS(nn.Module):
    """Project calendar features sampled at patch endpoints."""
    def __init__(self, d_time_in, d_model, dropout=0.1):
        super().__init__()
        self.time_embed = nn.Linear(d_time_in, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_time: torch.Tensor, selected_scales) -> dict:
        out_dict = {}
        for s in selected_scales:
            # Drop the oldest remainder to retain recent observations.
            rem = x_time.shape[1] % s
            x_time_s = x_time[:, rem:, :]
            x_time_end = x_time_s[:, s - 1::s, :]  # [B, N, d_time_in]
            H_time = self.time_embed(x_time_end)   # [B, N, d_model]
            out_dict[s] = self.dropout(H_time)
        return out_dict


class _FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class NullAwareCrossAttention(nn.Module):
    """Time-aligned cross-attention with a fixed zero key–value option."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.register_buffer("null_k", torch.zeros(1, 1, d_model))

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_sa, H_ex):
        # H_sa: [B, C_en, N, D]
        # H_ex: [B, N, C_ex, D]
        B, C_en, N, D = H_sa.shape
        _, _, C_ex, _ = H_ex.shape

        q = H_sa.permute(0, 2, 1, 3).contiguous().view(B * N, C_en, D)
        k_ex = H_ex.contiguous().view(B * N, C_ex, D)
        v_ex = H_ex.contiguous().view(B * N, C_ex, D)

        # Add the zero option after projection.
        K_real = self.k_proj(k_ex)
        null_k_exp = self.null_k.expand(B * N, 1, D)
        K_all = torch.cat([null_k_exp, K_real], dim=1) # [B*N, 1 + C_ex, D]

        V_real = self.v_proj(v_ex)
        null_v = torch.zeros(B * N, 1, D, device=H_sa.device, dtype=H_sa.dtype)
        V_all = torch.cat([null_v, V_real], dim=1)     # [B*N, 1 + C_ex, D]

        Q_m = self.q_proj(q).view(B * N, C_en, self.n_heads, self.head_dim).transpose(1, 2)
        K_m = K_all.view(B * N, 1 + C_ex, self.n_heads, self.head_dim).transpose(1, 2)
        V_m = V_all.view(B * N, 1 + C_ex, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q_m, K_m.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, V_m).transpose(1, 2).contiguous().view(B * N, C_en, D)
        ca_out = self.out_proj(out).view(B, N, C_en, D).permute(0, 2, 1, 3)
        return ca_out


class DualStreamBlock_NullAware(nn.Module):
    """GHI self-attention, weather cross-attention, and FFN."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_sa = nn.LayerNorm(d_model)
        self.cross = NullAwareCrossAttention(d_model, n_heads, dropout=dropout)
        self.norm_ca = nn.LayerNorm(d_model)
        self.ffn = _FFN(d_model, d_ff, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, H_en, H_ex):
        B, C_en, N, D = H_en.shape
        x = H_en.view(B * C_en, N, D)
        a, _ = self.sa(x, x, x)
        H_sa = self.norm_sa(H_en + self.drop(a.view(B, C_en, N, D)))

        if H_ex is not None and H_ex.shape[2] > 0:
            ca = self.cross(H_sa, H_ex)
            H_ca = self.norm_ca(H_sa + self.drop(ca))
        else:
            H_ca = H_sa

        return self.norm_ffn(H_ca + self.drop(self.ffn(H_ca)))


class EndogenousFixedScalesEncoder(nn.Module):
    def __init__(self, scales, d_model, c_en, dropout=0.1):
        super().__init__()
        self.scales = scales
        self.d_model = d_model
        self.projectors = nn.ModuleDict({
            str(s): nn.Conv1d(c_en, c_en * d_model, kernel_size=s, stride=s, groups=c_en)
            for s in scales
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_en):
        B, L, C_en = x_en.shape
        x_reshaped = x_en.transpose(1, 2)
        out_dict = {}
        for s in self.scales:
            # Drop the oldest remainder to retain recent observations.
            rem = L % s
            x_s = x_reshaped[:, :, rem:]
            H = self.projectors[str(s)](x_s)
            N = H.shape[-1]
            H = H.view(B, C_en, self.d_model, N).transpose(2, 3)
            out_dict[s] = self.dropout(H)
        return out_dict


class ExogenousEncoder(nn.Module):
    """Patch projection shared across weather variables."""
    def __init__(self, scales, d_model, dropout=0.1):
        super().__init__()
        self.projectors = nn.ModuleDict({
            str(s): nn.Conv1d(1, d_model, kernel_size=s, stride=s)
            for s in scales
        })
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x_ex, scales):
        if x_ex is None:
            return {s: None for s in scales}
        B, L, C_ex = x_ex.shape
        out_dict = {}
        if C_ex == 0:
            for s in scales:
                N = L // s
                out_dict[s] = torch.empty(B, N, 0, self.d_model, device=x_ex.device)
            return out_dict

        for s in scales:
            rem = L % s
            x_s = x_ex[:, rem:, :]
            x_reshaped = x_s.transpose(1, 2).reshape(B * C_ex, 1, x_s.shape[1])
            H = self.projectors[str(s)](x_reshaped)
            N = H.shape[-1]
            H = H.view(B, C_ex, self.d_model, N).permute(0, 3, 1, 2)
            out_dict[s] = self.dropout(H)
        return out_dict


class TAXA_Net(nn.Module):
    def __init__(self, seq_len: int = 480, pred_len: int = 96,
                 endo_cols: list = None, exo_cols: list = None,
                 d_model: int = 64, n_heads: int = 4, e_layers: int = 3, d_ff: int = 256,
                 scales = (16,), dropout: float = 0.2, **kwargs):
        super().__init__()
        endo_cols = endo_cols or ["Global_Horizontal_Radiation"]
        exo_cols = exo_cols or ["Wind_Speed", "Weather_Temperature_Celsius", "Weather_Relative_Humidity", "Weather_Daily_Rainfall"]

        if d_ff < 2 * d_model:
            d_ff = 2 * d_model
        self.c_en = len(endo_cols)
        self.c_ex = len(exo_cols)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scales = list(scales) if isinstance(scales, (list, tuple)) else [scales]
        self.d_model = d_model

        # Rainfall must be last to bypass RevIN.
        self.rainfall_col = kwargs.get("rainfall_col", "Weather_Daily_Rainfall")
        if self.rainfall_col in exo_cols:
            if exo_cols[-1] != self.rainfall_col:
                raise ValueError(
                    f"Rainfall feature '{self.rainfall_col}' must be positioned as the last exogenous column in exo_cols "
                    f"to properly bypass RevIN normalization (got {exo_cols})."
                )
            self.has_rainfall = True
        else:
            self.has_rainfall = False

        self.revin_en = RevIN(self.c_en, affine=False)
        num_cont_ex = (self.c_ex - 1) if (self.has_rainfall and self.c_ex > 1) else self.c_ex
        self.revin_ex = RevIN(num_cont_ex, affine=False) if num_cont_ex > 0 else None


        self.endo_embed = EndogenousFixedScalesEncoder(self.scales, d_model, self.c_en, dropout)
        self.exo_embed = ExogenousEncoder(self.scales, d_model, dropout)

        self.time_embed = TemporalEncoderSS(4, d_model, dropout)

        if self.c_ex > 0:
            self.exo_channel_embed = nn.Parameter(torch.empty(1, 1, self.c_ex, d_model))
            nn.init.normal_(self.exo_channel_embed, std=0.02)
        else:
            self.exo_channel_embed = None

        self.blocks = nn.ModuleList([
            DualStreamBlock_NullAware(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])

        for s in self.scales:
            num_patches = seq_len // s
            self.register_buffer(f"pos_embed_{s}", build_sinusoidal_pe(num_patches, d_model))

        total_flat_dim = sum((seq_len // s) * self.c_en * d_model for s in self.scales)
        self.pred_head = nn.Linear(total_flat_dim, pred_len)

    def forward(self, x_en: torch.Tensor, x_ex: torch.Tensor, x_time: torch.Tensor = None):
        """Predict GHI [B, pred_len] in W/m² from:
        x_en [B, seq_len, c_en], x_ex [B, seq_len, c_ex],
        and calendar features x_time [B, seq_len, 4].
        """
        x_en = self.revin_en(x_en, "normalize")
        if self.revin_ex is not None and x_ex is not None and x_ex.shape[-1] > 0:
            if self.has_rainfall:
                x_ex_cont = self.revin_ex(x_ex[:, :, :-1], "normalize")
                x_ex = torch.cat([x_ex_cont, x_ex[:, :, -1:]], dim=-1)
            else:
                x_ex = self.revin_ex(x_ex, "normalize")


        H_en_dict = self.endo_embed(x_en)
        if self.exo_embed is not None and x_ex is not None:
            H_ex_dict = self.exo_embed(x_ex, self.scales)
        else:
            H_ex_dict = {s: None for s in self.scales}

        if x_time is not None:
            H_time_dict = self.time_embed(x_time, self.scales)

        flat_feats = []
        for s in self.scales:
            H_en_s = H_en_dict[s] # [B, C_en, N, D]
            H_ex_s = H_ex_dict[s] # [B, N, C_ex, D]
            pos = getattr(self, f"pos_embed_{s}")

            # Calendar features are added only to GHI tokens.
            if x_time is not None:
                H_time_s = H_time_dict[s]
                H_en_s = H_en_s + H_time_s.unsqueeze(1)
            H_en_s = H_en_s + pos.unsqueeze(1)

            if H_ex_s is not None and H_ex_s.shape[2] > 0:
                if self.exo_channel_embed is not None:
                    H_ex_s = H_ex_s + self.exo_channel_embed
                H_ex_s = H_ex_s + pos.unsqueeze(2)

            for blk in self.blocks:
                H_en_s = blk(H_en_s, H_ex_s)

            flat_s = H_en_s.reshape(H_en_s.size(0), -1)
            flat_feats.append(flat_s)

        H_concat = torch.cat(flat_feats, dim=-1)
        out = self.pred_head(H_concat)
        out = self.revin_en(out.unsqueeze(-1), "denormalize").squeeze(-1)
        return F.relu(out)
