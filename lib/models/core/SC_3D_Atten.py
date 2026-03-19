import torch
import torch.nn as nn
import torch.nn.functional as F

class SC_3D_Atten(nn.Module):
    def __init__(self, dim, num_heads=8, k=64, lam_s=1.0, lam_f=1.0, qkv_bias=True, mlp_ratio=4.0):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.dk = dim // num_heads//4
        self.k = k
        self.lam_s = lam_s
        self.lam_f = lam_f
        
        self.attn_dim = dim//4

        self.q_proj = nn.Linear(dim, self.attn_dim , bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.attn_dim , bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.attn_dim , bias=qkv_bias)
        self.o_proj = nn.Linear(self.attn_dim , dim, bias=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(self.attn_dim  * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim , hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    @torch.no_grad()
    def _knn_3d(self, x, k, token_mask=None):
        B, N, _ = x.shape
        idx_list = []
        chunk = 1024 
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = torch.cdist(x[:, s:e], x, p=2)  # (B, e-s, N)
            if token_mask is not None:
                inv = (token_mask == 0).unsqueeze(1)  # (B,1,N)
                d = d.masked_fill(inv, float('inf'))
            _, nn_idx = torch.topk(d, k=k, dim=-1, largest=False)
            idx_list.append(nn_idx)
        return torch.cat(idx_list, dim=1)  # (B,N,k)

    def forward(self, feats, pos3d, feat_sem, token_mask=None):
        B, N, D = feats.shape
        D = D//4
        
        x = self.norm1(feats)
        Q = self.q_proj(x).view(B, N, self.h, self.dk).transpose(1, 2)  # (B,h,N,d)
        K = self.k_proj(x).view(B, N, self.h, self.dk).transpose(1, 2)  # (B,h,N,d)
        V = self.v_proj(x).view(B, N, self.h, self.dk).transpose(1, 2)  # (B,h,N,d)

        
        idx = self._knn_3d(pos3d, self.k, token_mask)           # (B,N,k)
        idx_h = idx.unsqueeze(1).expand(B, self.h, N, self.k)   # (B,h,N,k)

        def index_gather_BHNKd_from_BHNd(T, idx):
            B, H, N, D = T.shape
            k = idx.size(-1)
            device = T.device

            b = torch.arange(B, device=device)[:, None, None, None].expand(B, H, N, k)
            h = torch.arange(H, device=device)[None, :, None, None].expand(B, H, N, k)
            n = torch.arange(N, device=device)[None, None, :, None].expand(B, H, N, k)
            out = T[b, h, idx, :]  # (B,H,N,k,D)
            return out

        def index_gather_BNkD_from_BND(T, idx):
            B, N, D = T.shape
            k = idx.size(-1)
            device = T.device

            b = torch.arange(B, device=device)[:, None, None].expand(B, N, k)
            out = T[b, idx, :]  # (B,N,k,D)
            return out
        
        K_n = index_gather_BHNKd_from_BHNd(K, idx_h)
        V_n = index_gather_BHNKd_from_BHNd(V, idx_h)

    
        X_n = index_gather_BNkD_from_BND(pos3d,   idx)   
        F_n = index_gather_BNkD_from_BND(feat_sem, idx)  

        score_dot = (Q.unsqueeze(3) * K_n).sum(-1) / (self.dk ** 0.5)  # (B,h,N,k)

        F_q = F.normalize(feat_sem, dim=-1)
        F_nrm = F.normalize(F_n, dim=-1)
        d_feat = (F_q.unsqueeze(2) * F_nrm).sum(-1)  # (B,N,k)

        score = score_dot * (d_feat.unsqueeze(1).clamp(min=0))

        if token_mask is not None:
            neigh_mask = torch.gather(token_mask, 1, idx)  # (B,N,k)
            score = score.masked_fill(neigh_mask.unsqueeze(1) == 0, -1e9)

        attn = torch.softmax(score, dim=-1)                 # (B,h,N,k)
        y = (attn.unsqueeze(-1) * V_n).sum(dim=-2)          # (B,h,N,d)
        y = y.transpose(1, 2).contiguous().view(B, N, D)    # (B,N,D)

        out = feats + self.o_proj(y)
        out = out + self.mlp(self.norm2(out))

        if False:
            debug = {
                "attn": attn.detach(),      # (B,h,N,k)
                "idx": idx.detach(),        # (B,N,k)
                "pos3d": pos3d.detach(),    # (B,N,3)
                "feat_sem": feat_sem.detach() # (B,N,C)
            }
            return out, debug
        return out
    