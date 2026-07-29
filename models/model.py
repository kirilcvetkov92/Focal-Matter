import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models import focal


# ============================================================
# Utils
# ============================================================
def _reshape_tokens_to_map(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(tokens, 'b (h w) c -> b c h w', h=h, w=w)

def _reshape_map_to_tokens(feat: torch.Tensor):
    h, w = feat.shape[2:]
    return rearrange(feat, 'b c h w -> b (h w) c'), h, w


# ============================================================
# Lightweight Context Aggregation (top of U)
# ============================================================
class ContextAgg(nn.Module):
    def __init__(self, cin: int, cout: int, rates=(1, 2, 4, 8)):
        super().__init__()
        branches = []
        for r in rates:
            branches.append(nn.Sequential(
                nn.Conv2d(cin, cin, 3, 1, r, groups=cin, dilation=r, bias=True),  # depthwise
                nn.Conv2d(cin, cout // 4, 1, bias=True)  # pointwise
            ))
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.utils.spectral_norm(nn.Conv2d(cout, cout, 1, bias=True))
        self.act = nn.PReLU(cout)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        y = torch.cat(outs, dim=1)
        y = self.fuse(y)
        return self.act(y)


# ============================================================
# 2) Feature Align + Gate (skip attention, cheap)
# ============================================================
class SkipAlignGate(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.align = nn.utils.spectral_norm(nn.Conv2d(cin, cout, 1, bias=True))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cout, max(cout // 4, 16), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(cout // 4, 16), cout, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, skip, x_size):
        s = self.align(skip)
        if s.shape[-2:] != x_size:
            s = F.interpolate(s, size=x_size, mode='bilinear', align_corners=False)
        g = self.gate(s)
        return s * g


# ============================================================
# 3) Focal refinement wrapper (uses focal.BasicLayer with depth=3)
# ============================================================
class FocalRefine(nn.Module):
    def __init__(self, c: int, depth: int = 3, drop_path: float = 0.0, focal_window: int = 7):
        super().__init__()
        self.block = focal.BasicLayer(
            c, depth, focal_window=focal_window, drop_path=drop_path
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        tokens, _, _, *rest = self.block(tokens, h, w)  # ignore extras
        return rearrange(tokens, 'b (h w) c -> b c h w', h=h, w=w)


# ============================================================
# 4) Matting network with FocalNet encoder (U-shape decoder)
# ============================================================
class FocalMatter(nn.Module):
    """
    Image matting with a FocalNet-based encoder and a VRAM-efficient U-shape decoder.
    Encoder pyramid outputs: [96, 192, 384, 768].
    Decoder channels chosen to target ~25M trainable params in the decoder.
    """

    def __init__(self, encoder: nn.Module, encoder_weights_path: str = None):
        super().__init__()
        self.encoder = encoder

        #  Load encoder weights (optional) 
        if encoder_weights_path is not None:
            try:
                sd = torch.load(encoder_weights_path, map_location="cpu")
                sd = sd.get("model", sd)
                self.encoder.load_state_dict(sd, strict=False)
            except Exception as e:
                print("Warning loading encoder weights:", e)

        #  Stem (6-channel input = image + trimap) 
        self.stem_pre = nn.Sequential(
            nn.Conv2d(6, 48, 3, 1, 1, bias=True), nn.PReLU(48)
        )
        self.stem_down = nn.Sequential(
            nn.Conv2d(48, 64, 3, 2, 1, bias=True), nn.PReLU(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True), nn.PReLU(64)
        )

        # Adapt encoder input to accept 64-ch features (proj 64 -> 96)
        if hasattr(self.encoder, "patch_embed") and hasattr(self.encoder.patch_embed, "proj"):
            self.encoder.patch_embed.proj = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=True)
        else:
            raise RuntimeError("Encoder missing patch_embed.proj; cannot adjust input channels to 64.")

        # ---------------------------
        # Channel plan (decoder)
        # ---------------------------
        C4 = 512   
        C3 = 256
        C2 = 128
        C1 = 64

        self._C4, self._C3, self._C2, self._C1 = C4, C3, C2, C1  # for printing

        # ====== TOP: Context aggregation on l4 ======
        self.top_reduce = nn.utils.spectral_norm(nn.Conv2d(768, C4, 1, bias=True))
        self.top_ctx = ContextAgg(cin=768, cout=C4, rates=(1, 2, 4, 8))
        self.top_fuse = nn.utils.spectral_norm(nn.Conv2d(C4 + C4, C4, 1, bias=True))  # concat(reduce, ctx)

        # ====== Refinement at each scale (BasicLayer depth=3) ======
        self.refine4 = FocalRefine(C4, depth=3, drop_path=0.10, focal_window=7)  # h4,w4

        # Upsample 4->3 by transposed conv (sharper edges)
        self.up4to3 = nn.ConvTranspose2d(C4, C4, kernel_size=2, stride=2, bias=True)
        self.skip3 = SkipAlignGate(cin=384, cout=C3)
        self.fuse3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(C4 + C3, C3, 1, bias=True)),
            nn.PReLU(C3)
        )
        self.refine3 = FocalRefine(C3, depth=3, drop_path=0.08, focal_window=7)  # h3,w3

        # Upsample 3->2 by transposed conv
        self.up3to2 = nn.ConvTranspose2d(C3, C3, kernel_size=2, stride=2, bias=True)
        self.skip2 = SkipAlignGate(cin=192, cout=C2)
        self.fuse2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(C3 + C2, C2, 1, bias=True)),
            nn.PReLU(C2)
        )
        self.refine2 = FocalRefine(C2, depth=3, drop_path=0.06, focal_window=7)  # h2,w2

        # Upsample 2->1 by bilinear (cheaper) then fuse
        self.up2to1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.skip1 = SkipAlignGate(cin=96, cout=C1)
        self.fuse1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(C2 + C1, C1, 1, bias=True)),
            nn.PReLU(C1)
        )
        self.refine1 = FocalRefine(C1, depth=3, drop_path=0.04, focal_window=7)  # h1,w1

        # ====== Neck & Output heads ======
        self.neck = nn.Sequential(
            nn.Conv2d(64 + C1, 64, 3, 1, 1, bias=True), nn.PReLU(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True), nn.PReLU(64),
            nn.Conv2d(64, 48, 3, 1, 1, bias=True), nn.PReLU(48)
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(48 + 48 + 6, 32, 3, 1, 1, bias=True), nn.PReLU(32),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True), nn.PReLU(32),
            nn.Conv2d(32, 1, 3, 1, 1, bias=True)
        )

    # ------------------------------
    # Forward
    # ------------------------------
    def forward(self, image: torch.Tensor, trimap: torch.Tensor):
        # Inputs & stem
        six_channel_input = torch.cat((image, trimap), dim=1)         # (B,6,H,W)
        stem_l0 = self.stem_pre(six_channel_input)                     # (B,48,H,W)
        stem_l1 = self.stem_down(stem_l0)                              # (B,64,H/2,W/2)

        # Encoder pyramid (assumed: 4 stages -> 96, 192, 384, 768)
        l1, l2, l3, l4 = self.encoder(stem_l1)                         # h1,w1 ~ H/4; h4,w4 ~ H/32

        # ---- Top context aggregation + reduce ----
        top_ctx = self.top_ctx(l4)                                     # (B,C4,h4,w4) from 768
        top_red = self.top_reduce(l4)                                  # (B,C4,h4,w4)
        x4 = self.top_fuse(torch.cat([top_red, top_ctx], dim=1))       # (B,C4,h4,w4)
        x4 = self.refine4(x4)

        # ---- 4 -> 3 ----
        up3 = self.up4to3(x4)                                          # (B,C4,h3,w3)
        s3 = self.skip3(l3, up3.shape[-2:])                            # (B,C3,h3,w3)
        x3 = self.fuse3(torch.cat([up3, s3], dim=1))                   # (B,C3,h3,w3)
        x3 = self.refine3(x3)

        # ---- 3 -> 2 ----
        up2 = self.up3to2(x3)                                          # (B,C3,h2,w2)
        s2 = self.skip2(l2, up2.shape[-2:])                            # (B,C2,h2,w2)
        x2 = self.fuse2(torch.cat([up2, s2], dim=1))                   # (B,C2,h2,w2)
        x2 = self.refine2(x2)

        # ---- 2 -> 1 ----
        up1 = self.up2to1(x2)                                          # (B,C2,h1,w1)
        s1 = self.skip1(l1, up1.shape[-2:])                            # (B,C1,h1,w1)
        x1 = self.fuse1(torch.cat([up1, s1], dim=1))                   # (B,C1,h1,w1)
        x1 = self.refine1(x1)

        # ---- Neck back to stem scale and alpha head ----
        neck = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)  # (B,*, H/2, W/2)
        neck = torch.cat([stem_l1, neck], dim=1)                       # (B,64+C1,H/2,W/2)
        neck = self.neck(neck)                                         # (B,48,H/2,W/2)

        alpha_in = F.interpolate(neck, scale_factor=2, mode='bilinear', align_corners=False)  # (B,48,H,W)
        alpha_in = torch.cat([six_channel_input, stem_l0, alpha_in], dim=1)                   # (B,6+48+48,H,W)
        alpha = self.alpha_head(alpha_in)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha


# ============================================================
# 5) Sanity test + parameter accounting
# ============================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dummy_img = torch.randn(1, 3, 512, 512, device=device)
    dummy_trimap = torch.randn(1, 3, 512, 512, device=device)

    from models.focal import FocalNet as FocalFocus
    encoder = FocalFocus(
        in_chans=3,        
        embed_dim=96,
        depths=[2, 2, 6, 2],
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3]
    )

    model = FocalMatter(encoder=encoder).to(device).eval()

    # ---- parameter counters ----
    def count_parameters(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    def count_submodule(m, name):
        sub = getattr(m, name)
        total = sum(p.numel() for p in sub.parameters())
        trainable = sum(p.numel() for p in sub.parameters() if p.requires_grad)
        return total, trainable

    with torch.no_grad():
        out = model(dummy_img, dummy_trimap)

        total_params, trainable_params = count_parameters(model)
        enc_params, enc_trainable = count_submodule(model, "encoder")
        dec_params = total_params - enc_params
        dec_trainable = trainable_params - enc_trainable

        print(f"\n--- Parameter Summary ---")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Encoder parameters: {enc_params:,}")
        print(f"Decoder parameters: {dec_params:,}")
        print(f"Decoder widths (C4,C3,C2,C1): {model._C4}, {model._C3}, {model._C2}, {model._C1}\n")

        print(f"Input (image):  {tuple(dummy_img.shape)}")
        print(f"Input (trimap): {tuple(dummy_trimap.shape)}")
        print(f"Output alpha shape: {tuple(out.shape)}")
        print(f"Output alpha min/max: {out.min().item():.4f} / {out.max().item():.4f}")

        assert out.shape[1] == 1, "Output should be 1-channel alpha mask"
        assert 0.0 <= out.min() <= 1.0 and 0.0 <= out.max() <= 1.0, "Output"
        print("\nBasic output checks passed.\n")

