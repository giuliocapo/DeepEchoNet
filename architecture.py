import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from globals import RGB_img_res


# ============================================================
# 🔹 Blocchi di base
# ============================================================
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, padding=1, stride=stride, bias=bias
        ).to(device)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias).to(device)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1, device='cuda'):
        return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride=stride, padding=kernel_size//2, bias=False),
        nn.Conv2d(oup, oup, 1, bias=False),     # <-- come METER: 3x3 seguita da 1x1
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


# ============================================================
# 🔹 Transformer Block (semplificato)
# ============================================================
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads) for t in qkv]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth=1, heads=4, dim_head=32, mlp_dim=128, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ============================================================
# 🔹 Encoder Echo (MobileViT ridotto)
# ============================================================
class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res_connect = (stride == 1 and inp == oup)

        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, heads=4, dim_head=8, mlp_dim=mlp_dim, dropout=dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)
        return x


class MobileViT_Echo(nn.Module):
    def __init__(self, image_size, dims, channels, patch_size=(1, 1)):
        super().__init__()
        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)
        self.mv2_1 = MV2Block(channels[0], channels[1], stride=1)
        self.mv2_2 = MV2Block(channels[1], channels[2], stride=2)  # ↓ 64→32
        self.mv2_3 = MV2Block(channels[2], channels[3], stride=2)  # ↓ 32→16
        self.mvit1 = MobileViTBlock(dims[0], 1, channels[3], 3, (2,2), mlp_dim=int(dims[0]*2))
        self.bridge12 = conv_1x1_bn(channels[3], channels[4])  # dopo mvit1:  C3 -> C4
        self.mvit2 = MobileViTBlock(dims[1], 1, channels[4], 3, (2,2), mlp_dim=int(dims[1]*4))
        self.bridge23 = conv_1x1_bn(channels[4], channels[5])  # dopo mvit2:  C4 -> C5
        self.mvit3 = MobileViTBlock(dims[2], 1, channels[5], 3, (2,2), mlp_dim=int(dims[2]*4))
        self.conv2 = conv_1x1_bn(channels[5], channels[6])

    def forward(self, x):
        y0 = self.conv1(x)
        y1 = self.mv2_1(y0)
        y2 = self.mv2_2(y1)
        y3 = self.mv2_3(y2)

        x = self.mvit1(y3)      # out: C = channels[3]
        x = self.bridge12(x)    #      C = channels[4]
        x = self.mvit2(x)       # exp: channel=channels[4] ✅
        x = self.bridge23(x)    #      C = channels[5]
        x = self.mvit3(x)       # exp: channel=channels[5] ✅
        x = self.conv2(x)       # C5 -> C6
        return x, [y0, y1, y2, y3]


# ============================================================
# 🔹 Decoder Echo
# ============================================================
class UpSample_layer(nn.Module):
    def __init__(self, inp, oup, sep_conv_filters, device):
        super().__init__()
        self.conv2d_transpose = nn.ConvTranspose2d(inp, oup, kernel_size=3, stride=2,
                                                   padding=1, output_padding=1, bias=False)
        self.end_up_layer = nn.Sequential(
            SeparableConv2d(sep_conv_filters, oup, kernel_size=3, device=device),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, enc_layer):
        x = self.conv2d_transpose(x)
        if x.shape[-2:] != enc_layer.shape[-2:]:
            enc_layer = F.interpolate(enc_layer, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc_layer], dim=1)
        x = self.end_up_layer(x)
        return x


class Decoder_Echo(nn.Module):
    def __init__(self, device, channels):
        super().__init__()
        c_out = channels[-1]  # 112 per echo_s
        self.conv_in = nn.Conv2d(c_out, 96, 1, bias=False)

        # nuovo primo upsample (da 8×8 → 16×16)
        self.up0 = UpSample_layer(96, 80, sep_conv_filters=(80 + channels[3]), device=device)

        # upsample successivi come prima ma spostati
        self.up1 = UpSample_layer(80, 48, sep_conv_filters=(48 + channels[2]), device=device)
        self.up2 = UpSample_layer(48, 32, sep_conv_filters=(32 + channels[1]), device=device)
        self.fuse3 = nn.Sequential(nn.Conv2d(32 + channels[0], 16, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(16),nn.ReLU(inplace=True),)

        self.out = nn.Conv2d(16, 1, 3, padding=1, bias=False)

    def forward(self, x, enc_layers):
        # enc_layers: [y0, y1, y2, y3]
        x = self.conv_in(x)

        # --- DEBUG SHAPES (solo alla prima iterazione) ---
        if not hasattr(self, "_printed_shapes"):
            print("\n[DEBUG] Decoder input / skip connections:")
            print(f"   bottleneck: {x.shape}")
            for i, y in enumerate(enc_layers):
                print(f"   enc[{i}] → {tuple(y.shape)}")
            self._printed_shapes = True

        x = self.up0(x, enc_layers[3])  # nuovo skip: 8×8 → 16×16
        x = self.up1(x, enc_layers[2])  # 16×16 → 32×32
        x = self.up2(x, enc_layers[1])  # 32×32 → 64×64
        enc0 = enc_layers[0]
        if enc0.shape[-2:] != x.shape[-2:]:
            enc0 = F.interpolate(enc0, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc0], dim=1)
        x = self.fuse3(x)
        x = self.out(x)
        return x


# ============================================================
# 🔹 Costruttore multi-architettura
# ============================================================
def build_Echo_model(arch_type):
    cfgs = {
        "echo_xxs": dict(dims=[48, 64, 80], channels=[8, 12, 24, 32, 48, 64, 80]),
        "echo_xs":  dict(dims=[64, 80, 96], channels=[8, 16, 32, 48, 64, 80, 96]),
        "echo_s":   dict(dims=[80, 96, 112], channels=[8, 16, 32, 64, 80, 96, 112]),
    }
    if arch_type not in cfgs:
        raise ValueError(f"❌ Architettura non riconosciuta: {arch_type}")
    cfg = cfgs[arch_type]
    encoder = MobileViT_Echo((RGB_img_res[1], RGB_img_res[2]), cfg["dims"], cfg["channels"])
    return encoder, cfg["channels"]


# ============================================================
# 🔹 Modello finale
# ============================================================
class DeepEchoNet(nn.Module):
    def __init__(self, device, arch_type="echo_s"):
        super().__init__()
        self.encoder, channels = build_Echo_model(arch_type)
        self.decoder = Decoder_Echo(device, channels)

    def forward(self, x):
        x, enc = self.encoder(x)
        x = self.decoder(x, enc)
        return x
