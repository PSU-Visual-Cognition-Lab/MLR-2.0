import argparse
import contextlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
 
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# TODO: fix this import to wherever your Dataset class lives.
from MLR_src.dataset_builder import Dataset  # noqa: E402
from training_constants2 import training_datasets, training_components  # noqa: E402
# ----------------------------------------------------------------------------


# =============================================================================
# 1. Data sampling (your logic, wrapped)
# =============================================================================
def infinite(loader):
    """Re-iterates the DataLoader every epoch (reshuffles each time).
    itertools.cycle would instead cache the first epoch's batches in memory
    and replay them in the same order forever."""
    while True:
        for batch in loader:
            yield batch


def make_loaders(component: str, bs: int, train: bool) -> dict:
    datasets = training_components[component][0]
    loader_bs = max(bs // len(datasets), 1)
    loaders = {}
    for dataset in datasets:
        dataset_name = dataset.split('-')[0]
        dataset_transforms = training_datasets[dataset]
        loaders[dataset] = infinite(
            Dataset(dataset_name, dataset_transforms, True).get_loader(loader_bs))
    return loaders


def batch_samples(sample_dataloader_names: list, dataloaders: dict,
                  whichdecode_use: str, randomize: bool = True):
    samples = []
    for sample_dataloader_name in sample_dataloader_names:
        sample_dataloader = dataloaders[sample_dataloader_name]
        sample, sample_labels = next(sample_dataloader)
        samples += [sample]

    samples = torch.cat(samples, 0)
    if randomize:
        perm = torch.randperm(samples.shape[0])
        return samples[perm], 0
    return samples, 0


def get_clean_batch(loaders: dict, component: str) -> torch.Tensor:
    x, _ = batch_samples(list(loaders.keys()), loaders, component, True)
    return x.float()
 
 
 
# =============================================================================
# 2. Noise pipeline
# =============================================================================
@dataclass
class NoiseConfig:
    # each corruption is applied to a sample independently with this probability
    p_gauss: float = 0.7
    sigma_max: float = 0.3        # fraction of the data range
    p_blur: float = 0.3
    blur_sigma_max: float = 1.5   # pixels
    p_occlude: float = 0.3
    max_boxes: int = 3
    box_max_frac: float = 0.4     # max box side as fraction of image side
    p_crosstalk: float = 0.3      # blend in another image from the batch
    alpha_max: float = 0.5
 
 
def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(2 * math.ceil(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    C = x.shape[1]
    x = F.pad(x, (k // 2,) * 4, mode='reflect')
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(x, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
    return x
 
 
def random_boxes_mask(B, H, W, max_boxes, max_frac, device) -> torch.Tensor:
    """(B,1,H,W) bool mask: union of up to max_boxes random rectangles."""
    mask = torch.zeros(B, 1, H, W, dtype=torch.bool, device=device)
    ys = torch.arange(H, device=device).view(1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, W)
    for i in range(max_boxes):
        bh = (torch.rand(B, 1, 1, device=device) * max_frac * H).long().clamp(min=2)
        bw = (torch.rand(B, 1, 1, device=device) * max_frac * W).long().clamp(min=2)
        y0 = (torch.rand(B, 1, 1, device=device) * (H - bh)).long()
        x0 = (torch.rand(B, 1, 1, device=device) * (W - bw)).long()
        box = (ys >= y0) & (ys < y0 + bh) & (xs >= x0) & (xs < x0 + bw)
        if i > 0:  # first box always on, later ones with prob 0.5
            box = box & (torch.rand(B, 1, 1, device=device) < 0.5)
        mask |= box.unsqueeze(1)
    return mask
 
 
def add_noise(x: torch.Tensor, cfg: NoiseConfig, lo: float, hi: float) -> torch.Tensor:
    """Per-sample random mixture of corruptions. x: (B,C,H,W) clean images."""
    B, C, H, W = x.shape
    dev = x.device
    rng = hi - lo
 
    def pick(p):
        return torch.rand(B, 1, 1, 1, device=dev) < p
 
    m_x, m_b, m_o, m_g = (pick(cfg.p_crosstalk), pick(cfg.p_blur),
                          pick(cfg.p_occlude), pick(cfg.p_gauss))
    m_g = m_g | ~(m_x | m_b | m_o | m_g)  # every sample gets at least one corruption
 
    y = x.clone()
 
    # crosstalk: blend with another image from the batch
    alpha = torch.rand(B, 1, 1, 1, device=dev) * cfg.alpha_max
    other = x[torch.randperm(B, device=dev)]
    y = torch.where(m_x, (1 - alpha) * y + alpha * other, y)
 
    # blur
    sigma = random.uniform(0.5, cfg.blur_sigma_max)
    y = torch.where(m_b, gaussian_blur(y, sigma), y)
 
    # occlusion (filled with the data minimum)
    boxes = random_boxes_mask(B, H, W, cfg.max_boxes, cfg.box_max_frac, dev)
    y = torch.where(m_o & boxes, torch.full_like(y, lo), y)
 
    # additive gaussian, applied last
    s = torch.rand(B, 1, 1, 1, device=dev) * cfg.sigma_max * rng
    y = torch.where(m_g, y + torch.randn_like(y) * s, y)
 
    return y.clamp(lo, hi)
 
 
# =============================================================================
# 3. Models
# =============================================================================
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
    )
 
 
class UNet(nn.Module):
    """Small residual U-Net. Output = input + learned correction.
    Handles any H,W by reflect-padding to a multiple of 2**depth (28 -> 32)."""
 
    def __init__(self, in_ch=1, base=32, depth=3):
        super().__init__()
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth + 1)]  # e.g. 32,64,128,256
 
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.enc.append(conv_block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(chs[-2], chs[-1])
 
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        prev = chs[-1]
        for ch in reversed(chs[:-1]):
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(prev, ch, 1)))
            self.dec.append(conv_block(2 * ch, ch))
            prev = ch
 
        self.head = nn.Conv2d(chs[0], in_ch, 1)
        nn.init.zeros_(self.head.weight)  # start as identity mapping
        nn.init.zeros_(self.head.bias)
 
    def forward(self, x):
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-h) % m, (-w) % m
        xp = F.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x
 
        skips = []
        d = xp
        for blk in self.enc:
            d = blk(d)
            skips.append(d)
            d = self.pool(d)
        d = self.bottleneck(d)
        for up, dec in zip(self.up, self.dec):
            d = up(d)
            d = dec(torch.cat([d, skips.pop()], dim=1))
 
        return x + self.head(d)[..., :h, :w]
 
 
class DnCNN(nn.Module):
    """Plain full-resolution residual CNN baseline (no downsampling)."""
 
    def __init__(self, in_ch=1, width=64, layers=12):
        super().__init__()
        mods = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(layers - 2):
            mods += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                     nn.GroupNorm(8, width), nn.ReLU(inplace=True)]
        last = nn.Conv2d(width, in_ch, 3, padding=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        mods.append(last)
        self.net = nn.Sequential(*mods)
 
    def forward(self, x):
        return x + self.net(x)
 
 
# =============================================================================
# 4. Metrics
# =============================================================================
def ssim(x, y, rng, win=7, sigma=1.5):
    C = x.shape[1]
    ax = torch.arange(win, device=x.device, dtype=x.dtype) - win // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(C, 1, win, win).contiguous()
    f = lambda t: F.conv2d(t, k, groups=C)
    mx, my = f(x), f(y)
    vx, vy, cxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    C1, C2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    s = ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))
    return s.mean(dim=(1, 2, 3))
 
 
def psnr(x, y, rng):
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return 10 * torch.log10(rng ** 2 / mse)
 
 
@torch.no_grad()
def score(fn, noisy, clean, rng, chunk=500):
    """Mean PSNR/SSIM of fn(noisy) vs clean over the whole val set."""
    ps, ss, n = 0.0, 0.0, 0
    for i in range(0, len(clean), chunk):
        pred = fn(noisy[i:i + chunk])
        c = clean[i:i + chunk]
        ps += psnr(pred, c, rng).sum().item()
        ss += ssim(pred, c, rng).sum().item()
        n += len(c)
    return ps / n, ss / n
 
 
# =============================================================================
# 5. Training
# =============================================================================
def save_samples(model, noisy, clean, lo, hi, path, n=12):
    try:
        from torchvision.utils import save_image
    except ImportError:
        return
    model.eval()
    with torch.no_grad():
        den = model(noisy[:n]).clamp(lo, hi)
    to01 = lambda t: ((t - lo) / (hi - lo)).clamp(0, 1)
    rows = [torch.cat(list(to01(t[:n])), dim=-1) for t in (noisy, den, clean)]
    save_image(torch.cat(rows, dim=-2), path)  # rows: noisy / denoised / clean
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--component', default='shape')
    ap.add_argument('--model', choices=['unet', 'dncnn'], default='unet')
    ap.add_argument('--base', type=int, default=32, help='unet base channels')
    ap.add_argument('--depth', type=int, default=3, help='unet downsampling levels')
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--loss', choices=['l1', 'mse'], default='l1')
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--val_batches', type=int, default=8)
    ap.add_argument('--out', default='checkpoints/denoiser')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--run_name', default='test',
                    help='samples are saved to training_samples/denoiser_{run_name}/')
    ap.add_argument('--val_split', choices=['train', 'heldout'], default='train',
                    help="'train': validation images come from the train loaders (NOT held out, "
                         "but noise is fresh/fixed so fine for a feasibility check). "
                         "'heldout': build loaders with train=False (needs Dataset to support it).")
    args = ap.parse_args()
 
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample_dir = Path('training_samples') / f'denoiser_{args.run_name}'
    sample_dir.mkdir(parents=True, exist_ok=True)
    cfg = NoiseConfig()
 
    train_loaders = make_loaders(args.component, args.bs, train=True)
    if args.val_split == 'heldout':
        val_loaders = make_loaders(args.component, args.bs, train=False)
    else:
        print('WARNING: validation images are drawn from the training loaders (not held out).')
        val_loaders = train_loaders
 
    # ---- infer channels / data range from a first batch ----
    x0 = get_clean_batch(train_loaders, args.component)
    in_ch = x0.shape[1]
    lo, hi = x0.min().item(), x0.max().item()
    rng = hi - lo
    print(f'image shape {tuple(x0.shape[1:])}, data range [{lo:.3f}, {hi:.3f}]')
 
    # ---- fixed validation set with fixed noise (comparable across evals) ----
    clean_val = torch.cat([get_clean_batch(val_loaders, args.component)
                           for _ in range(args.val_batches)]).to(dev)
    g_state = torch.get_rng_state()
    torch.manual_seed(1234)
    noisy_val = add_noise(clean_val, cfg, lo, hi)
    torch.set_rng_state(g_state)
 
    ident = score(lambda t: t, noisy_val, clean_val, rng)
    blur = score(lambda t: gaussian_blur(t, 1.0), noisy_val, clean_val, rng)
    print(f'val  identity: PSNR {ident[0]:.2f} SSIM {ident[1]:.4f}')
    print(f'val  blur s=1: PSNR {blur[0]:.2f} SSIM {blur[1]:.4f}')
 
    # ---- model / optimiser ----
    model = (UNet(in_ch, args.base, args.depth) if args.model == 'unet'
             else DnCNN(in_ch)).to(dev)
    print(f'{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params')
 
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * it / args.iters)))
    loss_fn = F.l1_loss if args.loss == 'l1' else F.mse_loss
    use_amp = dev == 'cuda' and hasattr(torch, 'autocast') and torch.cuda.is_bf16_supported()
 
    best = -1.0
    run_loss = 0.0
    for it in range(1, args.iters + 1):
        model.train()
        clean = get_clean_batch(train_loaders, args.component).to(dev, non_blocking=True)
        noisy = add_noise(clean, cfg, lo, hi)
 
        amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                   if use_amp else contextlib.nullcontext())
        with amp_ctx:
            pred = model(noisy)
        loss = loss_fn(pred.float(), clean)
 
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += loss.item()
 
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            p, s = score(model, noisy_val, clean_val, rng)
            print(f'it {it:6d} | train loss {run_loss / args.eval_every:.4f} | '
                  f'val PSNR {p:.2f} (identity {ident[0]:.2f}) SSIM {s:.4f}')
            run_loss = 0.0
            save_samples(model, noisy_val, clean_val, lo, hi,
                         sample_dir / f'samples_it{it:06d}.png')
            if p > best:
                best = p
                torch.save({'model': model.state_dict(), 'args': vars(args),
                            'in_ch': in_ch, 'lo': lo, 'hi': hi}, out / 'best.pt')
 
    print(f'done. best val PSNR {best:.2f}  ->  {out / "best.pt"}')
 
 
if __name__ == '__main__':
    main()