import torch  # isort:skip
import json
from argparse import ArgumentParser
from contextlib import nullcontext
from types import SimpleNamespace

import tensorflow as tf
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

import commons
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import MultiPeriodDiscriminator, SynthesizerTrn
from tfloader import load_tfdata

tf.config.set_visible_devices([], "GPU")

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="config.json")
parser.add_argument("--tfdata", type=str, default="data/tfdata")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--compile", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
FLAGS = parser.parse_args()

# credit: https://github.com/karpathy/nanoGPT/blob/master/train.py#L72-L112
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(FLAGS.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = FLAGS.device
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = FLAGS.compile
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
# initialize a GradScaler. If enabled=False scaler is a no-op
print(dtype, ptdtype, ctx)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))


train_ds = load_tfdata(FLAGS.tfdata, "train")
ds = train_ds.bucket_by_sequence_length(
    lambda x: tf.shape(x["spec"])[0],
    bucket_boundaries=(32, 300, 400, 500, 600, 700, 800, 900, 1000),
    bucket_batch_sizes=[FLAGS.batch_size] * 10,
    pad_to_bucket_boundary=False,
)


with open(FLAGS.config, "rb") as f:
    hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
torch.manual_seed(hps.train.seed)

net_g = SynthesizerTrn(
    256,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **vars(hps.model),
).to(device)
net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)
optim_d = torch.optim.AdamW(
    net_d.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)

if compile:
    net_g = torch.compile(net_g)
    net_d = torch.compile(net_d)

epoch_str = 1
global_step = 0


scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
    optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
)

net_g.train()
net_d.train()


for batch in tqdm(ds.prefetch(1).as_numpy_iterator()):
    x = torch.from_numpy(batch["phone_idx"]).long().to(device, non_blocking=True)
    x_lengths = (
        torch.from_numpy(batch["phone_length"]).long().to(device, non_blocking=True)
    )
    spec = (
        torch.from_numpy(batch["spec"])
        .swapaxes(-1, -2)
        .float()
        .to(device, non_blocking=True)
    )
    spec = torch.log(1e-3 + spec)
    spec_lengths = (
        torch.from_numpy(batch["spec_length"]).long().to(device, non_blocking=True)
    )
    y = torch.from_numpy(batch["wav"]).float()[:, None, :].to(device, non_blocking=True)
    y_lengths = (
        torch.from_numpy(batch["wav_length"]).long().to(device, non_blocking=True)
    )
    duration = (
        torch.from_numpy(batch["phone_duration"]).float().to(device, non_blocking=True)
    )
    end_time = torch.cumsum(duration, dim=-1)
    start_time = end_time - duration
    start_frame = (
        start_time * hps.data.sampling_rate / hps.data.hop_length / 1000
    ).int()
    end_frame = (end_time * hps.data.sampling_rate / hps.data.hop_length / 1000).int()
    pos = torch.arange(0, spec.shape[-1], device=spec.device)
    attn = torch.logical_and(
        pos[None, :, None] >= start_frame[:, None, :],
        pos[None, :, None] < end_frame[:, None, :],
    )

    with ctx:
        (
            y_hat,
            l_length,
            attn,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        ) = net_g(x, x_lengths, attn.float(), spec, spec_lengths)

    mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
    )
    y_mel = commons.slice_segments(
        mel, ids_slice, hps.train.segment_size // hps.data.hop_length
    )
    y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
    )

    y = commons.slice_segments(
        y, ids_slice * hps.data.hop_length, hps.train.segment_size
    )  # slice

    with ctx:
        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with ctx:
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)
    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    print(loss_disc_all.item(), loss_gen_all.item())
