import tensorflow as tf
import torch

tf.config.set_visible_devices([], "GPU")


import json

import torch
from torch.nn import functional as F

import commons
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import MultiPeriodDiscriminator, SynthesizerTrn

with open("config.json", "rb") as f:
    hps = json.load(f)


def load_tfdata(root, split):
    files = tf.data.Dataset.list_files(f"{root}/{split}/part_*.tfrecords")
    files = files.repeat().shuffle(len(files))

    feature_description = {
        "phone_idx": tf.io.FixedLenFeature([], tf.string),
        "phone_duration": tf.io.FixedLenFeature([], tf.string),
        "phone_mask": tf.io.FixedLenFeature([], tf.string),
        "wav": tf.io.FixedLenFeature([], tf.string),
        "spec": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_tfrecord(r):
        r = tf.io.parse_example(r, feature_description)
        wav = tf.reshape(tf.io.parse_tensor(r["wav"], out_type=tf.float16), [-1])
        phone_mask = tf.reshape(
            tf.io.parse_tensor(r["phone_mask"], out_type=tf.bool), [-1]
        )
        spec = tf.io.parse_tensor(r["spec"], out_type=tf.float16)
        spec = tf.reshape(spec, [-1, tf.shape(spec)[-1]])
        return {
            "phone_idx": tf.reshape(
                tf.io.parse_tensor(r["phone_idx"], out_type=tf.int32), [-1]
            ),
            "phone_duration": tf.reshape(
                tf.io.parse_tensor(r["phone_duration"], out_type=tf.float32), [-1]
            ),
            "phone_mask": phone_mask,
            "phone_length": tf.shape(phone_mask)[0],
            "wav": wav,
            "wav_length": tf.shape(wav)[0],
            "spec": spec,
            "spec_length": tf.shape(spec)[0],
        }

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=4).map(
        parse_tfrecord, num_parallel_calls=4
    )
    return ds


train_ds = load_tfdata("tfdata", "train")
bs = 16
ds = train_ds.bucket_by_sequence_length(
    lambda x: tf.shape(x["spec"])[0],
    bucket_boundaries=(32, 300, 400, 500, 600, 700, 800, 900, 1000),
    bucket_batch_sizes=[bs] * 10,
    pad_to_bucket_boundary=False,
)


from utils import HParams

config_save_path = "config.json"
with open(config_save_path, "r") as f:
    data = f.read()
    config = json.loads(data)

hparams = HParams(**config)
hparams.model_dir = "./"
hps = hparams
torch.manual_seed(hps.train.seed)

from tqdm.auto import tqdm

net_g = SynthesizerTrn(
    256,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
)
net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
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
    x = torch.from_numpy(batch["phone_idx"]).long()
    x_lengths = torch.from_numpy(batch["phone_mask"]).sum(-1).long()
    spec = torch.from_numpy(batch["spec"]).swapaxes(-1, -2).float()
    spec = torch.log(1e-3 + spec)
    spec_lengths = torch.from_numpy(batch["spec_length"]).long()
    y = torch.from_numpy(batch["wav"]).float()[:, None, :]
    y_lengths = torch.from_numpy(batch["wav_length"]).long()

    duration = torch.from_numpy(batch["phone_duration"]).float()
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

    # Discriminator
    y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    loss_disc_all = loss_disc
    optim_d.zero_grad()
    loss_disc_all.backward()
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    optim_d.step()

    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

    # loss_dur = torch.sum(l_length.float())
    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)
    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

    optim_g.zero_grad()

    loss_gen_all.backward()
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    optim_g.step()

    loss_disc_all, loss_gen_all
