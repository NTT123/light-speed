import torch  # isort:skip
import tensorflow as tf

feature_description = {
    "phone_idx": tf.io.FixedLenFeature([], tf.string),
    "phone_duration": tf.io.FixedLenFeature([], tf.string),
    "wav": tf.io.FixedLenFeature([], tf.string),
    "spec": tf.io.FixedLenFeature([], tf.string),
}


def parse_tfrecord(r):
    r = tf.io.parse_example(r, feature_description)
    wav = tf.reshape(tf.io.parse_tensor(r["wav"], out_type=tf.float16), [-1])
    spec = tf.io.parse_tensor(r["spec"], out_type=tf.float16)
    spec = tf.reshape(spec, [-1, tf.shape(spec)[-1]])
    phone_idx = tf.reshape(tf.io.parse_tensor(r["phone_idx"], out_type=tf.int32), [-1])
    phone_duration = tf.reshape(
        tf.io.parse_tensor(r["phone_duration"], out_type=tf.float32), [-1]
    )
    return {
        "phone_idx": phone_idx,
        "phone_duration": phone_duration,
        "phone_length": tf.shape(phone_duration)[0],
        "wav": wav,
        "wav_length": tf.shape(wav)[0],
        "spec": spec,
        "spec_length": tf.shape(spec)[0],
    }


def load_tfdata(root, split, batch_size, seed, rank=0, world_size=1):
    files = tf.data.Dataset.list_files(
        f"{root}/{split}/part_*.tfrecords", shuffle=False
    )
    files = files.shuffle(len(files), seed=seed)
    files = files.shard(world_size, rank)
    return (
        tf.data.TFRecordDataset(files, num_parallel_reads=4)
        .map(parse_tfrecord, num_parallel_calls=4, deterministic=True)
        .shuffle(buffer_size=batch_size * 32, seed=seed)
        .bucket_by_sequence_length(
            lambda x: tf.shape(x["spec"])[0],
            bucket_boundaries=(200, 300, 400, 500, 600, 700, 800),
            bucket_batch_sizes=[batch_size] * 8,
            pad_to_bucket_boundary=False,
            drop_remainder=True,
        )
        .prefetch(1)
    )
