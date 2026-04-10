"""Hooked call_variants: runs inference with activation capture via HookedModel.

This replaces the standard call_variants binary, adding activation extraction
and caching through deepvariant.activation_hooks.HookedModel. The output
CallVariantsOutput tfrecord is fully compatible with postprocess_variants.

Usage:
  python scripts/call_variants_hooked.py \
    --examples /path/to/make_examples.tfrecord@1.gz \
    --checkpoint /opt/models/wgs \
    --outfile /path/to/call_variants_output.tfrecord.gz \
    --activation_cache_dir /path/to/activation_cache \
    --hook_layers mixed5 \
    --batch_size 512
"""

import json
import os
import sys
import time

# Add repo root to path so 'deepvariant' module can be imported
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

_EXAMPLES = flags.DEFINE_string(
    'examples', None, 'Path to make_examples tfrecord output.')
_CHECKPOINT = flags.DEFINE_string(
    'checkpoint', None,
    'Path to model checkpoint directory or .ckpt prefix.')
_OUTFILE = flags.DEFINE_string(
    'outfile', None, 'Output tfrecord.gz for CallVariantsOutput protos.')
_ACTIVATION_CACHE_DIR = flags.DEFINE_string(
    'activation_cache_dir', None,
    'Directory to write cached activation .npz files.')
_HOOK_LAYERS = flags.DEFINE_string(
    'hook_layers', 'mixed5',
    'Comma-separated layer names to capture activations from.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 512, 'Batch size for inference.')
_MAX_CACHE_ENTRIES = flags.DEFINE_integer(
    'max_cache_entries', 10000,
    'Maximum activation entries to hold in memory before eviction to disk.')
_SINGLE_PASS = flags.DEFINE_boolean(
    'single_pass', True,
    'If True, get predictions and activations in one forward pass.')


def _find_checkpoint_path(checkpoint_flag):
  """Resolve the checkpoint flag to the actual checkpoint prefix.

  The flag may point to a directory (saved model or checkpoint dir) or
  directly to a .ckpt prefix.
  """
  if tf.io.gfile.exists(os.path.join(checkpoint_flag, 'saved_model.pb')):
    return checkpoint_flag, True

  if os.path.isdir(checkpoint_flag):
    ckpt = tf.train.latest_checkpoint(checkpoint_flag)
    if ckpt:
      return ckpt, False
    for f in tf.io.gfile.listdir(checkpoint_flag):
      if f.endswith('.index'):
        return os.path.join(checkpoint_flag, f.replace('.index', '')), False

  return checkpoint_flag, False


def _load_example_info(checkpoint_path):
  """Load example_info.json from the checkpoint directory."""
  if os.path.isdir(checkpoint_path):
    info_path = os.path.join(checkpoint_path, 'example_info.json')
  else:
    info_path = os.path.join(
        os.path.dirname(checkpoint_path), 'example_info.json')

  if not tf.io.gfile.exists(info_path):
    raise FileNotFoundError(
        f'example_info.json not found at {info_path}. '
        'Check your --checkpoint path.')

  with tf.io.gfile.GFile(info_path, 'r') as f:
    return json.loads(f.read())


def _build_dataset(examples_path, example_shape, batch_size,
                    channel_indices=None):
  """Build a tf.data pipeline from the make_examples tfrecords."""
  from third_party.nucleus.io import sharded_file_utils

  proto_features = {
      'image/encoded': tf.io.FixedLenFeature((), tf.string),
      'variant/encoded': tf.io.FixedLenFeature((), tf.string),
      'alt_allele_indices/encoded': tf.io.FixedLenFeature((), tf.string),
  }

  def _parse(serialized):
    parsed = tf.io.parse_single_example(
        serialized=serialized, features=proto_features)
    image = tf.io.decode_raw(parsed['image/encoded'], tf.uint8)
    image = tf.reshape(image, example_shape)
    image = tf.cast(image, tf.float32)
    image = tf.subtract(image, 128.0)
    image = tf.math.divide(image, 128.0)
    if channel_indices:
      image = tf.gather(image, channel_indices, axis=-1)
    return (
        parsed['image/encoded'],
        image,
        parsed['variant/encoded'],
        parsed['alt_allele_indices/encoded'],
    )

  file_pattern = sharded_file_utils.normalize_to_sharded_file_pattern(
      examples_path)
  ds = tf.data.TFRecordDataset.list_files(file_pattern, shuffle=False)
  ds = ds.interleave(
      lambda f: tf.data.TFRecordDataset(f, compression_type='GZIP'),
      cycle_length=4,
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return ds


def _write_cvo(writer, variant_encoded, alt_allele_indices_encoded, gls):
  """Write a single CallVariantsOutput proto to the tfrecord writer."""
  from deepvariant.protos import deepvariant_pb2
  from third_party.nucleus.protos import variants_pb2

  variant = variants_pb2.Variant.FromString(variant_encoded)
  alt_allele_indices = (
      deepvariant_pb2.CallVariantsOutput.AltAlleleIndices.FromString(
          alt_allele_indices_encoded))

  cvo = deepvariant_pb2.CallVariantsOutput(
      variant=variant,
      alt_allele_indices=alt_allele_indices,
      genotype_probabilities=gls,
  )
  writer.write(cvo.SerializeToString())


def _round_gls(gls, precision=10):
  """Round genotype likelihoods so they sum to 1."""
  gls = list(gls)
  min_ix = int(np.argmin(gls))
  rounded = [round(float(g), precision) for g in gls]
  rounded[min_ix] = max(
      0.0,
      round(1 - sum(rounded[:min_ix] + rounded[min_ix + 1:]), precision))
  return rounded


def main(argv):
  del argv

  # Lazy imports so flag parsing happens first.
  from deepvariant import activation_hooks

  checkpoint_flag = _CHECKPOINT.value
  ckpt_path, is_saved_model = _find_checkpoint_path(checkpoint_flag)
  logging.info('Resolved checkpoint: %s (saved_model=%s)', ckpt_path,
               is_saved_model)

  if is_saved_model:
    raise NotImplementedError(
        'Activation hooks require a .ckpt checkpoint, not a SavedModel. '
        'Please point --checkpoint to a directory with .ckpt files.')

  example_info = _load_example_info(checkpoint_flag)
  example_shape = tuple(example_info['shape'])
  logging.info('Example shape from example_info.json: %s', example_shape)

  # Determine channel ablation indices (if any).
  channel_indices = []
  model_shape = list(example_shape)
  ablation_channels = example_info.get('ablation_channels', [])
  if ablation_channels:
    channels = example_info['channels']
    ablation_set = set(ablation_channels)
    channel_indices = [
        idx for idx, ch in enumerate(channels) if ch not in ablation_set
    ]
    model_shape[2] = len(channel_indices)
    logging.info('Channel ablation active: %d -> %d channels',
                 example_shape[2], model_shape[2])

  # Build model for inference only — skip add_l2_regularizers which uses
  # model.add_loss() (unsupported in Keras 3 Functional models).
  # L2 regularization has no effect during inference.
  from deepvariant import dv_constants
  input_shape = tuple(model_shape)
  backbone = tf.keras.applications.InceptionV3(
      include_top=False,
      weights=None,
      input_shape=input_shape,
      classes=dv_constants.NUM_CLASSES,
      pooling='avg',
  )
  hid = tf.keras.layers.Dropout(0.2)(backbone.output)
  head = tf.keras.layers.Dense(
      dv_constants.NUM_CLASSES, activation='softmax', dtype=tf.float32,
      name='classification',
  )(hid)
  model = tf.keras.Model(
      inputs=backbone.input, outputs=head, name='inceptionv3')
  checkpoint = tf.train.Checkpoint(model)
  checkpoint.restore(ckpt_path).expect_partial()
  logging.info('Loaded model weights from %s', ckpt_path)

  # Wrap with HookedModel.
  layer_names = [l.strip() for l in _HOOK_LAYERS.value.split(',')]
  hooked = activation_hooks.HookedModel(
      model=model,
      layer_names=layer_names,
      cache_dir=_ACTIVATION_CACHE_DIR.value,
      max_cache_entries=_MAX_CACHE_ENTRIES.value,
      single_pass=_SINGLE_PASS.value,
  )
  logging.info('HookedModel created for layers: %s', layer_names)

  # Build dataset.
  dataset = _build_dataset(
      _EXAMPLES.value, example_shape, _BATCH_SIZE.value,
      channel_indices=channel_indices or None)

  # Open output writer.
  outdir = os.path.dirname(_OUTFILE.value)
  if outdir:
    tf.io.gfile.makedirs(outdir)
  tf_options = tf.io.TFRecordOptions(compression_type='GZIP')
  writer = tf.io.TFRecordWriter(_OUTFILE.value, options=tf_options)

  is_gpu = bool(tf.config.list_physical_devices('GPU'))
  use_predict = not is_gpu

  n_examples = 0
  n_batches = 0
  start_time = time.time()

  for image_encodes, images, variants, alt_allele_indices in dataset:
    predictions, activations = hooked.predict_and_cache(
        images,
        use_predict=use_predict,
        metadata={'batch': n_batches},
    )

    for i in range(len(predictions)):
      gls = _round_gls(predictions[i])
      _write_cvo(
          writer,
          variant_encoded=variants[i].numpy(),
          alt_allele_indices_encoded=alt_allele_indices[i].numpy(),
          gls=gls,
      )
      n_examples += 1

    n_batches += 1
    elapsed = time.time() - start_time
    if n_batches % 10 == 0 or n_batches == 1:
      rate = (100 * elapsed) / max(n_examples, 1)
      logging.info(
          'Predicted %d examples in %d batches [%.3f sec per 100]. '
          'Layers: %s, activation shape: %s',
          n_examples, n_batches, rate,
          list(activations.keys()),
          {k: v.shape for k, v in activations.items()},
      )

  writer.close()
  logging.info('Wrote %d CallVariantsOutput protos to %s',
               n_examples, _OUTFILE.value)

  hooked.finalize()
  logging.info('Activation cache finalized: %d entries in %s',
               hooked.cache.total_stored, _ACTIVATION_CACHE_DIR.value)


if __name__ == '__main__':
  flags.mark_flags_as_required(['examples', 'checkpoint', 'outfile'])
  logging.use_python_logging()
  app.run(main)
