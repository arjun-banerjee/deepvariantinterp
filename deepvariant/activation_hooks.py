import collections
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from absl import logging
import numpy as np
import tensorflow as tf

from deepvariant import keras_modeling

LAYER_5_NAME = 'mixed5'

INCEPTION_MIXED_LAYERS = (
    'mixed0',
    'mixed1',
    'mixed2',
    'mixed3',
    'mixed4',
    'mixed5',
    'mixed6',
    'mixed7',
    'mixed8',
    'mixed9',
    'mixed10',
)


class ActivationHook:
  """Captures activations from specified layers during forward passes.

  Provides a PyTorch-style hook interface for TF/Keras models. Under the hood
  it builds a secondary Keras Model whose outputs are the requested
  intermediate layers, using `keras_modeling.get_activations_model`.

  Attributes:
    model: The base DeepVariant Keras model.
    layer_names: Names of layers to capture activations from.
    enabled: Whether the hook is currently active.
  """

  def __init__(
      self,
      model: tf.keras.Model,
      layer_names: Optional[Sequence[str]] = None,
      callback: Optional[Callable[[Dict[str, np.ndarray]], None]] = None,
  ):
    """Initializes the hook.

    Args:
      model: A DeepVariant InceptionV3 Keras model.
      layer_names: Layer names to hook. Defaults to ['mixed5'] (Layer 5).
      callback: Optional function called with the activations dict after each
        extraction.
    """
    if layer_names is None:
      layer_names = [LAYER_5_NAME]
    self.model = model
    self.layer_names = list(layer_names)
    self.callback = callback
    self._activation_model = keras_modeling.get_activations_model(
        model, self.layer_names
    )
    self._enabled = True

  @property
  def enabled(self) -> bool:
    return self._enabled

  def enable(self):
    self._enabled = True

  def disable(self):
    self._enabled = False

  def extract(
      self,
      inputs: tf.Tensor,
      use_predict: bool = False,
  ) -> Dict[str, np.ndarray]:
    """Run a forward pass through hooked layers and return activations.

    Args:
      inputs: Batch of input images as a tf.Tensor.
      use_predict: If True, use `predict_on_batch` (faster on CPU). Otherwise
        use direct model call (faster on GPU).

    Returns:
      Dict mapping layer name -> numpy activation array. Empty dict if hook is
      disabled.
    """
    if not self._enabled:
      return {}

    if use_predict:
      raw = self._activation_model.predict_on_batch(inputs)
    else:
      raw = self._activation_model(inputs, training=False)

    if isinstance(raw, dict):
      activations = {
          k: v.numpy() if hasattr(v, 'numpy') else v
          for k, v in raw.items()
      }
    else:
      val = raw.numpy() if hasattr(raw, 'numpy') else raw
      activations = {self.layer_names[0]: val}

    if self.callback is not None:
      self.callback(activations)

    return activations

  def __call__(
      self, inputs: tf.Tensor, **kwargs
  ) -> Dict[str, np.ndarray]:
    return self.extract(inputs, **kwargs)


class ActivationCache:
  """Thread-safe cache for layer activations with memory and disk persistence.

  Activations are held in an ordered in-memory buffer. When the buffer exceeds
  `max_memory_entries`, the oldest entries are evicted and (optionally) flushed
  to disk as compressed `.npz` files. Disk entries can be read back on demand.

  Attributes:
    size: Number of entries currently in memory.
    total_stored: Total number of entries stored since creation.
  """

  def __init__(
      self,
      cache_dir: Optional[str] = None,
      max_memory_entries: int = 1000,
      flush_interval: int = 100,
  ):
    """Initializes the cache.

    Args:
      cache_dir: Directory for on-disk persistence. If None, evicted entries
        are dropped.
      max_memory_entries: Maximum entries to hold in memory before eviction.
      flush_interval: Write in-memory entries to disk every N stores.
    """
    self._cache_dir = cache_dir
    self._max_memory_entries = max_memory_entries
    self._flush_interval = flush_interval
    self._memory: collections.OrderedDict = collections.OrderedDict()
    self._lock = threading.Lock()
    self._counter = 0
    self._flushed_up_to = 0

    if cache_dir:
      tf.io.gfile.makedirs(cache_dir)

  @property
  def size(self) -> int:
    with self._lock:
      return len(self._memory)

  @property
  def total_stored(self) -> int:
    return self._counter

  def store(
      self,
      activations: Dict[str, np.ndarray],
      metadata: Optional[Dict[str, Any]] = None,
  ) -> int:
    """Store activations in the cache.

    Args:
      activations: Dict mapping layer name to numpy array.
      metadata: Optional metadata dict to associate with this entry.

    Returns:
      The integer index assigned to this entry.
    """
    with self._lock:
      idx = self._counter
      self._counter += 1

      self._memory[idx] = {
          'activations': activations,
          'metadata': metadata or {},
          'timestamp': time.time(),
      }

      while len(self._memory) > self._max_memory_entries:
        evicted_idx, evicted = self._memory.popitem(last=False)
        if self._cache_dir:
          self._write_entry(evicted_idx, evicted)

      if (
          self._cache_dir
          and self._flush_interval > 0
          and self._counter % self._flush_interval == 0
      ):
        self._flush_locked()

      return idx

  def get(self, index: int) -> Optional[Dict[str, np.ndarray]]:
    """Retrieve activations by index. Checks memory first, then disk.

    Args:
      index: The entry index returned by `store`.

    Returns:
      Dict of layer activations, or None if not found.
    """
    with self._lock:
      if index in self._memory:
        return self._memory[index]['activations']

    if self._cache_dir:
      return self._read_entry(index)
    return None

  def get_batch(
      self, start: int, end: int
  ) -> Dict[int, Dict[str, np.ndarray]]:
    """Retrieve a contiguous range of entries [start, end).

    Args:
      start: First index (inclusive).
      end: Last index (exclusive).

    Returns:
      Dict mapping index -> activations dict.
    """
    results = {}
    for i in range(start, end):
      entry = self.get(i)
      if entry is not None:
        results[i] = entry
    return results

  def flush_to_disk(self):
    """Write all in-memory entries to disk."""
    with self._lock:
      self._flush_locked()

  def _flush_locked(self):
    if not self._cache_dir:
      return
    for idx, entry in self._memory.items():
      if idx >= self._flushed_up_to:
        self._write_entry(idx, entry)
    self._flushed_up_to = self._counter

  def _entry_path(self, index: int) -> str:
    return os.path.join(self._cache_dir, f'activations_{index:08d}.npz')

  def _write_entry(self, index: int, entry: Dict):
    path = self._entry_path(index)
    np.savez_compressed(path, **entry['activations'])
    logging.vlog(3, 'Wrote activation cache entry %d to %s', index, path)

  def _read_entry(self, index: int) -> Optional[Dict[str, np.ndarray]]:
    path = self._entry_path(index)
    if not tf.io.gfile.exists(path):
      return None
    with open(path, 'rb') as f:
      data = np.load(f)
      return dict(data)

  def clear_memory(self):
    """Drop all in-memory entries (disk entries are untouched)."""
    with self._lock:
      self._memory.clear()

  def clear_all(self):
    """Drop in-memory entries and delete on-disk cache files."""
    with self._lock:
      self._memory.clear()
      self._counter = 0
      self._flushed_up_to = 0
    if self._cache_dir and tf.io.gfile.exists(self._cache_dir):
      for fname in tf.io.gfile.listdir(self._cache_dir):
        if fname.startswith('activations_') and fname.endswith('.npz'):
          tf.io.gfile.remove(os.path.join(self._cache_dir, fname))
        if fname == 'manifest.json':
          tf.io.gfile.remove(os.path.join(self._cache_dir, fname))

  def save_manifest(self):
    """Write a JSON manifest describing the cache state to `cache_dir`."""
    if not self._cache_dir:
      return
    manifest_path = os.path.join(self._cache_dir, 'manifest.json')
    manifest = {
        'total_entries': self._counter,
        'flushed_up_to': self._flushed_up_to,
        'in_memory_entries': len(self._memory),
        'layer_names': [],
        'timestamp': time.time(),
    }
    if self._memory:
      first_entry = next(iter(self._memory.values()))
      manifest['layer_names'] = list(first_entry['activations'].keys())
    with tf.io.gfile.GFile(manifest_path, 'w') as f:
      json.dump(manifest, f, indent=2)
    logging.info('Saved activation cache manifest to %s', manifest_path)


class HookedModel:
  """Wraps a DeepVariant model to capture Layer 5 activations during inference.

  In single-pass mode (default), a combined Keras Model is built that returns
  both the classification prediction and the intermediate layer activations in
  one forward pass -- avoiding the cost of running inference twice.

  All captured activations are automatically stored in an ActivationCache.

  Attributes:
    model: The underlying DeepVariant Keras model.
    layer_names: Names of hooked layers.
    cache: The ActivationCache holding captured values.
  """

  def __init__(
      self,
      model: tf.keras.Model,
      layer_names: Optional[Sequence[str]] = None,
      cache_dir: Optional[str] = None,
      max_cache_entries: int = 10000,
      single_pass: bool = True,
  ):
    """Initializes the hooked model.

    Args:
      model: A DeepVariant InceptionV3 Keras model.
      layer_names: Layers to capture. Defaults to ['mixed5'] (Layer 5).
      cache_dir: Directory for disk persistence. None = memory only.
      max_cache_entries: Max entries to hold in the memory cache.
      single_pass: If True, build a combined model that produces predictions
        and activations in one forward pass. If False, use a separate
        ActivationHook (two forward passes).
    """
    if layer_names is None:
      layer_names = [LAYER_5_NAME]

    self.model = model
    self.layer_names = list(layer_names)
    self.cache = ActivationCache(
        cache_dir=cache_dir,
        max_memory_entries=max_cache_entries,
    )
    self._single_pass = single_pass

    if single_pass:
      self._combined_model = self._build_combined_model()
    else:
      self._hook = ActivationHook(model, layer_names)

  def _build_combined_model(self) -> tf.keras.Model:
    """Build a model returning both predictions and layer activations."""
    layer_outputs = {
        name: self.model.get_layer(name).output for name in self.layer_names
    }
    all_outputs = {'prediction': self.model.output}
    all_outputs.update(layer_outputs)
    return tf.keras.Model(inputs=self.model.input, outputs=all_outputs)

  def predict_and_cache(
      self,
      inputs: tf.Tensor,
      metadata: Optional[Dict[str, Any]] = None,
      use_predict: bool = False,
  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run inference, capture and cache layer activations.

    Args:
      inputs: Batch of pileup image tensors.
      metadata: Optional metadata dict to store alongside the activations.
      use_predict: If True, use `predict_on_batch` (faster on CPU).

    Returns:
      Tuple of (predictions, activations) where predictions is a numpy array
      of shape [batch, 3] and activations is a dict mapping layer name to
      numpy array.
    """
    if self._single_pass:
      return self._predict_single_pass(inputs, metadata, use_predict)
    return self._predict_dual_pass(inputs, metadata, use_predict)

  def _predict_single_pass(
      self,
      inputs: tf.Tensor,
      metadata: Optional[Dict[str, Any]],
      use_predict: bool,
  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if use_predict:
      outputs = self._combined_model.predict_on_batch(inputs)
    else:
      outputs = self._combined_model(inputs, training=False)

    outputs = {
        k: v.numpy() if hasattr(v, 'numpy') else v
        for k, v in outputs.items()
    }
    predictions = outputs.pop('prediction')
    activations = outputs

    self.cache.store(activations, metadata=metadata)
    return predictions, activations

  def _predict_dual_pass(
      self,
      inputs: tf.Tensor,
      metadata: Optional[Dict[str, Any]],
      use_predict: bool,
  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if use_predict:
      predictions = self.model.predict_on_batch(inputs)
    else:
      predictions = self.model(inputs, training=False)
    if hasattr(predictions, 'numpy'):
      predictions = predictions.numpy()

    activations = self._hook(inputs, use_predict=use_predict)
    self.cache.store(activations, metadata=metadata)
    return predictions, activations

  def finalize(self):
    """Flush remaining cached activations to disk and write a manifest."""
    self.cache.flush_to_disk()
    self.cache.save_manifest()
    logging.info(
        'Finalized activation cache: %d total entries.', self.cache.total_stored
    )


def create_layer5_hook(
    model: tf.keras.Model,
    cache_dir: Optional[str] = None,
    single_pass: bool = True,
) -> HookedModel:
  """Convenience: hook Layer 5 (mixed5) of an InceptionV3 model with caching.

  Args:
    model: A DeepVariant InceptionV3 Keras model.
    cache_dir: Directory for on-disk persistence. None = memory only.
    single_pass: Use a single forward pass for both predictions and
      activations.

  Returns:
    A HookedModel targeting the mixed5 layer.
  """
  return HookedModel(
      model=model,
      layer_names=[LAYER_5_NAME],
      cache_dir=cache_dir,
      single_pass=single_pass,
  )


def list_hookable_layers(model: tf.keras.Model) -> List[str]:
  """Return names of all layers in the model that can be hooked."""
  return [layer.name for layer in model.layers]
