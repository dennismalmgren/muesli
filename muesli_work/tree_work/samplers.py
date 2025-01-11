from __future__ import annotations

import json
import textwrap
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy, deepcopy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from tensordict import MemoryMappedTensor, TensorDict
from tensordict.utils import NestedKey

from torchrl._extension import EXTENSION_WARNING

from torchrl._utils import _replace_last, logger
from torchrl.data.replay_buffers.storages import Storage, StorageEnsemble, TensorStorage
from torchrl.data.replay_buffers.utils import _auto_device, _is_int, unravel_index

try:
    from torchrl._torchrl import (
        MinSegmentTreeFp32,
        MinSegmentTreeFp64,
        SumSegmentTreeFp32,
        SumSegmentTreeFp64,
    )
except ImportError:
    warnings.warn(EXTENSION_WARNING)

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."

from torchrl.data.replay_buffers import SliceSampler, SamplerWithoutReplacement

class SliceSamplerWithoutReplacement(SliceSampler, SamplerWithoutReplacement):
    """Samples slices of data along the first dimension, given start and stop signals, without replacement.

    In this context, ``without replacement`` means that the same element (NOT trajectory) will not be sampled twice
    before the counter is automatically reset. Within a single sample, however, only one slice of a given trajectory
    will appear (see example below).

    This class is to be used with static replay buffers or in between two
    replay buffer extensions. Extending the replay buffer will reset the
    the sampler, and continuous sampling without replacement is currently not
    allowed.

    .. note:: `SliceSamplerWithoutReplacement` can be slow to retrieve the trajectory indices. To accelerate
        its execution, prefer using `end_key` over `traj_key`, and consider the following
        keyword arguments: :attr:`compile`, :attr:`cache_values` and :attr:`use_gpu`.

    Keyword Args:
        drop_last (bool, optional): if ``True``, the last incomplete sample (if any) will be dropped.
            If ``False``, this last sample will be kept.
            Defaults to ``False``.
        num_slices (int): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``.
        slice_len (int): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``.
        end_key (NestedKey, optional): the key indicating the end of a
            trajectory (or episode). Defaults to ``("next", "done")``.
        traj_key (NestedKey, optional): the key indicating the trajectories.
            Defaults to ``"episode"`` (commonly used across datasets in TorchRL).
        ends (torch.Tensor, optional): a 1d boolean tensor containing the end of run signals.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        trajectories (torch.Tensor, optional): a 1d integer tensor containing the run ids.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        truncated_key (NestedKey, optional): If not ``None``, this argument
            indicates where a truncated signal should be written in the output
            data. This is used to indicate to value estimators where the provided
            trajectory breaks. Defaults to ``("next", "truncated")``.
            This feature only works with :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer`
            instances (otherwise the truncated key is returned in the info dictionary
            returned by the :meth:`~torchrl.data.replay_buffers.ReplayBuffer.sample` method).
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch. If ``True``, trajectories shorted
            than required will be filtered out.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`~torchrl.collectors.split_trajectories`. Defaults to ``True``.
        shuffle (bool, optional): if ``False``, the order of the trajectories
            is not shuffled. Defaults to ``True``.
        compile (bool or dict of kwargs, optional): if ``True``, the bottleneck of
            the :meth:`~sample` method will be compiled with :func:`~torch.compile`.
            Keyword arguments can also be passed to torch.compile with this arg.
            Defaults to ``False``.
        use_gpu (bool or torch.device): if ``True`` (or is a device is passed), an accelerator
            will be used to retrieve the indices of the trajectory starts. This can significanlty
            accelerate the sampling when the buffer content is large.
            Defaults to ``False``.

    .. note:: To recover the trajectory splits in the storage,
        :class:`~torchrl.data.replay_buffers.samplers.SliceSamplerWithoutReplacement` will first
        attempt to find the ``traj_key`` entry in the storage. If it cannot be
        found, the ``end_key`` will be used to reconstruct the episodes.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
        >>> from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
        >>>
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(1000),
        ...     # asking for 10 slices for a total of 320 elements, ie, 10 trajectories of 32 transitions each
        ...     sampler=SliceSamplerWithoutReplacement(num_slices=10),
        ...     batch_size=320,
        ... )
        >>> episode = torch.zeros(1000, dtype=torch.int)
        >>> episode[:300] = 1
        >>> episode[300:550] = 2
        >>> episode[550:700] = 3
        >>> episode[700:] = 4
        >>> data = TensorDict(
        ...     {
        ...         "episode": episode,
        ...         "obs": torch.randn((3, 4, 5)).expand(1000, 3, 4, 5),
        ...         "act": torch.randn((20,)).expand(1000, 20),
        ...         "other": torch.randn((20, 50)).expand(1000, 20, 50),
        ...     }, [1000]
        ... )
        >>> rb.extend(data)
        >>> sample = rb.sample()
        >>> # since we want trajectories of 32 transitions but there are only 4 episodes to
        >>> # sample from, we only get 4 x 32 = 128 transitions in this batch
        >>> print("sample:", sample)
        >>> print("trajectories in sample", sample.get("episode").unique())

    :class:`~torchrl.data.replay_buffers.SliceSamplerWithoutReplacement` is default-compatible with
    most of TorchRL's datasets, and allows users to consume datasets in a dataloader-like fashion:

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data.datasets import RobosetExperienceReplay
        >>> from torchrl.data import SliceSamplerWithoutReplacement
        >>>
        >>> torch.manual_seed(0)
        >>> num_slices = 10
        >>> dataid = list(RobosetExperienceReplay.available_datasets)[0]
        >>> data = RobosetExperienceReplay(dataid, batch_size=320,
        ...     sampler=SliceSamplerWithoutReplacement(num_slices=num_slices))
        >>> # the last sample is kept, since drop_last=False by default
        >>> for i, batch in enumerate(data):
        ...     print(batch.get("episode").unique())
        tensor([ 5,  6,  8, 11, 12, 14, 16, 17, 19, 24])
        tensor([ 1,  2,  7,  9, 10, 13, 15, 18, 21, 22])
        tensor([ 0,  3,  4, 20, 23])

    When requesting a large total number of samples with few trajectories and small span, the batch will contain
    only at most one sample of each trajectory:

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.collectors.utils import split_trajectories
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler, SliceSamplerWithoutReplacement
        >>>
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(max_size=1000),
        ...                   sampler=SliceSamplerWithoutReplacement(
        ...                       slice_len=5, traj_key="episode",strict_length=False
        ...                   ))
        ...
        >>> ep_1 = TensorDict(
        ...     {"obs": torch.arange(100),
        ...     "episode": torch.zeros(100),},
        ...     batch_size=[100]
        ... )
        >>> ep_2 = TensorDict(
        ...     {"obs": torch.arange(51),
        ...     "episode": torch.ones(51),},
        ...     batch_size=[51]
        ... )
        >>> rb.extend(ep_1)
        >>> rb.extend(ep_2)
        >>>
        >>> s = rb.sample(50)
        >>> t = split_trajectories(s, trajectory_key="episode")
        >>> print(t["obs"])
        tensor([[14, 15, 16, 17, 18],
                [ 3,  4,  5,  6,  7]])
        >>> print(t["episode"])
        tensor([[0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1.]])
        >>>
        >>> s = rb.sample(50)
        >>> t = split_trajectories(s, trajectory_key="episode")
        >>> print(t["obs"])
        tensor([[ 4,  5,  6,  7,  8],
                [26, 27, 28, 29, 30]])
        >>> print(t["episode"])
        tensor([[0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1.]])

    """

    def __init__(
        self,
        *,
        num_slices: int | None = None,
        slice_len: int | None = None,
        drop_last: bool = False,
        end_key: NestedKey | None = None,
        traj_key: NestedKey | None = None,
        ends: torch.Tensor | None = None,
        trajectories: torch.Tensor | None = None,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
        shuffle: bool = True,
        compile: bool | dict = False,
        use_gpu: bool | torch.device = False,
    ):
        SliceSampler.__init__(
            self,
            num_slices=num_slices,
            slice_len=slice_len,
            end_key=end_key,
            traj_key=traj_key,
            cache_values=True,
            truncated_key=truncated_key,
            strict_length=strict_length,
            ends=ends,
            trajectories=trajectories,
            compile=compile,
            use_gpu=use_gpu,
        )
        SamplerWithoutReplacement.__init__(self, drop_last=drop_last, shuffle=shuffle)

    def __repr__(self):
        if self._sample_list is not None:
            perc = len(self._sample_list) / self.len_storage * 100
        else:
            perc = 0
        return (
            f"{self.__class__.__name__}("
            f"num_slices={self.num_slices}, "
            f"slice_len={self.slice_len}, "
            f"end_key={self.end_key}, "
            f"traj_key={self.traj_key}, "
            f"truncated_key={self.truncated_key}, "
            f"strict_length={self.strict_length},"
            f"{perc}% sampled)"
        )

    def _empty(self):
        self._cache = {}
        SamplerWithoutReplacement._empty(self)

    def _storage_len(self, storage):
        return self._storage_len_buffer

    def sample(
        self, storage: Storage, batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        if self._batch_size_multiplier is not None:
            batch_size = batch_size * self._batch_size_multiplier
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        # we have to make sure that the number of dims of the storage
        # is the same as the stop/start signals since we will
        # use these to sample the storage
        if start_idx.shape[1] != storage.ndim:
            raise RuntimeError(
                f"Expected the end-of-trajectory signal to be "
                f"{storage.ndim}-dimensional. Got a {start_idx.shape[1]} tensor "
                "instead."
            )
        self._storage_len_buffer = len(start_idx)
        # first get indices of the trajectories we want to retrieve
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        indices, _ = SamplerWithoutReplacement.sample(self, storage, num_slices)
        storage_length = storage.shape[0]

        # traj_idx will either be a single tensor or a tuple that can be reorganized
        # like a non-zero through stacking.
        def tuple_to_tensor(traj_idx, lengths=lengths):
            if isinstance(traj_idx, tuple):
                traj_idx = torch.arange(len(storage), device=lengths.device).view(
                    storage.shape
                )[traj_idx]
            return traj_idx

        idx, info = self._sample_slices(
            lengths,
            start_idx,
            stop_idx,
            seq_length,
            num_slices,
            storage_length,
            traj_idx=tuple_to_tensor(indices),
            storage=storage,
        )
        return idx, info

    def state_dict(self) -> Dict[str, Any]:
        return SamplerWithoutReplacement.state_dict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return SamplerWithoutReplacement.load_state_dict(self, state_dict)

