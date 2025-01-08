
from __future__ import annotations

import functools
import importlib.util
import multiprocessing as mp
import warnings
from copy import copy
from enum import IntEnum
from functools import wraps
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

import torch

from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NonTensorData,
    set_lazy_legacy,
    TensorDict,
    TensorDictBase,
    unravel_key,
    unravel_key_list,
)
from tensordict.nn import dispatch, TensorDictModuleBase
from tensordict.utils import (
    _unravel_key_to_tuple,
    _zip_strict,
    expand_as_right,
    expand_right,
    NestedKey,
)
from torch import nn, Tensor
from torch.utils._pytree import tree_map

from torchrl._utils import (
    _append_last,
    _ends_with,
    _make_ordinal_device,
    _replace_last,
    implement_for,
)

from torchrl.data.tensor_specs import (
    Binary,
    Bounded,
    Categorical,
    Composite,
    ContinuousBox,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    TensorSpec,
    Unbounded,
)
from torchrl.envs.common import _do_nothing, _EnvPostInit, EnvBase, make_tensordict
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
    check_finite,
)
from torchrl.envs.utils import _sort_keys, _update_during_reset, step_mdp
from torchrl.objectives.value.functional import reward2go

_has_tv = importlib.util.find_spec("torchvision", None) is not None

IMAGE_KEYS = ["pixels"]
_MAX_NOOPS_TRIALS = 10

FORWARD_NOT_IMPLEMENTED = "class {} cannot be executed without a parent environment."

T = TypeVar("T", bound="Transform")


from torchrl.envs.transforms import Transform
def _apply_to_composite(function):
    @wraps(function)
    def new_fun(self, observation_spec):
        if isinstance(observation_spec, Composite):
            _specs = observation_spec._specs
            in_keys = self.in_keys
            out_keys = self.out_keys
            for in_key, out_key in _zip_strict(in_keys, out_keys):
                if in_key in observation_spec.keys(True, True):
                    _specs[out_key] = function(self, observation_spec[in_key].clone())
            return Composite(
                _specs, shape=observation_spec.shape, device=observation_spec.device
            )
        else:
            return function(self, observation_spec)

    return new_fun

class KalmanReward(Transform):
    """Affine transform of the reward.

     The reward is transformed according to:

    .. math::
        reward = reward * scale + loc

    Args:
        loc (number or torch.Tensor): location of the affine transform
        scale (number or torch.Tensor): scale of the affine transform
        standard_normal (bool, optional): if ``True``, the transform will be

            .. math::
                reward = (reward-loc)/scale

            as it is done for standardization. Default is `False`.
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["reward", "prediction_error"]
        if out_keys is None:
            out_keys = copy(in_keys)

        super().__init__(in_keys=in_keys, out_keys=out_keys)
      
    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        #if "reward" in tensordict:
        #    tensordict["reward"] -= 0.01 * torch.square(tensordict["prediction_error"]).sum(-1, keepdim=True)
        return tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        #reward = tensordict["reward"]
        #prediction_error = tensordict["prediction_error"]
        #tensordict["reward"] -= 0.01 * torch.square(prediction_error).sum(-1, keepdim=True)
        return tensordict

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        return output_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return input_spec
