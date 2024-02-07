# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType
from ..lora.config import LoftQConfig

@dataclass
class RosaConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`RosaModel`].

    Args:
        r (`int`):
            Lora attention dimension (the "rank").
        d (`float`):
            The density of SpA.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        impl (`str`):
            Which implementation of RoSA to use. Options are 'spmm', 'sp_add', and 'auto'.
            Normally, sp_add should be faster and more memory efficient, but it may
            be slower when the weights are quantized.
        spa_store_transpose (`bool`):
            Whether to store the transposed SpA matrix. Default is True. Ignored when impl='sp_add'.
        spa_num_grads (`int`):
            number of gradients to use for SpA mask generation
        grad_acc_mode (`str`):
            how to accumulate gradients for mask generation. Options are 'mean_squared' (diagonal fisher) and 'mean'.
        mask_load_path (`str`):
            A path to load SpA masks at the outset of training.
        mask_save_path (`str`):
            A path to store SpA masks once generated.
        rosa_dtype (`str`):
            The data type for the RoSA parameters. Can be 'fp32', 'fp16', or 'bf16'.
        terminate_after_mask_generation (`bool`):
            Whether to terminate right after the SpA masks are generated. This is particularly useful when we need
            to restart the training with the generated masks.
        schedule (`str`):
            The RoSA schedule. Current options are 'lora_only', 'spa_only', 'wl{warmup}'
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for RoSA. Can be 'none', 'all' or 'rosa_only'. If 'all' or 'rosa_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        use_rslora (`bool`):
            When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a> which
            sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it was proven to work better.
            Otherwise, it will use the original default value of `lora_alpha/r`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_lora_weights (`bool` | `Literal["gaussian", "loftq"]`):
            How to initialize the weights of the adapter layers. Passing True (default) results in the default
            initialization from the reference implementation from Microsoft. Passing 'gaussian' results in Gaussian
            initialization scaled by the LoRA rank for linear and layers. Setting the initialization to False leads to
            completely random initialization and is discouraged. Pass `'loftq'` to use LoftQ initialization.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        density_pattern (`dict`):
            The mapping from layer names or regexp expression to densities which are different from the default density
            specified by `d`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `lora_alpha`.
        megatron_config (`Optional[dict]`):
            The TransformerConfig arguments for Megatron. It is used to create LoRA's parallel linear layer. You can
            get it like this, `core_transformer_config_from_args(get_args())`, these two functions being from Megatron.
            The arguments will be used to initialize the TransformerConfig of Megatron. You need to specify this
            parameter when you want to apply LoRA to the ColumnParallelLinear and RowParallelLinear layers of megatron.
        megatron_core (`Optional[str]`):
            The core module from Megatron to use, defaults to `"megatron.core"`.
        loftq_config (`Optional[LoftQConfig]`):
            The configuration of LoftQ. If this is not None, then LoftQ will be used to quantize the backbone weights
            and initialize Lora layers. Also pass `init_lora_weights='loftq'`. Note that you should not pass a
            quantized model in this case, as LoftQ will quantize the model itself.
    """

    r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    d: int = field(default=0.003, metadata={"help": "SpA density"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with RoSA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you shoud specify the target modules manually."
            ),
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    impl: str = field(default='auto', metadata={"help": (
        "Which implementation of RoSA to use. Options are 'spmm' and 'sp_add'."
        "Normally, sp_add should be faster and more memory efficient, but it may"
        "be slower when the weights are quantized."
    )})
    spa_store_transpose: bool = field(default=True, metadata={"help": "Whether to store the transposed SpA matrix"})
    spa_num_grads: int = field(default=1, metadata={"help": "number of gradients to use for SpA mask generation"})
    grad_acc_mode: str = field(default='mean_squared', metadata={"help": "how to accumulate gradients for mask generation. Options are 'mean_squared' (diagonal fisher) and 'mean'."})
    mask_load_path: str = field(default=None, metadata={"help": "A path to load SpA masks at the outset of training."})
    mask_save_path: str = field(default=None, metadata={"help": "A path to store SpA masks once generated."})
    rosa_dtype: str = field(default='bf16', metadata={"help": "The data type for the RoSA parameters. Can be 'fp32', 'fp16', or 'bf16'."})
    terminate_after_mask_generation: bool = field(default=False, metadata={"help": (
        "Whether to terminate right after the SpA masks are generated. This is particularly useful when we need"
        "to restart the training with the generated masks."
    )})
    schedule: str = field(default=None, metadata={"help": "The RoSA schedule. Current options are 'lora_only', 'spa_only', 'wl{warmup}'"})

    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "rosa_only"] = field(
        default="none", metadata={"help": "Bias type for RoSA. Can be 'none', 'all' or 'rosa_only'"}
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses Rank-Stabilized LoRA doi.org/10.48550/arXiv.2312.03732"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from RoSA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool | Literal["gaussian", "loftq"] = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. Passing True (default) results in the default "
                "initialization from the reference implementation from Microsoft. Passing 'gaussian' results "
                "in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization "
                "to False leads to completely random initialization and is discouraged."
                "Pass `'loftq'` to use LoftQ initialization"
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    density_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to densities which are different from the default density specified by `d`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 0.006`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
    megatron_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The TransformerConfig from Megatron. It is used to create LoRA's parallel linear layer."
                "You can get it like this, `core_transformer_config_from_args(get_args())`, "
                "these two functions being from Megatron."
                "You need to specify this parameter when you want to apply LoRA to the ColumnParallelLinear and "
                "RowParallelLinear layers of megatron."
                "It should be noted that we may not be able to use the `save_pretrained` and `from_pretrained` "
                "functions, because TransformerConfig may not necessarily be serialized."
                "But when using megatron, we can use `get_peft_model_state_dict` function and "
                "megatron's framework, they can also save and load models and configurations."
            )
        },
    )
    megatron_core: Optional[str] = field(
        default="megatron.core",
        metadata={
            "help": (
                "The core module from Megatron, it is used to create LoRA's parallel linear layer. "
                "It only needs to be passed in when you need to use your own modified megatron core module. "
                "Otherwise, it will use the default value `megatron.core`. "
            )
        },
    )
    # dict type is used when loading config.json
    loftq_config: Union[LoftQConfig, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The configuration of LoftQ. If this is passed, then LoftQ will be used to quantize the backbone "
                "weights and initialize Lora layers. Also set `init_lora_weights='loftq'` in this case."
            )
        },
    )
    
    def __post_init__(self):
        self.peft_type = PeftType.ROSA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # handle init_lora_weights and loftq_config
        if self.init_lora_weights == "loftq":
            import importlib

            if not importlib.util.find_spec("scipy"):
                raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
            if self.loftq_config is None:
                raise ValueError("`loftq_config` must be specified when `init_lora_weights` is 'loftq'.")

        # convert loftq_config to dict
        if self.loftq_config and not isinstance(self.loftq_config, dict):
            self.loftq_config = vars(self.loftq_config)