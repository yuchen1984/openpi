"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    # When set explicitly (True/False) on a base_config passed to a DataConfigFactory, the explicit
    # value wins; None means "auto" (quantile norm for every non-PI0 model type, the upstream default).
    use_quantile_norm: bool | None = None

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            # Respect an explicit base_config.use_quantile_norm override (e.g. the *_meanstd
            # fine-tune variants); otherwise keep the upstream default of quantile norm for
            # every non-PI0 model type.
            use_quantile_norm=(
                self.base_config.use_quantile_norm
                if self.base_config is not None and self.base_config.use_quantile_norm is not None
                else model_config.model_type != ModelType.PI0
            ),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    # Number of action dimensions in the dataset.  Standard LIBERO = 7,
    # sim fine-tune with done signal = 8.
    action_dim: int = 7

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_inputs = [
            _transforms.RepackTransform(
                {
                    "observation/image": "image",
                    "observation/wrist_image": "wrist_image",
                    "observation/state": "state",
                    "actions": "actions",
                    "prompt": "prompt",
                }
            )
        ]
        # Truncate actions to action_dim so that e.g. a 7-dim config training
        # on an 8-dim dataset (with done signal) only sees the first 7 dims.
        if self.action_dim < 8:
            repack_inputs.append(_transforms.TruncateActions(dim=self.action_dim))
        repack_transform = _transforms.Group(inputs=repack_inputs)

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs(action_dim=self.action_dim)],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDJointPosDataConfig(DataConfigFactory):
    """
    Data config for custom DROID dataset in LeRobot format with absolute joint position actions.

    Unlike LeRobotDROIDDataConfig (which assumes joint velocity actions), this config applies
    DeltaActions/AbsoluteActions transforms to convert absolute joint positions to deltas for
    training and back to absolute positions at inference — matching RLDSDroidDataConfig's
    JOINT_POSITION action space handling.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # Absolute joint position actions: convert to delta for training, back to absolute at inference.
        # mask: first 7 dims (joints) are delta, last dim (gripper) stays absolute.
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # Maximum number of recent checkpoints to keep (older ones are pruned unless matching keep_period).
    max_to_keep: int = 1

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # Fine-tuning pi05-DROID joint position model on Isaac Sim demonstrations (LoRA).
    # Uses absolute joint position actions with delta/absolute transforms.
    #
    TrainConfig(
        name="pi05_droid_jointpos_sim_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="local/sim_droid_jointpos_50ep",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Reuse original DROID jointpos norm stats from the polaris checkpoint.
                assets_dir="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=5_000,
        batch_size=2,
        save_interval=500,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=5_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # pi0.5 LIBERO sim fine-tune config (7-dim actions, no done signal).
    # For fine-tuning on simulated UF850 cloth pick-and-place data in LIBERO format.
    # Data: convert_sim_to_lerobot.py --format libero --repo-id local/sim_libero_50ep
    #
    TrainConfig(
        name="pi05_libero_sim_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/sim_cloth_102ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_libero/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=5_000,
        batch_size=2,
        save_interval=500,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=5_000,
            decay_lr=2e-6,
        ),
    ),
    # pi0.5 LIBERO sim fine-tune with done signal (8-dim actions).
    # Same as above but action_dim=8 to include the done termination signal.
    # Data: convert_sim_to_lerobot.py --format libero (produces 8-dim actions with done)
    #
    TrainConfig(
        name="pi05_libero_sim_finetune_done",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/sim_cloth_102ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_libero/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=5_000,
        batch_size=2,
        save_interval=500,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=5_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # NERO single-arm pocket pick fine-tuning (mirrors the UF850 cloth
    # config but points at the NERO v4 LeRobot dataset and uses its own
    # asset_id so norm stats stay correct).
    #
    TrainConfig(
        name="pi05_libero_sim_finetune_nero_pick",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 4x weight on dim 6 (gripper_cmd) to force the model to
            # learn confident close/open decisions instead of
            # defaulting to the +1 (open) majority class. The other
            # 31 dims keep weight 1. See
            # uf850-experiment/docs/plan_gripper_loss_weighting.md.
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6) — was 4.0
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/sim_nero_pick_100ep_compact",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_libero/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=8_000,
        batch_size=4,
        save_interval=1_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=8_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # NERO single-arm pocket pick — pi05_BASE variant.
    # Same hyperparameters as pi05_libero_sim_finetune_nero_pick but
    # loads the unspecialised pi05_base checkpoint. Hypothesis: LIBERO
    # post-training overwrites the cloth/laundry priors in pi05_base
    # (openpi#692); starting from pi05_base should retain them and
    # improve placement precision on the NERO suction-cloth task.
    # See docs/vla_base_model_research.md for the rationale.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_pick",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/sim_nero_pick_100ep_compact",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=8_000,
        batch_size=4,
        save_interval=1_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=8_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # NERO single-arm pick — fine-tune from pi0.5-base on the mixed
    # LHS/RHS negY_flipcheck dataset (201 compacted episodes: ~103 negY
    # + ~97 posY). Same model/loss as pi05_base_sim_finetune_nero_pick
    # (pi05, gripper loss-weight 2.0, LoRA); only repo_id + the 24k/2k
    # schedule differ.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_negY_flipcheck",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/negY_flipcheck_201ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=24_000,
        batch_size=4,
        save_interval=2_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=24_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # MEAN-STD A/B variant of pi05_base_sim_finetune_negY_flipcheck (2026-06-12
    # endgame-precision Tier-0 gate). Identical model/data/schedule; the ONLY
    # delta is use_quantile_norm=False → z-score (mean/std) normalization.
    # Rationale: the flipcheck action stats are asymmetric in delta-x
    # (q01=-0.0164 / q99=+0.0077) so quantile norm maps one transport direction
    # out of the flow expert's [-1,1] band, and the quantile range (~0.022) is
    # ~6x the std (~0.0035) so ~±2mm endgame corrections normalize ~3x weaker
    # than under mean-std. openpi issues #763/#799/#817 report fine-tune quality
    # restored by disabling quantile norm on small datasets. Norm stats file is
    # a copy of the sibling's (mean/std fields come from the same sweep; the
    # quantile fields simply go unused). Train FROM SCRATCH — never --resume.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_negY_flipcheck_meanstd",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/negY_flipcheck_201ep",
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=False),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=24_000,
        batch_size=4,
        save_interval=2_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=24_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # FULL fine-tuning of the negY_flipcheck CLOTH pick (non-LoRA). Primarily a
    # SERVING config for the A100 full-FT cloth checkpoints (nero_cloth_full_24k):
    # non-LoRA gemma_2b + gemma_300m, no freeze_filter. Architecture must match the
    # full-FT checkpoint; params load from --policy.dir, norm-stats from the ckpt
    # assets / local/negY_flipcheck_201ep.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_negY_flipcheck_full",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            action_dim_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, *([1.0] * 24)),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/negY_flipcheck_201ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False, action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/pi05_base/params"),
        ema_decay=0.99, num_train_steps=24_000, batch_size=8,
        save_interval=1_000, keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=1e-5, decay_steps=24_000, decay_lr=1e-6),
    ),
    #
    # REAL-ROBOT UF850 cube pick — LoRA fine-tune from pi0.5-base on
    # real_cube_v4 (106 episodes, 31,773 frames @10Hz; one global + one
    # wrist camera, both 256x256; step-wise delta actions). Repacked from
    # the nero-exp v3.0 LeRobot dataset to v2.1 via
    # scripts/convert_real_v3_to_openpi.py → local/real_cube_v4_106ep.
    # Schema is the standard LIBERO one (image, wrist_image, state(8),
    # actions(8)); identical model/loss/LoRA to the sim cube config.
    # NB: the recorded GRIPPER channel is CONSTANT (state grip=[0,0],
    # action grip=-1 throughout) — the model learns position deltas + the
    # done flag; gripper loss-weight 2.0 is harmless on a constant target.
    # 4090 24 GB → batch_size=2 (verify it/s + no OOM in first ~100 steps;
    # drop to 1 if OOM). 30k smooth cosine FROM SCRATCH. save 2k / keep 4k.
    #
    TrainConfig(
        name="pi05_base_finetune_real_cube_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/real_cube_v4_106ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=2,
        save_interval=2_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # SAME real_cube_v4 LoRA fine-tune, but starting from the LIBERO-tuned
    # pi0.5 checkpoint (./checkpoints/pi05_libero/params) instead of pi0.5-base
    # — a base-vs-libero start-checkpoint A/B (cf. docs nero_cube_base_vs_libero).
    # Identical dataset / model / LoRA / 30k schedule; ONLY the weight_loader
    # differs.
    #
    TrainConfig(
        name="pi05_libero_finetune_real_cube_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/real_cube_v4_106ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_libero/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=2,
        save_interval=2_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # real_cube_v4 with a RECONSTRUCTED gripper channel. The original
    # real_cube_v4 recording captured NO gripper telemetry (state[6:8]=[0,0],
    # action[6]=-1 for every frame — a dead channel, so the policy can't learn
    # grasp/release). scripts/convert_real_v3_to_openpi.py --reconstruct-gripper
    # synthesizes a binary open/close signal from the EE-z trajectory (open ->
    # close at the pick descent -> closed through transport -> open at the place
    # descent) and writes local/real_cube_v4_106ep_grip. Everything else is
    # identical to pi05_base/libero_finetune_real_cube_v4; the gripper dim (6)
    # already carries loss-weight 2.0, now over a VARYING target. base-vs-libero
    # start A/B preserved. 30k LoRA, batch 2, save 2k / keep 4k.
    #
    TrainConfig(
        name="pi05_base_finetune_real_cube_v4_grip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/real_cube_v4_106ep_grip",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=2,
        save_interval=2_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # SAME reconstructed-gripper real_cube_v4 fine-tune, started from the
    # LIBERO-tuned pi0.5 checkpoint instead of pi0.5-base (start-checkpoint A/B).
    #
    TrainConfig(
        name="pi05_libero_finetune_real_cube_v4_grip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/real_cube_v4_106ep_grip",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_libero/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=2,
        save_interval=2_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # NERO single-arm AgileX rigid-cube pick — fine-tune from pi0.5-base
    # on the mixed cube dataset (100 episodes: red/green cube, start side
    # negY/posY, target outline left/right all randomized; rot180 folded
    # layout, hybrid FixedJoint+jaw grasp). Success is centroid distance,
    # not IoU. Same model/loss as the cloth pick configs (pi05, gripper
    # loss-weight 2.0, LoRA); 12k steps, ckpt every 1k, permanent every 4k.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_mixed_200ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # Extended 24k → 40k (resume from the 24k ckpt with --resume). Cosine
        # decay_steps also → 40k, so the LR re-extends (mild warm-restart bump
        # ~8e-6 at the 24k resume point, re-decaying to 2e-6 by 40k).
        num_train_steps=40_000,
        batch_size=4,
        save_interval=1_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=40_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # 2 cm cube + LEFT-side low grazing global camera (real_cube_v4-like view),
    # 2026-06-21. A fresh 200-ep mixed dataset (100 negY + 100 posY, red/green
    # random, rot180 folded) recorded with --cube-size-m 0.02 and the left
    # camera; converted with --overhead-rot-deg 0 (the left view is already
    # upright). Same pi0.5 LoRA / model / loss (gripper dim-6 weight 2.0) as
    # pi05_base_sim_finetune_nero_cube; 24k SMOOTH cosine FROM SCRATCH (no
    # warm-restart). Inference parity: serve with --cube-size-m 0.02 and
    # --overhead-rot-deg 0.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_2cm_leftcam",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_2cm_leftcam_200ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=24_000,
        batch_size=4,
        save_interval=1_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=24_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # 2 cm left-cam + symmetric posY + GRASP-PHASE OVERSAMPLE (2026-06-23, Option
    # B for the posY grasp-initiation gap). Same as ..._2cm_leftcam but the
    # dataset is re-converted with --grasp-oversample 6 --grasp-window 4
    # --grasp-oversample-side posY: the posY gripper open→close window is
    # duplicated 6× inline so the grasp-initiation decision (which the pure
    # policy fails on posY — it reaches the cube but never closes) is upweighted.
    # repo_id local/nero_cube_2cm_leftcam_graspos_200ep. 24k LoRA, batch 4.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_2cm_leftcam_graspos",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_2cm_leftcam_graspos_200ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=24_000,
        batch_size=4,
        save_interval=1_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=24_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # v4 ENDGAME-OVERSAMPLE (Tier-1 endgame-precision lever, 2026-06-18). Same
    # LoRA/model/loss as pi05_base_sim_finetune_nero_cube; only the dataset differs
    # → local/nero_cube_v4_eg3_200ep adds 2x endgame-only episodes per source
    # (converter --endgame-oversample 3 --endgame-frac 0.2) to upweight the
    # precision-critical descent+place. Clean A/B: 10 Hz (same inference parity),
    # 24k SMOOTH cosine FROM SCRATCH (no warm-restart). Tests whether endgame
    # oversampling improves late-stage placement vs the baseline LoRA (best 9.3 cm
    # negY / 14.3 cm posY, 0 success) — RPD closed, full FT didn't beat LoRA.
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_v4_eg",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                2.0,
                1.0,
                *([1.0] * 24),
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_v4_eg3_200ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=24_000,
        batch_size=4,
        save_interval=1_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=24_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # COMBINED dataset (Phase-3 rework, 2026-06-11): local/nero_cube_combined_v2
    # = mixed-200 + 100 new posY full trajectories + 136 RESample recovery clips,
    # converted with side tokens in the prompt and negY oversampled 2x
    # (--side-in-prompt --side-repeat negY:2 → ~288 negY / 292 posY ≈ 1:1; the raw
    # set was already ~2:1 posY after the new data). The combined dataset's
    # negY:2-weighted delta-Y quantiles are SYMMETRIC (posY -Y transport p05
    # normalizes to -0.90, negY +Y p95 to +0.38 — both well in-band), so the
    # quantile-norm out-of-band mechanism that plagued the cloth set is absent
    # here → no _meanstd variant needed; this run tests the STEP-LIMIT lever
    # (posY kept improving with steps). 60k SMOOTH cosine FROM SCRATCH (never a
    # warm-restart resume — twice-documented regression). Same LoRA/model/loss as
    # pi05_base_sim_finetune_nero_cube. save 4k / keep 8k retains 24k/40k/48k/56k.
    # NB: 4090 24 GB — batch_size starts at 2 (verify it/s + no OOM in the first
    # ~100 steps; drop to 1 if OOM); ~2x A100 step time expected.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_combined",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_combined_v2",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=60_000,
        batch_size=2,
        save_interval=4_000,
        keep_period=8_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=60_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # v4b ENDGAME-OVERSAMPLE 2x on COMBINED (clean retry, 2026-06-19). The v4
    # 3x-on-mixed test was confounded (mixed lacks combined_v2's posY transport
    # fix; 3x over-weighted the place → oscillation). This removes both: same
    # combined recipe (posY-fixed dataset, side prompts) but dataset
    # local/nero_cube_v4b_eg2_combined = combined raw + --side-repeat negY:2 +
    # --endgame-oversample 2 --endgame-frac 0.2 (gentler 2x). 24k SMOOTH (sweet
    # spot) to compare against combined_v2 @24k (negY 9.3 cm best). Last endgame-
    # precision BC lever before stopping.
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_v4b_eg2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                2.0,
                1.0,
                *([1.0] * 24),
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_v4b_eg2_combined",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=24_000,
        batch_size=2,
        save_interval=2_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=24_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # MEAN-STD A/B control for pi05_base_sim_finetune_nero_cube_combined
    # (2026-06-12 endgame-precision Tier-0). combined_v2's quantiles are
    # side-SYMMETRIC (the out-of-band mechanism is absent), but the quantile
    # range (~0.027) is still ~5x the std (~0.0055), so quantile norm weakens
    # the ~±2mm endgame-correction signal ~2.5x vs mean-std. This run isolates
    # that fine-delta effect on the cube task. Identical 60k smooth schedule;
    # the ONLY delta is use_quantile_norm=False. FROM SCRATCH — never --resume.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_combined_meanstd",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_combined_v2",
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=False),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=60_000,
        batch_size=2,
        save_interval=4_000,
        keep_period=8_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=60_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # CHUNK-WISE DELTA variant (2026-06-12 endgame-precision direction B).
    # Dataset local/nero_cube_combined_v3_chunkwise = same source episodes as
    # combined_v2 but converted with --delta-mode chunkwise: actions store the
    # ABSOLUTE next-frame EE pose [pos(3), axis_angle(3)] (same convention as
    # the state vector). extra_delta_transform=True pushes
    # DeltaActions(make_bool_mask(6,-1)) which subtracts the CHUNK-START state
    # broadcast over the whole horizon → chunk-wise deltas (within-chunk error
    # O(1) instead of O(k); "Demystifying Action Space Design" ICLR 2026:
    # chunk-wise > step-wise by >10pp). Serving inverts via AbsoluteActions →
    # the websocket returns ABSOLUTE EE pose targets; the demo must run with
    # the matching --action-mode chunkwise. Norm stats MUST be computed with
    # compute_norm_stats on THIS config (the sweep then sees post-DeltaActions
    # chunk-wise deltas) — never copied from a stepwise sibling. Mean-std norm:
    # chunk-wise deltas mix horizons k=1..10 so quantiles are even less
    # meaningful, and the eval ladder isolates variables as
    # v2+quantile → v2+meanstd → v3chunkwise+meanstd. FROM SCRATCH.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_chunkwise",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # chunkwise delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_combined_v3_chunkwise",
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=False),
            extra_delta_transform=True,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=60_000,
        batch_size=2,
        save_interval=4_000,
        keep_period=8_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=60_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # CHUNK-WISE + QUANTILE "best-of-both" (2026-06-14 ablation). The chunkwise
    # run (mean-std) gave the best negY precision but mean-std broke cube posY
    # GRASP (gripper dim mean-std mean=+0.31 open-bias suppresses the close
    # command on the weaker posY side); the v2 quantile baseline grasped posY.
    # This variant keeps the chunk-wise action representation (negY precision)
    # but uses QUANTILE norm (symmetric gripper ±1 → posY grasp restored).
    # Identical to pi05_base_sim_finetune_nero_cube_chunkwise except
    # use_quantile_norm is left auto (→True for pi05). Reuses the v3 chunkwise
    # dataset + its norm_stats.json (which already contains quantile fields).
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_chunkwise_quantile",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # chunkwise delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_combined_v3_chunkwise",
            base_config=DataConfig(prompt_from_task=True),  # use_quantile_norm auto→True
            extra_delta_transform=True,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=60_000,
        batch_size=2,
        save_interval=4_000,
        keep_period=8_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=60_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # FULL fine-tuning sibling of pi05_base_sim_finetune_nero_cube (the LoRA
    # config above). Same data/model/loss, but trains ALL weights instead of
    # LoRA adapters: NON-lora gemma variants (gemma_2b + gemma_300m), NO
    # freeze_filter (nothing frozen), EMA on, and a LOWER LR (1e-5 vs the LoRA
    # 2e-5) to avoid catastrophic forgetting of the pi0.5-base prior on this
    # small (200-episode) dataset. Select it by NAME — this is the full-vs-LoRA
    # switch:
    #   LoRA : uv run scripts/train.py pi05_base_sim_finetune_nero_cube      --exp-name X
    #   FULL : uv run scripts/train.py pi05_base_sim_finetune_nero_cube_full --exp-name X
    # MEMORY: full fine-tuning of pi0.5 (~3.3B params) + AdamW + EMA needs an
    # 80 GB A100 (40 GB will OOM). batch_size=8 is a conservative single-A100
    # start; override per your GPU with `--batch-size N` (tyro), or shard across
    # GPUs. If memory is tight, drop EMA with `--ema-decay None`.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_full",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            # NOTE: no *_lora variants → full gemma_2b + gemma_300m (all weights
            # trainable).
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_mixed_200ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        # No freeze_filter → every parameter is fine-tuned (full FT).
        ema_decay=0.99,
        num_train_steps=20_000,
        batch_size=8,
        save_interval=1_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-5,
            decay_steps=20_000,
            decay_lr=1e-6,
        ),
    ),
    #
    # posY-ONLY sanity model — same LoRA config as pi05_base_sim_finetune_nero_cube
    # but trained on the 100 posY episodes alone (local/nero_cube_posY_100ep). Tests
    # whether posY fails in the mixed model from BIMODAL confusion (negY+posY need
    # opposite swings) vs a fundamental posY difficulty: if posY-only succeeds, it's
    # the former. Only the data differs (LoRA/loss/LR identical), so the comparison
    # is clean. 16k steps.
    #
    TrainConfig(
        name="pi05_base_sim_finetune_nero_cube_posY",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # delta_pos, delta_ori
                2.0,                            # gripper_cmd (dim 6)
                1.0,                            # done (dim 7)
                *([1.0] * 24),                  # padding dims 8..31
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_posY_100ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=16_000,
        batch_size=4,
        save_interval=1_000,
        keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=16_000,
            decay_lr=2e-6,
        ),
    ),
    #
    # Per-side single-side cube models for the base-vs-LIBERO 2x2 study. Each is
    # the SAME LoRA recipe trained on ONE side's 100 episodes; the only knobs that
    # vary are the base weights (pi05_base vs pi05_libero) and the data side
    # (negY vs posY). posY/base is pi05_base_sim_finetune_nero_cube_posY above;
    # these three complete the matrix. 16k steps.
    #
    TrainConfig(  # negY-only, pi0.5-BASE
        name="pi05_base_sim_finetune_nero_cube_negY",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, *([1.0] * 24)),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_negY_100ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False, action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None, num_train_steps=16_000, batch_size=4,
        save_interval=1_000, keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2e-5, decay_steps=16_000, decay_lr=2e-6),
    ),
    TrainConfig(  # negY-only, pi0.5-LIBERO
        name="pi05_libero_sim_finetune_nero_cube_negY",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, *([1.0] * 24)),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_negY_100ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False, action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/pi05_libero/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None, num_train_steps=16_000, batch_size=4,
        save_interval=1_000, keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2e-5, decay_steps=16_000, decay_lr=2e-6),
    ),
    TrainConfig(  # posY-only, pi0.5-LIBERO
        name="pi05_libero_sim_finetune_nero_cube_posY",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
            action_dim_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, *([1.0] * 24)),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/nero_cube_posY_100ep",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False, action_dim=8,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/pi05_libero/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None, num_train_steps=16_000, batch_size=4,
        save_interval=1_000, keep_period=4_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2e-5, decay_steps=16_000, decay_lr=2e-6),
    ),
    #
    # ALOHA Sim fine-tuning config for bimanual cloth manipulation.
    #
    TrainConfig(
        name="pi05_aloha_sim_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=4,
            action_dim=14,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="local/sim_bimanual_cloth_50ep",
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="pick up the cloth and place it on the green target",
            use_delta_joint_actions=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "cam_high",
                                "cam_left_wrist": "cam_left_wrist",
                                "cam_right_wrist": "cam_right_wrist",
                            },
                            "state": "qpos",
                            "actions": "actions",
                        }
                    )
                ]
            ),
            action_sequence_keys=("actions",),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_aloha/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=8000,
        batch_size=2,
        save_interval=1000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-5,
            decay_steps=8000,
            decay_lr=2e-6,
        ),
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
