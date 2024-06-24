import copy
import os
from collections import OrderedDict
from functools import partial

import PIL.Image
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchvision import transforms as T
from transformers.image_processing_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import TensorType

from dreambench_plus.constants import MODEL_ZOOS, PATH_TO_MODEL_NAME
from dreambench_plus.utils.loguru import logger
from dreambench_plus.utils.misc import omageconf_safe_update_dict

from ..eva_clip import modeling_eva_vit, modeling_eva_vit_emu2

try:
    from apex.normalization import FusedLayerNorm
except:
    FusedLayerNorm = nn.LayerNorm
    logger.error("Please 'pip install apex'")


DEFAULT_CONFIG = DictConfig(
    content=dict(
        eva_model_name=None,
        image_size=224,
        patch_size=16,
        width=768,
        layers=12,
        head_width=64,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=None,  # drop path rate
        fusedLN=False,
        layer_norm_eps=1e-6,
        ls_init_value=None,  # layer scale initial value
        patch_dropout=0.0,  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
        rope=False,
        global_average_pool=False,  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
        xattn=False,
        postnorm=False,
        pt_hw_seq_len=16,  # 224/14
        intp_freq=False,
        naiveswiglu=False,
        subln=False,
    ),
    flags={"allow_objects": True},
)


VISION_CONFIG = OmegaConf.create(flags={"allow_objects": True})

VISION_CONFIG.eva02_clip_e_psz14_plus_s9B_emu2 = dict(
    eva_model_name="eva02_clip_e_psz14_plus_s9B_emu2",
    image_size=448,
    patch_size=14,
    width=1792,
    layers=64,
    head_width=112,
    mlp_ratio=8.571428571428571,
    qkv_bias=True,
    drop_path_rate=0,
    layer_norm_eps=1e-6,
    xattn=True,
    postnorm=True,
)

VISION_CONFIG.eva02_clip_e_psz14_plus_s9B = copy.deepcopy(DEFAULT_CONFIG)
omageconf_safe_update_dict(
    VISION_CONFIG.eva02_clip_e_psz14_plus_s9B,
    dict(
        eva_model_name="eva02_clip_e_psz14_plus_s9B",
        image_size=224,
        patch_size=14,
        num_classes=1024,
        width=1792,
        layers=64,
        head_width=112,
        mlp_ratio=8.571428571428571,
        drop_path_rate=0,
        layer_norm_eps=1e-6,
        xattn=True,
        postnorm=True,
        fusedLN=True,
    ),
)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class EVACLIPImageProcessorWrapper:
    def __init__(self):
        super().__init__()

        self.transform = None
        self.image_mean = OPENAI_DATASET_MEAN
        self.image_std = OPENAI_DATASET_STD

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        if pretrained_model_name_or_path in PATH_TO_MODEL_NAME:
            model_name = PATH_TO_MODEL_NAME[pretrained_model_name_or_path]
        else:
            model_name = pretrained_model_name_or_path
        assert model_name in VISION_CONFIG, f"Unsupported {model_name} model"

        vision_config = VISION_CONFIG[model_name]

        image_processor = cls()
        transform = T.Compose(
            [
                T.Resize(vision_config.image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(vision_config.image_size),
                T.ToTensor(),
                T.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ]
        )
        setattr(image_processor, "transform", transform)

        return image_processor

    def __call__(
        self,
        images: PIL.Image.Image | list[PIL.Image.Image],
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        if not isinstance(images, list):
            images = [images]

        images = [self.transform(image) for image in images]
        images = torch.stack(images)
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)


class EVACLIPVisionConfigWrapper:
    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        if pretrained_model_name_or_path in PATH_TO_MODEL_NAME:
            model_name = PATH_TO_MODEL_NAME[pretrained_model_name_or_path]
        else:
            model_name = pretrained_model_name_or_path
        assert model_name in VISION_CONFIG, f"Unsupported {model_name} model"

        vision_config = VISION_CONFIG[model_name]

        return vision_config


class EVACLIPVisionModelWrapper:
    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        if pretrained_model_name_or_path in PATH_TO_MODEL_NAME:
            model_name = PATH_TO_MODEL_NAME[pretrained_model_name_or_path]
        else:
            model_name = pretrained_model_name_or_path
        assert model_name in VISION_CONFIG, f"Unsupported {model_name} model"

        vision_config = VISION_CONFIG[model_name]
        # In order to be compatible with CLIPVisionConfig
        vision_config.hidden_size = vision_config.width

        if model_name == "eva02_clip_e_psz14_plus_s9B_emu2":
            model = modeling_eva_vit_emu2.EVAVisionTransformer(
                img_size=vision_config.image_size,
                patch_size=vision_config.patch_size,
                embed_dim=vision_config.width,
                depth=vision_config.layers,
                num_heads=vision_config.width // vision_config.head_width,
                mlp_ratio=vision_config.mlp_ratio,
                qkv_bias=vision_config.qkv_bias,
                drop_path_rate=vision_config.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=vision_config.layer_norm_eps),
                xattn=vision_config.xattn,
                postnorm=vision_config.postnorm,
            )
            state_dict = torch.load(MODEL_ZOOS[model_name], map_location="cpu")
            model.load_state_dict(state_dict)

            setattr(model, "config", vision_config)

            def _forward(self, pixel_values: torch.FloatTensor | None = None, **kwargs):
                # [B, n_patch, C]
                last_hidden_state, all_hidden_states = self.forward_features(pixel_values)
                pooled_output = last_hidden_state[:, 0, :]

                return BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=all_hidden_states,
                    attentions=None,
                )

            model.forward = _forward.__get__(model, modeling_eva_vit_emu2.EVAVisionTransformer)

        elif model_name == "eva02_clip_e_psz14_plus_s9B":
            if vision_config.rope:
                os.environ["RoPE"] = "1"
            else:
                os.environ["RoPE"] = "0"

            model = modeling_eva_vit.EVAVisionTransformer(
                img_size=vision_config.image_size,
                patch_size=vision_config.patch_size,
                num_classes=vision_config.num_classes,
                embed_dim=vision_config.width,
                depth=vision_config.layers,
                num_heads=vision_config.width // vision_config.head_width,
                mlp_ratio=vision_config.mlp_ratio,
                qkv_bias=vision_config.qkv_bias,
                drop_path_rate=vision_config.drop_path_rate,
                norm_layer=(
                    partial(FusedLayerNorm, eps=vision_config.layer_norm_eps)
                    if vision_config.fusedLN
                    else partial(nn.LayerNorm, eps=vision_config.layer_norm_eps)
                ),
                init_values=vision_config.ls_init_value,
                patch_dropout=vision_config.patch_dropout,
                rope=vision_config.rope,
                use_mean_pooling=vision_config.global_average_pool,  # False
                xattn=vision_config.xattn,
                postnorm=vision_config.postnorm,
                pt_hw_seq_len=vision_config.pt_hw_seq_len,  # 224/14
                intp_freq=vision_config.intp_freq,
                naiveswiglu=vision_config.naiveswiglu,
                subln=vision_config.subln,
            )
            state_dict = torch.load(MODEL_ZOOS[model_name], map_location="cpu")

            def remove_prefix_from_keys(odict, prefix):
                new_odict = OrderedDict()
                for key, value in odict.items():
                    if key.startswith(prefix):
                        new_key = key[len(prefix) :]
                    else:
                        new_key = key
                    new_odict[new_key] = value
                return new_odict

            state_dict = remove_prefix_from_keys(state_dict, "visual.")
            model.load_state_dict(state_dict, strict=False)

            setattr(model, "config", vision_config)

            def _forward(self, pixel_values: torch.FloatTensor | None = None, **kwargs):
                # [B, n_patch, C]
                last_hidden_state, all_hidden_states = self.forward_features(pixel_values, return_all_features=True)
                pooled_output = last_hidden_state[:, 0, :]

                return BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=all_hidden_states,
                    attentions=None,
                )

            model.forward = _forward.__get__(model, modeling_eva_vit.EVAVisionTransformer)
        else:
            raise NotImplementedError(f"`{model_name}` is not supported yet.")

        return model
