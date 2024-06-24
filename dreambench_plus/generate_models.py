import os
from typing import Literal

import fire

from dreambench_plus.constants import DREAMBENCH_PLUS_DIR
from dreambench_plus.dreambench_plus_dataset import DreamBenchPlus

DEFAULT_PARAMS = dict(
    dreambooth_sd=dict(bs=1, learning_rate=2.5e-6, max_train_steps=250),
    dreambooth_lora_sd=dict(bs=1, learning_rate=1e-4, max_train_steps=100),
    dreambooth_lora_sdxl=dict(bs=1, learning_rate=5e-5, max_train_steps=500),
    textual_inversion_sd=dict(bs=1, learning_rate=5e-4, max_train_steps=3000),
)


def model_generator(
    method: Literal[
        "dreambooth_sd",
        "dreambooth_sdxl",
        "dreambooth_lora_sd",
        "dreambooth_lora_sdxl",
        "textual_inversion_sd",
        "textual_inversion_sdxl",
    ],
    output_dir: str | None = None,
    start: int | None = None,
    end: int | None = None,
):
    dreambench_plus = DreamBenchPlus(dir=DREAMBENCH_PLUS_DIR)

    if output_dir is None:
        output_dir = method

    bs = DEFAULT_PARAMS[method]["bs"]
    learning_rate = DEFAULT_PARAMS[method]["learning_rate"]
    max_train_steps = DEFAULT_PARAMS[method]["max_train_steps"]

    if method == "dreambooth_sd":
        model_name_or_path = "runwayml/stable-diffusion-v1-5"
        for i, sample in enumerate(dreambench_plus):
            if i >= start and i < end:
                cmd = f"""torchrun dreambench_plus/training_scripts/train_dreambooth.py \
--pretrained_model_name_or_path="'{model_name_or_path}'" \
--instance_data_dir="'{sample.image_path}'" \
--output_dir="'work_dirs/dreambench_plus/{output_dir}/{sample.collection_id}'" \
--instance_prompt="'a photo of {sample.subject}'" \
--resolution=512 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--learning_rate={learning_rate} \
--lr_scheduler="'constant'" \
--lr_warmup_steps=0 \
--max_train_steps={max_train_steps} \
--validation_steps=99999 \
--seed=42
"""
                os.system(cmd)

    elif method == "dreambooth_lora_sd":
        model_name_or_path = "runwayml/stable-diffusion-v1-5"
        for i, sample in enumerate(dreambench_plus):
            if i >= start and i < end:
                cmd = f"""torchrun dreambench_plus/training_scripts/train_dreambooth_lora.py \
--pretrained_model_name_or_path="'{model_name_or_path}'" \
--instance_data_dir="'{sample.image_path}'" \
--output_dir="'work_dirs/dreambench_plus/{output_dir}/{sample.collection_id}'" \
--instance_prompt="'a photo of {sample.subject}'" \
--resolution=512 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--learning_rate={learning_rate} \
--lr_scheduler="'constant'" \
--lr_warmup_steps=0 \
--max_train_steps={max_train_steps} \
--validation_epochs=99999 \
--seed=42
"""
                os.system(cmd)

    elif method == "dreambooth_lora_sdxl":
        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
        for i, sample in enumerate(dreambench_plus):
            if i >= start and i < end:
                cmd = f"""torchrun dreambench_plus/training_scripts/train_dreambooth_lora_sdxl.py \
--pretrained_model_name_or_path="'{model_name_or_path}'"  \
--instance_data_dir="'{sample.image_path}'" \
--pretrained_vae_model_name_or_path="'{vae_name_or_path}'" \
--output_dir="'work_dirs/dreambench_plus/{output_dir}/{sample.collection_id}'" \
--mixed_precision="fp16" \
--instance_prompt="'a photo of {sample.subject}'" \
--resolution=1024 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--learning_rate={learning_rate} \
--lr_scheduler="'constant'" \
--lr_warmup_steps=0 \
--max_train_steps={max_train_steps} \
--validation_epochs=99999 \
--seed=42
"""
                os.system(cmd)

    elif method == "textual_inversion_sd":
        model_name_or_path = "runwayml/stable-diffusion-v1-5"
        for i, sample in enumerate(dreambench_plus):
            _class_single_token = sample.subject.split(" ")[0]
            if "style" in sample.image_path:
                learnable_property = "style"
            else:
                learnable_property = "object"
            if i >= start and i < end:
                cmd = f"""torchrun dreambench_plus/training_scripts/textual_inversion.py \
--pretrained_model_name_or_path="'{model_name_or_path}'" \
--train_data_dir="'{sample.image_path}'" \
--learnable_property="'{learnable_property}'" \
--placeholder_token="'<sks>'" \
--initializer_token="{_class_single_token}" \
--resolution=512 \
--train_batch_size={bs} \
--gradient_accumulation_steps=1 \
--max_train_steps={max_train_steps} \
--learning_rate={learning_rate} \
--scale_lr \
--lr_scheduler="'constant'" \
--lr_warmup_steps=0 \
--output_dir="'work_dirs/dreambench_plus/{output_dir}/{sample.collection_id}'"
"""
                os.system(cmd)


if __name__ == "__main__":
    fire.Fire(model_generator)
