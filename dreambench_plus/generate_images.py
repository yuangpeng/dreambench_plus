import math
import os
from typing import Literal

import fire
import megfile
import torch
from accelerate import PartialState
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPVisionModelWithProjection, LlamaTokenizerFast

from dreambench_plus.constants import DREAMBENCH_PLUS_DIR, MODEL_ZOOS
from dreambench_plus.dreambench_plus_dataset import DreamBenchPlus
from dreambench_plus.metrics.clip_score import multigpu_eval_clipi_score, multigpu_eval_clipt_score
from dreambench_plus.metrics.dino_score import multigpu_eval_dino_score
from dreambench_plus.models.emu.modeling_emu import EmuForCausalLM
from dreambench_plus.pipelines.blip_diffusion.pipeline_blip_diffusion import BlipDiffusionPipeline
from dreambench_plus.pipelines.emu.pipeline_emu2_gen import EmuVisualGenerationPipeline
from dreambench_plus.utils.image_utils import save_image
from dreambench_plus.utils.loguru import logger

NEGATIVE_PROMPT_TEMPLATE = dict(
    blip_diffusion="over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate",
    ip_adapter_sd="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    ip_adapter_sdxl="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
)


DEFAULT_PARAMS = dict(
    dreambooth_sd=dict(guidance_scale=7.5, num_inference_steps=100, negative_prompt=None),
    dreambooth_lora_sd=dict(guidance_scale=7.5, num_inference_steps=100, negative_prompt=None),
    dreambooth_lora_sdxl=dict(guidance_scale=7.5, num_inference_steps=100, negative_prompt=None),
    textual_inversion_sd=dict(guidance_scale=7.5, num_inference_steps=100, negative_prompt=None),
    blip_diffusion=dict(guidance_scale=7.5, num_inference_steps=100, negative_prompt=NEGATIVE_PROMPT_TEMPLATE["blip_diffusion"]),
    emu2=dict(guidance_scale=3, num_inference_steps=50, negative_prompt=None),
    ip_adapter_sd=dict(
        guidance_scale=7.5,
        num_inference_steps=100,
        negative_prompt=NEGATIVE_PROMPT_TEMPLATE["ip_adapter_sd"],
        ip_adapter_scale=0.6,
    ),
    ip_adapter_sdxl=dict(
        guidance_scale=7.5,
        num_inference_steps=100,
        negative_prompt=NEGATIVE_PROMPT_TEMPLATE["ip_adapter_sdxl"],
        ip_adapter_scale=0.6,
    ),
    ip_adapter_plus_sdxl=dict(
        guidance_scale=7.5,
        num_inference_steps=100,
        negative_prompt=NEGATIVE_PROMPT_TEMPLATE["ip_adapter_sdxl"],
        ip_adapter_scale=0.6,
    ),
)


def dreambench_plus(
    method: Literal[
        "blip_diffusion",
        "emu2",
        "ip_adapter_sd",
        "ip_adapter_sdxl",
        "ip_adapter_plus_sdxl",
        "dreambooth_sd",
        "dreambooth_sdxl",
        "dreambooth_lora_sd",
        "dreambooth_lora_sdxl",
        "textual_inversion_sd",
        "textual_inversion_sdxl",
    ],
    dreambench_plus_dir: str = DREAMBENCH_PLUS_DIR,
    db_or_ti_output_dir: str | None = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 100,
    num_images_per_prompt: int = 1,
    negative_prompt: str | None = None,
    ip_adapter_scale: float | None = 0.6,
    ip_adapter_encoder: Literal["huge", "giant"] = "huge",
    seed: int = 42,
    torch_dtype: str | torch.dtype = "float16",
    save_dir: str = "./samples",
    enable_xformers: bool = False,
    local_files_only: bool = False,
    use_default_params: bool = False,
):
    distributed_state = PartialState()
    dreambench_plus = DreamBenchPlus(dir=dreambench_plus_dir)

    if use_default_params:
        logger.warning("Using default params for the pipeline, ignoring the provided params.")
        guidance_scale = DEFAULT_PARAMS[method].get("guidance_scale", guidance_scale)
        num_inference_steps = DEFAULT_PARAMS[method].get("num_inference_steps", num_inference_steps)
        negative_prompt = DEFAULT_PARAMS[method].get("negative_prompt", negative_prompt)
        ip_adapter_scale = DEFAULT_PARAMS[method].get("ip_adapter_scale", ip_adapter_scale)

    generator = torch.Generator(device=distributed_state.device).manual_seed(seed)

    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    logger.info(f"building pipeline for {method} ...")
    if method == "blip_diffusion":
        pipeline = BlipDiffusionPipeline.from_pretrained(
            MODEL_ZOOS["salesforce/blipdiffusion"],
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
    elif method == "emu2":
        if torch_dtype != torch.bfloat16:
            logger.warning("Emu2 only supports bfloat16")
            torch_dtype = torch.bfloat16

        model_name_or_path = MODEL_ZOOS["BAAI/Emu2-Gen"]
        multimodal_encoder = EmuForCausalLM.from_pretrained(
            f"{model_name_or_path}/multimodal_encoder",
            use_safetensors=True,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            variant="bf16",
            low_cpu_mem_usage=True,
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(f"{model_name_or_path}/tokenizer", local_files_only=local_files_only)
        pipeline = EmuVisualGenerationPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="bf16",
            multimodal_encoder=multimodal_encoder,
            tokenizer=tokenizer,
            local_files_only=local_files_only,
        )
    elif method == "ip_adapter_sd":
        pass
    elif method == "ip_adapter_sdxl":
        if ip_adapter_encoder == "huge":
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                MODEL_ZOOS["h94/IP-Adapter"],
                subfolder="models/image_encoder",
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
        else:
            image_encoder = None
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ZOOS["stabilityai/stable-diffusion-xl-base-1.0"],
            image_encoder=image_encoder,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        if ip_adapter_encoder == "huge":
            pipeline.load_ip_adapter(
                MODEL_ZOOS["h94/IP-Adapter"],
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl_vit-h.bin",
                local_files_only=local_files_only,
            )
        elif ip_adapter_encoder == "giant":
            pipeline.load_ip_adapter(
                MODEL_ZOOS["h94/IP-Adapter"],
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
                local_files_only=local_files_only,
            )
        pipeline.set_ip_adapter_scale(ip_adapter_scale)
    elif method == "ip_adapter_plus_sdxl":
        if ip_adapter_encoder == "huge":
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                MODEL_ZOOS["h94/IP-Adapter"],
                subfolder="models/image_encoder",
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
        else:
            image_encoder = None
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ZOOS["stabilityai/stable-diffusion-xl-base-1.0"],
            image_encoder=image_encoder,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        if ip_adapter_encoder == "huge":
            pipeline.load_ip_adapter(
                MODEL_ZOOS["h94/IP-Adapter"],
                subfolder="sdxl_models",
                weight_name="ip-adapter-plus_sdxl_vit-h.bin",
                local_files_only=local_files_only,
            )
        pipeline.set_ip_adapter_scale(ip_adapter_scale)
    elif method == "dreambooth_sd":
        assert db_or_ti_output_dir is not None, "`db_or_ti_output_dir` must be provided"
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ZOOS["runwayml/stable-diffusion-v1-5"],
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
    elif method == "dreambooth_lora_sd":
        assert db_or_ti_output_dir is not None, "`db_or_ti_output_dir` must be provided"
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ZOOS["runwayml/stable-diffusion-v1-5"],
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
    elif method == "dreambooth_lora_sdxl":
        assert db_or_ti_output_dir is not None, "`db_or_ti_output_dir` must be provided"
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ZOOS["stabilityai/stable-diffusion-xl-base-1.0"],
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            add_watermarker=False,
        )
    elif method == "textual_inversion_sd":
        assert db_or_ti_output_dir is not None, "`db_or_ti_output_dir` must be provided"
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ZOOS["runwayml/stable-diffusion-v1-5"],
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipeline = pipeline.to(distributed_state.device)
    pipeline.set_progress_bar_config(disable=True)

    if enable_xformers and method != "ip_adapter_sd" and method != "ip_adapter_sdxl" and method != "ip_adapter_plus_sdxl":
        try:
            import xformers

            pipeline.unet.enable_xformers_memory_efficient_attention()
        except:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # fmt: off
    if method.startswith("ip_adapter"):
        additional_info = f"_vit_{ip_adapter_encoder}_scale{ip_adapter_scale}"
    else:
        additional_info = ""
    method_dir = f"{method}{additional_info}_gs{guidance_scale}_step{num_inference_steps}_seed{seed}_{torch_dtype}".replace(".", "_")
    with megfile.smart_open(os.path.join(save_dir, method_dir, "negative_prompt.txt"), "w") as f:
        if negative_prompt is None:
            f.write("")
        else:
            f.write(negative_prompt)
    # fmt: on

    logger.info(f"Method: {method}, save_dir: {method_dir}")
    pbar = tqdm(
        total=math.ceil(len(dreambench_plus) * len(dreambench_plus[0].captions) / distributed_state.num_processes),
        desc=f"Generating Images on DreamBenchPlus",
        disable=not distributed_state.is_local_main_process,
    )

    params = []
    for sample in dreambench_plus:
        if (
            method == "blip_diffusion"
            or method == "ip_adapter_sd"
            or method == "ip_adapter_sdxl"
            or method == "ip_adapter_plus_sdxl"
            or method == "dreambooth_sd"
            or method == "dreambooth_sdxl"
            or method == "dreambooth_lora_sd"
            or method == "dreambooth_lora_sdxl"
            or method == "emu2"
        ):
            for i, (text_prompt_input, prompt_gt) in enumerate(zip(sample.captions, sample.captions)):
                params.append((sample.id, sample.subject, sample.image, i, text_prompt_input, prompt_gt))
        # elif method == "emu2":
        #     pass
        elif method == "textual_inversion_sd" or method == "textual_inversion_sdxl":
            for i, (text_prompt_input, prompt_gt) in enumerate(zip(sample.captions, sample.captions)):
                text_prompt_input = text_prompt_input.replace(subject, "<sks>")
                params.append((sample.id, sample.subject, sample.image, i, text_prompt_input, prompt_gt))
        else:
            pass

    with distributed_state.split_between_processes(params) as sub_params:
        for _param in sub_params:
            id, subject, cond_image, i, text_prompt_input, prompt_gt = _param

            if method == "blip_diffusion":
                prompt = text_prompt_input
                pipe_kwargs = dict(
                    reference_image=cond_image,
                    source_subject_category=subject,
                    target_subject_category=None,
                    height=512,
                    width=512,
                )

            elif method == "emu2":
                try:
                    # NOTE: official prompt generation method, but need the images that kosmos-g has selected
                    # center_mask = EmuVisualGenerationPipeline.gen_mask()
                    # # <grouding><phrase>a {class}</phrase><object>[center_mask]</object>[cond_image] in the jungle
                    # text1, text2 = tuple(text_prompt_input.split(subject))
                    # prompt = [
                    #     "<grounding><phrase>" + f"{text1} {subject}" + "</phrase><object>",
                    #     center_mask,
                    #     "</object>",
                    #     cond_image,
                    #     " " + text2,
                    # ]

                    # kosmosg
                    text1, text2 = tuple(text_prompt_input.split(subject))
                    prompt = [text1, cond_image, text2]
                    pipe_kwargs = dict(height=1024, width=1024)
                except:
                    logger.warning(
                        f"Failed to split the text prompt, using the original text prompt.\nsubject: {subject}, text_prompt_input: {text_prompt_input}"
                    )
                    prompt = [cond_image, text_prompt_input]
                    pipe_kwargs = dict(height=1024, width=1024)

            elif method == "ip_adapter_sd" or method == "ip_adapter_sdxl" or method == "ip_adapter_plus_sdxl":
                prompt = text_prompt_input
                pipe_kwargs = dict(ip_adapter_image=cond_image, height=1024, width=1024)

            elif method == "dreambooth_sd":
                prompt = text_prompt_input
                pipeline = StableDiffusionPipeline.from_pretrained(
                    os.path.join(db_or_ti_output_dir, id),
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                    safety_checker=None,
                    requires_safety_checker=False,
                )

                pipeline = pipeline.to(distributed_state.device)
                pipeline.set_progress_bar_config(disable=True)

                if enable_xformers:
                    try:
                        import xformers

                        pipeline.unet.enable_xformers_memory_efficient_attention()
                    except:
                        raise ValueError("xformers is not available. Make sure it is installed correctly")

                pipe_kwargs = dict(height=512, width=512)

            elif method == "dreambooth_lora_sdxl":
                prompt = text_prompt_input
                pipeline.load_lora_weights(os.path.join(db_or_ti_output_dir, id))
                pipe_kwargs = dict(height=1024, width=1024)

            elif method == "textual_inversion_sd":
                prompt = text_prompt_input
                pipeline.tokenizer = CLIPTokenizer.from_pretrained(
                    MODEL_ZOOS["runwayml/stable-diffusion-v1-5"], subfolder="tokenizer", local_files_only=local_files_only
                )
                pipeline.load_textual_inversion(os.path.join(db_or_ti_output_dir, id))
                pipe_kwargs = dict(height=512, width=512)

            output = pipeline(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                **pipe_kwargs,
            ).images

            for j, _output in enumerate(output):
                save_image(cond_image, path=os.path.join(save_dir, method_dir, "src_image", id, f"{i}_{j}.jpg"))
                save_image(_output, path=os.path.join(save_dir, method_dir, "tgt_image", id, f"{i}_{j}.jpg"))
                with megfile.smart_open(os.path.join(save_dir, method_dir, "text", id, f"{i}_{j}.txt"), "w") as f:
                    f.write(prompt_gt)

            pbar.update(1)

    distributed_state.wait_for_everyone()

    dinov1_score = multigpu_eval_dino_score(
        os.path.join(save_dir, method_dir, "src_image"),
        os.path.join(save_dir, method_dir, "tgt_image"),
        distributed_state=distributed_state,
        version="v1",
    )
    dinov2_score = multigpu_eval_dino_score(
        os.path.join(save_dir, method_dir, "src_image"),
        os.path.join(save_dir, method_dir, "tgt_image"),
        distributed_state=distributed_state,
        version="v2",
    )
    clipi_score = multigpu_eval_clipi_score(
        os.path.join(save_dir, method_dir, "src_image"),
        os.path.join(save_dir, method_dir, "tgt_image"),
        distributed_state=distributed_state,
    )
    clipt_score = multigpu_eval_clipt_score(
        os.path.join(save_dir, method_dir, "text"),
        os.path.join(save_dir, method_dir, "tgt_image"),
        distributed_state=distributed_state,
    )
    logger.info(f"Method: {method}, save_dir: {method_dir}")
    logger.info(f"DINOv1 score: {dinov1_score}")
    logger.info(f"DINOv2 score: {dinov2_score}")
    logger.info(f"CLIP-I score: {clipi_score}")
    logger.info(f"CLIP-T score: {clipt_score}")


if __name__ == "__main__":
    fire.Fire(dreambench_plus)
