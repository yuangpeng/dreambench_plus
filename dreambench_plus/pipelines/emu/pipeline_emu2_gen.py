# -*- coding: utf-8 -*-

# ===========================================================================================
#
#    Copyright (c) Beijing Academy of Artificial Intelligence (BAAI). All rights reserved.
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-19 10:45
#    Last Modified : 2023-12-25 07:59
#    File Name     : pipeline_emu2_gen.py
#    Description   :
#
# ===========================================================================================

from dataclasses import dataclass
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline, EulerDiscreteScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torchvision import transforms as TF
from transformers import CLIPImageProcessor, LlamaTokenizerFast

from dreambench_plus.models.emu.modeling_emu import EmuForCausalLM

EVA_IMAGE_SIZE = 448
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"


@dataclass
class EmuVisualGenerationPipelineOutput(BaseOutput):
    images: Image.Image | list[Image.Image]
    nsfw_content_detected: Optional[bool]


class EmuVisualGenerationPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "multimodal_encoder->unet->vae"

    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        multimodal_encoder: EmuForCausalLM,
        scheduler: EulerDiscreteScheduler,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        feature_extractor: CLIPImageProcessor,
        safety_checker: StableDiffusionSafetyChecker,
        eva_size=EVA_IMAGE_SIZE,
        eva_mean=OPENAI_DATASET_MEAN,
        eva_std=OPENAI_DATASET_STD,
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            multimodal_encoder=multimodal_encoder,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.transform = TF.Compose(
            [
                TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
                TF.ToTensor(),
                TF.Normalize(mean=eva_mean, std=eva_std),
            ]
        )

        self.negative_prompt = {}

        self.seq_mode = False

    def get_device(self, module):
        return next(module.parameters()).device

    def get_dtype(self, module):
        return next(module.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        negative_prompt: str = "",
        guidance_scale: float = 3.0,
        num_images_per_prompt: Optional[int] = 1,
        crop_info: List[int] = [0, 0],
        original_size: List[int] = [1024, 1024],
    ):
        seed = generator.initial_seed()
        device = generator.device

        try:
            if not self.seq_mode:
                return self.iner_call(
                    inputs=inputs,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    crop_info=crop_info,
                    original_size=original_size,
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.seq_mode = True
            else:
                raise e

        if self.seq_mode:
            images = []
            for i in range(num_images_per_prompt):
                output = self.iner_call(
                    inputs=inputs,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator(device=device).manual_seed(seed + i),
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                    crop_info=crop_info,
                    original_size=original_size,
                )
                images.append(output.images[0])
            return EmuVisualGenerationPipelineOutput(images=images, nsfw_content_detected=None)

    @torch.no_grad()
    def iner_call(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        negative_prompt: str | None = "",
        guidance_scale: float = 3.0,
        num_images_per_prompt: Optional[int] = 1,
        crop_info: List[int] = [0, 0],
        original_size: List[int] = [1024, 1024],
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if negative_prompt is None:
            negative_prompt = ""

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.get_device(self.unet)
        dtype = self.get_dtype(self.unet)

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        prompt_embeds = (
            self._prepare_and_encode_inputs(
                inputs,
                negative_prompt,
                do_classifier_free_guidance,
                num_images_per_prompt,
            )
            .to(dtype)
            .to(device)
        )
        batch_size = prompt_embeds.shape[0] // 2 // num_images_per_prompt if do_classifier_free_guidance else prompt_embeds.shape[0] // num_images_per_prompt

        unet_added_conditions = {}
        time_ids = torch.LongTensor(original_size + crop_info + [height, width]).to(device)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([time_ids, time_ids], dim=0)
        else:
            add_time_ids = time_ids
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        unet_added_conditions["time_ids"] = add_time_ids
        unet_added_conditions["text_embeds"] = torch.mean(prompt_embeds, dim=1)

        # 2. Prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels=self.unet.config.in_channels,
            height=height // self.vae_scale_factor,
            width=width // self.vae_scale_factor,
            generator=generator,
            latents=None,
            dtype=self.unet.dtype,
            device=device,
        )

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 4. Denoising loop
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            # 2B x 4 x H x W
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Post-processing
        images = self.decode_latents(latents)

        # 6. Run safety checker
        # images, has_nsfw_concept = self.run_safety_checker(images)

        # 7. Convert to PIL
        images = self.numpy_to_pil(images)
        return EmuVisualGenerationPipelineOutput(
            images=images,
            nsfw_content_detected=None,
            # nsfw_content_detected=None if has_nsfw_concept is None else has_nsfw_concept[0],
        )

    def _prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        negative_prompt: str = "",
        do_classifier_free_guidance: bool = False,
        num_images_per_prompt: Optional[int] = 1,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        device = self.get_device(self.multimodal_encoder.model.visual)
        dtype = self.get_dtype(self.multimodal_encoder.model.visual)
        has_image, has_text = False, False
        text_prompt, image_prompt = "", []
        for x in inputs:
            if isinstance(x, str):
                has_text = True
                text_prompt += x
            else:
                has_image = True
                text_prompt += placeholder
                image_prompt.append(self.transform(x))

        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.stack(image_prompt)
            image_prompt = image_prompt.type(dtype).to(device)

        if has_image and not has_text:
            prompt = self.multimodal_encoder.model.encode_image(image=image_prompt)
            bs_embed, seq_len, _ = prompt.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt = prompt.repeat(1, num_images_per_prompt, 1)
            prompt = prompt.view(bs_embed * num_images_per_prompt, seq_len, -1)
            if do_classifier_free_guidance:
                key = "[NULL_IMAGE]"
                if key not in self.negative_prompt:
                    negative_image = torch.zeros_like(image_prompt)
                    self.negative_prompt[key] = self.multimodal_encoder.model.encode_image(image=negative_image)
                uncond_prompt = self.negative_prompt[key]
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                uncond_prompt = uncond_prompt.repeat(1, num_images_per_prompt, 1)
                uncond_prompt = uncond_prompt.view(bs_embed * num_images_per_prompt, seq_len, -1)
                prompt = torch.cat([prompt, uncond_prompt], dim=0)
        else:
            prompt = self.multimodal_encoder.generate_image_stream(text=[text_prompt], image=image_prompt, tokenizer=self.tokenizer)
            bs_embed, seq_len, _ = prompt.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt = prompt.repeat(1, num_images_per_prompt, 1)
            prompt = prompt.view(bs_embed * num_images_per_prompt, seq_len, -1)
            if do_classifier_free_guidance:
                key = negative_prompt
                if key not in self.negative_prompt:
                    self.negative_prompt[key] = self.multimodal_encoder.generate_image_stream(text=[negative_prompt], tokenizer=self.tokenizer)
                uncond_prompt = self.negative_prompt[key]
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                uncond_prompt = uncond_prompt.repeat(1, num_images_per_prompt, 1)
                uncond_prompt = uncond_prompt.view(bs_embed * num_images_per_prompt, seq_len, -1)
                prompt = torch.cat([prompt, uncond_prompt], dim=0)

        return prompt

    # Copied from diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_safety_checker(self, images: np.ndarray):
        if self.safety_checker is not None:
            device = self.get_device(self.safety_checker)
            dtype = self.get_dtype(self.safety_checker)
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(images),
                return_tensors="pt",
            ).to(device)
            images, has_nsfw_concept = self.safety_checker(
                images=images,
                clip_input=safety_checker_input.pixel_values.to(dtype),
            )
        else:
            has_nsfw_concept = None
        return images, has_nsfw_concept

    @staticmethod
    def gen_mask(left: int = 110, top: int = 110, right: int = 340, bottom: int = 340, size: int = EVA_IMAGE_SIZE):
        mask = np.zeros((size, size, 3), dtype=np.uint8)
        mask = cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), 3)
        mask = Image.fromarray(mask)
        return mask
