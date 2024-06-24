import math
import os
from typing import Literal

import fire
import megfile
import torch
from accelerate import PartialState
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from dreambench_plus.constants import LOCAL_FILES_ONLY, MODEL_ZOOS
from dreambench_plus.utils.comm import all_gather
from dreambench_plus.utils.image_utils import IMAGE_EXT, ImageType, load_image
from dreambench_plus.utils.loguru import logger

_DEFAULT_MODEL: str = MODEL_ZOOS["openai/clip-vit-base-patch32"]
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32


class CLIPScore:

    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = LOCAL_FILES_ONLY,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = CLIPModel.from_pretrained(model_or_name_path, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_or_name_path, local_files_only=local_files_only)

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)

        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)

    @torch.no_grad()
    def get_text_features(self, text: str | list[str], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
        inputs = self.processor(text=text, padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(
            inputs["input_ids"].to(self.device),
            inputs["attention_mask"].to(self.device),
        )
        if norm:
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(inputs["pixel_values"].to(self.device, dtype=self.dtype))
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def clipi_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."

        images1_features = self.get_image_features(images1, norm=True)
        images2_features = self.get_image_features(images2, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (images1_features * images2_features).sum(axis=-1)
        return score.sum(0).float(), len(images1)

    @torch.no_grad()
    def clipt_score(self, texts: str | list[str], images: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(texts, list):
            texts = [texts]
        if not isinstance(images, list):
            images = [images]
        assert len(texts) == len(images), f"Number of texts ({len(texts)}) and images {(len(images))} should be same."

        texts_features = self.get_text_features(texts, norm=True)
        images_features = self.get_image_features(images, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (texts_features * images_features).sum(axis=-1)
        return score.sum(0).float(), len(texts)


def multigpu_eval_clipi_score(
    image1_dir: str,
    image2_dir: str,
    distributed_state: PartialState | None = None,
    clip_score: CLIPScore | None = None,
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    if clip_score is None:
        clip_score = CLIPScore(device=distributed_state.device)

    if image1_dir[-1] != "/":
        image1_dir = image1_dir + "/"

    if image2_dir[-1] != "/":
        image2_dir = image2_dir + "/"

    image1_files = []
    for _ext in IMAGE_EXT:
        image1_files.extend(megfile.smart_glob(os.path.join(image1_dir, f"**/*.{_ext}")))
    image1_files = sorted(image1_files)

    image2_files = []
    for _ext in IMAGE_EXT:
        image2_files.extend(megfile.smart_glob(os.path.join(image2_dir, f"**/*.{_ext}")))
    image2_files = sorted(image2_files)

    assert len(image1_files) == len(image2_files), f"Number of image1 files {len(image1_files)} != number of image2 files {len(image2_files)}."

    params = []
    for image1_file, image2_file in zip(image1_files, image2_files):
        assert (
            image1_file.split(image1_dir)[-1].split(".")[0] == image2_file.split(image2_dir)[-1].split(".")[0]
        ), f"Image1 file {image1_file} and image2 file {image2_file} do not match."

        params.append((image1_file, image2_file))

    pbar = tqdm(
        total=math.ceil(len(image1_files) / distributed_state.num_processes),
        desc="Evaluating CLIP-I Score",
        disable=not distributed_state.is_local_main_process,
    )

    with distributed_state.split_between_processes(params) as sub_params:
        score = 0
        for _param in sub_params:
            image1_file, image2_file = _param
            image1, image2 = load_image(image1_file), load_image(image2_file)
            score += clip_score.clipi_score(image1, image2)[0]
            pbar.update(1)

    scores = all_gather(score)
    return (sum(scores) / len(image1_files)).item()


def multigpu_eval_clipt_score(
    text_dir: str,
    image_dir: str,
    distributed_state: PartialState | None = None,
    clip_score: CLIPScore | None = None,
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    if clip_score is None:
        clip_score = CLIPScore(device=distributed_state.device)

    text_files = megfile.smart_glob(os.path.join(text_dir, "**/*.txt"))
    text_files = sorted(text_files)

    if text_dir[-1] != "/":
        text_dir = text_dir + "/"

    if image_dir[-1] != "/":
        image_dir = image_dir + "/"

    image_files = []
    for _ext in IMAGE_EXT:
        image_files.extend(megfile.smart_glob(os.path.join(image_dir, f"**/*.{_ext}")))
    image_files = sorted(image_files)

    assert len(text_files) == len(image_files), f"Number of text files {len(text_files)} != number of image files {len(image_files)}."

    params = []
    for text_file, image_file in zip(text_files, image_files):
        assert (
            text_file.split(text_dir)[-1].split(".")[0] == image_file.split(image_dir)[-1].split(".")[0]
        ), f"Text file {text_file} and image file {image_file} do not match."

        params.append((text_file, image_file))

    pbar = tqdm(
        total=math.ceil(len(text_files) / distributed_state.num_processes),
        desc="Evaluating CLIP-T Score",
        disable=not distributed_state.is_local_main_process,
    )

    with distributed_state.split_between_processes(params) as sub_params:
        score = 0
        for _param in sub_params:
            text_file, image_file = _param
            with megfile.smart_open(text_file, "r") as f:
                text = f.read()
            image = load_image(image_file)
            score += clip_score.clipt_score(text, image)[0]
            pbar.update(1)

    scores = all_gather(score)
    return (sum(scores) / len(text_files)).item()


def clip_eval(mode: Literal["clipi", "clipt"], dir1: str, dir2: str):
    if mode == "clipi":
        logger.info(f"CLIP-I Score: {multigpu_eval_clipi_score(dir1, dir2)}")
    elif mode == "clipt":
        logger.info(f"CLIP-T Score: {multigpu_eval_clipt_score(dir1, dir2)}")


if __name__ == "__main__":
    fire.Fire(clip_eval)
