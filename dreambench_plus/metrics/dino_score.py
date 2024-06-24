import math
import os
from typing import Literal

import fire
import megfile
import torch
from accelerate import PartialState
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import BitImageProcessor, Dinov2Model

from dreambench_plus.constants import LOCAL_FILES_ONLY, MODEL_ZOOS
from dreambench_plus.utils.comm import all_gather
from dreambench_plus.utils.image_utils import IMAGE_EXT, ImageType, load_image
from dreambench_plus.utils.loguru import logger

_DEFAULT_MODEL_V1: str = "dino_vits8"
_DEFAULT_MODEL_V2: str = MODEL_ZOOS["facebook/dinov2-small"]
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32


class DinoScore:
    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL_V1,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = LOCAL_FILES_ONLY,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = torch.hub.load("facebookresearch/dino:main", model_or_name_path).to(device, dtype=torch_dtype)
        self.model.eval()
        self.processor = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)

        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = [self.processor(i) for i in image]
        inputs = torch.stack(inputs).to(self.device, dtype=self.dtype)
        image_features = self.model(inputs)
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def dino_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
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


class Dinov2Score(DinoScore):
    # NOTE: noqa, in version 1, the performance of the official repository and HuggingFace is inconsistent.
    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL_V2,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = Dinov2Model.from_pretrained(model_or_name_path, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.processor = BitImageProcessor.from_pretrained(model_or_name_path, local_files_only=local_files_only)

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model(inputs["pixel_values"].to(self.device, dtype=self.dtype)).last_hidden_state[:, 0, :]
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features


def multigpu_eval_dino_score(
    image1_dir: str,
    image2_dir: str,
    distributed_state: PartialState | None = None,
    dino_score: DinoScore | Dinov2Score | None = None,
    version: Literal["v1", "v2"] = "v1",
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    if dino_score is None:
        if version == "v1":
            dino_score = DinoScore(device=distributed_state.device)
        elif version == "v2":
            dino_score = Dinov2Score(device=distributed_state.device)
        else:
            raise ValueError(f"Invalid version {version}")

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
        desc=f"Evaluating Dino{version} Score",
        disable=not distributed_state.is_local_main_process,
    )

    with distributed_state.split_between_processes(params) as sub_params:
        score = 0
        for _param in sub_params:
            image1_file, image2_file = _param
            image1, image2 = load_image(image1_file), load_image(image2_file)
            score += dino_score.dino_score(image1, image2)[0]
            pbar.update(1)

    scores = all_gather(score)
    return (sum(scores) / len(image1_files)).item()


def dino_eval(dir1: str, dir2: str, version: Literal["v1", "v2"] = "v1"):
    logger.info(f"Dino{version} Score: {multigpu_eval_dino_score(dir1, dir2, version=version)}")


if __name__ == "__main__":
    fire.Fire(dino_eval)
