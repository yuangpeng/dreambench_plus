import glob
import os
from dataclasses import dataclass
from typing import Literal

from torch.utils.data import Dataset

from dreambench_plus.constants import DREAMBENCH_PLUS_DIR
from dreambench_plus.utils.image_utils import ImageType, load_image


@dataclass
class CollectionInfo:
    collection_id: str  # {category}_{index}_{subject}
    category: Literal["animal", "human", "object", "style"]
    subject: str
    image: ImageType
    captions: list[str]
    image_path: str
    caption_path: str


class DreamBenchPlus(Dataset):

    def __init__(self, dir: str = DREAMBENCH_PLUS_DIR):
        super().__init__()
        self.dir = os.path.abspath(dir)

        self.image_files = glob.glob(f"{os.path.join(self.dir, 'images')}/**/*.jpg", recursive=True)
        self.image_files = sorted(self.image_files)
        self.images = [load_image(image) for image in self.image_files]

        self.caption_files = glob.glob(f"{os.path.join(self.dir, 'captions')}/**/*.txt", recursive=True)
        self.caption_files = sorted(self.caption_files)

        self.captions = []
        self.subject = []

        self.collection_id = []

        for file, image_file in zip(self.caption_files, self.image_files):
            if file.split("captions")[-1].split(".")[0] != image_file.split("images")[-1].split(".")[0]:
                raise ValueError(f"Image and caption file mismatch: {file} != {image_file}.")

            with open(file, "r") as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
            self.captions.append(lines[1:])
            self.subject.append(lines[0])

            _category = file.split("/")[-2]
            if _category == "animal" or _category == "human":
                _category = f"live_subject_{_category}"
            _index = file.split("/")[-1].split(".")[0]
            _subject = lines[0].lower().replace(" ", "_")
            self.collection_id.append("_".join([_category, _index, _subject]))

    @property
    def collections(self):
        _collections = {
            self.collection_id[i]: CollectionInfo(
                collection_id=self.collection_id[i],
                category=self.image_files[i].split("/")[-2],
                subject=self.subject[i],
                image=self.images[i],
                captions=self.captions[i],
                image_path=self.image_files[i],
                caption_path=self.caption_files[i],
            )
            for i in range(len(self.images))
        }
        _collections = dict(sorted(_collections.items()))
        return _collections

    @property
    def collection_list(self):
        return list(self.collections.values())

    def __len__(self):
        return len(self.collections)

    def __getitem__(self, idx):
        return self.collection_list[idx]
