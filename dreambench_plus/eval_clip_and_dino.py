import os

import fire
from accelerate import PartialState

from dreambench_plus.metrics.clip_score import multigpu_eval_clipi_score, multigpu_eval_clipt_score
from dreambench_plus.metrics.dino_score import multigpu_eval_dino_score
from dreambench_plus.utils.loguru import logger


def eval_clip_and_dino(dir):
    distributed_state = PartialState()
    dinov1_score = multigpu_eval_dino_score(
        os.path.join(dir, "src_image"),
        os.path.join(dir, "tgt_image"),
        distributed_state=distributed_state,
        version="v1",
    )
    dinov2_score = multigpu_eval_dino_score(
        os.path.join(dir, "src_image"),
        os.path.join(dir, "tgt_image"),
        distributed_state=distributed_state,
        version="v2",
    )
    clipi_score = multigpu_eval_clipi_score(
        os.path.join(dir, "src_image"),
        os.path.join(dir, "tgt_image"),
        distributed_state=distributed_state,
    )
    clipt_score = multigpu_eval_clipt_score(
        os.path.join(dir, "text"),
        os.path.join(dir, "tgt_image"),
        distributed_state=distributed_state,
    )
    logger.info(f"DINOv1 score: {dinov1_score}")
    logger.info(f"DINOv2 score: {dinov2_score}")
    logger.info(f"CLIP-I score: {clipi_score}")
    logger.info(f"CLIP-T score: {clipt_score}")


if __name__ == "__main__":
    fire.Fire(eval_clip_and_dino)
