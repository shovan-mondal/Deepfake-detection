# -*- coding: utf-8 -*-
import logging
import shutil
import random
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def process_dataset(input_dir: Path, output_dir: Path):
    """
    Build a perfectly balanced DeepFake dataset (FaceForensics++ + optional CelebDF_V2)
    - Train: 3000 real, 3000 fake
    - Test:  1000 real, 1000 fake
    """

    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting dataset processing for FaceForensics++ format")

    ffpp_dir = input_dir / "FaceForencis++"
    celeba_dir = input_dir / "CelebDF_V2" / "Celeb_V2"
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png"}

    # ✅ Collect all real and fake images recursively
    real_images = [p for p in (ffpp_dir / "real").rglob("*") if p.suffix.lower() in valid_exts]
    fake_images = [p for p in (ffpp_dir / "fake").rglob("*") if p.suffix.lower() in valid_exts]

    logger.info(f"Found {len(real_images)} real and {len(fake_images)} fake images in FaceForencis++")

    if celeba_dir.exists():
        celeb_images = [p for p in celeba_dir.rglob("*") if p.suffix.lower() in valid_exts]
        logger.info(f"Found {len(celeb_images)} images in CelebDF_V2 (optional inclusion)")

    TARGETS = {
        "train": {"real": 3000, "fake": 3000},
        "test": {"real": 1000, "fake": 1000},
    }

    random.seed(42)

    def sample_images(images, target_count):
        if len(images) >= target_count:
            return random.sample(images, target_count)
        else:
            repeated = images * (target_count // len(images))
            remaining = target_count - len(repeated)
            repeated += random.sample(images, remaining)
            return repeated

    random.shuffle(real_images)
    random.shuffle(fake_images)

    # Split and sample
    train_real = sample_images(real_images, TARGETS["train"]["real"])
    train_fake = sample_images(fake_images, TARGETS["train"]["fake"])
    test_real = sample_images(real_images, TARGETS["test"]["real"])
    test_fake = sample_images(fake_images, TARGETS["test"]["fake"])

    for split, real_list, fake_list in [
        ("train", train_real, train_fake),
        ("test", test_real, test_fake),
    ]:
        split_dir = output_dir / split
        # 🧹 clean old data before writing new
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "real").mkdir(parents=True, exist_ok=True)
        (split_dir / "fake").mkdir(parents=True, exist_ok=True)

        def safe_copy(src_path, dest_dir, prefix):
            dest_name = f"{prefix}_{src_path.parent.name}_{src_path.name}"
            dest_path = dest_dir / dest_name
            shutil.copy2(src_path, dest_path)

        # Copy all with unique names
        for src in real_list:
            safe_copy(src, split_dir / "real", "real")
        for src in fake_list:
            safe_copy(src, split_dir / "fake", "fake")

        # Verify counts
        rc = len(list((split_dir / "real").glob("*")))
        fc = len(list((split_dir / "fake").glob("*")))
        logger.info(f"✅ {split.capitalize()} processed: {rc} real, {fc} fake")

    logger.info("🏁 Balanced dataset successfully created in data/processed/")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / "data" / "raw"
    output_path = project_dir / "data" / "processed"

    process_dataset(input_path, output_path)
