import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# Define paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Corrected dataset paths
celeb_folder = RAW_DIR / "CelebDF_V2" / "Celeb_V2"
ffpp_folder = RAW_DIR / "FaceForencis++"

# Parameters
SPLITS = {
    "train": 4000,
    "val": 500,
    "test": 500
}

def gather_images(base_path, label_type):
    """Collect all image file paths (jpg, png, jpeg) recursively."""
    exts = (".jpg", ".png", ".jpeg")
    image_list = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(exts):
                image_list.append(Path(root) / f)
    print(f"Found {len(image_list):,} {label_type} images in {base_path}")
    return image_list

def prepare_dir_structure():
    """Create train/val/test and fake/real folders."""
    for split in SPLITS:
        for label in ["real", "fake"]:
            out_dir = PROCESSED_DIR / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

def copy_samples(images, split, label, n):
    """Copy sampled images to the processed folder."""
    out_dir = PROCESSED_DIR / split / label
    samples = random.sample(images, min(n, len(images)))
    for img_path in tqdm(samples, desc=f"Copying {split}/{label}"):
        shutil.copy(img_path, out_dir / img_path.name)

def main():
    print("Combining datasets...")

    # CelebDF structure is already divided into train/val/test
    celeb_real = gather_images(celeb_folder / "Train" / "real", "CelebDF real") + \
                 gather_images(celeb_folder / "Val" / "real", "CelebDF real") + \
                 gather_images(celeb_folder / "Test" / "real", "CelebDF real")

    celeb_fake = gather_images(celeb_folder / "Train" / "fake", "CelebDF fake") + \
                 gather_images(celeb_folder / "Val" / "fake", "CelebDF fake") + \
                 gather_images(celeb_folder / "Test" / "fake", "CelebDF fake")

    # FaceForensics++: real images are inside subfolders 000_001_..., same for fake
    ffpp_real = gather_images(ffpp_folder / "real", "FF++ real")
    ffpp_fake = gather_images(ffpp_folder / "fake", "FF++ fake")

    # Combine all
    real_images = celeb_real + ffpp_real
    fake_images = celeb_fake + ffpp_fake

    print(f"Total real images found: {len(real_images):,}")
    print(f"Total fake images found: {len(fake_images):,}")

    prepare_dir_structure()

    # Randomize and split
    random.shuffle(real_images)
    random.shuffle(fake_images)

    for split, n in SPLITS.items():
        copy_samples(real_images, split, "real", n)
        copy_samples(fake_images, split, "fake", n)

    print(f"\n✅ Combined dataset ready at: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()