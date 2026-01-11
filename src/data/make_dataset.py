import logging
import shutil
import random
import os
from pathlib import Path
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv

def gather_images_recursive(base_path, valid_exts=(".jpg", ".jpeg", ".png")):
    """Helper to recursively find images."""
    if not base_path.exists():
        return []
    return [p for p in base_path.rglob("*") if p.suffix.lower() in valid_exts]

def process_dataset(input_dir: Path, output_dir: Path):
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Universal Dataset Merge (FF++ / CelebDF / CIFAKE)")

    # Define Paths
    ffpp_dir = input_dir / "FaceForencis++"
    celeb_dir = input_dir / "CelebDF_V2" 
    cifake_dir = input_dir / "CIFAKE"

    # --- 1. HARVEST IMAGES ---
    real_images = []
    fake_images = []

    # A. FaceForensics++
    if ffpp_dir.exists():
        logger.info("📂 Harvesting FaceForensics++...")
        ff_real = gather_images_recursive(ffpp_dir / "real")
        ff_fake = gather_images_recursive(ffpp_dir / "fake")
        # Tag them so we know source later (path, prefix)
        real_images.extend([(p, "ffpp") for p in ff_real])
        fake_images.extend([(p, "ffpp") for p in ff_fake])

    # B. Celeb-DF v2
    if celeb_dir.exists():
        logger.info("📂 Harvesting Celeb-DF v2...")
        # Check specific folders mentioned in your script
        c_real = gather_images_recursive(celeb_dir / "Celeb_V2" / "Celeb-real") + \
                 gather_images_recursive(celeb_dir / "Celeb_V2" / "YouTube-real")
        c_fake = gather_images_recursive(celeb_dir / "Celeb_V2" / "Celeb-synthesis")
        
        # Fallback if your folder structure is flatter (like in your snippet)
        if not c_real: 
             c_real = gather_images_recursive(celeb_dir / "Celeb_V2" / "Train" / "real") + \
                      gather_images_recursive(celeb_dir / "Celeb_V2" / "Val" / "real") + \
                      gather_images_recursive(celeb_dir / "Celeb_V2" / "Test" / "real")
        if not c_fake:
             c_fake = gather_images_recursive(celeb_dir / "Celeb_V2" / "Train" / "fake") + \
                      gather_images_recursive(celeb_dir / "Celeb_V2" / "Val" / "fake") + \
                      gather_images_recursive(celeb_dir / "Celeb_V2" / "Test" / "fake")

        real_images.extend([(p, "celeb") for p in c_real])
        fake_images.extend([(p, "celeb") for p in c_fake])

    # C. CIFAKE
    if cifake_dir.exists():
        logger.info("📂 Harvesting CIFAKE...")
        # Case insensitive search for REAL/FAKE folders
        all_cifake = gather_images_recursive(cifake_dir)
        for p in all_cifake:
            parent_name = p.parent.name.upper()
            if "REAL" in parent_name:
                real_images.append((p, "cifake"))
            elif "FAKE" in parent_name:
                fake_images.append((p, "cifake"))

    # --- 2. VALIDATE ---
    total_real = len(real_images)
    total_fake = len(fake_images)
    logger.info(f"📊 Total Found: {total_real} Real, {total_fake} Fake")
    
    if total_real == 0 or total_fake == 0:
        raise ValueError("❌ No images found. Check raw paths.")

    # --- 3. SHUFFLE & SPLIT (80/10/10) ---
    # This prevents Data Leakage by ensuring exclusive sets
    random.seed(42)
    random.shuffle(real_images)
    random.shuffle(fake_images)

    def get_splits(data_list):
        total = len(data_list)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)
        return {
            "train": data_list[:train_end],
            "val": data_list[train_end:val_end],
            "test": data_list[val_end:]
        }

    splits_real = get_splits(real_images)
    splits_fake = get_splits(fake_images)

    # --- 4. WRITE TO DISK ---
    # Clean output directory first
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        logger.info(f"💾 Writing {split} set...")
        
        # Create directories
        (output_dir / split / "real").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "fake").mkdir(parents=True, exist_ok=True)

        # Helper to copy with rename
        def copy_files(file_list, label):
            dest_dir = output_dir / split / label
            for (src_path, prefix) in tqdm(file_list, desc=f"{split}/{label}", leave=False):
                # Rename to avoid collisions: "cifake_image001.jpg"
                new_name = f"{prefix}_{src_path.name}"
                shutil.copy2(src_path, dest_dir / new_name)

        copy_files(splits_real[split], "real")
        copy_files(splits_fake[split], "fake")

    logger.info("🏁 Dataset Generation Complete!")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())

    import sys
    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / "data" / "raw"
    output_path = project_dir / "data" / "processed"

    process_dataset(input_path, output_path)