from pathlib import Path
import numpy as np

from skimage import io
from skimage.transform import resize


TRAIN_INPUT = Path("dataset/train")
TEST_INPUT = Path("dataset/test")

TRAIN_OUTPUT = Path("dataset_resized/train_resized")
TEST_OUTPUT = Path("dataset_resized/test_resized")


def process_images(input_root: Path, output_root: Path, size=(256, 256)):
    for image_path in input_root.rglob("*.png"):

        # Stessa struttura sottocartelle
        rel_path = image_path.relative_to(input_root)
        output_path = output_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Legge il PNG mantenendo dtype e range originali
        image = io.imread(image_path)

        # Resize
        image_resized = resize(
            image,
            size,
            anti_aliasing=True,
            preserve_range=True,
        )
        image_resized = np.rint(image_resized).astype(image.dtype)

        
        io.imsave(output_path, image_resized)


process_images(TRAIN_INPUT,TRAIN_OUTPUT,size=(256, 256))

process_images(TEST_INPUT,TEST_OUTPUT,size=(256, 256))

print("Resize completato.")