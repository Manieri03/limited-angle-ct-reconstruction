from pathlib import Path

import numpy as np
from skimage import io

#Funzione di normalizzazione in [0,1]
def load_normalized_image(path: Path):
    image = io.imread(path)
    return image.astype(np.float32) / 255.0