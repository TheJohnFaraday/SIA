import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from PIL import Image


def mc_matrix(subset: int = 8):
    images_folder = "./dataset/minecraft-faces"
    images_shape = (20, 20)
    x = []

    images = [f for f in os.listdir(images_folder) if f.endswith(".png")]
    processed = 0

    for image in images[:subset]:
        full_path = os.path.join(images_folder, image)
        img = Image.open(full_path)
        img = img.convert("L")
        img = img.resize(images_shape)
        img = (np.asarray(img, dtype=np.float64) - 127.5) / 127.5
        x.append(img)
        processed += 1

    print(f"Loaded {processed} out of {len(images)} images with shape {images_shape}")

    i = 0
    for xi in x:
        plt.imshow(xi, cmap="Blues")
        plt.axis("off")
        plt.savefig(f"./plots/example-fig-{i}.png")
        i += 1

    x = np.reshape(x, (processed, images_shape[0] * images_shape[1], 1))
    return x
