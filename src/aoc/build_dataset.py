import os

import pandas as pd
import skimage.io
import skimage.feature
import skimage.transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv

from aoc.utils import (
    extract_coin,
    gaussuian_mask,
    convolve_mask,
)

# READ CSV
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "data"

df = pd.read_csv(os.path.join(current_dir, data_dir, "items.csv"), sep=";")
size_before = df.shape[0]
df.dropna(inplace=True, ignore_index=True)
assert df.shape[0] == size_before
df = df.astype({"x1": int, "y1": int, "x2": int, "y2": int})


# RESIZE
resize_shape = (200, 200)
# HIGH PASS FILTER
mask_sigma = 15
# HISTOGRAM OF ORIENTATED GRADIENTS
orientations = 8
pixels_per_cell = (16, 16)
cells_per_block = (3, 3)

# PREPARE DATASET FOR ROTATIONS
rotations = list(range(0, 360, 45))
rotations_df = pd.DataFrame(rotations, columns=["degrees"])
df = pd.merge(df, rotations_df, how="cross")

# PREPARE HIGH PASS FILTER MASK
mask = gaussuian_mask(resize_shape, mask_sigma)

features_list = []

loaded_image = None
loaded_image_path = None

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    bounding_box = (row["x1"], row["y1"], row["x2"], row["y2"])
    degrees = row["degrees"]

    image_path = os.path.join(current_dir, data_dir, row["img_dir"])
    # LOAD IMAGE BUT ONLY IF NECESSARY
    if image_path != loaded_image_path:
        loaded_image = cv.imread(image_path)
        loaded_image_path = image_path
        # BGR to RGB
        loaded_image = loaded_image[:, :, ::-1]
        # EXTRACT COIN AND CONVERT TO GRAYSCALE
        loaded_image = extract_coin(loaded_image, bounding_box, resize_shape)
        # APPLY HIGH PASS FILTER
        loaded_image = convolve_mask(loaded_image, mask)

    # APPLY ROTATION
    image = skimage.transform.rotate(loaded_image, degrees)

    # EXTRACT FEATURES
    hog_features = skimage.feature.hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        channel_axis=None,
    )
    features_list.append(hog_features)

df["features"] = features_list

df.to_parquet(os.path.join(current_dir, data_dir, "dataset.parquet"))