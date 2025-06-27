from tqdm import tqdm
from glob import glob

import pandas as pd
import numpy as np
import cv2
import os

DIR_DATA = "data"
DIR_DEBUG = "debug"
DIR_MASKS = "masks"

EXT_DATA = ".JPG"
EXT_MASK = ".png"

HIGHLIGHT_COLOR = (50, 50, 130)


def sort_by_row_and_column(mask_cnt) -> int:
    cx, cy, cw, ch = cv2.boundingRect(mask_cnt)
    return cx + 4 * cy


os.makedirs(DIR_DEBUG, exist_ok=True)

png_files = [f for f in os.listdir(DIR_MASKS) if f.lower().endswith(EXT_MASK)]
filenames = [os.path.splitext(f)[0] for f in png_files]
mask_file = None

while True:
    print("Choose one of the following files by number:")
    for idx, name in enumerate(filenames):
        print(f"{idx}: {name}")

    if (choice := input("\nType the number of your choice: ").strip()).isdigit():
        index = int(choice)
        if 0 <= index < len(filenames):
            mask_file = filenames[index]
            print(f"You selected: {mask_file}{EXT_MASK}")
            break

    print("You wrote an invalid input, please try again.\n")

image_mask = cv2.imread(os.path.join(DIR_MASKS, mask_file + EXT_MASK), cv2.IMREAD_GRAYSCALE)
image_mask_contours, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_mask_contours = sorted(image_mask_contours, key=sort_by_row_and_column)

df_f, df_x, df_y = [], [], []

for i in tqdm(glob(os.path.join(DIR_DATA, f"*{EXT_DATA}"))):
    image = cv2.imread(i)
    image = cv2.medianBlur(image, 15)  # Blur out bubbles

    # Otsu threshold
    image_plate = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Detect contours
    image_plate_contours, _ = cv2.findContours(image_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find top left, top right, bottom left and bottom right
    x_min, x_max, y_min, y_max = image.shape[1], 0, image.shape[0], 0
    for cnt in image_plate_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if x < x_min:
            x_min = x
        if x + w > x_max:
            x_max = x + w
        if y < y_min:
            y_min = y
        if y + h > y_max:
            y_max = y + h

    # cropped_bgr = image[y_min:y_max, x_min:x_max]
    cropped_bgr = image[y_min:y_min + image_mask.shape[0], x_min:x_min + image_mask.shape[1]]
    cropped_bgr = cv2.bitwise_and(cropped_bgr, cropped_bgr, mask=image_mask)
    cropped_hsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)

    # Select for saturated color that is not blue, remove noise
    threshold = cv2.inRange(cropped_hsv, (15, 90, 30), (100, 255, 255))
    threshold = cv2.dilate(cv2.erode(threshold, None, iterations=3), None, iterations=3)

    cropped_bgr[threshold >= 255] = HIGHLIGHT_COLOR

    # Find contours
    for n, cnt in enumerate(image_mask_contours):
        n_human = n + 1

        x, y, w, h = cv2.boundingRect(cnt)

        mask_selection = np.zeros(threshold.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_selection, [cnt], -1, 255, cv2.FILLED)

        to_count = cv2.bitwise_and(threshold, threshold, mask=mask_selection)
        counted = cv2.countNonZero(to_count)

        cv2.putText(cropped_bgr, str(n_human), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, HIGHLIGHT_COLOR, 2)

        df_f.append(i)
        df_x.append(n_human)
        df_y.append(counted)

    cv2.imwrite(f"{os.path.join(DIR_DEBUG, i.split('/')[-1])}_debug.jpg", cropped_bgr)

pd.DataFrame({"file": df_f, "index": df_x, "pixel_count": df_y}).to_csv("result.csv", index=False)
