import os
from PIL import Image
import numpy as np

### Paths

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../../../data/KEGG_Pathway_Image")
input_dir = os.path.join(data_dir, "Images")
output_dir = os.path.join(data_dir, "new_Images")

if not os.path.exists(data_dir):
  raise FileNotFoundError(f"Data directory not found: {data_dir}! Generate dataset first!")

os.makedirs(output_dir, exist_ok=True)

def find_black_pixel(row):
    """
    Returns the index of the last red pixel or length of the row if no black pixel is found.
    """
    for i in range(len(row)-1, -1, -1):
        if row[i] != 0:
            return i

def get_max_rows_lengths(images_files):
    """
    Gets processed images.
    Returns sorted dictionary {ID, length} where ID is number of the row and length is max length of the row from all of the images.
    """
    max_rows_lengths = {}
    for file in images_files:
        img_path = os.path.join(input_dir, file)
        img = Image.open(img_path)
        img = img.convert("L")
        img_array = np.array(img)
        for row_id, row in enumerate(img_array):
            max_length = find_black_pixel(row)
            if row_id not in max_rows_lengths or max_length > max_rows_lengths[row_id]:
                max_rows_lengths[row_id] = max_length

    return sorted(max_rows_lengths.items(), key=lambda x: x[1], reverse=True)

def group_rows(max_rows_lengths, img_size=224*224, patch_size=16*16):
    """
    Group rows into two groups:
    - RG: first 'n' rows with the longest max length, where 'n' is num of patches in img.
    - B: rest of the rows.
    """
    num_patches = img_size // patch_size
    rg_rows = [row_id for row_id, _ in max_rows_lengths[:num_patches]]
    b_rows = [row_id for row_id, _ in max_rows_lengths[num_patches:]]
    return rg_rows, b_rows

if __name__ == "__main__":
    images = os.listdir(input_dir)
    max_rows_lengths = get_max_rows_lengths(images)
    rg_rows, b_rows = group_rows(max_rows_lengths)