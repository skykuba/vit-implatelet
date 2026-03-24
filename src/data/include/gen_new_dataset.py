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

def get_image_array(file, scale="L"):
    """
    Loads an image and converts it to a numpy array.
    """
    image_path = os.path.join(input_dir, file)
    img = Image.open(image_path)
    img = img.convert(scale)
    return np.array(img)

def find_last_red_pixel(row):
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
        img_array = get_image_array(file)
        for row_id, row in enumerate(img_array):
            max_length = find_last_red_pixel(row)
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

def generate_RG_patch_from_row(row, patch_length=16):
    """
    Generates patch of size patch_length x patch_length from the given row in RG channels.
    Firstly fills the patch with red pixels from left-top. 
    If there is no space for red pixels switch to green channel and fill patch from left-top with green pixels.
    """
    patch = np.zeros((patch_length, patch_length, 3), dtype=int)
    length = find_last_red_pixel(row)

    # RED CHANNEL
    for y in range(patch_length):
        for x in range(patch_length):
            # y*patch_length + x is the index of the pixel in the row; to 255
            if y*patch_length + x <= length:
                patch[y, x, 0] = row[y*patch_length + x]
            else:
                return patch
            
    if length > patch_length*patch_length - 1:
        # GREEN CHANNEL
        for y in range(patch_length):
            for x in range(patch_length):
                # patch_length**2 + y*patch_length + x is the index of the pixel in the row; from 256
                if patch_length**2 + y*patch_length + x <= length:
                    patch[y, x, 1] = row[patch_length**2 + y*patch_length + x]
                else:
                    return patch


if __name__ == "__main__":
    images = os.listdir(input_dir)
    max_rows_lengths = get_max_rows_lengths(images)
    rg_rows, b_rows = group_rows(max_rows_lengths)