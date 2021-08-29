import os
import glob
import nrrd as nrr
import numpy as np
from tqdm import tqdm
from scipy.ndimage import rotate as rot
from skimage.exposure import equalize_hist, equalize_adapthist
from PIL import Image

from image import read_image, get_padded_image


def save_img(img, folder, plate_no, degree):
    output_path = os.path.join("data", str(folder), str(plate_no))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file_name = f"{plate_no}_{degree}.jpg"
    output_path = os.path.join(output_path, output_file_name)
    if "nissl" in folder:
        im = Image.fromarray(img / 255.0)
    else:
        im = Image.fromarray(img)
    im.convert("L").save(output_path, "JPEG", quality=100)


def get_2D_slices(atlas: str):
    """
    Slice and save 2D plates at different angles
    :param atlas: "avg" or "nissl"
    """
    if atlas == "avg":
        AVGT, metaAVGT = nrr.read("nrrd/average_template_10.nrrd")
        folder = "avg"
    elif atlas == "nissl":
        AVGT, metaAVGT = nrr.read("nrrd/ara_nissl_10.nrrd")
        folder = "nissl"
    else:
        raise Exception("Wrong argument for var atlas")

    degress = [i for i in range(-15, 16, 3)]

    for degree in tqdm(degress):
        rotated = rot(AVGT, angle=degree, mode="nearest", order=0, reshape=True)
        print(degree)
        for plate_no in tqdm(range(300, 1300, 5)):
            save_img(rotated[plate_no], folder=folder, plate_no=plate_no, degree=degree)


def preprocess(img):
    def crop(image):
        y_nonzero, x_nonzero = np.nonzero(image)
        return image[
            np.min(y_nonzero) : np.max(y_nonzero), np.min(x_nonzero) : np.max(x_nonzero)
        ]

    def equalize(image, bins=255, clip_limit=0.01):
        image = equalize_adapthist(image, nbins=bins, clip_limit=clip_limit)
        return image

    def pad_imgage(img):
        return get_padded_image(img)

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def resize_proportional(image, size=512):
        image = Image.fromarray(image)

        w, h = image.size
        # max_size = max(w, h)
        max_size = size

        if float(w) < max_size and float(h) < max_size:
            raise Exception(f"Both dims lower than {str(max_size)}, {str(w)}, {str(h)}")
            # raise Exception('Both dims lower than ', str(max_size), str(w), str(h))

        if h >= w:
            height_percent = max_size / float(h)
            width_size = int((float(w) * float(height_percent)))
            image = image.resize((width_size, max_size), Image.BICUBIC)
        else:
            width_percent = max_size / float(w)
            height_size = int((float(h) * float(width_percent)))
            image = image.resize((max_size, height_size), Image.BICUBIC)

        return np.array(image)

    img[img < 15] = 0
    img = crop(img)
    img = equalize(img, clip_limit=0.05) * 255
    # img = NormalizeData(img)
    img = pad_imgage(img)
    img = resize_proportional(img)
    return img


def preprocess_all(input_path="data/avg/*/*.jpg"):
    for file_path in tqdm(glob.glob(input_path)):
        img = read_image(file_path)
        try:
            img = preprocess(img)
        except Exception as err:
            print(f"{file_path}: {str(err)}")
            continue

        output_path = file_path.replace("nissl", "nissl_processed")
        output_path = output_path.replace("avg", "avg_processed")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        im = Image.fromarray(img)
        im.convert("L").save(output_path, "JPEG", quality=100)


if __name__ == "__main__":
    # # get 2D plates from 3D ref atlas
    # atlas = "avg"
    # get_2D_slices(atlas)

    # preprocess 2D atlas plates
    preprocess_all()
