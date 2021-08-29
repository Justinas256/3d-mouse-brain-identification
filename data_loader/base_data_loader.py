from utils.image import (
    load_image,
    get_sub_dir_image_paths,
)
import os
from paths import PATHS


class BaseDataLoader(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

        self._load_allen_atlas()
        self.nissl_images = None

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError

    def augment_data(self, images):
        raise NotImplementedError

    # load allen mouse brain average template atlas (2D plates)
    def _load_allen_atlas(self):
        # loading images
        print("Loading Allen Mouse brain atlas images...")
        self.avgt_images = self._get_images(PATHS.ALLEN_AVGT)
        total_images = len(list(self.avgt_images.keys()))
        if total_images > 0:
            print("Loaded. Number of images: ", total_images)
        else:
            raise Exception("No images were found in dir ", PATHS.ALLEN_AVGT)

    # load training dataset (Nissl images)
    def _load_training_dataset(self):
        print("Loading training dataset...")
        self.nissl_images = self._get_images(PATHS.TRAIN_NISSL)
        total_images = len(list(self.nissl_images.keys()))
        if total_images > 0:
            print("Loaded. Number of images: ", total_images)
        else:
            raise Exception("No images were found in dir ", PATHS.TRAIN_NISSL)

    def _get_images(self, dir: str):
        """
        Load and process (pad, resize) images
        :param dir: Path to the directory where images are located
        :return: dict[slice_number] = image
        """
        img_dict = {}

        for path in get_sub_dir_image_paths(dir):
            slice_no = os.path.basename(path).split(".")[0]
            img_dict[slice_no] = load_image(path, input_shape=self.input_shape)

        return img_dict
