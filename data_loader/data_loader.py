from data_loader.base_data_loader import BaseDataLoader
import numpy as np
import random
import imgaug.augmenters as iaa


class TripletDataLoader(BaseDataLoader):
    def __init__(self, input_shape, augmentation: bool = False, batch_size: int = 16):
        super(TripletDataLoader, self).__init__(input_shape)
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.seq = None

    def get_train_data(self):
        if not self.nissl_images:
            self._load_training_dataset()
        train_file_names = list(self.nissl_images.keys())

        while True:
            images = []
            labels = []

            for i in range(self.batch_size):
                while True:
                    atlas_no = random.choice(train_file_names)
                    if atlas_no not in labels:
                        break
                for u in range(2):
                    labels.append(atlas_no)
                images.append(self.avgt_images[atlas_no])
                if self.augmentation:
                    images.append(self.augment_data([self.nissl_images[atlas_no]])[0])
                else:
                    images.append(self.nissl_images[atlas_no])

            yield np.array(images), np.array(labels)

    def get_test_data(self):
        return None

    def augment_data(self, images):
        if not self.seq:
            self.seq = iaa.Sequential(
                [
                    iaa.Affine(rotate=(-10, 10)),
                    iaa.Affine(scale=(1.0, 1.8)),
                    iaa.CropAndPad(percent=(-0.10, 0.10)),
                    iaa.CoarsePepper(0.1, size_percent=(0.01, 0.01)),
                ]
            )
        return self.seq(images=images)
