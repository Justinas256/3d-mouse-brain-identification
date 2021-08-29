from tqdm import tqdm
import numpy as np
import os, sys, glob

sys.path.insert(0, os.getcwd())

from trainers.base_trainer import BaseTrain
from models.base_model import BaseModel
from data_loader.base_data_loader import BaseDataLoader
from data_loader.data_loader import TripletDataLoader
from models.resnet50 import ResNet50V2Model
from models.efficient_net import EfficientNetModel
from utils.metrics import Metrics
from paths import PATHS
from utils.cuda import set_cuda_memory
from utils.helper import create_folder_if_not_exists


class MainTrain(BaseTrain):
    def __init__(self, model: BaseModel, data_loader: BaseDataLoader):
        super().__init__(model, data_loader)

    def train(self, iters: int = 20000):
        self.model.compile_model()
        metrics = Metrics(
            self.data_loader, model=self.model, dataset_path=PATHS.VAL_NISSL
        )

        # variables for saving models and logs
        save_dir = "output"
        base_file_name = (
            f"{self.model.get_model_name()}_{self.data_loader.input_shape[0]}"
        )
        log_path = os.path.join(save_dir, "logs", f"{base_file_name}.txt")
        create_folder_if_not_exists(log_path)

        # store logs of loss and mean absolute error
        losses = []
        avg_losses = []
        mae_val_list = []
        best_mae_list = []
        best_mae = None

        print(
            f"Training settings: input shape: {self.model.input_shape}, network: {self.model.get_model_name()}"
        )

        for i in tqdm(range(iters)):
            x, y = next(self.data_loader.get_train_data())
            loss = self.model.model.train_on_batch(x, y)
            losses.append(loss)

            if i % 5 == 0 and i != 0:
                avg_losses.append(np.mean(losses[-100:]))
                print(f"Average train loss in last 100 iterations: {avg_losses[-1]}")

                # compute Mean Absolute Error
                mae_val = metrics.compute()

                # if mae improved
                if not best_mae or mae_val < best_mae:
                    best_mae = mae_val
                    # delete existing models
                    for filename in glob.glob(f"{save_dir}/models/{base_file_name}*"):
                        print(f"Deleting {filename}")
                        os.remove(filename)
                    # save a new model
                    path = os.path.join(
                        save_dir,
                        "models",
                        base_file_name + f"_{best_mae}.hdf5",
                    )
                    create_folder_if_not_exists(path)
                    self.model.save(path)

                best_mae_list.append(best_mae)
                mae_val_list.append(mae_val)

                # save logs
                f = open(log_path, "w")
                for u in range(len(avg_losses)):
                    f.write(f"{avg_losses[u]}; {mae_val_list[u]}; {best_mae_list[u]}\n")
                f.close()


def train_several_models(
    freeze: bool = False, augmentation: bool = False, imagenet: bool = True
):
    # input shape 224
    input_shape = (224, 224, 3)
    data_loader = TripletDataLoader(input_shape=input_shape, augmentation=augmentation)
    for model in [
        ResNet50V2Model(input_shape=input_shape, freeze=freeze, imagenet=imagenet),
        EfficientNetModel(
            input_shape=input_shape, architecture=0, freeze=freeze, imagenet=imagenet
        ),
        EfficientNetModel(
            input_shape=input_shape, architecture=4, freeze=freeze, imagenet=imagenet
        ),
    ]:
        trainer = MainTrain(model=model, data_loader=data_loader)
        trainer.train()

    # input shape 380
    input_shape = (380, 380, 3)
    data_loader = TripletDataLoader(
        input_shape=input_shape, augmentation=augmentation, batch_size=15
    )
    model = EfficientNetModel(
        input_shape=input_shape, architecture=4, freeze=freeze, imagenet=imagenet
    )
    trainer = MainTrain(model=model, data_loader=data_loader)
    trainer.train()

    # input shape 512
    input_shape = (512, 512, 3)
    data_loader = TripletDataLoader(input_shape=input_shape, augmentation=augmentation)
    for model in [
        ResNet50V2Model(input_shape=input_shape, freeze=freeze, imagenet=imagenet),
        EfficientNetModel(
            input_shape=input_shape, architecture=0, freeze=freeze, imagenet=imagenet
        ),
    ]:
        trainer = MainTrain(model=model, data_loader=data_loader)
        trainer.train()


def fine_tune(checkpoint: str):
    input_shape = (512, 512, 3)
    data_loader = TripletDataLoader(input_shape=input_shape, batch_size=8)
    model = ResNet50V2Model(
        input_shape=input_shape, freeze=True, weights_path=checkpoint
    )
    trainer = MainTrain(model=model, data_loader=data_loader)
    trainer.train(iters=3000)


if __name__ == "__main__":
    set_cuda_memory()
    train_several_models(imagenet=False)
    # fine_tune(checkpoint='output_augmented/models/ResNet50v2_512_2.94.hdf5')
    # fine_tune(checkpoint='output_imagenet/models/ResNet50v2_512_0.46.hdf5')
