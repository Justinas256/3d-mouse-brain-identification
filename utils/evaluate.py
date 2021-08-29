import os, sys

sys.path.insert(0, os.getcwd())

import glob
from utils.metrics import Metrics
from data_loader.base_data_loader import BaseDataLoader
from data_loader.data_loader import TripletDataLoader
from models.base_model import BaseModel
from models.resnet50 import ResNet50V2Model
from models.efficient_net import EfficientNetModel
from models.simple_cnn import SimpleCNNModel
from paths import PATHS


def evaluate(model: BaseModel, data_loader: BaseDataLoader, visualize: bool = False):
    input_shape = data_loader.input_shape
    model.compile_model()
    checkpoint = [
        filename
        for filename in glob.glob(
            f"output/models/{model.get_model_name()}_{input_shape[0]}*"
        )
    ][0]
    model.load(checkpoint)

    # compute MAE
    metrics_val = Metrics(data_loader, model=model, dataset_path=PATHS.TEST_NISSL)
    mae = metrics_val.compute(visualize=visualize)

    return mae


def evaluate_several():
    results_list = []

    # input shape 224
    input_shape = (224, 224, 3)
    data_loader = TripletDataLoader(input_shape=input_shape)
    for model in [
        ResNet50V2Model(input_shape=input_shape, imagenet=False),
        EfficientNetModel(input_shape=input_shape, architecture=0, imagenet=False),
        EfficientNetModel(input_shape=input_shape, architecture=4, imagenet=False),
    ]:
        mae = evaluate(model, data_loader)
        results_list.append((model.get_model_name(), input_shape[0], mae))

    # input shape 380
    input_shape = (380, 380, 3)
    data_loader = TripletDataLoader(input_shape=input_shape)
    model = EfficientNetModel(input_shape=input_shape, architecture=4, imagenet=False)
    mae = evaluate(model, data_loader)
    results_list.append((model.get_model_name(), input_shape[0], mae))

    # input shape 512
    input_shape = (512, 512, 3)
    data_loader = TripletDataLoader(input_shape=input_shape)
    for model in [
        ResNet50V2Model(input_shape=input_shape),
        EfficientNetModel(input_shape=input_shape, architecture=0),
    ]:
        mae = evaluate(model, data_loader)
        results_list.append((model.get_model_name(), input_shape[0], mae))

    print(results_list)


if __name__ == "__main__":
    evaluate_several()
