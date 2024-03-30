import os
from IPython.display import display, Image
import yaml
from roboflow import Roboflow
from src.data.download_dataset import get_and_prepare_dataset

with open("model_config.yaml") as model_config_stream:
    CONFIG_DICT = yaml.safe_load(model_config_stream)
with open(CONFIG_DICT["creds_config_path"]) as creds_stream:
    CREDS = yaml.safe_load(creds_stream)

HOME = os.getcwd()


def initialize_yolo_v8():
    from ultralytics import YOLO

    return YOLO("yolov8x.pt")


def initialize_yolo_v5():
    os.system("git clone https://github.com/ultralytics/yolov5")
    os.system("cd yolov5")
    os.system("pip install -r requirements.txt")

    import torch
    import utils

    os.system(f"cd {HOME}")


def initialize_model(model_type: str, **kwargs):
    if model_type == "yolov8":
        initialize_yolo_v8()
    elif model_type == "yolov5":
        initialize_yolo_v5()


def train(
    model_type: str,
    roboflow_project: str,
    roboflow_project_version: int,
    train_params: dict,
    **kwargs,
):
    dataset_path = get_and_prepare_dataset(
        roboflow_api_key=CREDS["roboflow_api_key"], **CONFIG_DICT
    )
    initialize_model(**CONFIG_DICT)
    project = f"../../models/{roboflow_project}-{roboflow_project_version}-{model_type}"

    TRAINING_CLI_STRING = f"""
    yolo task=detect mode=train model={train_params['model']}
    data={dataset_path}/data.yaml
    epochs={train_params['epochs']}
    patience={train_params['patience']}
    batch={train_params['batch']}
    imgsz={train_params['imgsz']}
    save={train_params['save']}
    optimizer={train_params['optimizer']}
    project={project}
    plots=True
    """

    os.system(TRAINING_CLI_STRING)


if __name__ == "__main__":
    train(**CONFIG_DICT)
