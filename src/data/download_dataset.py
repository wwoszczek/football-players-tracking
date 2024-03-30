import os
from roboflow import Roboflow
import yaml

with open("model_config.yaml") as model_config_stream:
    CONFIG_DICT = yaml.safe_load(model_config_stream)
with open(CONFIG_DICT["creds_config_path"]) as creds_stream:
    CREDS = yaml.safe_load(creds_stream)

HOME = os.getcwd()


def get_and_prepare_dataset(
    roboflow_api_key: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_project_version: int,
    model_type: str,
    **kwargs,
):
    if not os.path.exists(f"../../data/{roboflow_project}-{roboflow_project_version}"):
        rf = Roboflow(api_key=roboflow_api_key)
        project = rf.workspace(roboflow_workspace).project(roboflow_project)
        version = project.version(roboflow_project_version)
        dataset = version.download(
            model_type,
            location=f"../../data/{roboflow_project}-{roboflow_project_version}",
        )
        return dataset.location
    else:
        return f"../../data/{roboflow_project}-{roboflow_project_version}"


if __name__ == "__main__":
    get_and_prepare_dataset(roboflow_api_key=CREDS["roboflow_api_key"], **CONFIG_DICT)
