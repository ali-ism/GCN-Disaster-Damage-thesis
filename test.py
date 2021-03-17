import json
import torch
from twostream_resnet50_diff import TwoStream_Resnet50_Diff
from download import download_weights

default_dict = {
        "model": {
            "type": "onestage",
            "name": "twostream_resnet50",
            "model_args": {"in_channels": 6, "n_classes": 5, "seperate_loss": False, "pretrained": False, "output_features": False},
        },
        "data": {
            "data_folder": ['datasets/resize128/','datasets/resize128_3/'],
            "viz_folder": ['datasets/viz/all/'],
            "holdout_folder": ['datasets/hold/all/'],
            "img_size": 128,
            "batch_size": 44,
            "disasters": "all",
            "augment_plus": True,
            "adabn": False,
            "adabn_train": False
        },
        "objective": {
            "name": "CE",
            "params": {
                "weights": [0.1197, 0.7166, 1.2869, 1.0000, 1.3640],
            }
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001,
            "sheduler": {
                "patience": 2,
                "factor": 0.1
            },
            "longshedule": False
        },
        "epochs": 25,
        "seed": 42
}


if __name__ == "__main__":

    setting_name = 'table_1_plain'
    setting_path = "./table_1_plain.json"

    weight_path = download_weights(setting_name)
    
    print("Loading settings")

    with open(setting_path, 'r') as JSON:
        setting_dict = json.load(JSON)

    for k, v in default_dict.items():
        if k not in setting_dict:
            setting_dict[k] = v
        else:
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk not in setting_dict[k]:
                        setting_dict[k][kk] = vv

    # Model

    model_args = setting_dict["model"]["model_args"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = TwoStream_Resnet50_Diff(**model_args)

    model.eval()
    model.load_state_dict(torch.load(weight_path)["state_dict"])
        
    model.to(device)