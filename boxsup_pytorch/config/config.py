"""Config Module."""
from pathlib import Path

import toml


class GlobalConfig:
    """Config Class for global config providing."""

    def __init__(self) -> None:
        """Construct class GlobalConfig.

        Reads the example.toml file and sets the values
        """
        __path = Path(__file__).resolve().parent / "example.toml"
        self.__conf_dict = toml.load(__path)

        # DATASET
        self.dataset = self.__conf_dict["DATASET"].get("dataset", "pascal2012")
        self.root = Path(
            self.__conf_dict["DATASET"].get(
                "root", "/mnt/data1/datasets/image/voc/VOCdevkit/VOC2012/"
            )
        )
        self.num_classes = self.__conf_dict["DATASET"].get("num_classes", 20)

        # DATASET.count
        self.train_count = self.__conf_dict["DATASET"]["count"].get("train", 0)
        self.val_count = self.__conf_dict["DATASET"]["count"].get("train", 0)

        # MCG
        self.mcg_num_candidates = self.__conf_dict["MCG"].get("num_candidates", 30)
        self.mcg_path = self.root / self.__conf_dict["MCG"].get("mcg_path", "MCG")
        self.rerun_process = self.__conf_dict["MCG"].get("rerun_process", False)

        # TRAINER
        self.batchsize = self.__conf_dict["TRAINER"].get("batchsize", 20)
        self.image_size = self.__conf_dict["TRAINER"].get("image_size", 224)
        self.epochs = self.__conf_dict["TRAINER"].get("epochs", 80)
        self.num_iter = self.__conf_dict["TRAINER"].get("num_iter", 20)
        self.optimizer = self.__conf_dict["TRAINER"].get("optimizer", "ADAM")
        self.num_worker = self.__conf_dict["TRAINER"].get("num_worker", 4)

        # HYPER
        self.learning_rate = self.__conf_dict["HYPER"].get("lr", 0.001)
        self.hyperparams = dict()
        self.hyperparams["ADAM"] = {
            "betas": self.__conf_dict["HYPER"].get("betas", (0.9, 0.999)),
            "weight_decay": self.__conf_dict["HYPER"].get("weight_decay", 0),
        }
        self.hyperparams["SGD"] = {
            "momentum": self.__conf_dict["HYPER"].get("momentum", 0.9),
            "weight_decay": self.__conf_dict["HYPER"].get("weight_decay", 0),
        }
        self.hyperparams["StepLR"] = {
            "step_size": self.__conf_dict["HYPER"].get("step_size", 7),
            "gamma": self.__conf_dict["HYPER"].get("gamma", 0.1),
        }

        # MASKS
        self.strat = self.__conf_dict["MASKS"].get("strat", "greedy")
