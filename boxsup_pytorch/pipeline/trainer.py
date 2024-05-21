"""Module trainer.py.

copyright Matti Kaupenjohann, 2023
"""

import copy
import time
from typing import Any, Dict, Tuple, Union

import matplotlib
import torch
from torch import nn
from torch.backends import cudnn
from torch.optim import lr_scheduler

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.core import optimizers
from boxsup_pytorch.data.dataloader import BoxSupDataloader
from boxsup_pytorch.model.network import FCN8s
from boxsup_pytorch.utils.common import print_progress_bar
from boxsup_pytorch.utils.losses import get_segmentation_loss

matplotlib.use("Agg")


class Trainer:
    """Trainer class which handles the data and train the network."""

    def __init__(self, model: nn.Module) -> None:
        """Construct the trainer class.

        Args:
            model (nn.Module): Networkmodel (FCN8s)
        """
        self.model = model
        self.optimizer = optimizers.provider.create(GLOBAL_CONFIG.optimizer)(
            model.parameters(),
            GLOBAL_CONFIG.learning_rate,
            **GLOBAL_CONFIG.hyperparams[GLOBAL_CONFIG.optimizer],
        )
        self.criterion = get_segmentation_loss()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, **GLOBAL_CONFIG.hyperparams["StepLR"])
        self.loader = BoxSupDataloader(type="NET")
        self.dataloaders = self.loader.get_data_loader()
        self.dataset_sizes = {"train": 0, "val": 0}

    def train_model(self) -> nn.Module:
        """Train the network based on the parameters provided by the config.

        Returns:
            _type_: _description_
        """
        since = time.time()

        device = _device()
        self.model = self.model.to(device)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        self.dataset_sizes["train"] = len(self.loader.train_dataset)
        self.dataset_sizes["val"] = len(self.loader.val_dataset)

        for epoch in range(GLOBAL_CONFIG.epochs):
            print(f"Epoch {epoch}/{GLOBAL_CONFIG.epochs - 1}")
            print("-" * 10)

            self.model.train()
            # self.model.eval()

            self.run_train_epoch(device=device)

            if (epoch + 1) % 10 == 0:
                self.model.eval()
                new_model_wts = self.run_val_epoch(device=device, best_acc=best_acc)
                if new_model_wts:
                    best_model_wts = new_model_wts

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def run_train_epoch(self, device: torch.device) -> None:
        """Run a single epoch.

        Args:
            phase (str): _description_
            device (torch.device): _description_

        Returns:
            Tuple[float, float]: epoch_loss, epoch_acc
        """
        phase = "train"

        self.iterate_dataset(phase=phase, device=device)

    def run_val_epoch(self, device: torch.device, best_acc: float) -> Union[Dict[str, Any], None]:

        phase = "val"

        epoch_results = self.iterate_dataset(phase=phase, device=device)
        epoch_acc = epoch_results[1]

        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            return best_model_wts

    def iterate_dataset(self, phase: str, device: torch.device) -> Tuple[float, float]:
        running_loss = 0.0
        running_corrects = 0

        for idx, (inputs, gt_label, gt_pseudo) in enumerate(self.dataloaders[phase]):
            inputs = inputs.to(device)
            gt_label = gt_label.to(device)
            gt_pseudo = gt_pseudo.to(device)

            self.optimizer.zero_grad()

            # Forward Propagation
            with torch.set_grad_enabled(phase == "train"):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs[0], 1)

                if phase == "train":
                    loss = self.criterion(outputs, gt_pseudo)
                    loss["loss"].backward()
                    self.optimizer.step()
                else:
                    loss = self.criterion(outputs, gt_label)

            # statistics
            running_loss += loss["loss"].item() * inputs.size(0)
            running_corrects += torch.sum(preds == gt_label.data)
            print_progress_bar(
                idx,
                len(self.dataloaders[phase]) - 1,
                title=f"RUN {phase}: {idx}/{len(self.dataloaders[phase]) - 1}",
            )

        if phase == "train":
            self.scheduler.step()

        epoch_loss = float(running_loss / self.dataset_sizes[phase])
        # Calculate Accuracy
        image_size = self.loader.image_size
        epoch_acc = float(running_corrects / (self.dataset_sizes[phase] * image_size**2))

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        return (epoch_loss, epoch_acc)


def _device():
    if torch.cuda.is_available():
        cudnn.benchmark = True
        use_device = "cuda"
    else:
        use_device = "cpu"
    return torch.device(use_device)


if __name__ == "__main__":
    model = FCN8s(nclass=21, device=_device())
    trainer = Trainer(model)

    trainer.train_model()
