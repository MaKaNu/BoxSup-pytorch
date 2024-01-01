"""Dataloader Module.

TODO:
- make it faster
- some ideas:
    - https://stackoverflow.com/questions/64694489
"""

from typing import Dict

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import yaml

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.core.dataset_factory import dataset_factory


class BoxSupDataloader:
    """BoxSup initiate training class."""

    def __init__(self, type: str) -> None:
        """Initialize instance of BoxSupDataloader.

        Args:
            type (str): defines which kind of dataloader should be initilaized
                options: ["ALL", "NET", "MASK", "IOU"]

        Raises:
            ValueError: raised if invalid/not implemented type
        """
        if type not in ["ALL", "NET", "MASK", "IOU"]:
            raise ValueError(f"type {type} is not correct, allowed is 'ALL', 'NET', 'MASK', 'IOU'")
        self.dataset_root = GLOBAL_CONFIG.root
        self.image_size = GLOBAL_CONFIG.image_size
        dataset_mean_file = self.dataset_root / "ImageSets/BoxSup/mean_std.yml"

        # Load mean and std of dataset
        with open(dataset_mean_file, "r") as file:
            statistic_data = yaml.safe_load(file)
        mean = statistic_data["mean"]
        std = statistic_data["std"]
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.target_transform = transforms.Compose(
            [
                # WOW SUCH IMPORTANT THIS INTERPOLATIONMODE WHAHAHAHAHAAHAHAHA
                transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
                transforms.CenterCrop(self.image_size),
            ]
        )

        self.train_dataset = dataset_factory.get_dataset(type)(
            self.dataset_root,
            type="train",
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.val_dataset = dataset_factory.get_dataset(type)(
            self.dataset_root,
            type="val",
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def get_data_loader(self) -> Dict[str, DataLoader]:
        """Return a dictionary of datasets for train and validation sets.

        Returns:
            Dict[str, DataLoader]: Train/Val-Dataset Dict
        """
        batch_size = GLOBAL_CONFIG.batchsize
        num_workers = GLOBAL_CONFIG.num_worker

        dataloaders = {
            # TODO: Use Dataloader for testing purposes instead of MultiEpochsDataloader
            x.type: DataLoader(x, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for x in [self.train_dataset, self.val_dataset]
        }
        return dataloaders

    def get_dataset_sizes(self) -> Dict[str, int]:
        """Return dict of dataset size for train and val.

        Returns:
            Dict[str, int]: dataset size dict
        """
        return {x.type: len(x) for x in [self.train_dataset, self.val_dataset]}

    def get_class_names(self) -> Dict[str, Dict[str, int]]:
        """Return dict for train/val with class mapping to int.

        Returns:
            Dict[str, Dict[str, int]]: class mapping dict
        """
        return {x.type: x.classes for x in [self.train_dataset, self.val_dataset]}


class MultiEpochsDataLoader(DataLoader):
    """Dataloader which constantly loads data between epochs.

    Args:
        DataLoader (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)  # type: ignore[assignment, arg-type]

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


if __name__ == "__main__":
    loader = BoxSupDataloader("MASK")

    print(loader.get_dataset_sizes())
    print(loader.get_class_names())
    dataloaders = loader.get_data_loader()

    print(len(dataloaders["train"]))
    for idx, (inputs, masks, bboxes) in enumerate(dataloaders["train"]):
        print()
