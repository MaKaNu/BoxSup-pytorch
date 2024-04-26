"""BoxSup Dataset Module."""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from xml.etree.ElementTree import Element as ET_Element
from xml.etree.ElementTree import parse as ET_parse
import random

# from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import toml
import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor

from boxsup_pytorch.utils.common import get_larger


class BoxSupBaseDataset(VisionDataset):
    """Base Dataset."""

    def __init__(
        self,
        root: str,
        type: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        if type not in ["train", "trainval", "val"]:
            raise ValueError(f"Selected type '{type}' is invalid!")
        data_dir = Path(root) / f"ImageSets/BoxSup/{type}.txt"
        with open(data_dir, "r") as data_file:
            self.file_stems = data_file.read().split("\n")
        self.root = Path(root)
        self.classes = toml.load(self.root / "classes.toml")
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        self.type = type
        self.max_bboxes = self.get_max_bboxes()

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: len of image list
        """
        return len(self.file_stems)

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(BoxSupBaseDataset.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def get_max_bboxes(self) -> int:
        num_bboxes = 0
        for file_stem in self.file_stems:
            if not file_stem:
                continue
            annotation_path = self.root / f"Annotations/{file_stem}.xml"
            bbox_dict = self.parse_voc_xml(ET_parse(annotation_path).getroot())
            num_bboxes_next = len(bbox_dict["annotation"]["object"])
            num_bboxes = get_larger(num_bboxes, num_bboxes_next)
        return num_bboxes

    def _generate_mask_from_dict(self, annotation: Dict[str, Any]) -> torch.Tensor:
        # We need to get w/h of original Image
        width = int(annotation["annotation"]["size"]["width"])
        height = int(annotation["annotation"]["size"]["height"])

        # Initiate Tensor
        # but u and v coordinate need to be switched
        bbox_tensor = torch.zeros((self.max_bboxes, height, width))
        for idx, object in enumerate(annotation["annotation"]["object"]):
            bbox_class = self.classes[object["name"]]
            # Get BBox Corners
            xmin = int(object["bndbox"]["xmin"])
            ymin = int(object["bndbox"]["ymin"])
            xmax = int(object["bndbox"]["xmax"])
            ymax = int(object["bndbox"]["ymax"])

            # So x and y needs to be switched aswell (probably)
            bbox_tensor[idx, ymin:ymax, xmin:xmax] = bbox_class
        return bbox_tensor.flip(0)  # reverse bbox order
    
    
class BoxSupDatasetAll(BoxSupBaseDataset):
    """The BoxSupDataset Class."""

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        image_path = self.root / f"JPEGImages/{self.file_stems[index]}.jpg"
        image = Image.open(image_path).convert("RGB")
        masks_path = self.root / f"MCG_processed/{self.file_stems[index]}.npz"
        masks = torch.from_numpy(np.load(masks_path)["masks"])
        bbox_path = self.root / f"Annotations/{self.file_stems[index]}.xml"
        bbox_dict = self.parse_voc_xml(ET_parse(bbox_path).getroot())
        bboxes = self._generate_mask_from_dict(bbox_dict)
        gt_label_path = self.root / f"SegmentationClass/{self.file_stems[index]}.png"
        gt_label = Image.open(gt_label_path)
        #gt_label = read_image(str(gt_label_path)).long()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            masks = self.target_transform(masks)
            bboxes = self.target_transform(bboxes)
            gt_label = self.target_transform(gt_label)
            gt_label = ToTensor()(gt_label).squeeze().long()

        return image, masks, bboxes, gt_label


class BoxSupDatasetUpdateMask(BoxSupBaseDataset):
    """The BoxSupDataset Class for Update Mask."""

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:

        image_path = self.root / f"JPEGImages/{self.file_stems[index]}.jpg"
        image = read_image(str(image_path))
        masks_path = self.root / f"MCG_processed/{self.file_stems[index]}.npz"
        masks = torch.from_numpy(np.load(masks_path)["masks"])
        bbox_path = self.root / f"Annotations/{self.file_stems[index]}.xml"
        bbox_dict = self.parse_voc_xml(ET_parse(bbox_path).getroot())
        bboxes = self._generate_mask_from_dict(bbox_dict)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            masks = self.target_transform(masks)
            bboxes = self.target_transform(bboxes)

        return image, masks, bboxes


class BoxSupDatasetUpdateNet(BoxSupBaseDataset):
    """The BoxSupDataset Class for Update Mask."""

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:

        image_path = self.root / f"JPEGImages/{self.file_stems[index]}.jpg"
        image = read_image(str(image_path))
        gt_label_path = self.root / f"SegmentationClass/{self.file_stems[index]}.png"
        gt_label = read_image(str(gt_label_path))

        if self.transform:
            image = self.transform(image)
class BoxSupDatasetUpdateIOU(BoxSupBaseDataset):
    """The BoxSupDataset Class for Update Mask."""

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str]:
        """Return One Sample as Tuple of Dataset.

        The Tuple for this dataset includes all data necessary for IoU Calculation.

        Args:
            index (int): index of the Sample

        Returns:
            Tuple[Tensor, Tensor, str]: _description_
        """
        masks_path = self.root / f"MCG_processed/{self.file_stems[index]}.npz"
        masks = torch.from_numpy(np.load(masks_path)["masks"])
        bbox_path = self.root / f"Annotations/{self.file_stems[index]}.xml"
        bbox_dict = self.parse_voc_xml(ET_parse(bbox_path).getroot())
        bboxes = self._generate_mask_from_dict(bbox_dict)

        if self.target_transform:
            masks = self.target_transform(masks)
            bboxes = self.target_transform(bboxes)

        return masks, bboxes, self.file_stems[index]
        if self.target_transform:
            gt_label = self.target_transform(gt_label)

        return image, gt_label


if __name__ == "__main__":
    dataset = BoxSupDatasetAll("boxsup_pytorch/data/datasets/pascal2012")
    items = dataset.__getitem__(0)
    print(items)
