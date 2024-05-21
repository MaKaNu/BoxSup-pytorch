"""BoxSup Dataset Module."""

from __future__ import annotations

import collections
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from xml.etree.ElementTree import Element as ET_Element
from xml.etree.ElementTree import parse as ET_parse

# from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import toml
import torch
from torch import Tensor
import torchvision
from torchvision.datasets import VisionDataset

from boxsup_pytorch.utils.common import get_larger


class BoxSupBaseDataset(VisionDataset):
    """Base Dataset."""

    def __init__(
        self,
        root: PathLike,
        variant: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Create BoxSupDatasets.

        Args:
            root (str): root directory of dataset
            variant (str, optional): define evaulation phase.
                Defaults to "train".
            transforms (Optional[Callable], optional): transforms for input and target.
                Defaults to None.
            transform (Optional[Callable], optional): transforms for input.
                Defaults to None.
            target_transform (Optional[Callable], optional): transfroms for target.
                Defaults to None.

        Raises:
            ValueError: is raised if invalid variant is selected.
        """
        super().__init__(str(root), transforms, transform, target_transform)
        if variant not in ["train", "trainval", "val"]:
            raise ValueError(f"Selected variant '{variant}' is invalid!")
        data_dir = Path(root) / f"ImageSets/BoxSup/{variant}.txt"
        with open(data_dir, "r") as data_file:
            self.file_stems = data_file.read().split("\n")
        self.root = Path(root)
        self.classes = toml.load(self.root / "classes.toml")
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        self.img2tensor = torchvision.transforms.ToTensor()
        self.variant = variant
        self.max_bboxes = self.get_max_bboxes()

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: len of image list
        """
        return len(self.file_stems)

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        """Convert xml voc structure to dict.

        Args:
            node (ET_Element): xml voc object

        Returns:
            Dict[str, Any]: dict represantation of voc xml
        """
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
        """Count the max number of bboxes per images.

        Returns:
            int: max number bboxes
        """
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

    def _get_bbox_class(self, bbox_dict: Dict[str, Any]) -> Tensor:
        bbox_class = torch.zeros(self.max_bboxes)
        for idx, obj in enumerate(bbox_dict["annotation"]["object"]):
            bbox_class[idx] = self.classes[obj["name"]]
        return bbox_class


class BoxSupDatasetAll(BoxSupBaseDataset):
    """The BoxSupDataset Class."""

    def __getitem__(
        self, index: int
    ) -> Tuple[Image.Image | Tensor, Tensor, Tensor, Image.Image | Tensor, Tensor]:
        """Return One Sample as Tuple of Dataset.

        The Tuple for this dataset includes all data for complete Process.

        Args:
            index (int): index of the Sample

        Returns:
            Tuple[Image.Image | Tensor, Tensor, Tensor, Image.Image | Tensor, Tensor]: Sample
                        ^                 ^        ^       ^                    ^
                     Image              masks   bboxes gt_masks             gt_pseudo_masks
        """
        image_path = self.root / f"JPEGImages/{self.file_stems[index]}.jpg"
        image = Image.open(image_path).convert("RGB")
        masks_path = self.root / f"MCG_processed/{self.file_stems[index]}.npz"
        masks = torch.from_numpy(np.load(masks_path)["masks"])
        bbox_path = self.root / f"Annotations/{self.file_stems[index]}.xml"
        bbox_dict = self.parse_voc_xml(ET_parse(bbox_path).getroot())
        bboxes = self._generate_mask_from_dict(bbox_dict)
        gt_mask_path = self.root / f"SegmentationClass/{self.file_stems[index]}.png"
        gt_masks = Image.open(gt_mask_path)
        gt_pseudo_mask_path = self.root / f"PseudoSegmentationClass/{self.file_stems[index]}.pt"
        gt_pseudo_masks = torch.load(gt_pseudo_mask_path)

        if self.transform:
            image = self.transform(image)
            gt_masks = self.transform(gt_masks)
        if self.target_transform:
            masks = self.target_transform(masks)
            bboxes = self.target_transform(bboxes)
            gt_pseudo_masks = self.target_transform(gt_pseudo_masks)

        return image, masks, bboxes, gt_masks, gt_pseudo_masks


class BoxSupDatasetUpdateMask(BoxSupBaseDataset):
    """The BoxSupDataset Class for Update Mask."""

    def __getitem__(
        self, index: int
    ) -> Tuple[Image.Image | Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return One Sample as Tuple of Dataset.

        The Tuple for this dataset includes all data necessary for the UpdateMask Process.

        Args:
            index (int): index of the Sample

        Returns:
            Tuple[Image.Image | Tensor, Tensor, Tensor, Tensor, Tensor]: Sample as Tuple
        """
        image_path = self.root / f"JPEGImages/{self.file_stems[index]}.jpg"
        image = Image.open(image_path).convert("RGB")
        masks_path = self.root / f"MCG_processed/{self.file_stems[index]}.npz"
        masks = torch.from_numpy(np.load(masks_path)["masks"])
        masks_iou_path = self.root / f"boxsup_iou/{self.file_stems[index]}.pt"
        masks_cls_path = self.root / f"boxsup_iou/{self.file_stems[index]}_cls.pt"
        masks_iou = torch.load(masks_iou_path)
        masks_cls = torch.load(masks_cls_path)
        bbox_path = self.root / f"Annotations/{self.file_stems[index]}.xml"
        bbox_dict = self.parse_voc_xml(ET_parse(bbox_path).getroot())
        bbox_class = self._get_bbox_class(bbox_dict)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            masks = self.target_transform(masks)

        if masks_cls_path.exists():
            masks_class = torch.load(masks_cls_path)
        else:
            masks_class = torch.zeros(masks.shape[0])

        return image, masks, masks_cls, masks_iou, bbox_class


class BoxSupDatasetUpdateNet(BoxSupBaseDataset):
    """The BoxSupDataset Class for Update Mask."""

    def __getitem__(self, index: int) -> Tuple[Image.Image | Tensor, Image.Image | Tensor, Tensor]:
        """Return One Sample as Tuple of Dataset.

        The Tuple for this dataset includes all data necessary for the UpdateNet Process.

        Args:
            index (int): index of the Sample

        Returns:
            Tuple[Image.Image | Tensor, Image.Image | Tensor, Tensor]: Sample as Tuple
        """
        image_path = self.root / f"JPEGImages/{self.file_stems[index]}.jpg"
        image = Image.open(image_path).convert("RGB")
        gt_mask_path = self.root / f"SegmentationClass/{self.file_stems[index]}.png"
        gt_masks = Image.open(gt_mask_path)
        gt_pseudo_mask_path = self.root / f"PseudoSegmentationClass/{self.file_stems[index]}.pt"
        gt_pseudo_masks = torch.load(gt_pseudo_mask_path)

        if self.transform:
            image = self.transform(image)
            gt_masks = self.transform(gt_masks)
        if self.target_transform:
            gt_pseudo_masks = self.target_transform(gt_pseudo_masks)

        return image, gt_masks, gt_pseudo_masks


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


class BoxSupDatasetTrain(BoxSupBaseDataset):
    """The Training Dataset Sub Class."""

    def __init__(
        self,
        root: PathLike,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize BoxSupDatasetTrain.

        Args:
            root (PathLike): dataset root path
            transforms (Optional[Callable], optional): Combined Transforms for data and gt.
                Defaults to None.
            transform (Optional[Callable], optional): image transformations. Defaults to None.
            target_transform (Optional[Callable], optional): gt transformations. Defaults to None.
        """
        super().__init__(
            root,
            variant="train",
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get image candidates and gt_bboxes for training.

        Args:
            index (int): points to data index which should be returned.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple of image, candidates and gt_bboxes.
        """
        # Load Image and convert to Tensor
        image_path = self.root / f"JPEGImages/{self.file_stems[index]}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = self.img2tensor(image)

        # Load candidates as tensors
        candidates_path = self.root / f"MCG_processed/{self.file_stems[index]}.npz"
        candidates = torch.from_numpy(np.load(candidates_path)["masks"])

        # Load Ground Truth BBoxes and create Masks
        gt_bbox_path = self.root / f"Annotations/{self.file_stems[index]}.xml"
        gt_bbox_dict = self.parse_voc_xml(ET_parse(gt_bbox_path).getroot())
        gt_bboxes = self._generate_mask_from_dict(gt_bbox_dict)

        if self.target_transform:
            candidates = self.target_transform(candidates)
            gt_bboxes = self.target_transform(gt_bboxes)

        if self.transform:
            image = self.transform(image)

        return (image, candidates, gt_bboxes)


if __name__ == "__main__":
    pass
