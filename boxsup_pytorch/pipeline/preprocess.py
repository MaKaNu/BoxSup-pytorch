"""PreProcess Modul.

This file is part of "boxsup-pytorch" which is released under

                GNU GENERAL PUBLIC LICENSE
                        Version 3.
See file LICENSE or go to https://github.com/MaKaNu/boxsup-pytorch/blob/main/LICENSE
for full license details.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy
import torch
from torch import Tensor

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.data.dataloader import BoxSupDataloader
from boxsup_pytorch.utils.check import check_path_exists_msg
from boxsup_pytorch.utils.common import counter, print_progress_bar
from boxsup_pytorch.utils.losses import Losses


@dataclass
class PreProcessMCG:
    """PreProcess Masks Candidtates.

    The mask candidates are created with a MATLAB algorithm called
    MCG (Multiscale Combinatorial Grouping). To actually use the candidates created by the
    algorithm it is necessary to transform them to numpy array.
    """

    def __init__(self, mask_location: Path) -> None:
        """Init method of PreProcessMCG.

        Converts MCG values to masks

        Args:
            mask_location (Path): location where the MCG MATLAB file is located

        Raises:
            FileNotFoundError: if the mask file does not exsists.
        """
        if not mask_location.exists():
            raise FileNotFoundError(check_path_exists_msg(mask_location))
        self.mask_location: Path = mask_location

    def update_logic(self):
        """Update the candidate masks from mat-files to npz-files."""
        self.out_masks_location = self.mask_location.parent / "MCG_processed/"
        not_processed_files = self._get_unprocessed_files()
        self._load_and_process_matfiles(not_processed_files)

    @counter
    def _get_unprocessed_files(self) -> List[str]:
        """Create a list of file stems which are not processed.

        After checking if the in-/outputs of the Pipeline Process are set,
        the method creates the masks location if the location not yet exists.
        Depending on the config 'rerun_process' boolean, the npz-files in the target
        location will be deleted.

        Returns:
            List[str]: Depending on the comparison between the files in the
                       'self.in_mask_location' and the files in the
                       'self.out_masks_location'  a List of file stems
        """
        if not self.out_masks_location.exists():
            self.out_masks_location.mkdir()

        files = [file.stem for file in self.mask_location.glob("*.mat")]
        processed_files = [file.stem for file in self.out_masks_location.glob("*.npz")]

        if GLOBAL_CONFIG.rerun_process:
            for file in processed_files:
                (self.out_masks_location / (file + ".npz")).unlink()

        not_processed_files = [file for file in files if file not in processed_files]
        return not_processed_files

    def _load_and_process_matfiles(self, list_of_files) -> None:
        """Iterate over all MATLAB files and generate the candidate masks.

        The method extracts the 'scores', 'superpixels' and 'labels' from the MATLAB-file dict.
        Based on the 'scores' the TOP N candidates are choosen, while N is defined in the config.
        The mask creation is based on the 'superpixels' 2D array which is compared to the 'labels'.
        The stacked masked are saved as compressed numpy file at the 'self.out_masks_location'.

        Args:
            list_of_files (List[str]): The list, which is created by the '_get_unprocessed_files.
        """

        def _process_matfiles():
            top_n = GLOBAL_CONFIG.mcg_num_candidates
            top_n_idx = np.argpartition(scores.squeeze(), -top_n)[-top_n:]
            masks = []
            for idx in top_n_idx:
                mask = np.isin(superpixels, labels[idx, 0])
                mask = mask.reshape(superpixels.shape)
                mask = mask.astype(np.float32)
                masks.append(mask)
            masks = np.stack(masks)
            save_path = self.out_masks_location / (mat_file + ".npz")
            np.savez_compressed(save_path, masks=masks)

        if self._get_unprocessed_files.invocations == 0:
            raise RuntimeError("'self._get_unprocessed_files' not called!")

        total_steps = len(list_of_files)
        for idx, mat_file in enumerate(list_of_files):
            mcg_mat = scipy.io.loadmat(self.mask_location / (mat_file + ".mat"))
            scores = mcg_mat["scores"]
            superpixels = mcg_mat["superpixels"]
            labels = mcg_mat["labels"]
            _process_matfiles()
            print_progress_bar(idx, total_steps, title=f"File: {mat_file}")


@dataclass
class PreProcessIoU:
    """PreProcess for calculating IoU.

    The IoU can be calculated before the actuall process, since the position and size of the
    bboxes is fixed in the dataset aswell as the masks which are created by the MCG or any other
    Algorithm, which might be used for masks generation.

    Further this class provides the initial class selecetion for each IoU.
    """

    per_bbox_iou: Tensor = torch.tensor([])
    mask_classes: Tensor = torch.tensor([])

    def update_logic(self, dataloader: BoxSupDataloader):
        """Run the loading of data and calculates the IoU for Masks and BBoxes.

        Args:
            dataloaders (BoxSupDataloader): The main dataloader.
        """
        stage = "train"
        dataloaders = dataloader.get_data_loader()
        for run, data_tuple in enumerate(dataloaders[stage]):
            batched_masks, batched_bboxes, batched_file_stems = data_tuple
            total_steps = batched_bboxes.shape[0]
            for idx, bbox_mask_tuple in enumerate(zip(batched_bboxes, batched_masks)):
                sample_bboxes, sample_masks = bbox_mask_tuple

                num_bboxes = batched_bboxes.shape[1]
                num_masks = batched_masks.shape[1]

                self.per_bbox_iou = torch.zeros((num_bboxes, num_masks))
                self.mask_classes = torch.zeros(num_masks)
                for box_idx, sample_bbox in enumerate(sample_bboxes):
                    if sample_bbox.max() == 0:
                        continue
                    self._calculate_iou(box_idx, sample_masks, sample_bbox)
                    self._assign_sample_to_class(box_idx, sample_bbox)

                file_stem = batched_file_stems[idx]
                file_path_iou = Path(GLOBAL_CONFIG.root) / f"boxsup_iou/{file_stem}.pt"
                file_path_cls = Path(GLOBAL_CONFIG.root) / f"boxsup_iou/{file_stem}_cls.pt"
                torch.save(self.per_bbox_iou, file_path_iou)
                torch.save(self.mask_classes, file_path_cls)
                print_progress_bar(idx + 1, total_steps, title=f"PreIoU RUN: {run}")

    def _calculate_iou(self, box_idx, masks, bbox):
        if self.per_bbox_iou.nelement() == 0:
            raise RuntimeError("Class paramter 'per_bbox_iou' is not allocated!")

        losses = Losses()

        iou_values = losses.inter_over_union(masks, bbox)
        self.per_bbox_iou[box_idx, :] = iou_values

    def _assign_sample_to_class(self, box_idx, bbox):
        """Assign a sample to a class.

        if IoU is zero or less than threshold, the class is not provided for the candidate.
        if the IoU is greater than zero or above threshold, the class is provided for the candidate
            only if no class is already provided.
        if the IoU is greater than zero or above threshold, the provided class is decided based on
            higher IoU.

        Raises:
            FileNotFoundError: _description_
        """
        # Guarding IFs
        if self.per_bbox_iou.nelement() == 0:
            raise RuntimeError("Class paramter 'per_bbox_iou' is not allocated!")
        if self.mask_classes.nelement() == 0:
            raise RuntimeError("Class paramter 'mask_classes' is not allocated!")

        # Read Data
        ious = self.per_bbox_iou[box_idx]
        cls = bbox.max()
        saved_iou = torch.cat((self.per_bbox_iou[:box_idx], self.per_bbox_iou[box_idx + 1 :]))

        # Compare saved values
        iou_mask = (ious > 0) * (~torch.greater(saved_iou, ious)).prod(0).bool()

        # Assign cls to compared iou values
        self.mask_classes[iou_mask] = cls.expand(self.mask_classes.shape)[iou_mask]


@dataclass
class PrePorcessTensor:
    """TEST."""

    def __init__(self, mask_location: Path) -> None:
        """Init Method of PreProcessTensor.

        Args:
            mask_location (Path): WTF

        Raises:
            FileNotFoundError: WTF
        """
        if not self.masks_location.exists():
            raise FileNotFoundError(check_path_exists_msg(mask_location))
        self.masks_location: Path = mask_location


if __name__ == "__main__":
    # set input
    inputs = {"mask_location": GLOBAL_CONFIG.mcg_path}
    process_instance = PreProcessMCG(**inputs)
    # update
    process_instance.update_logic()

    dataloader = BoxSupDataloader("IOU")
    iou_preprocess = PreProcessIoU()
    iou_preprocess.update_logic(dataloader)
