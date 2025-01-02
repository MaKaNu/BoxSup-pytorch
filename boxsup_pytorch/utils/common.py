import sys

import torch
from torch import Tensor


def counter(fn):
    def _counted(*largs, **kargs):
        _counted.invocations += 1
        return fn(*largs, **kargs)

    _counted.invocations = 0
    return _counted


def print_progress_bar(step, total_steps, bar_width=80, title="", print_perc=True):
    # correct len of bar_width
    if len(title) > 0:
        bar_width = bar_width - len(title) - 2
    if print_perc:
        bar_width = bar_width - 9

    # UTF-8 left blocks: 1, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8
    utf_8s = ["█", "▏", "▎", "▍", "▌", "▋", "▊", "█"]
    perc = 100 * float(step) / float(total_steps)
    max_ticks = bar_width * 8
    num_ticks = int(round(perc / 100 * max_ticks))
    full_ticks = num_ticks / 8  # Number of full blocks
    part_ticks = num_ticks % 8  # Size of partial block (array index)

    disp = bar = ""  # Blank out variables
    bar += utf_8s[0] * int(full_ticks)  # Add full blocks into Progress Bar

    # If part_ticks is zero, then no partial block, else append part char
    if part_ticks > 0:
        bar += utf_8s[part_ticks]

    # Pad Progress Bar with fill character
    bar += "▒" * int((max_ticks / 8 - float(num_ticks) / 8.0))

    if len(title) > 0:
        disp = title + ": "  # Optional title to progress display

    # Print progress bar in green: https://stackoverflow.com/a/21786287/6929343
    disp += "\x1b[0;32m"  # Color Green
    disp += bar  # Progress bar to progress display
    disp += "\x1b[0m"  # Color Reset
    if print_perc:
        # If requested, append percentage complete to progress display
        if perc > 100.0:
            perc = 100.0  # Fix "100.04 %" rounding error
        disp += " {:6.2f}".format(perc) + " %"

    # Output to terminal repetitively over the same line using '\r'.
    sys.stdout.write("\r" + disp)
    sys.stdout.flush()


def get_larger(previous_value, value):
    return previous_value if value < previous_value else value


def squash_mask_layer(mask_layers: Tensor, label: Tensor) -> Tensor:
    if len(mask_layers.shape) != 3:
        raise RuntimeError("Mask layer tensor has not the expected shape")
    if mask_layers.shape[0] == 1:
        return mask_layers * label
    cumsum_mask = mask_layers.cumsum(dim=0)
    valid_mask = ~(cumsum_mask / cumsum_mask).nan_to_num().bool()
    first_mask = mask_layers[0, :, :]
    follow_masks = mask_layers[1:, :, :]
    follow_masks_cut = follow_masks * valid_mask[:-1, :, :]
    first_mask = (first_mask * label[0, :, :]).unsqueeze(0)
    follow_masks_cut = follow_masks_cut * label[1:, :, :]
    return torch.cat([first_mask, follow_masks_cut]).sum(0)


def squash_mask_layer2(mask_layers: Tensor, label: Tensor) -> Tensor:
    if len(mask_layers.shape) != 3:
        raise RuntimeError("Mask layer tensor has not the expected shape")
    if mask_layers.shape[0] == 1:
        return mask_layers * label
    inverted_mask_layers = mask_layers == 0
    valid_mask = inverted_mask_layers.cumprod(dim=0)
    first_mask = mask_layers[0, :, :]
    follow_masks = mask_layers[1:, :, :]
    follow_masks_cut = follow_masks * valid_mask[:-1, :, :]
    first_mask = (first_mask * label[0, :, :]).unsqueeze(0)
    follow_masks_cut = follow_masks_cut * label[1:, :, :]
    return torch.cat([first_mask, follow_masks_cut]).sum(0)
