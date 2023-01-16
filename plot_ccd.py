# -*- coding: utf-8 -*-
"""
@Time    : 11/01/2023 18:22
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : plot_ccd
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# mintpy modules
import mintpy
from mintpy.objects import ifgramStack
from mintpy.utils import readfile, writefile, plot, utils as ut


def plot_ccd\
                (ccd_data,
             event_date,
             #metadata,
             subset=None,
             method='difference',
             coherence_threshold=None,
             title=True,
             colorbar=True):
    """Plot coherence change detection results."""

    # Check the length of input array to define number of subplots
    nimg = ccd_data.shape[0] if ccd_data.ndim > 2 else 1
    fig, axs = plt.subplots(1, nimg, sharey=True, figsize=(10, 12), dpi=100)

    # if subset not use select whole scene to plot
    if subset is None:
        subset = [0, ccd_data.shape[1], 0, ccd_data.shape[0 ]]

    for i in range(nimg):
        # Single plot
        if nimg == 1:
            plot_method = method
            data = ccd_data
            ax = axs
        # Multiple plots
        else:
            plot_method = method[i]
            data = ccd_data[i]
            ax = axs[i]

        img = ax.imshow(data[subset[2]:subset[3], subset[0]:subset[1]], cmap='RdGy_r')
        if colorbar == True:
            fig.colorbar(img, ax=ax, location='bottom', pad=0.05)

        # flip left_right and up-down based on the orbit direction for radar_coded images
        #plot.auto_flip_direction(metadata, ax, print_msg=False)

        # Title On/Off
        if title == True:
            if coherence_threshold is not None:
                txt = f'CCD on {event_date} \n {plot_method} \n w/ thresh. {coherence_threshold}'
            else:
                txt = f'CCD on {event_date} \n {plot_method}'
            ax.set_title(txt)
    return

