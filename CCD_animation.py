# -*- coding: utf-8 -*-
"""
This script animates a stack of tiffs & saves to mp4
"""
"""
@Time    : 11/01/2023 14:52
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : CCD_animation
"""

from osgeo import gdal
import rasterio
import rasterio as rasta
from rasterio.merge import merge
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def ccd_animation(opened_rasta_stack, savepath=None,titles=None):

    fig, ax = plt.subplots()
    #plt.suptitle('')
    ims =[]

    for i in range(len(opened_rasta_stack.read())):
        im = ax.imshow(opened_rasta_stack.read( i +1), animated=True)
        if i== 0:
            ax.imshow(opened_rasta_stack.read(i + 1))
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
    ## Add title above each image..
    if titles is not None:
        plt.suptitle(str(titles[0])+':'+str(titles[-1]))
    if savepath is not None:
        writer = animation.FFMpegWriter(
            fps=3, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("CCD_animated.mp4", writer=writer)
    plt.show()
    plt.pause(10000)

