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
## TODO: Add title above each image..
##  Add scale bar to each image..

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np
import datetime
def ccd_animation(opened_rasta_stack,coh_path_list, savepath=None,titles=None):

    fig, ax = plt.subplots()

    titles = [f[17:25] for f in os.listdir(coh_path_list[0]) if f.endswith('.tif')]
    ims =[]

    for i in range(len(opened_rasta_stack.read())):  #39
        im = ax.imshow(opened_rasta_stack.read( i +1), animated=True, cmap="gray")
        if i== 0:
            ax.imshow(opened_rasta_stack.read(i + 1),cmap="gray")#, vmin=0, vmax=1)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)#,repeat_delay=10000)
    ## Add title above each image..
    if titles is not None:
        plt.suptitle(f"InSAR Correlation Coefficient\n \n {datetime.datetime.strptime(titles[0], '%Y%m%d').strftime('%Y-%m-%d')} - {datetime.datetime.strptime(titles[-1], '%Y%m%d').strftime('%Y-%m-%d')}")

        #plt.suptitle(f"Coherence Coefficient,  {datetime.datetime.strptime(titles[0], '%Y%m%d').strftime('%Y-%m-%d')} - {datetime.datetime.strptime(titles[-1], '%Y%m%d').strftime('%Y-%m-%d')}")#, fontsize=16)
    if savepath is not None:
        writer = animation.FFMpegWriter(
            fps=3, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("CCD_animated.mp4", writer=writer)
    plt.show()
    plt.pause(10000)

