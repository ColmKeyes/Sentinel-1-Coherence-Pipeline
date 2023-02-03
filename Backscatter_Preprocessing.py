# -*- coding: utf-8 -*-
 #NEED TO CHANGE THIS NAME....
"""
This script processes SLC data to coherence or backscatter GeoTiffs
"""
"""
@Time    : 18/01/2023 16:02
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Backscatter_Preprocessing
"""

import Sentinel_1_SLC_Preprocessing as slc

pols = 'VV' #'VH,VV'
iw_swath = 'IW2'
first_burst_index = 4
mode = 'coherence'
last_burst_index = 7
product_type = 'GeoTIFF'
outpath = 'D:\Data\Results\Coherence_Results'
if mode == 'backscatter':
     outpath_window = '_multilook_window_'
elif mode == 'coherence':
     outpath_window = '_coherence_window_'
multilook_window_size = [[10,50]]#[4,20,] [5,25],[6,30],[7,35],[8,40],[10,50]]      #[2,10],[3,15],[4,20]]

for ix, i in enumerate(multilook_window_size):
    slc.main(pols,
         iw_swath,
         first_burst_index,
         last_burst_index,
         multilook_window_size[ix],
         mode=mode,
         speckle_filter='Lee',
         speckle_filter_size=[5,5],
         product_type=product_type,
         outpath = outpath + '\\pol_'+str(pols) + str(outpath_window) + str(multilook_window_size[ix][0]*multilook_window_size[ix][1]))

