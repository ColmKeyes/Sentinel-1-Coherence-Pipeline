# -*- coding: utf-8 -*-
"""
This script provides a class and methods for building data-cubes from SLC processed coherence and backscatter data
"""
"""
@Time    : 17/02/2023 15:32
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Coherence-Time-Series
"""

import numpy as np
import numpy.ma as ma
import os
from matplotlib.pyplot import pause
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import pandas as pd
import rioxarray
import xarray
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform ,reproject
from rasterio.windows import get_data_window, transform
from rasterio.enums import Resampling
from rasterio.mask import mask
from geocube.api.core import make_geocube
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Daily
from statsmodels.tsa.seasonal import seasonal_decompose
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#plt.style.use('dark_background')


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class CoherenceTimeSeries:
    """
    A class for generating coherence and backscatter time series data-cube from Sentinel-1 SLC products.
    """

    def __init__(self, asf_df, path, stack_path_list, window_size,window, shp=None, normalised=False):
        """
        Args:
            asf_df (pandas.DataFrame): ASF catalog search results.
            path (list): List of file paths to Sentinel-1 SLC Tiff products.
            stack_path_list (str): Path to directory where coherence and backscatter raster stacks are stored.
            window_size (int): pixel spacing in metres.
            shp (geopandas.GeoDataFrame, optional): GeoDataFrame containing study area polygon.
            normalised (bool, optional): Whether to normalise output data-stacks. Defaults to False.

        Attributes:
            cube (None or xarray.DataArray): Multi-dimensional, named array for storing data.
        """

        # Initialize class variables
        self.asf_df = asf_df
        self.path = path
        self.stack_path_list = stack_path_list
        self.cube = None
        self.grouped = None
        self.grouped_ordered = None
        self.window_size = window_size
        self.window = window
        self.normalised = normalised
        self.mask = None
        self.shp = shp
        # Get titles of files ending with '.tif' in path
        self.titles = [f[17:25] for f in os.listdir(self.path[0]) if f.endswith('.tif')]



    def write_rasterio_stack(self, write=True):
        """
        Write folder of GeoTIFFs to a GeoTIFF stack file.
        https://gist.github.com/prakharcode/b83caaaa2fc6d2d62b7fe558656df0d1#file-resample-py-L14
        """

        for path in self.path:
            files = [f for f in os.listdir(path) if f.endswith('.tif')]

            # Read the source image and the GCPs
            src_image = rasterio.open(os.path.join(path, files[0]), "r+")

            dst_crs = src_image.crs
            dst_transform, dst_width, dst_height = calculate_default_transform(src_image.crs, dst_crs, src_image.width, src_image.height, *src_image.bounds)

            # Create a VRT dataset with the transformation applied
            dst_profile = src_image.profile.copy()
            dst_profile.update({
                "driver": "GTiff",
                "count": len(files),
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height
            })

            if not os.path.exists(f"{self.stack_path_list}"):
                os.makedirs(f"{self.stack_path_list}")
            with rasterio.open(f'{self.stack_path_list}\\{path[45:]}.tif', 'w', **dst_profile) as dst:
                for i, file in enumerate(files, start=1):

                    src = rasterio.open(os.path.join(path, file), "r+")
                    src.nodata = 0
                    src.close()

                    with rasterio.open(os.path.join(path, file), "r") as src:
                        profile = src.profile.copy()
                        data_window = get_data_window(src.read(masked=True))
                        data_transform = transform(data_window, src.transform)
                        profile.update(
                            transform=data_transform,
                            height=data_window.height,
                            width=data_window.width)

                        data = src.read(window=data_window)
                        src.close()

                    if write:
                        if self.normalised:
                            dst.write(self.pct_clip(data[0],[0.02,99.98]), i)
                        else:
                            dst.write(data[0], i)
                dst.close()

    def fix_dimension(self, ds, target_dims):

        for dim in target_dims:
            if ds[dim].size < target_dims[dim]:
                pad_width = target_dims[dim] - ds[dim].size
                ds = ds.pad({dim: (0, pad_width)}, constant_values=0)
            elif ds[dim].size > target_dims[dim]:
                ds = ds.isel({dim: slice(0, target_dims[dim])})
        return ds

    def build_cube(self):
        """
        Method to build a cube by reading the coherence and backscatter stacks from rasterio.
        """

        if self.shp is not None:
            # Clipping the stack using the given shp geometry
            shp_stacks = [rioxarray.open_rasterio(os.path.join(self.stack_path_list, stack), masked=True).rio.clip(self.shp.geometry.values, self.shp.crs, from_disk=True) for stack
                          in os.listdir(self.stack_path_list)]
        else:
            shp_stacks = [rioxarray.open_rasterio(os.path.join(self.stack_path_list, stack), masked=True) for stack in os.listdir(self.stack_path_list)]

        shp_stack_backscatter_vh = shp_stacks[0]
        shp_stack_coh_vh = shp_stacks[1] if len(shp_stacks) >= 2 else None
        shp_stack_backscatter_vv = shp_stacks[2] if len(shp_stacks) >= 3 else None
        shp_stack_coh_vv = shp_stacks[3] if len(shp_stacks) >= 4 else None

        self.cube = make_geocube(self.shp, like=shp_stack_coh_vh, measurements=["code"])

        target_dims = {'y': self.cube['y'].size, 'x': self.cube['x'].size}

        if shp_stack_coh_vv is not None:
            shp_stack_coh_vv = self.fix_dimension(shp_stack_coh_vv, target_dims)
            shp_stack_coh_vv["code"] = shp_stack_coh_vv.band + 1
            self.cube['coherence_VV'] = (shp_stack_coh_vv.dims, shp_stack_coh_vv.values, shp_stack_coh_vv.attrs, shp_stack_coh_vv.encoding)

        if shp_stack_coh_vh is not None:
            shp_stack_coh_vh = self.fix_dimension(shp_stack_coh_vh, target_dims)
            shp_stack_coh_vh["code"] = shp_stack_coh_vh.band + 1
            self.cube["coherence_VH"] = (shp_stack_coh_vh.dims, shp_stack_coh_vh.values, shp_stack_coh_vh.attrs, shp_stack_coh_vh.encoding)

        if shp_stack_backscatter_vv is not None:
            shp_stack_backscatter_vv = self.fix_dimension(shp_stack_backscatter_vv, target_dims)
            shp_stack_backscatter_vv["code"] = shp_stack_backscatter_vv.band + 1
            self.cube["backscatter_VV"] = (shp_stack_backscatter_vv.dims, shp_stack_backscatter_vv.values, shp_stack_backscatter_vv.attrs, shp_stack_backscatter_vv.encoding)

        if shp_stack_backscatter_vh is not None:
            shp_stack_backscatter_vh = self.fix_dimension(shp_stack_backscatter_vh, target_dims)
            shp_stack_backscatter_vh["code"] = shp_stack_backscatter_vh.band + 1
            self.cube["backscatter_VH"] = (shp_stack_backscatter_vh.dims, shp_stack_backscatter_vh.values, shp_stack_backscatter_vh.attrs, shp_stack_backscatter_vh.encoding)

        coh_dates = pd.to_datetime(pd.Series(self.titles))
        self.cube['dates'] = coh_dates

        self.grouped = self.cube.groupby('code')
        self.grouped_ordered = list(self.cube.groupby('code'))
        order_dict = {6: 1, 1: 2, 3: 3, 2: 4, 4: 5, 5: 6}

        self.grouped_ordered.sort(key=lambda x: order_dict[x[0]])

    def build_date_mask(self):
        ## Set up a mask for dates where the data is too far apart in the time series plots..
        dates_to_mask = [
            ('2021-05-01', '2021-07-24'),
            ('2022-04-26', '2022-06-13'),
            ('2022-06-25', '2022-08-12'),
        ]

        mask = np.zeros(len(self.cube.dates), dtype=bool)

        for start_date, end_date in dates_to_mask:
            start_idx = np.where(self.cube.dates == np.datetime64(start_date))[0][0]
            end_idx = np.where(self.cube.dates == np.datetime64(end_date))[0][0]

            # Update the mask for the new gap
            if start_idx < end_idx:  # Prevents out of bounds and negative indexing issues
                mask[start_idx:end_idx] = True

        self.mask = mask

    def build_significant_date_mask(self, significant_dates):
        ## Set up a mask for dates where the data is too far apart in the time series plots..
        dates_to_mask = [
            ('2021-05-01', '2021-07-24'),
            ('2022-04-26', '2022-06-13'),
            ('2022-06-25', '2022-08-12'),
        ]

        # Check for non-datetime elements in significant_dates and remove them
        significant_dates = np.array([date for date in significant_dates if isinstance(date, np.datetime64)])

        mask = np.zeros(len(significant_dates), dtype=bool)

        for start_date, end_date in dates_to_mask:
            start_date = np.datetime64(start_date)
            end_date = np.datetime64(end_date)

            # Update the mask for the new gap
            mask[(significant_dates >= start_date) & (significant_dates <= end_date)] = True

        return mask

    def single_plot(self, titles,plot_code):
        """
        Plot the coherence time series for a single shp code in the Cube.
        """
        # Group cube by polygon code
        #grouped = self.cube.groupby('code')

        # Define the index range of the gap
        start_idx = np.where(self.cube.dates == np.datetime64('2021-05-01'))[0][0]
        end_idx = np.where(self.cube.dates == np.datetime64('2021-07-24'))[0][0]

        # Create a mask for the gap
        mask = np.zeros(len(self.cube.dates), dtype=bool)
        mask[start_idx:end_idx] = True
        grouped_list = list(self.cube.groupby('code'))
        order_dict = {6: 1, 1: 2, 3: 3, 2: 4, 4: 5, 5: 6}

        grouped_list.sort(key=lambda x: order_dict[x[0]])

        for i, (code, ds) in enumerate(grouped_list):
            if i == plot_code:  ## "Intact Forest" code: 5
                plt.plot(ds.dates[:,0], ds.coherence_VH, label=f'{ds.coherence_VH.name}')
                plt.plot(ds.dates[:,0], ds.coherence_VV, label=f'{ds.coherence_VV.name}')
                plt.title(f'{titles[i]}, Coherence Window: {self.window} ',fontsize=18)  #Disturbance Event {plot_code}
                plt.legend(prop={'size': 11})
                plt.ylim([0, 1])
                plt.xlabel('Dates',fontsize=18)
                plt.ylabel("γ",fontsize=18)
                plt.show()
                plt.pause(100)

    def multiple_plots(self, titles=None):
        """
        Plot the coherence time series for each single polygon code in the Cube.
        """
        event_dates = [None,['2020-07-01', '2021-03-01'],None, '2021-02-01', None,'2020-11-01']

        # Some not-so readable code to Convert non-None dates to numerical format
        event_dates_num = []
        for sublist in event_dates:
            if isinstance(sublist, list):
                sublist_dates = []
                for d in sublist:
                    try:
                        sublist_dates.append(datetime.strptime(d, '%Y-%m-%d'))
                    except (ValueError, TypeError):
                        sublist_dates.append(None)
                event_dates_num.append(sublist_dates)
            else:
                try:
                    event_dates_num.append(datetime.strptime(sublist, '%Y-%m-%d'))
                except (ValueError, TypeError):
                    event_dates_num.append(None)

        # Define the index range of the gap
        start_idx = np.where(self.cube.dates == np.datetime64('2021-05-01'))[0][0]
        end_idx = np.where(self.cube.dates == np.datetime64('2021-07-24'))[0][0]

        # Create a mask for the gap
        mask = np.zeros(len(self.cube.dates), dtype=bool)
        mask[start_idx:end_idx] = True
        self.mask=mask
        #mask = self.cube['dates'].isnull()

        # Mask the `dates` variable and other variables along the `dim_0` dimension
        masked_dates = ma.array(self.cube['dates'].values, mask=mask)
        #grouped = self.cube.where(~mask).drop('dim_0').groupby('code')  #.assign_coords(dates=masked_dates)


        coh_bsc_vars = [var for var in self.cube.variables if "coherence" in var]  #"backscatter" in var or
        # Group by the 'code' variable

        grouped_list = list(self.cube.groupby('code'))
        order_dict = {6: 1, 1: 2, 3: 3, 2: 4, 4: 5, 5: 6}

        grouped_list.sort(key=lambda x: order_dict[x[0]])

        # Loop through the grouped data and plot each variable
        fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharey=True, sharex=True)
        for var in coh_bsc_vars:
            ax = ax.flatten()
            for i, ((code, data), event_date) in enumerate(zip(self.grouped_ordered, event_dates_num)):
                if code==5:
                    continue
                print(i)
                #data1=data.where(~mask)
                ax[i].plot(data['dates'][:,0].where(~self.mask), data.drop('dates')[var], label=str(var))#.rolling(band=3,min_periods=2, center=True).std()[var], label=str(var))
                ax[i].set_xlabel('Dates',fontsize=18)
                ax[i].set_ylabel("γ",fontsize=18)#"Correlation Coefficient" if "coherence" in var else "Backscatter (dB)",fontsize=18)
                ax[i].set_ylim([0, 1])
                ax[i].tick_params(axis='both', which='both', length=5, width=2)
                #ax[i].autoscale(enable=True, axis='both', tight=True)
                if titles:
                    ax[i].set_title(titles[i])
                # if event_date is not None:
                #     if isinstance(event_date, list):
                #         ax[i].axvline(x=event_date[0], color='r')
                #         ax[i].axvline(x=event_date[1], color='r')#,label="Manually Detected Event")
                #     else:
                #         ax[i].axvline(x=event_date, color='r')
                ax[i].legend(prop={'size': 11})

                # if event_dates and i < len(event_dates):
                #     event_date = event_dates[i]
                #     # Convert event date to Matplotlib format
                #     event_date = mdates.date2num(event_date)
                #     ax[i].axvline(x=event_date, color='red')

                fig.suptitle(f"Disturbance Analysis: Coherence Window: {self.window}",fontsize=18)

                plt.tight_layout()

        plt.show()
        pause(100)

    def stats(self, titles=None):
        '''
        Plotting coherence disturbance events
        ## grouped[5]  ## 3rd Disturbed
        ## grouped[6]  ## Intact Forest
        '''

        event_dates = [None,['2020-07-01', '2021-03-01'],None, '2021-02-01', None, '2020-11-01']

        # Some not-so readable code to Convert non-None dates to numerical format
        event_dates_num = []
        for sublist in event_dates:
            if isinstance(sublist, list):
                sublist_dates = []
                for d in sublist:
                    try:
                        sublist_dates.append(datetime.strptime(d, '%Y-%m-%d'))
                    except (ValueError, TypeError):
                        sublist_dates.append(None)
                event_dates_num.append(sublist_dates)
            else:
                try:
                    event_dates_num.append(datetime.strptime(sublist, '%Y-%m-%d'))
                except (ValueError, TypeError):
                    event_dates_num.append(None)

        coh_bsc_vars = [var for var in self.cube.variables if "coherence" in var]  # "backscatter" in var or


        ##########################
        ## Multiplot loop
        ##########################
        # Loop through the grouped data and plot each variable
        fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharey=True, sharex=True)
        for var in coh_bsc_vars:
            ax = ax.flatten()



            for i, ((code, data), event_date) in enumerate(zip(self.grouped_ordered, event_dates_num)):
                if code==4:
                    continue
                amplitude_diff = data.drop('dates').coherence_VH.values - self.grouped[6].drop('dates').coherence_VH.values
                std_diff = np.abs(np.abs(amplitude_diff) / (np.abs(self.grouped[6].drop('dates').rolling(band=5, min_periods=3, center=True).std().coherence_VH.values))) ## 3 Sigma


                # Find the indices where std_diff is greater than 1
                significant_indices = np.where(std_diff > 3)

                # Extract the corresponding dates for the significant indices
                try:
                    significant_dates = data['dates'][significant_indices].values[:, 0]
                except IndexError:
                    significant_dates = []

                # Create the mask for significant_dates
                #sig_dates_mask = self.build_significant_date_mask(significant_dates)

                # Use the mask to filter significant_dates
                significant_dates = significant_dates#[~sig_dates_mask]

                #ax[i].plot(data['dates'][:, 0].where(~mask), data.drop('dates').rolling(band=3, min_periods=2, center=True).std()[var], label=str(var))
                ax[i].plot(data['dates'][:, 0].where(~self.mask), data[var] , label=str(var))
                #ax[i].plot(data['dates'][:, 0].where(~self.mask), data.coherence_VH , label=str(var))
                ax[i].set_xlabel('Dates',fontsize=18)
                ax[i].set_ylabel("γ",fontsize=18)#"Correlation Coefficient",fontsize=18) #"STD from I.F.")
                ax[i].legend(fontsize=14)
                ax[i].set_ylim([0, 1])
                ax[i].tick_params(axis='both', which='both', length=5, width=2)
                # ax[i].autoscale(enable=True, axis='both', tight=True)
                if titles:
                    ax[i].set_title(titles[i],fontsize=18)
                if event_date is not None:
                    if isinstance(event_date, list):
                        ax[i].axvline(x=event_date[0], color='r')
                        ax[i].axvline(x=event_date[1], color='r')
                    else:
                        ax[i].axvline(x=event_date, color='r')

                    # Create a Line2D object for the legend
                    red_line = Line2D([], [], color='r', label='Manual Event Date')
                # Add vertical lines for the significant dates
                if significant_indices[0].size > 0 and np.any(significant_indices[0] != 0):
                    for date in significant_dates:
                        ax[i].axvline(x=date, color='g', linestyle='--')
                    green_line = Line2D([], [], color='g', linestyle='--',label='Estimated Events')

                try:
                    ax[i].legend(handles=[red_line,green_line] + ax[i].get_legend_handles_labels()[0],prop={'size': 11})
                except UnboundLocalError:
                    continue

        ##########################

                fig.suptitle(f"Disturbance Event Detection\n \n3σ From Intact Forest  \n\nWindow: {self.window}",fontsize=18)
                #fig.suptitle(f"Disturbance Analysis \n  Coherence Window: {self.window}",fontsize=18)
                plt.tight_layout()

        plt.show()
        pause(100)

    def ccd_animation(self, opened_rasta_stack, coh_path_list, coherence_time_series_dates, savepath=None, titles=None):

        def on_click(event):
            if event.inaxes == ax_std_diff:
                clicked_date = np.datetime64(datetime.utcfromtimestamp(event.xdata.astype(int)), 'ns')
                nearest_idx = np.abs(dates - clicked_date).argmin()
                selected_img = opened_rasta_stack.read([nearest_idx + 1])[0]  # Extract the first element
                ax_animation.imshow(selected_img, cmap="gray")
                ax_std_diff.lines[0].set_xdata(dates[nearest_idx])
                fig.canvas.draw()


        for i, (code, data) in enumerate(self.grouped):
            if code==4:
                amplitude_diff = data.drop('dates').coherence_VH.values - self.grouped[6].drop('dates').coherence_VH.values
                std_diff = np.abs(np.abs(amplitude_diff) / (np.abs(self.grouped[6].drop('dates').rolling(band=5, min_periods=3, center=True).std().coherence_VH.values) * 3))  ## 3 Sigma

        # Create a 2x1 grid of subplots
        fig, (ax_animation, ax_std_diff) = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))

        # Add the event handler for mouse clicks on the ax_std_diff plot
        fig.canvas.mpl_connect('button_press_event', on_click)

        # Find the indices where std_diff is greater than 1
        significant_indices = np.where(std_diff > 1)

        # Extract the corresponding dates for the significant indices
        try:
            significant_dates = data['dates'][significant_indices].values[:, 0]
        except IndexError:
            significant_dates = []

        titles = [f[17:25] for f in os.listdir(coh_path_list[0]) if f.endswith('.tif')]
        ims = []

        for i in range(len(opened_rasta_stack.read())):  # 39
            im = ax_animation.imshow(opened_rasta_stack.read(i + 1), animated=True, cmap="gray_r", vmin=0.2, vmax=.4)
            if i == 0:
                ax_animation.imshow(opened_rasta_stack.read(i + 1), cmap="gray_r", vmin=0.2, vmax=.4)

            dates = coherence_time_series_dates.values
            vline = ax_std_diff.axvline(x=dates[i], color='r', animated=True)#, label= "5σ Difference")
            ims.append([im, vline])


        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)

        if titles is not None:
            plt.suptitle(
                f"InSAR Correlation Coefficient\n Window: {self.window} \n Dates: {datetime.strptime(titles[0], '%Y%m%d').strftime('%Y-%m-%d')} - {datetime.strptime(titles[-1], '%Y%m%d').strftime('%Y-%m-%d')}")

        # Plot the std_diff data
        ax_std_diff.plot(coherence_time_series_dates, std_diff)

        # Add vertical lines for the significant dates
        if significant_indices[0].size > 0 and np.any(significant_indices[0] != 0):
            for date in significant_dates:
                ax_std_diff.axvline(x=date, color='g', linestyle='--')
            green_line = Line2D([], [], color='g', linestyle='--', label='3σ Difference')

        ax_std_diff.legend(handles=[green_line] + ax_std_diff.get_legend_handles_labels()[0])
        ax_std_diff.set_xlabel('Dates')
        ax_std_diff.set_ylabel('STD from Intact Forest')
        ax_std_diff.set_title('STD Difference between Intact Forest & 1st Disturbance Event')#'Intact Forest STDs for 1st Disturbance Event')

        if savepath is not None:
            writer = animation.FFMpegWriter(
                fps=3, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(f"{savepath}\CCD_animated.mp4", writer=writer)
        plt.show()
        plt.pause(10000)

    def change_mapping(self,opened_rasta_stack):
        '''
        Map forest disturbance events
        somplete the above differencing with the intact forest as baseline, but now I need to rope in the whole image,
        so that it maps forest changes for the whole iamge.

        rasterio.open(f'{stack_path_list}\\{os.listdir(stack_path_list)[0]}'

        '''

        ## for every level/image stacked, compare each pixel in that image to the corresponding single value of coherence obtained from the intact forest values.

        amplitude = opened_rasta_stack.read()

        amplitude = np.where(amplitude > 0, amplitude, np.nan)

        xarray_values = self.grouped[4].drop('dates').coherence_VH.values

        # Reshape the xarray_values to have the same dimensions as amplitude_diff
        xarray_values_reshaped = xarray_values[:, np.newaxis]


        # Perform the subtraction
        amplitude_diff = amplitude - xarray_values_reshaped
        #std_diff = np.abs(np.abs(amplitude_diff) / (np.abs(self.grouped[6].drop('dates').rolling(band=3, min_periods=2, center=True).std().coherence_VH.values) * 3))  ## 3 Sigma

        amplitude_diff = np.abs(amplitude_diff)

        std_xarray_values = np.abs(self.grouped[4].drop('dates').rolling(band=5, min_periods=3, center=True).std().coherence_VH.values)
        std_xarray_values_reshaped = std_xarray_values[:, np.newaxis]

        std_diff = amplitude_diff / std_xarray_values_reshaped

        # Check where the std_diff exceeds 1
        exceeds_threshold = np.where(std_diff > 3, True, np.where(np.isnan(std_diff), np.nan, False))

        # Create a mask of where the NaN values are
        nan_mask = np.isnan(exceeds_threshold)
        not_nan_mask = ~np.isnan(exceeds_threshold)
        # Replace NaN values with 0
        exceeds_threshold_no_nan = np.where(np.isnan(exceeds_threshold), 0, exceeds_threshold)

        # Convert to integer
        std_diff_xr = xarray.DataArray(exceeds_threshold_no_nan.astype(int), dims=['band', 'height', 'width'])

        exceed_counts = std_diff_xr.rolling(band=23, center=False, min_periods=23).sum()
        #exceed_counts = std_diff_xr.rolling(band=21, min_periods=21).sum()
        #std_diff_xr = xarray.DataArray(exceeds_threshold.astype(int), dims=['band', 'height', 'width'])
        #exceed_counts = std_diff_xr.rolling(band=21,min_periods=21).sum()

        exceed_counts_thresholded = np.where(exceed_counts >= 3,True,np.where(np.isnan(exceed_counts), np.nan, False))

        exceed_counts_height = exceed_counts_thresholded.shape[1]
        exceed_counts_width = exceed_counts_thresholded.shape[2]

        #############
        ## Adding Hansen Treecover2000
        #############

        path = 'D:\Data\\'
        hansen_treecover2000 = rasterio.open(path + 'Hansen_GFC-2021-v1.9_treecover2000_00N_110E.tif')
        hansen_lossyear = rasterio.open(path + 'Hansen_GFC-2022-v1.10_lossyear_00N_110E.tif')


        with opened_rasta_stack as exceed_counts_thresholded_ds:
            exceed_counts_thresholded_crs = exceed_counts_thresholded_ds.crs
            exceed_counts_thresholded_transform = exceed_counts_thresholded_ds.transform
            exceed_counts_thresholded_width = exceed_counts_thresholded_ds.width
            exceed_counts_thresholded_height = exceed_counts_thresholded_ds.height
            exceed_counts_thresholded_band_count = exceed_counts_thresholded_ds.count


        # Open the forest data
        with rasterio.open('D:\Data\Hansen_GFC-2021-v1.9_treecover2000_00N_110E.tif') as forest_map_ds:
            forest_mask = forest_map_ds.read()

            # Allocate memory for the reprojected forest mask (will have the same dimensions as exceed_counts_thresholded)
            forest_mask_reprojected = np.empty((exceed_counts_thresholded_band_count, exceed_counts_thresholded_height, exceed_counts_thresholded_width))

            # Reproject the forest mask to match the CRS, transformation and dimensions of exceed_counts_thresholded
            reproject(
                source=forest_mask,
                destination=forest_mask_reprojected,
                src_transform=forest_map_ds.transform,
                src_crs=forest_map_ds.crs,
                dst_transform=exceed_counts_thresholded_transform,
                dst_crs=exceed_counts_thresholded_crs,
                resampling=Resampling.nearest
            )
        forest_mask_boolean = forest_mask_reprojected > 80

        # Convert the forest_mask_boolean to xarray dataarray
        forest_mask_xr = xarray.DataArray(forest_mask_boolean, dims=['band', 'height', 'width'])
        forest_mask_xr_filled = xarray.concat([forest_mask_xr[0]] * 62, dim='band')

        exceed_counts_thresholded_masked = np.where((forest_mask_xr_filled) & (exceed_counts_thresholded > 0) & not_nan_mask, exceed_counts_thresholded, np.nan)
        below_counts_thresholded_masked = np.where((forest_mask_xr_filled) & (exceed_counts_thresholded == 0) & not_nan_mask, exceed_counts_thresholded, np.nan)

        # Assign non-NaN values to 1
        below_counts_thresholded_masked = np.where(~np.isnan(below_counts_thresholded_masked), 1, np.nan)


        ## Total : 89526 pixels - Hansen 252m
        #Average incidence angle over the area is 38.5 degrees.


        ### Predicted values from coherence map

        ## ConFusion Matrix

        with rasterio.open('D:\Data\Hansen_GFC-2022-v1.10_lossyear_00N_110E.tif') as forest_loss_map_ds:
            forest_loss_mask = forest_loss_map_ds.read()

            # Allocate memory for the reprojected forest mask (will have the same dimensions as exceed_counts_thresholded)
            forest_loss_mask_reprojected = np.empty((exceed_counts_thresholded_band_count, exceed_counts_thresholded_height, exceed_counts_thresholded_width))

            # Reproject the forest mask to match the CRS, transformation and dimensions of exceed_counts_thresholded
            reproject(
                source=forest_loss_mask,
                destination=forest_loss_mask_reprojected,
                src_transform=forest_loss_map_ds.transform,
                src_crs=forest_loss_map_ds.crs,
                dst_transform=exceed_counts_thresholded_transform,
                dst_crs=exceed_counts_thresholded_crs,
                resampling=Resampling.nearest
            )


            ##############################
            ## Sample Selection
            ##############################
            # Set a seed for reproducibility
            seed_value = 42
            np.random.seed(seed_value)

            # Filtering based on the conditions
            ## This has an improper shape, i thas a full empty array of 62..
            forest_loss_mask_boolean = np.where((forest_loss_mask_reprojected >= 21) & (forest_loss_mask_reprojected < 22) & forest_mask_xr_filled & not_nan_mask,
                                                forest_loss_mask_reprojected, np.nan)
            disturbed_indices_hansen = np.argwhere(forest_loss_mask_boolean[0] == 21)


            non_disturbed_condition = np.where(((forest_loss_mask_reprojected[0] == 0) | (forest_loss_mask_reprojected[0] >= 22)) & forest_mask_xr_filled & not_nan_mask, forest_loss_mask_reprojected,
                                               np.nan)
            non_disturbed_indices_hansen = np.argwhere((non_disturbed_condition[0])>=0)

            # len(disturbed_indices_hansen)
            # 3287
            # len(non_disturbed_indices_hansen)
            # 310915

            exceed_counts_thresholded_masked_only21 = np.where((forest_loss_mask_reprojected[0] >= 21) & (forest_loss_mask_reprojected[0] < 22), exceed_counts_thresholded_masked[36], np.nan)
            below_counts_thresholded_masked_only21 = np.where((forest_loss_mask_reprojected[0] == 0) | (forest_loss_mask_reprojected[0] >= 22), below_counts_thresholded_masked[36], np.nan)

            # np.nansum(exceed_counts_thresholded_masked_only21)
            # 1381.0
            # np.nansum(below_counts_thresholded_masked_only21)
            # 202315.0

            # np.nansum(exceed_counts_thresholded_masked[36])
            # 64631
            # np.nansum(below_counts_thresholded_masked[36])
            # 294498



            random_disturbed_samples = disturbed_indices_hansen[np.random.choice(len(disturbed_indices_hansen), 348, replace=False)]
            random_non_disturbed_samples = non_disturbed_indices_hansen[np.random.choice(len(non_disturbed_indices_hansen), 50, replace=False)]


            ## sample coordinates:
            # For disturbed samples
            disturbed_rows = random_disturbed_samples[:, 0]
            disturbed_cols = random_disturbed_samples[:, 1]
            disturbed_coords = [forest_loss_mask_reprojected.xy(row, col) for row, col in zip(disturbed_rows, disturbed_cols)]

            # For non-disturbed samples
            non_disturbed_rows = random_non_disturbed_samples[:, 0]
            non_disturbed_cols = random_non_disturbed_samples[:, 1]
            non_disturbed_coords = [forest_loss_mask_reprojected.xy(row, col) for row, col in zip(non_disturbed_rows, non_disturbed_cols)]


            ## rest
            # Extract the actual (reference) values from the Hansen map for the sampled pixels
            actual_disturbed_values = np.ones(len(random_disturbed_samples))
            actual_non_disturbed_values = np.zeros(len(random_non_disturbed_samples))

            # Extract predicted values from the coherence map for the same sampled pixels
            # predicted_disturbed_values_coh = exceed_counts_thresholded_masked[36][tuple(random_disturbed_samples.T)]
            # predicted_non_disturbed_values_coh = exceed_counts_thresholded_masked[36][tuple(random_non_disturbed_samples.T)]


            predicted_disturbed_values_coh = exceed_counts_thresholded_masked_only21[tuple(random_disturbed_samples.T)]
            predicted_non_disturbed_values_coh = exceed_counts_thresholded_masked_only21[tuple(random_non_disturbed_samples.T)]



            # Convert predicted values into binary classification (1 for disturbed, 0 for non-disturbed)
            predicted_disturbed_values_coh = (predicted_disturbed_values_coh > 0).astype(int)
            predicted_non_disturbed_values_coh = (predicted_non_disturbed_values_coh > 0).astype(int)

            # Combine actual and predicted values for generating the confusion matrix
            actual_values = np.concatenate([actual_disturbed_values, actual_non_disturbed_values])
            predicted_values = np.concatenate([predicted_disturbed_values_coh, predicted_non_disturbed_values_coh])




        combined_matrix = np.full(exceed_counts_thresholded_masked[36].shape, np.nan)

        # Assigning 1 where exceed_counts_thresholded_masked is not NaN
        combined_matrix[~np.isnan(exceed_counts_thresholded_masked[36])] = 1

        # Assigning 0 where below_counts_thresholded_masked is 1
        combined_matrix[below_counts_thresholded_masked[36] == 1] = 0



        #
        # ####################
        # ## plotting producer's map colourblind image with labels.
        # ####################
        import matplotlib.colors as mcolors

        # 1. Create binary masks
        disturbed_mask = np.where(forest_loss_mask_boolean[0] == 21, 1, np.nan)
        non_disturbed_mask = np.where((non_disturbed_condition[0] >= 0), 0, np.nan)

        # 2. Combine the masks
        combined_mask = np.where(~np.isnan(disturbed_mask), disturbed_mask, non_disturbed_mask)

        # # 3. Plot the combined matrix
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create a custom colormap
        cmap = mcolors.ListedColormap(['#377eb8', '#ff7f00'])
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Use the transform to set x and y limits
        x_min = 0 + exceed_counts_thresholded_transform.c
        x_max = combined_mask.shape[1] * exceed_counts_thresholded_transform.a + exceed_counts_thresholded_transform.c
        y_min = 0 + exceed_counts_thresholded_transform.f
        y_max = combined_mask.shape[0] * exceed_counts_thresholded_transform.e + exceed_counts_thresholded_transform.f

        # Display the image
        cax = ax.imshow(combined_mask, cmap=cmap, norm=norm, extent=[x_min, x_max, y_min, y_max])

        # Set labels, title, and other plot parameters
        ax.set_xlabel('Longitude', fontsize=16)
        ax.set_ylabel('Latitude', fontsize=16)
        ax.set_title('GLAD Fores loss - 2021', fontsize=18)

        # Create a legend
        from matplotlib.patches import Patch
        legend_labels = {
            '#377eb8': 'Non-Disturbed',
            '#ff7f00': 'Disturbed'
        }
        patches = [Patch(color=color, label=label) for color, label in legend_labels.items()]
        ax.legend(handles=patches, loc='upper right', fontsize=14, frameon=True)

        plt.tight_layout()
        plt.show()
        plt.pause(10)

        ###################
        ## plotting user's map colourblind image with labels.
        ####################
        import matplotlib.patches as mpatches

        # Extract the transform
        transform = exceed_counts_thresholded_transform
        fig, ax = plt.subplots(figsize=(12, 10))

        # Calculate the extent
        left = transform.c
        right = transform.c + transform.a * exceed_counts_thresholded_width
        bottom = transform.f + transform.e * exceed_counts_thresholded_height
        top = transform.f

        extent = [left, right, bottom, top]

        # Create a custom colormap for colorblind-friendly visualization
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#377eb8', '#ff7f00'])  # blue for 'Below Count', orange for 'Exceed Count'

        # Plotting
        plt.imshow(combined_matrix, extent=extent, cmap=cmap)  # Assuming combined_matrix is a 3D array and you're visualizing the first band
        plt.xlabel('Longitude',fontsize=16)
        plt.ylabel('Latitude',fontsize=16)
        plt.title('3σ Disturbance Detections 2021',fontsize=18)

        # Create legend
        legend_labels = {0: 'Non-Disturbed', 1: 'Disturbed'}
        patches = [mpatches.Patch(color='#377eb8', label=legend_labels[0]),
                   mpatches.Patch(color='#ff7f00', label=legend_labels[1])]
        legend=plt.legend(handles=patches, loc='upper right', fontsize=14)
        for text in legend.get_texts():
            text.set_fontsize(14)
        plt.show()
        plt.pause(10)



    def pct_clip(self, array, pct=[2, 98]):
        """
        Clips the input array between percentiles defined by the input percentage list.

        Args:
        - array: input array to be clipped
        - pct: list of two percentiles (low and high)

        Returns:
        - clip: clipped array
        """
        array_min, array_max = np.nanpercentile(array, pct[0]), np.nanpercentile(array, pct[1])
        clip = (array - array_min) / (array_max - array_min)
        clip[clip > 1] = 1
        clip[clip < 0] = 0
        return clip

    def precip_perpdist_plot(self, perp_dist_diff, titles):
        """
        plot mean precipitation data over coherence time period
        along with perpendicular distance between coherence pairs.
         ,
        Args:
        - perp_dist_diff: pd.series of perpendicular distance between coherence pairs
        - coh_mean_df: pd.dataframe of mean coherence values for each date

        """

        grouped = self.cube.groupby('code')

        start = datetime(2021, 1, 1)
        end = datetime(2023, 1, 31)

        data = Daily(96655, start, end)
        data = data.fetch()

        prcp = data.groupby(pd.cut(data.index, self.cube.dates)).mean()['prcp'].to_frame()
        # one less date as this is change between dates
        prcp['dates'] = self.cube.dates[:-1]
        prcp.name = 'Mean Precipitation'


        for i, (code, ds) in enumerate(grouped):
            if i == 0:  ## "Intact Forest" code: 5, sand == 2
                #plt.plot(ds.dates, (ds.coherence_VH+ ds.coherence_VV)/2, label="Average Coherence"  )
                plt.figure(figsize=(15,5))
                plt.plot(ds.dates,  ds.coherence_VV, label="Coherence VV"  )
                plt.plot(ds.dates, ds.coherence_VH, label="Coherence VH"  )
                plt.plot(self.cube.dates[:-1], self.pct_clip(perp_dist_diff), label='Perpendicular Distance',linestyle='--')
                plt.scatter(prcp['dates'], self.pct_clip(prcp.prcp), label='Mean Precipitation')

                #plt.plot( self.cube.dates[:-1],  self.cube["coherence_VH"][:-1].values, label=self.cube["coherence_VH"].name)
                plt.ylim([0, 1])
                plt.legend(prop={'size': 11})
                plt.xlabel('Dates',fontsize=18)
                plt.ylabel('Normalised Values',fontsize=18)
                plt.title(f'{titles[i]}: Effect of Perpendicular Baseline and Precipitation',fontsize=18)#Perp Baseline & Mean Precipitation')
                plt.show()
                plt.pause(1000)

    def radd_alert_plot(self):
        """
        Extracts the RADD alert data for the area of interest and plots the alert dates on the coherence time series.
        """

        print("The `radd_alert_data` method is still under construction.")
        #return

        if self.shp is not None:
            radd = rioxarray.open_rasterio("D:/Data/Radd_Alert.tif", masked=True)#.rio.clip(
                #self.shp.geometry.values, self.shp.crs, from_disk=True)

            radd_cube = make_geocube(self.shp, like=radd, measurements=['code'])
        else:
            return
        radd_cube["alert_date"] = (radd.dims, radd.values, radd.attrs, radd.encoding)
        radd_stats = radd_cube.groupby(radd_cube.code)

        radd_count = radd_stats.count()
        radd_count["alert_dates"] = (radd.dims, radd.values, radd.attrs, radd.encoding)

        radd_count['dates'] = datetime.strptime(radd_cube.alert_date, '%y%j')

        # two slightly different ways to build an xarray datacube

        unique_codes = np.unique(radd_cube.code)[np.isfinite(np.unique(radd_cube.code))]  ## or [~np.isnan()]
        radd_array = pd.DataFrame()
        for ix, i in enumerate(unique_codes):
            polygon = radd_cube.where(radd_cube.code == i).alert_date[1]  # .plot.imshow() plt.pause(100)
            polygon_counts = np.unique(polygon, return_counts=True)
            radd_array[f'polygon{i}_dates'] = pd.to_datetime(pd.Series(polygon_counts[0].astype(str)), format="%y%j.0", errors='coerce')
            radd_array[f'polygon{i}_values'] = pd.Series(polygon_counts[1])

        radd_array1 = radd_array.iloc[:, :2]
        radd_array2 = radd_array.iloc[:, 2:4]
        # radd_array3 = radd_array.iloc[:, 4:6]##only NaTTypes
        radd_array4 = radd_array.iloc[:, 6:8]
        radd_array5 = radd_array.iloc[:, 8:10]
        radd_array6 = radd_array.iloc[:, 10:12]

        radd_arrays_list = [radd_array1, radd_array2, radd_array4, radd_array5, radd_array6]
        # radd_array1.groupby([pd.Grouper(key='polygon3.0_dates', freq='W-MON')]).mean()['polygon3.0_values'].to_frame()

        ##trying to get the radd alert with the highest count in the polygon?
        radd_arrays = [radd_df.groupby([pd.Grouper(key=str(radd_df.columns[0]), freq='D')]).mean()[str(radd_df.columns[1])].to_frame() for radd_df in radd_arrays_list]  # W-MON

        #radd_xarray = pd.concat(radd_arrays).to_xarray()

        ax[i, j].scatter(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_dates'][7:61] ,self.pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_values'][7:61],[.2,99.8]),label='Radd Alert Detections')#raddy_array.index,radd_xarray[f'polygon{np.unique(radd_cube.code)[a]}_values'],[0,100]))#f'polygon{np.unique(radd_cube.code)[ a]}_dates'
        plt.scatter(radd_array[f'polygon{np.unique(radd_cube.code)[4]}_dates'][7:61], self.pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[4]}_values'][7:61], [.2, 99.8]), label='Radd Alert Detections')

        plt.imshow(radd[1])
        plt.title("Radd Alert Detections 2020 - 2023")
        plt.pause(100)

    def hansen_forest_reprojection(self,opened_rasta_stack):

        path = 'D:\Data\\'
        #hansen_treecover2000 = rasterio.open(path + 'Hansen_GFC-2021-v1.9_treecover2000_00N_110E.tif')
        #hansen_lossyear = rasterio.open(path+'Hansen_GFC-2021-v1.9_lossyear_00N_110E.tif')
        hansen_forest = rasterio.open('D:\Data\Hansen_GFC-2021-v1.9_treecover2000_00N_110E.tif')
        forest_mask = hansen_forest.read(1)


        with opened_rasta_stack as exceed_counts_thresholded_ds:
            exceed_counts_thresholded_crs = exceed_counts_thresholded_ds.crs
            exceed_counts_thresholded_transform = exceed_counts_thresholded_ds.transform
            exceed_counts_thresholded_width = exceed_counts_thresholded_ds.width
            exceed_counts_thresholded_height = exceed_counts_thresholded_ds.height
            exceed_counts_thresholded_band_count = exceed_counts_thresholded_ds.count



            # Open the forest data
            #with rasterio.open('D:\Data\Hansen_GFC-2021-v1.9_treecover2000_00N_110E.tif') as forest_map_ds:
            #    forest_mask = forest_map_ds.read(1)

                # Allocate memory for the reprojected forest mask (will have the same dimensions as exceed_counts_thresholded)
            forest_mask_reprojected = np.empty((exceed_counts_thresholded_band_count, exceed_counts_thresholded_height, exceed_counts_thresholded_width))

            # Reproject the forest mask to match the CRS, transformation and dimensions of exceed_counts_thresholded
            reproject(
                source=forest_mask,
                destination=forest_mask_reprojected,
                src_transform=hansen_forest.transform,
                src_crs=hansen_forest.crs,
                dst_transform=exceed_counts_thresholded_transform,
                dst_crs=exceed_counts_thresholded_crs,
                resampling=Resampling.bilinear
            )


            forest_mask_boolean = forest_mask_reprojected > 80

            # Create a boolean mask
            #forest_mask_boolean = forest_mask_reprojected > 66

            # Convert the forest_mask_boolean to xarray dataarray
            forest_mask_xr = xarray.DataArray(forest_mask_boolean, dims=[ 'bands','height', 'width'])
            forest_mask_xr_filled = xarray.concat([forest_mask_xr[0]] * 62, dim='bands')

            opened_rasta_array = xarray.DataArray(opened_rasta_stack.read(), dims=['bands','height', 'width'])
            exceed_counts_thresholded_masked = opened_rasta_array.where((forest_mask_xr_filled) & (opened_rasta_array>0))

    def seasonal_decomposition(self, variable, code, freq=None, model='additive'):
        """
        Applies seasonal decomposition to a variable in the datacube for a given polygon code.

        Args:
            variable (str): The variable to apply seasonal decomposition on (e.g., 'coherence_VH').
            code (int): The polygon code to perform the decomposition on.
            freq (int, optional): The frequency of the seasonal component. Defaults to None (automatic detection).
            model (str, optional): The type of decomposition to perform ('additive' or 'multiplicative'). Defaults to 'additive'.

        Returns:
            decomposed (xr.Dataset): A dataset containing the trend, seasonal, and residual components.
         """
        # Get the time series for the given variable and polygon code
        ts = self.get_time_series(variable, code)

        if freq is None:
            # Automatically detect the frequency of the seasonal component
            freq = self.detect_seasonal_frequency(ts)

        # Apply seasonal decomposition
        decomposition = seasonal_decompose(ts, freq=freq, model=model, extrapolate_trend='freq')

        # Create a new xarray Dataset with the decomposition components
        # decomposed = xarray.Dataset({
        #     'trend': (['time'], decomposition.trend),
        #     'seasonal': (['time'], decomposition.seasonal),
        #     'residual': (['time'], decomposition.resid)
        # }, coords={
        #     'time': ts['time']
        # })

        # return decomposed

    def detect_seasonal_frequency(self, time_series):
        # Implement a function to automatically detect the seasonal frequency
        # of the time series (e.g., using autocorrelation or periodogram)
        # This is a placeholder for your custom implementation
        # raise NotImplementedError("Please implement a method to automatically detect the seasonal frequency")

        """
        Automatically detects the seasonal frequency of the time series using the Lomb-Scargle periodogram.

        Args:
            time_series (xr.DataArray): The input time series.

        Returns:
            freq (int): The detected seasonal frequency.
        """
        # Convert time to floating point representation (number of days since 1970-01-01)
        time_float = time_series['time'].astype(float)

        # Normalize time
        time_norm = (time_float - time_float.min()) / (time_float.max() - time_float.min())

        # Calculate the Lomb-Scargle periodogram
        f = fftfreq(len(time_series), time_norm[1] - time_norm[0])
        f = f[f > 0]
        pgram = lombscargle(time_norm, time_series, f)

        # Find the frequency with the highest power
        dominant_freq = f[np.argmax(pgram)]

        # Convert the dominant frequency to the corresponding period (integer number of days

        # between observations)
        dominant_period = int(round(1 / dominant_freq))

        return dominant_period
        ## Please note that this method assumes that the time series is irregularly sampled. If your time series is regularly sampled (e.g., daily, weekly, or monthly), you can use other techniques, such as the Fast Fourier Transform (FFT), to find the dominant frequency more efficiently.


