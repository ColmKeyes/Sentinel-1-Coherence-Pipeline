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
import os
from matplotlib.pyplot import pause
import pandas as pd
import rioxarray
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform
from rasterio.windows import get_data_window, transform
from geocube.api.core import make_geocube
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Daily
from statsmodels.tsa.seasonal import seasonal_decompose
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class CoherenceTimeSeries:
    """
    A class for generating coherence and backscatter time series data-cube from Sentinel-1 SLC products.
    """

    def __init__(self, asf_df, path, stack_path_list, window_size, shp=None, normalised=False):
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
        self.window_size = window_size
        self.normalised = normalised
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

        shp_stack_coh_vh = shp_stacks[0]
        shp_stack_coh_vv = shp_stacks[1] if len(shp_stacks) >= 2 else None
        shp_stack_backscatter_vh = shp_stacks[2] if len(shp_stacks) >= 3 else None
        shp_stack_backscatter_vv = shp_stacks[3] if len(shp_stacks) >= 4 else None


        self.cube = make_geocube(self.shp, like=shp_stack_coh_vh, measurements=["code"])

        # Add variables to cube if corresponding data is available
        if shp_stack_coh_vv is not None:
            shp_stack_coh_vv["code"] = shp_stack_coh_vv.band + 1
            self.cube['coherence_VV'] = (shp_stack_coh_vv.dims, shp_stack_coh_vv.values, shp_stack_coh_vv.attrs, shp_stack_coh_vv.encoding)

        if shp_stack_coh_vh is not None:
            shp_stack_coh_vh["code"] = shp_stack_coh_vh.band + 1
            self.cube["coherence_VH"] = (shp_stack_coh_vh.dims, shp_stack_coh_vh.values, shp_stack_coh_vh.attrs, shp_stack_coh_vh.encoding)

        if shp_stack_backscatter_vv is not None:
            shp_stack_backscatter_vv["code"] = shp_stack_backscatter_vv.band + 1
            self.cube["backscatter_VV"] = (shp_stack_backscatter_vv.dims, shp_stack_backscatter_vv.values, shp_stack_backscatter_vv.attrs, shp_stack_backscatter_vv.encoding)

        if shp_stack_backscatter_vh is not None:
            shp_stack_backscatter_vh["code"] = shp_stack_backscatter_vh.band + 1
            self.cube["backscatter_VH"] = (shp_stack_backscatter_vh.dims, shp_stack_backscatter_vh.values, shp_stack_backscatter_vh.attrs, shp_stack_backscatter_vh.encoding)

        coh_dates = pd.to_datetime(pd.Series(self.titles))
        self.cube['dates'] = coh_dates

    def single_plot(self, titles,plot_code=5):
        """
        Plot the coherence time series for a single shp code in the Cube.
        """
        # Group cube by polygon code
        grouped = self.cube.groupby('code')

        for i, (code, ds) in enumerate(grouped):
            if i == plot_code:  ## "Intact Forest" code: 5
                plt.plot(ds.dates, ds.coherence_VH, label=f'{titles[i]}_VH')
                plt.plot(ds.dates, ds.coherence_VV, label=f'{titles[i]}_VV')
                plt.title(f'{titles[i]}, {self.window_size}m Pixel Spacing')  #Disturbance Event {plot_code}
                plt.legend()
                plt.ylim([0, 1])
                plt.xlabel('Dates')
                plt.ylabel("Correlation Coefficient")
                plt.show()
                plt.pause(100)



    def multiple_plots(self, titles=None):
        """
        Plot the coherence time series for each single polygon code in the Cube.
        """

        coh_bsc_vars = [var for var in self.cube.variables if "coherence" in var or "backscatter" in var]
        # Group by the 'code' variable
        grouped = self.cube.groupby('code')

        # Loop through the grouped data and plot each variable
        fig, ax = plt.subplots(3, 2, figsize=(21, 7), sharey=True, sharex=True)
        for var in coh_bsc_vars:
            ax = ax.flatten()
            for i, (code, data) in enumerate(grouped):
                ax[i].plot(data['dates'], data[var], label=str(var))
                ax[i].set_xlabel('Dates')
                ax[i].set_ylabel("Correlation Coefficient" if "coherence" in var else "Backscatter (dB)")
                ax[i].legend()
                ax[i].set_ylim([0, 1])
                ax[i].tick_params(axis='both', which='both', length=5, width=2)
                #ax[i].autoscale(enable=True, axis='both', tight=True)
                if titles:
                    ax[i].set_title(titles[i])

                fig.suptitle("Disturbance Analysis")
                plt.tight_layout()

        plt.show()
        pause(100)

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

    def precip_perpdist_plot(self, perp_dist_diff):
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
            if i == 5:  ## "Intact Forest" code: 5
                plt.scatter(prcp['dates'], self.pct_clip(prcp.prcp), label='Mean Precipitation')
                plt.scatter(self.cube.dates[:-1], self.pct_clip(perp_dist_diff), label='Perpendicular Distance')
                plt.plot(ds.dates, ds.coherence_VH, label=self.cube["coherence_VH"].name)
                
    #            plt.plot( self.cube.dates[:-1],  self.cube["coherence_VH"][:-1].values, label=self.cube["coherence_VH"].name)
                plt.yticks([0, 1])
                plt.legend()
                plt.show()
                plt.pause(1000)

    def radd_alert_plot(self):
        """
        Extracts the RADD alert data for the area of interest and plots the alert dates on the coherence time series.
        """

        print("The `radd_alert_data` method is still under construction.")
        return

        if self.shp:
            radd = rioxarray.open_rasterio("D:/Data/Radd_Alert.tif", masked=True).rio.clip(
                self.shp.geometry.values, self.shp.crs, from_disk=True)

            radd_cube = make_geocube(self.shp, like=radd, measurements=['code'])
        #else:

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

        radd_arrays = [radd_df.groupby([pd.Grouper(key=str(radd_df.columns[0]), freq='D')]).mean()[str(radd_df.columns[1])].to_frame() for radd_df in radd_arrays_list]  # W-MON

        radd_xarray = pd.concat(radd_arrays).to_xarray()

        # ax[i, j].scatter(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_dates'][7:61] ,pct_clip(radd_array[f'polygon{np.unique(radd_cube.code)[a]}_values'][7:61],[.2,
        # 99.8]),label='Radd Alert Detections')#raddy_array.index,radd_xarray[f'polygon{np.unique(radd_cube.code)[a]}_values'],[0,100]))#f'polygon{np.unique(radd_cube.code)[
        # a]}_dates'




























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
        decomposed = xarray.Dataset({
            'trend': (['time'], decomposition.trend),
            'seasonal': (['time'], decomposition.seasonal),
            'residual': (['time'], decomposition.resid)
        }, coords={
            'time': ts['time']
        })

        return decomposed

    def detect_seasonal_frequency(self, time_series):
        # Implement a function to automatically detect the seasonal frequency
        # of the time series (e.g., using autocorrelation or periodogram)
        # This is a placeholder for your custom implementation
        raise NotImplementedError("Please implement a method to automatically detect the seasonal frequency")

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












