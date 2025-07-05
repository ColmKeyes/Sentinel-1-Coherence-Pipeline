# Sentinel-1 Coherence Pipeline

A Python package for processing Sentinel-1 SLC data to produce coherence and backscatter time series for forest disturbance monitoring and InSAR analysis.

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This package provides a complete pipeline for processing Sentinel-1 SAR data to generate coherence and backscatter time series for forest disturbance detection and monitoring. Originally developed for MSc thesis research on "Assessing Sentinel-1 Coherence Measures for Tropical Forest Disturbance Mapping" in Central Kalimantan, Indonesia.

## Features

- **Automated Sentinel-1 Processing**: Complete workflow from scene catalog generation to processed products
- **SNAP Integration**: Leverages ESA SNAP toolbox through SNAPPY Python interface
- **Coherence Analysis**: Multi-temporal coherence calculation with configurable window sizes
- **Backscatter Processing**: Calibrated backscatter time series generation
- **Time Series Analysis**: Advanced analysis tools for disturbance detection
- **Scalable Processing**: Handles large-scale regional processing
- **Flexible API**: Easy integration into existing workflows

## Installation

### Prerequisites

- Python >= 3.7
- ESA SNAP with SNAPPY Python interface configured
- GDAL/OGR libraries
- Earthdata account for ASF downloads

### Install Package

```bash
git clone https://github.com/ColmKeyes/Sentinel-1-Coherence-Pipeline.git
cd Sentinel-1-Coherence-Pipeline
pip install -e .
```

### Dependencies

All dependencies are automatically installed via pip. Key packages include:
- `esa-snappy`: SNAP Python interface
- `asf-search`: ASF API for Sentinel-1 data discovery
- `rasterio`, `xarray`, `geopandas`: Geospatial data processing
- `pyarrow`: Efficient data storage and retrieval

## Quick Start

### Basic Workflow

```python
import sentinel1_coherence as s1c

# 1. Generate Sentinel-1 catalog for your area of interest
catalog_dir = s1c.generate_s1_catalog(
    bbox=(108.0, -4.5, 119.0, 7.0),  # Borneo region (west, south, east, north)
    start_date="2021-06-01",
    end_date="2025-03-31",
    earthdata_token="your_earthdata_token"
)

# 2. Select optimal scene pairs for processing
pairs_csv = s1c.select_scene_pairs(
    catalog_dir=catalog_dir,
    min_days=12,                    # Minimum temporal baseline
    max_perp_baseline=200.0         # Maximum perpendicular baseline (meters)
)

# 3. Download scenes (optional - can use existing data)
downloaded_files = s1c.download_s1_scenes(
    pairs_csv=pairs_csv,
    download_dir="/path/to/slc/data"
)

# 4. Process coherence time series
coherence_results = s1c.run_coherence_workflow(
    pairs_csv=pairs_csv,
    slc_path="/path/to/slc/data",
    output_path="/path/to/output",
    polarizations=['VH', 'VV'],
    window_sizes=[[2, 8], [9, 34], [18, 69]]  # Multiple resolution levels
)
```

### Advanced Analysis

```python
from sentinel1_coherence.core import CoherenceTimeSeries
import pandas as pd

# Load processed coherence data for analysis
pairs_df = pd.read_csv("scene_pairs.csv")
cts = CoherenceTimeSeries(
    asf_df=pairs_df,
    path=["/path/to/coherence/data"],
    stack_path_list="/path/to/stacks",
    window_size=252,  # 252m pixel spacing
    window=[18, 69]   # Coherence estimation window
)

# Build multi-dimensional data cube
cts.build_cube()

# Perform disturbance analysis
cts.multiple_plots(titles=["Intact Forest", "Disturbed Area 1", "Disturbed Area 2"])
cts.stats()  # Statistical analysis and event detection
```

## Package Structure

```
sentinel1_coherence/
├── core/                    # Core processing modules
│   ├── sentinel1slc.py     # SNAP interface for SAR processing
│   ├── coherence_time_series.py  # Time series analysis class
│   ├── sar_model_input_processor.py
│   └── sar_processing_prep.py
├── processing/              # High-level workflow functions
│   └── workflow.py         # Complete processing workflows
└── utils/                   # Utility functions
    ├── calc_coherence_change.py
    ├── ccd_animation.py
    └── plot_ccd.py
```

## Processing Workflow

The package implements a complete Sentinel-1 processing chain:

1. **Catalog Generation** (`generate_s1_catalog`): Query ASF API for available scenes
2. **Pair Selection** (`select_scene_pairs`): Optimal temporal/spatial baseline selection
3. **Data Download** (`download_s1_scenes`): Automated scene retrieval
4. **SAR Processing** (`run_coherence_workflow`): SNAP-based processing pipeline
5. **Time Series Analysis** (`CoherenceTimeSeries`): Multi-dimensional analysis

### Processing Steps (SNAP Chain)

- TOPSAR Split
- Apply Orbit File
- Back-Geocoding (for coherence)
- Coherence Estimation / Calibration (for backscatter)
- TOPSAR Deburst
- Multi-look / Speckle Filtering
- Terrain Correction

## Research Background

This package was developed as part of MSc thesis research focusing on Central Kalimantan, Indonesia - a region known for extensive deforestation activities. The study area was selected using Global Forest Watch RADD alerts and examines forest disturbance events between 2021-2022.

### Key Research Findings

- **Multi-scale Analysis**: Coherence window sizes [2,8], [9,34], [18,69] provide different spatial resolutions
- **Disturbance Detection**: 3-sigma statistical approach for automated event detection
- **Temporal Patterns**: Clear seasonal patterns in agricultural areas vs. forest disturbances
- **Baseline Comparison**: Intact forest areas serve as reference for disturbance quantification

### Study Area: Central Kalimantan, Borneo

<p align="center">
  <img src="images/other/Planet_Disturbance_Event.gif" alt="Forest Disturbance Time Series" width="100%">
</p>

The research demonstrates the effectiveness of coherence measures for forest monitoring, complementing existing backscatter-based systems like Global Forest Watch.

## Configuration

### Window Sizes and Spatial Resolution

The package supports multiple coherence estimation windows:

| Window Size | Pixel Spacing | Use Case |
|-------------|---------------|----------|
| [2, 8] | 28m | High-resolution analysis |
| [9, 34] | 126m | Balanced resolution/noise |
| [18, 69] | 252m | Regional monitoring |

### Environment Variables

```bash
export EARTHDATA_TOKEN="your_earthdata_token"
export SNAP_HOME="/path/to/snap"
```

## Processing Scripts

The `bin/` directory contains the complete processing pipeline as individual scripts, organized in sequential order:

### Core Processing Pipeline
- `step_01_generate_s1_catalog.py` - Generate Sentinel-1 scene catalog using ASF API
- `step_02_scene_pair_selector.py` - Select optimal scene pairs for InSAR processing
- `step_03_compute_pair_baselines.py` - Calculate temporal and spatial baselines
- `step_04_download_s1_scenes.py` - Download Sentinel-1 SLC data from ASF
- `step_05_sentinel1slc_bsc_coh_processing.py` - Main SAR processing (coherence/backscatter)
- `step_06_record_downloaded_slcs.py` - Record processing metadata

### Advanced Processing
- `step_07_sar_model_run_input_processor.py` - SAR model input preparation
- `step_08_sar_run_processing_prep.py` - Advanced processing preparation

### Analysis Scripts
- `step_09_coherence_function_analysis.py` - Coherence function analysis
- `step_10_coherence_time_series_analysis.py` - Time series analysis and visualization

These scripts can be run individually or the equivalent functionality is available through the package API for integration into other projects.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Citation

If you use this package in your research, please cite:

```
Keyes, C. (2023). Assessing Sentinel-1 Coherence Measures for Tropical Forest Disturbance Mapping. 
MSc Thesis, Trinity College Dublin.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ESA for Sentinel-1 data and SNAP toolbox
- ASF for data access and search capabilities
- Global Forest Watch for validation data
- Trinity College Dublin for research support

## Contact

**Colm Keyes**  
Email: keyesco@tcd.ie  
GitHub: [@ColmKeyes](https://github.com/ColmKeyes)
