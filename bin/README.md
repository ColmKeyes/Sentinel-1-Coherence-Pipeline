# Processing Scripts

This directory contains the complete Sentinel-1 coherence processing pipeline as individual scripts, organized in sequential order.

## Processing Pipeline

### Core Processing Steps

1. **step_01_generate_s1_catalog.py**
   - Generate Sentinel-1 scene catalog using ASF API
   - Creates partitioned Parquet dataset for efficient querying
   - Requires: Earthdata token, bounding box coordinates

2. **step_02_scene_pair_selector.py**
   - Select optimal scene pairs for InSAR processing
   - Filters by temporal and perpendicular baselines
   - Outputs: CSV file with master/slave pairs

3. **step_03_compute_pair_baselines.py**
   - Calculate detailed temporal and spatial baselines
   - Enhances pair selection with precise baseline metrics

4. **step_04_download_s1_scenes.py**
   - Download Sentinel-1 SLC data from ASF
   - Automated retrieval based on scene pairs
   - Requires: Earthdata authentication

5. **step_05_sentinel1slc_bsc_coh_processing.py**
   - Main SAR processing using SNAP/SNAPPY
   - Generates coherence and backscatter products
   - Supports multiple window sizes and polarizations

6. **step_06_record_downloaded_slcs.py**
   - Record processing metadata and status
   - Track completed processing runs

### Advanced Processing

7. **step_07_sar_model_run_input_processor.py**
   - Prepare inputs for SAR modeling workflows
   - Process metadata for advanced analysis

8. **step_08_sar_run_processing_prep.py**
   - Advanced processing preparation
   - Setup for specialized SAR analysis

### Analysis Scripts

9. **step_09_coherence_function_analysis.py**
   - Analyze coherence functions and statistics
   - Generate coherence-specific metrics

10. **step_10_coherence_time_series_analysis.py**
    - Complete time series analysis and visualization
    - Disturbance detection and mapping
    - Generate final analysis products

## Usage

### Individual Script Execution
```bash
# Run individual steps
python step_01_generate_s1_catalog.py
python step_02_scene_pair_selector.py
# ... continue with subsequent steps
```

### Package API (Recommended)
```python
import sentinel1_coherence as s1c

# Use high-level workflow functions instead
catalog_dir = s1c.generate_s1_catalog(...)
pairs_csv = s1c.select_scene_pairs(...)
results = s1c.run_coherence_workflow(...)
```

## Dependencies

- Python >= 3.7
- ESA SNAP with SNAPPY configured
- All dependencies listed in `requirements.txt`
- Earthdata account for ASF downloads

## Environment Setup

```bash
export EARTHDATA_TOKEN="your_token"
export SNAP_HOME="/path/to/snap"
```

## Notes

- Scripts are designed to be run in sequence
- Each step depends on outputs from previous steps
- The package API provides equivalent functionality with better error handling
- For new projects, use the package API rather than individual scripts
