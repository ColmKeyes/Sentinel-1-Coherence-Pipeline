import os
from src.sar_model_input_processor import SARLoader

sen2_stack_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\15000_minalerts"
output_dir = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd_sar"


# Set this to "coh" or "bsc" based on which data you want to process
data_type = "coh"

sar_model_processing = SARLoader(sen2_stack_dir, output_dir,data_type)


# Step 1: Calculate Global Statistics for SAR Bands
global_min, global_max = sar_model_processing.compute_global_min_max(output_dir, bands=[6, 7])

# Step 2: Normalize SAR bands in each stack using global statistics
for file in os.listdir(output_dir):
    if file.endswith(f'_sentinel_agb_normalized_sar.tif') and data_type in file and "T49MDU" in file:
        combined_stack_path = os.path.join(output_dir, file)
        sar_model_processing.apply_mask_and_save_to_sar_bands(combined_stack_path)
        combined_stack_path = os.path.join(output_dir, file)

        # Normalize SAR bands using the global statistics
        normalized_output_path = combined_stack_path.replace('.tif', '_masked_normalized.tif')

        sar_model_processing.normalize_images_global(combined_stack_path,normalized_output_path, global_min, global_max, bands=[6, 7] )

        print(f"Processed and normalized {combined_stack_path} for data type: {data_type}")

# Step 3: Calculate global means and standard deviations after normalization
new_global_means, new_global_stds = sar_model_processing.compute_global_mean_std(output_dir, bands=[6, 7])
print(f"Global means after normalization: {new_global_means}")
print(f"Global standard deviations after normalization: {new_global_stds}")

sar_model_processing.rename_processed_files()
sar_model_processing.convert_dates_to_doy()


