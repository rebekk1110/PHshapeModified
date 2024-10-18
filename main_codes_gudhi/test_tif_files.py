import os
from mdl1_bolPH_gu import main_basicOL

# Define the test files
test_files = {
    'testBig': 'test_images/trd_1_crop_188_3-kopi.tif',
    'testSmall': 'test_images/trd_1_crop_224_4-kopi.tif'
}

# Set up parameters
data_root_folder = 'test_images'
data_type = '.tif'
out_folder = 'output'
bld_list = list(test_files.keys())
pre_cloud_num = 5000
down_sample_num = 400
bfr_tole = 0.5
bfr_otdiff = 0.01
is_use_saved_bfr = False
savename_bfr = 'bfr_optim.csv'
is_unrefresh_save = False
is_Debug = True

# Create output folder if it doesn't exist
os.makedirs(out_folder, exist_ok=True)

# Run main_basicOL for each test file
main_basicOL(
    data_root_folder=data_root_folder,
    data_type=data_type,
    out_folder=out_folder,
    bld_list=bld_list,
    pre_cloud_num=pre_cloud_num,
    down_sample_num=down_sample_num,
    bfr_tole=bfr_tole,
    bfr_otdiff=bfr_otdiff,
    is_use_saved_bfr=is_use_saved_bfr,
    savename_bfr=savename_bfr,
    is_unrefresh_save=is_unrefresh_save,
    is_Debug=is_Debug
)

print("Processing complete. Check the 'output' folder for results.")

