## Typical_set

Code for the Typical Set experiments.

I modified sample_batch to ensure compatibility with the batch structure and masking pipeline used in my previous setup.
The img_utils file is an updated version of the original img_utils in the util folder, adapted to support different masking strategies.

I also use super_pixel_generation.py for generating superpixels.

Super_pixel_categorized.ipynb contains the implementation of the Typical Set experiments for face datasets.

New_dataset.ipynb contains the implementation for the defect dataset (e.g., MVTec AD).

The script for faces experiment is run_script.h and for MVTec is run_with_mask.sh. 

The directory for data need to be given and the folder for data is only one data per time so that is why we move a sample image from sample_next to the directory to be sampled.
