# #!/bin/bash

# IMAGE_SIZE=256
# PATCH_SIZE=64
# BASE_SAVE_DIR="./results_patches"

# # Define all 16 patches with GPU assignments
# patches=(
#     "0 0 1 64 0 64 6"      # row col top bottom left right gpu
#     "1 1 1 64 64 128 4"
#     "2 2 1 64 128 192 3"
#     "3 3 1 64 192 256 5"
#     "4 0 1 128 0 64 5"
#     "5 1 1 128 64 128 5"
#     "6 2 1 128 128 192 5"
#     "7 3 1 128 192 256 2"
#     "8 0 1 192 0 64 1"
#     "9 1 1 192 64 128 1"
#     "10 2 1 192 128 192 2"
#     "11 3 1 192 192 256 2"
#     "12 0 1 256 0 64 1"
#     "13 1 1 256 64 128 1"
#     "14 2 1 256 128 192 7"
#     "15 3 1 256 192 256 7"
# )
# for num_samples in $(seq 0 1); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/baseline_scar/new_label_5" \
#             --seed $(((num_samples - 1) * 15 + row)) \
#             --box_coords $top $bottom $left $right &

#     done
#     wait
    
# done
# #     CUDA_VISIBLE_DEVICES=7 python3 sample_condition.py \
# #         --model_config=configs/model_config.yaml \
# #         --diffusion_config=configs/diffusion_config.yaml \
# #         --task_config=configs/inpainting_config.yaml \
# #         --save_dir="${BASE_SAVE_DIR}/baseline_scar/new_label_2" \
# #         --seed ${num_samples} \
# #         --box_coords 1 1 0 1 &

# echo "Done!"

###################################################
# python3 super_pixel_generation.py \
#     --input_image=/home/akheirandish3/diffusion-posterior-sampling/data/samples/00092_1.png
# wait

# IMAGE_SIZE=256
# PATCH_SIZE=64
# BASE_SAVE_DIR="./results_patches"

# # Define all 16 patches with GPU assignments
# patches=(
#     "0 0 0 64 0 64 2"      # row col top bottom left right gpu
#     "0 1 1 64 64 128 2"
#     "0 2 2 64 128 192 2"
#     "0 3 3 64 192 256 2"
#     "1 0 4 128 0 64 1"
#     "1 1 5 128 64 128 1"
#     "1 2 6 128 128 192 1"
#     "1 3 7 128 192 256 1"
#     "2 0 8 192 0 64 5"
#     "2 1 9 192 64 128 5"
#     "2 2 10 192 128 192 5"
#     "2 3 11 192 192 256 5"
#     "3 0 12 256 0 64 1"
#     "3 1 13 256 64 128 1"
#     "3 2 14 256 128 192 7"
#     "3 3 15 256 192 256 7"
#     "4 0 16 320 0 64 6"
#     "4 1 17 320 64 128 6"
#     "4 2 18 320 128 192 5"
#     "4 3 19 320 192 256 5"
#     "5 0 20 384 0 64 2"
#     "5 1 21 384 64 128 2"
#     "5 2 22 384 128 192 7"
#     "5 3 23 384 192 256 7"
#     "4 1 24 320 64 128 4"
#     "4 2 25 320 128 192 7"
#     "4 3 26 320 192 256 7"
#     "5 0 27 384 0 64 6"
#     "5 1 28 384 64 128 6"
#     "5 2 29 384 128 192 3"
#     "5 3 30 384 192 256 3"
#     "5 0 31 384 0 64 3"
#     "5 1 32 384 64 128 3"
#     "5 2 33 384 128 192 3"
#     "5 3 34 384 192 256 3"
# )
# for num_samples in $(seq 1 2); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/superpixel_scar_${top}" \
#             --seed ${num_samples} \
#             --box_coords $top $bottom $left $right \
#             --mask_prob 1.0 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png &

#     done
#     wait
# done

# #     --mask_prob 2.0 \
# #     --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png 
# wait
# python3 step_1.py \
#   --global_label_path "/home/akheirandish3/diffusion-posterior-sampling/data/samples/00092_1.png" \
#   --mask_path "/home/akheirandish3/diffusion-posterior-sampling/data/mask.png" \
#   --output_path "/home/akheirandish3/diffusion-posterior-sampling/data/uncertanity_map_1.png" \
#   --n_superpixels 34

# wait

# patches=(
#     "0 0 1 64 0 64 6"      # row col top bottom left right gpu
#     "1 1 1 64 64 128 4"
#     "2 2 1 64 128 192 3"
#     "3 3 1 64 192 256 5"
#     "4 0 1 128 0 64 5"
#     "5 1 1 128 64 128 5"
#     "6 2 1 128 128 192 5"
#     "7 3 1 128 192 256 2"
#     "8 0 1 192 0 64 1"
#     "9 1 1 192 64 128 1"
#     "10 2 1 192 128 192 2"
#     "11 3 1 192 192 256 2"
#     "12 0 1 256 0 64 1"
#     "13 1 1 256 64 128 1"
#     "14 2 1 256 128 192 7"
#     "15 3 1 256 192 256 7"
# )
# for num_samples in $(seq 0 1); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/baseline_scar/new_label_5" \
#             --seed $(((num_samples - 1) * 15 + row)) \
#             --box_coords $top $bottom $left $right \
#             --mask_prob 2.0 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/uncertanity_map_1.png &
#     done
#     wait
    
# done


########################################### #
# #!/bin/bash


# python3 super_pixel_generation_1.py \
#     --input_image=/home/akheirandish3/diffusion-posterior-sampling/data/samples/00092_1.png 

# wait

# IMAGE_SIZE=256
# PATCH_SIZE=64
# BASE_SAVE_DIR="./results_patches"

# # Define all 16 patches with GPU assignments
# patches=(
#     "0 0 0 0 0 64 2"      # row col top bottom left right gpu
#     "0 1 1 0 64 128 2"
#     "0 2 2 0 128 192 2"
#     "0 3 3 0 192 256 2"
#     "1 0 4 0 0 64 1"
#     "1 1 5 0 64 128 1"
#     "1 2 6 0 128 192 1"
#     "1 3 7 0 192 256 1"
#     "2 0 8 0 0 64 5"
#     "2 1 9 0 64 128 5"
#     "2 2 10 0 128 192 5"
#     "2 3 11 0 192 256 5"
#     "3 0 12 0 0 64 1"
#     "3 1 13 0 64 128 1"
#     "3 2 14 0 128 192 7"
#     "3 3 15 0 192 256 7"
#     "4 0 16 0 0 64 6"
#     "4 1 17 0 64 128 6"
#     "4 2 18 0 128 192 5"
#     "4 3 19 0 192 256 5"
#     "5 0 20 0 0 64 2"
#     "5 1 21 0 64 128 2"
#     "5 2 22 0 128 192 7"
#     "5 3 23 0 192 256 7"
#     "4 1 24 0 64 128 4"
#     "4 2 25 0 128 192 7"
#     "4 3 26 0 192 256 7"
#     "5 0 27 0 0 64 6"
# )
# for num_samples in $(seq 1 4); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/superpixel_scar_${top}_${bottom}" \
#             --seed ${num_samples} \
#             --box_coords $top $bottom $left $right \
#             --mask_prob 2.0 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png &

#     done
#     wait
# done

# patches=(
#     "0 0 0 1 0 64 2"      # row col top bottom left right gpu
#     "0 1 1 1 64 128 2"
#     "0 2 2 1 128 192 2"
#     "0 3 3 1 192 256 2"
#     "1 0 4 1 0 64 1"
#     "1 1 5 1 64 128 1"
#     "1 2 6 1 128 192 1"
#     "1 3 7 1 192 256 1"
#     "2 0 8 1 0 64 5"
#     "2 1 9 1 64 128 5"
#     "2 2 10 1 128 192 5"
#     "2 3 11 1 192 256 5"
#     "3 0 12 1 0 64 1"
#     "3 1 13 1 64 128 1"
#     "3 2 14 1 128 192 7"
#     "3 3 15 1 192 256 7"
#     "4 0 16 1 0 64 6"
#     "4 1 17 1 64 128 6"
#     "4 2 18 1 128 192 5"
#     "4 3 19 1 192 256 5"
#     "5 0 20 1 0 64 2"
#     "5 1 21 1 64 128 2"
#     "5 2 22 1 128 192 7"
#     "5 3 23 1 192 256 7"
#     "4 1 24 1 64 128 4"
#     "4 2 25 1 128 192 7"
#     "4 3 26 1 192 256 7"
#     "5 0 27 1 0 64 6"
# )
# for num_samples in $(seq 1 4); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/superpixel_scar_${top}_${bottom}" \
#             --seed ${num_samples} \
#             --box_coords $top $bottom $left $right \
#             --mask_prob 2.0 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png &

#     done
#     wait
# done

# patches=(
#     "0 0 0 2 0 64 2"      # row col top bottom left right gpu
#     "0 1 1 2 64 128 2"
#     "0 2 2 2 128 192 2"
#     "0 3 3 2 192 256 2"
#     "1 0 4 2 0 64 1"
#     "1 1 5 2 64 128 1"
#     "1 2 6 2 128 192 1"
#     "1 3 7 2 192 256 1"
#     "2 0 8 2 0 64 5"
#     "2 1 9 2 64 128 5"
#     "2 2 10 2 128 192 5"
#     "2 3 11 2 192 256 5"
#     "3 0 12 2 0 64 1"
#     "3 1 13 2 64 128 1"
#     "3 2 14 2 128 192 7"
#     "3 3 15 2 192 256 7"
#     "4 0 16 2 0 64 6"
#     "4 1 17 2 64 128 6"
#     "4 2 18 2 128 192 5"
#     "4 3 19 2 192 256 5"
#     "5 0 20 2 0 64 2"
#     "5 1 21 2 64 128 2"
#     "5 2 22 2 128 192 7"
#     "5 3 23 2 192 256 7"
#     "4 1 24 2 64 128 4"
#     "4 2 25 2 128 192 7"
#     "4 3 26 2 192 256 7"
#     "5 0 27 2 0 64 6"
# )
# for num_samples in $(seq 1 4); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/superpixel_scar_${top}_${bottom}" \
#             --seed ${num_samples} \
#             --box_coords $top $bottom $left $right \
#             --mask_prob 2.0 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png &

#     done
#     wait
# done

##############

#!/bin/bash

IMAGE_SIZE=256
PATCH_SIZE=64
BASE_SAVE_DIR="./results_patches"
SAMPLES_DIR="./data/samples"
SAMPLES_NEXT_DIR="./data/samples_next"

# List of sample files to process
SAMPLE_FILES=(
    "00092_1.png"
    "00105_1.png"
    "00107.png"
    "00015_out_pnegg.png"
    "00129_1.png"
    "00098_1.png"
    "00080_1.png"
    "00040_1.png"
    "00044_1.png"
    "00061_1.png"
    "00049_1.png"
    "00065_1.png"
)

# Function to get patches with dynamic bottom coordinate
# get_patches() {
#     local bottom=$1
    
#     local patches=(
#         "0 0 0 $bottom 0 64 4"
#         "0 1 1 $bottom 64 128 0"
#         "0 2 2 $bottom 128 192 1"
#         "0 3 3 $bottom 192 256 1"
#         "1 0 4 $bottom 0 64 3"
#         "1 1 5 $bottom 64 128 3"
#         "1 2 6 $bottom 128 192 3"
#         "1 3 7 $bottom 192 256 3"
#         "2 0 8 $bottom 0 64 6"
#         "2 1 9 $bottom 64 128 5"
#         "2 2 10 $bottom 128 192 5"
#         "2 3 11 $bottom 192 256 1"
#         "3 0 12 $bottom 0 64 6"
#         "3 1 13 $bottom 64 128 6"
#         "3 2 14 $bottom 128 192 6"
#         "3 3 15 $bottom 192 256 3"
#         "4 0 16 $bottom 0 64 5"
#         "4 1 17 $bottom 64 128 6"
#         "4 2 18 $bottom 128 192 6"
#         "4 3 19 $bottom 192 256 6"
#         "5 0 20 $bottom 0 64 6"
#         "5 1 21 $bottom 64 128 7"
#         "5 2 22 $bottom 128 192 7"
#         "5 3 23 $bottom 192 256 7"
#         "4 1 24 $bottom 64 128 7"
#         "4 2 25 $bottom 128 192 7"
#         "4 3 26 $bottom 192 256 7"
#         "5 0 27 $bottom 0 64 7"
#     )
    
#     printf '%s\n' "${patches[@]}"
# }


# get_patches() {
#     local bottom=$1
    
#     local patches=(
#         "0 0 28 $bottom 0 64 4"
#         "0 1 29 $bottom 64 128 0"
#         "0 2 30 $bottom 128 192 1"
#         "0 3 31 $bottom 192 256 1"
#         "1 0 32 $bottom 0 64 3"
#         "1 1 33 $bottom 64 128 3"
#         "1 2 34 $bottom 128 192 7"
#         "1 3 35 $bottom 192 256 7"
#         "2 0 36 $bottom 0 64 6"
#         "2 1 37 $bottom 64 128 5"
#         "2 2 38 $bottom 128 192 5"
#         "2 3 39 $bottom 192 256 1"
#         "3 0 40 $bottom 0 64 6"
#         "3 1 41 $bottom 64 128 7"
#         "3 2 42 $bottom 128 192 6"
#         "3 3 43 $bottom 192 256 3"
#         "4 0 44 $bottom 0 64 5"
#         "4 1 45 $bottom 64 128 6"
#         # "4 2 18 $bottom 128 192 6"
#         # "4 3 19 $bottom 192 256 6"
#         # "5 0 20 $bottom 0 64 6"
#         # "5 1 21 $bottom 64 128 7"
#         # "5 2 22 $bottom 128 192 7"
#         # "5 3 23 $bottom 192 256 7"
#         # "4 1 24 $bottom 64 128 7"
#         # "4 2 25 $bottom 128 192 7"
#         # "4 3 26 $bottom 192 256 7"
#         # "5 0 27 $bottom 0 64 7"
#     )
    
#     printf '%s\n' "${patches[@]}"
# }

get_patches() {
    local bottom=$1
    
    local patches=(
        "0 0 0 $bottom 0 64 3"
        "0 1 1 $bottom 64 128 3"
        "0 2 2 $bottom 128 192 3"
        "0 3 3 $bottom 192 256 4"
        "1 0 4 $bottom 0 64 1"
        "1 1 5 $bottom 64 128 4"
        "1 2 6 $bottom 128 192 4"
        "1 3 7 $bottom 192 256 5"
        "2 0 8 $bottom 0 64 5"
        "2 1 9 $bottom 64 128 5"
        "2 2 10 $bottom 128 192 5"
        "2 3 11 $bottom 192 256 6"
        "3 0 12 $bottom 0 64 6"
        "3 1 13 $bottom 64 128 6"
        "3 2 14 $bottom 128 192 6"
        "3 3 15 $bottom 192 256 7"
        "4 0 16 $bottom 0 64 7"
        "4 1 17 $bottom 64 128 2"
        "4 2 18 $bottom 128 192 7"
        "4 3 19 $bottom 192 256 2"
        "5 0 20 $bottom 0 64 2"
        "5 1 21 $bottom 64 128 1"
        "5 2 22 $bottom 128 192 1"
        # "5 3 23 $bottom 192 256 7"
        # "4 1 24 $bottom 64 128 7"
        # "4 2 25 $bottom 128 192 7"
        # "4 3 26 $bottom 192 256 7"
        # "5 0 27 $bottom 0 64 7"
    )
    
    printf '%s\n' "${patches[@]}"
}

# Function to process patches for a given bottom coordinate
process_patches() {
    local bottom=$1
    local sample_name=$2
    local num_samples=$3
    
    echo "Processing patches with bottom=$bottom for sample=$sample_name"
    
    while IFS= read -r patch; do
        read -r row col top _ left right gpu <<< "$patch"
        
        CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
            --model_config=configs/model_config.yaml \
            --diffusion_config=configs/diffusion_config.yaml \
            --task_config=configs/inpainting_config.yaml \
            --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/scar_very_big_sigma_blurred_${top}_${bottom}" \
            --data_root="${SAMPLES_DIR}" \
            --seed ${num_samples} \
            --box_coords $top $bottom $left $right \
            --mask_prob 0.9 \
            --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png &
    done < <(get_patches "$bottom")
    
    wait
}

process_patches_afterward() {
    local bottom=$1
    local sample_name=$2
    local num_samples=$3
    local seed=$4
    
    echo "Processing patches with bottom=$bottom for sample=$sample_name"
    local seed_save=$(printf "%.0f" $(echo "$num_samples * 10" | bc))
    while IFS= read -r patch; do
        read -r row col top _ left right gpu <<< "$patch"
        
        CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
            --model_config=configs/model_config.yaml \
            --diffusion_config=configs/diffusion_config.yaml \
            --task_config=configs/inpainting_config.yaml \
            --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/scar_second_${top}_${bottom}/${seed_save}" \
            --data_root="${BASE_SAVE_DIR}/samples_${sample_name}/scar_${top}_${bottom}/inpainting/recon" \
            --seed ${seed} \
            --box_coords $top $bottom $left $right \
            --mask_prob ${num_samples} \
            --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png &
    done < <(get_patches "$bottom")
    
    wait
}

process_patches_afterward_with_candidate() {
    local bottom=$1
    local sample_name=$2
    local num_samples=$3
    local seed=$4
    local candidate_idx=$5  # New parameter for specific candidate index
    
    echo "Processing candidate patch $candidate_idx with bottom=$bottom for sample=$sample_name"
    local seed_save=$(printf "%.0f" $(echo "$num_samples * 10" | bc))
    
    # Get all patches into an array
    mapfile -t all_patches < <(get_patches "$bottom")
    
    # Debug: show total patches
    echo "Total patches available: ${#all_patches[@]}"
    
    # Get the specific patch for this candidate index
    if [ "$candidate_idx" -ge "${#all_patches[@]}" ]; then
        echo "Error: candidate_idx $candidate_idx out of range (max: $((${#all_patches[@]} - 1)))"
        return 1
    fi
    
    local patch="${all_patches[$candidate_idx]}"
    
    if [ -z "$patch" ]; then
        echo "Warning: No patch found for index $candidate_idx"
        return 1
    fi
    
    # Debug: show the patch string
    echo "Patch string: '$patch'"
    
    # Parse the patch - note: underscore is placeholder for bottom
    read -r row col top bottom_placeholder left right gpu <<< "$patch"
    
    # Debug: show parsed values
    echo "Parsed values: row=$row col=$col top=$top bottom=$bottom left=$left right=$right gpu=$gpu"
        if ! [[ "$top" =~ ^[0-9]+$ ]] || ! [[ "$bottom" =~ ^[0-9]+$ ]] || ! [[ "$left" =~ ^[0-9]+$ ]] || ! [[ "$right" =~ ^[0-9]+$ ]]; then
        echo "Error: Non-numeric box coordinates detected!"
        echo "  top='$top' bottom='$bottom' left='$left' right='$right'"
        return 1
    fi
    
    # CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
    #     --model_config=configs/model_config.yaml \
    #     --diffusion_config=configs/diffusion_config.yaml \
    #     --task_config=configs/inpainting_config.yaml \
    #     --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/scar_second_${candidate_idx}_${bottom}/${seed_save}" \
    #     --data_root="${BASE_SAVE_DIR}/samples_${sample_name}/scar_${candidate_idx}_${bottom}/inpainting/recon" \
    #     --seed ${seed} \
    #     --box_coords $candidate_idx $bottom $left $right \
    #     --mask_prob ${num_samples} \
    #     --mask_path "/home/akheirandish3/diffusion-posterior-sampling/data/ood_regions_samples_${sample_name}.png"
    #     # --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png

    CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/diffusion_config.yaml \
        --task_config=configs/inpainting_config.yaml \
        --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/scar_third_${candidate_idx}_${bottom}/${seed_save}" \
        --data_root="/home/akheirandish3/diffusion-posterior-sampling/results_patches/img_blur/inpainting/label" \
        --seed ${seed} \
        --box_coords $candidate_idx $bottom $left $right \
        --mask_prob ${num_samples} \
        --mask_path "/home/akheirandish3/diffusion-posterior-sampling/data/ood_regions_samples_${sample_name}.png"
        # --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png
        
}


process_blurry_with_candidate() {
    local bottom=$1
    local sample_name=$2
    local num_samples=$3
    local seed=$4
    local candidate_idx=$5  # New parameter for specific candidate index
    
    echo "Processing candidate patch $candidate_idx with bottom=$bottom for sample=$sample_name"
    local seed_save=$(printf "%.0f" $(echo "$num_samples * 10" | bc))
    
    # Get all patches into an array
    mapfile -t all_patches < <(get_patches "$bottom")
    
    # Debug: show total patches
    echo "Total patches available: ${#all_patches[@]}"
    
    # Get the specific patch for this candidate index
    if [ "$candidate_idx" -ge "${#all_patches[@]}" ]; then
        echo "Error: candidate_idx $candidate_idx out of range (max: $((${#all_patches[@]} - 1)))"
        return 1
    fi
    
    local patch="${all_patches[$candidate_idx]}"
    
    if [ -z "$patch" ]; then
        echo "Warning: No patch found for index $candidate_idx"
        return 1
    fi
    
    # Debug: show the patch string
    echo "Patch string: '$patch'"
    
    # Parse the patch - note: underscore is placeholder for bottom
    read -r row col top bottom_placeholder left right gpu <<< "$patch"
    
    # Debug: show parsed values
    echo "Parsed values: row=$row col=$col top=$top bottom=$bottom left=$left right=$right gpu=$gpu"
        if ! [[ "$top" =~ ^[0-9]+$ ]] || ! [[ "$bottom" =~ ^[0-9]+$ ]] || ! [[ "$left" =~ ^[0-9]+$ ]] || ! [[ "$right" =~ ^[0-9]+$ ]]; then
        echo "Error: Non-numeric box coordinates detected!"
        echo "  top='$top' bottom='$bottom' left='$left' right='$right'"
        return 1
    fi
    # python3 blur_image.py  --read_path "${BASE_SAVE_DIR}/samples_${sample_name}/scar_third_${candidate_idx}_${bottom}/${seed_save}/inpainting/recon/1_00000.png" --save_path "${BASE_SAVE_DIR}/samples_${sample_name}/scar_third_${candidate_idx}_${bottom}/${seed_save}/inpainting/label/1_00000.png"
    # CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
    #     --model_config=configs/model_config.yaml \
    #     --diffusion_config=configs/diffusion_config.yaml \
    #     --task_config=configs/inpainting_config.yaml \
    #     --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/scar_second_${candidate_idx}_${bottom}/${seed_save}" \
    #     --data_root="${BASE_SAVE_DIR}/samples_${sample_name}/scar_${candidate_idx}_${bottom}/inpainting/recon" \
    #     --seed ${seed} \
    #     --box_coords $candidate_idx $bottom $left $right \
    #     --mask_prob ${num_samples} \
    #     --mask_path "/home/akheirandish3/diffusion-posterior-sampling/data/ood_regions_samples_${sample_name}.png"
    #     # --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png

    CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/diffusion_config.yaml \
        --task_config=configs/inpainting_config.yaml \
        --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/scar_forth_${candidate_idx}_${bottom}/${seed_save}" \
        --data_root="${BASE_SAVE_DIR}/samples_${sample_name}/scar_third_${candidate_idx}_${bottom}/${seed_save}/inpainting/recon" \
        --seed ${seed} \
        --box_coords $candidate_idx $bottom $left $right \
        --mask_prob ${num_samples} \
        --mask_path "/home/akheirandish3/diffusion-posterior-sampling/data/ood_regions_samples_${sample_name}.png"
        # --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png
        
}

# Function to process a single sample
process_sample() {
    local filename=$1
    local sample_name="${filename%.*}"
    
    echo "=========================================="
    echo "Processing sample: $filename"
    echo "=========================================="
    # python3 blur_image.py --read_path "/home/akheirandish3/diffusion-posterior-sampling/data/samples_next/${sample_name}.png" --save_path "/home/akheirandish3/diffusion-posterior-sampling/data/samples/${sample_name}_blurred.png"
    
    # Move file from samples_next to samples with new naming
    # if [ -f "${SAMPLES_NEXT_DIR}/${filename}" ]; then
    #     cp "${SAMPLES_NEXT_DIR}/${filename}" "${SAMPLES_DIR}/${filename}"
    #     echo "✓ Moved: ${SAMPLES_NEXT_DIR}/${filename} -> ${SAMPLES_DIR}/${filename}"
    # else
    #     echo "✗ Warning: ${SAMPLES_NEXT_DIR}/${filename} not found!"
    #     return 1
    # fi
    
    # Generate superpixels for this sample
    echo "Generating superpixels..."
    python3 super_pixel_generation.py \
        --input_image="${SAMPLES_DIR}/${sample_name}.png"
    
    wait
    
    # Process patches for different bottom values (0, 1, 2)
    for bottom in 4; do
        for num_samples in $(seq 1 10); do
            process_patches "$bottom" "$sample_name" "$num_samples"
        done
        wait
    done
    # CANDIDATE_INDICES=($(python3 candidat.py --sample_name "samples_$sample_name" 2>&1 | tail -n 1))

    # CANDIDATE_INDICES=($(python3 candidate_p1.py --sample_name "samples_$sample_name" 2>&1 | tail -n 1 | tr -d '[],' ))
    # python3 blur_image.py --sample_name "samples_$sample_name" --save_path "/home/akheirandish3/diffusion-posterior-sampling/data/samples"
    echo "Selected candidates: ${CANDIDATE_INDICES[@]}"
    # for bottom in 4; do
        
    #     for i in {9..10}; do
    #         for candidate_idx in "${CANDIDATE_INDICES[@]}"; do
    #             for seed in {1..3}; do
    #                 num_samples=$(echo "scale=1; $i / 10" | bc)
    #                 process_patches_afterward_with_candidate "$bottom" "$sample_name" "$num_samples" "$seed" "$candidate_idx" &
    #             done
                
    #         done
    #         wait
    #     done
    # done
    # wait
    echo "✓ Completed sample: $filename"
    echo "Moving file back to samples_next..."
    if [ -f "${SAMPLES_DIR}/${sample_name}_blurred.png" ]; then
        mv "${SAMPLES_DIR}/${sample_name}_blurred.png" "${SAMPLES_NEXT_DIR}/${sample_name}_blurred.png"
        echo "✓ Moved back: ${SAMPLES_DIR}/${sample_name}_blurred.png -> ${SAMPLES_NEXT_DIR}/${sample_name}_blurred.png"
    else
        echo "✗ Warning: Could not find ${SAMPLES_DIR}/${filename} to move back!"
    fi
    # for bottom in 4; do
        
    #     for i in {9..10}; do
    #         for candidate_idx in "${CANDIDATE_INDICES[@]}"; do
    #             for seed in {1..10}; do
    #                 num_samples=$(echo "scale=1; $i / 10" | bc)
    #                 process_blurry_with_candidate "$bottom" "$sample_name" "$num_samples" "$seed" "$candidate_idx" &
    #             done
                
    #             wait
    #         done
            
    #     done
    # done
    
    # for bottom in 4; do
    #     for i in {6..10}; do
    #         for seed in {1..3}; do
    #             num_samples=$(echo "scale=1; $i / 10" | bc)
    #             process_patches_afterward "$bottom" "$sample_name" "$num_samples" "$seed"
    #         done
    #     done
    # done

}
process_patches_batch() {
    local bottom=$1
    local sample_name=$2
    local num_samples=$3
    echo "Processing patches with bottom=$bottom for sample=$sample_name"
    
    while IFS= read -r patch; do
        read -r row col top _ left right gpu <<< "$patch"
        
        CUDA_VISIBLE_DEVICES=$gpu python3 sample_batch.py \
            --model_config=configs/model_config.yaml \
            --diffusion_config=configs/diffusion_config.yaml \
            --task_config=configs/inpainting_config.yaml \
            --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/New_masking_eights_sigma_batched_${top}_${bottom}" \
            --data_root="${SAMPLES_DIR}" \
            --seed ${num_samples} \
            --box_coords $top $bottom $left $right \
            --mask_prob 0.9 \
            --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png \
            --num_measurements 4 &
    done < <(get_patches "$bottom")
    
    wait
}

process_sample_batch() {
    local filename=$1
    local sample_name="${filename%.*}"
    
    echo "=========================================="
    echo "Processing sample: $filename"
    echo "=========================================="
    # Move file from samples_next to samples with new naming
    if [ -f "${SAMPLES_NEXT_DIR}/${filename}" ]; then
        cp "${SAMPLES_NEXT_DIR}/${filename}" "${SAMPLES_DIR}/${filename}"
        echo "✓ Moved: ${SAMPLES_NEXT_DIR}/${filename} -> ${SAMPLES_DIR}/${filename}"
    else
        echo "✗ Warning: ${SAMPLES_NEXT_DIR}/${filename} not found!"
        return 1
    fi
    # Generate superpixels for this sample
    echo "Generating superpixels..."
    python3 super_pixel_generation.py \
        --input_image="${SAMPLES_DIR}/${sample_name}.png"
    
    wait
    # Process patches for different bottom values (0, 1, 2)
    for bottom in 4; do
        process_patches_batch "$bottom" "$sample_name" "1"
    done
    echo "✓ Completed sample: $filename"
    echo "Moving file back to samples_next..."
    if [ -f "${SAMPLES_DIR}/${sample_name}.png" ]; then
        mv "${SAMPLES_DIR}/${sample_name}.png" "${SAMPLES_NEXT_DIR}/${sample_name}.png"
        echo "✓ Moved back: ${SAMPLES_DIR}/${sample_name}.png -> ${SAMPLES_NEXT_DIR}/${sample_name}.png"
    else
        echo "✗ Warning: Could not find ${SAMPLES_DIR}/${filename} to move back!"
    fi
}
# Main execution
echo "Starting batch processing..."

# for sample_file in "${SAMPLE_FILES[@]}"; do
#     process_sample "$sample_file"
# done
for sample_file in "${SAMPLE_FILES[@]}"; do
    process_sample_batch "$sample_file"
    wait
done
echo "=========================================="
echo "All samples processed successfully!"
echo "=========================================="