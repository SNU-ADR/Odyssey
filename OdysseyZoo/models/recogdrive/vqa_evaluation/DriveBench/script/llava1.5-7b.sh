GPU=$1

# Define corresponding output names and corruption values in order.
corruptions=(
    ""              # clean
    "NoImage"       # noimage
    "BitError"      # biterror
    "CameraCrash"   # cameracrash
    "Fog"           # fog
    "H256ABRCompression"  # h256
    "LowLight"      # lowlight
    "Rain"          # rain
    "Snow"          # snow
    "Brightness"    # bright
    "ColorQuant"    # colorquant
    "FrameLost"     # framelost
    "LensObstacleCorruption"  # lens
    "MotionBlur"    # motion
    "Saturate"      # saturate
    "ZoomBlur"      # zoom
    "WaterSplashCorruption"   # water
)

outputs=(
    "clean"
    "noimage"
    "biterror"
    "cameracrash"
    "fog"
    "h256"
    "lowlight"
    "rain"
    "snow"
    "bright"
    "colorquant"
    "framelost"
    "lens"
    "motion"
    "saturate"
    "zoom"
    "water"
)

# Loop over the arrays and run the commands.
for i in "${!outputs[@]}"; do
    python inference/internvl.py \
        --model '/high_perf_store3/world-model/yongkangli/data/NAVSIM/internvl_chat/work_dirs/VLA-Navsim/internvl3_8b_finetune_full_baseline_com_his_mix_training_final_no_llava_all_llava_clean_pipelinev5_internvl3' \
        --data data/drivebench-test-final.json \
        --output "res/llava-1.5-7b/${outputs[i]}" \
        --system_prompt prompt.txt \
        --num_processes "${GPU}" \
        --max_model_len 4096 \
        --corruption "${corruptions[i]}"
done
