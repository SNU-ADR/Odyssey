#!/bin/bash

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "tmux could not be found. Please install tmux to continue."
    exit 1
fi

rm -rf ./tmp_download
mkdir -p ./tmp_download

# Create a new tmux session
SESSION_NAME="download_session"
tmux new-session -d -s "$SESSION_NAME" -n "main_window"

# Function to download and extract files in a tmux pane
download_file_in_pane() {
    url=$1
    filename=$2
    tmux send-keys -t "$SESSION_NAME" "echo Downloading $url as $filename...; aria2c -x 16 -s 64 -j 8 -d ./tmp_download -o '$filename' '$url'" C-m
}

# Function to extract and sync files in a tmux pane
extract_and_sync_in_pane() {
    tar_file=$1
    folder_name=$2
    tmux send-keys -t "$SESSION_NAME" "echo 'Extracting $tar_file...'; tar -xzf '$tar_file' -C ./tmp_download; rm '$tar_file'" C-m

    # Sync extracted files from the folder inside tmp_download to the target directory
    tmux send-keys -t "$SESSION_NAME" "echo Syncing files from $folder_name to trainval_sensor_blobs/trainval...; rsync -rv ./tmp_download/$folder_name/ ./trainval_sensor_blobs/trainval" C-m

    # Remove the extracted folder after syncing
    tmux send-keys -t "$SESSION_NAME" "echo Cleaning up $folder_name...; rm -rf ./tmp_download/$folder_name" C-m
}

# Function to handle processing each split file in its own tmux pane
process_split_in_pane() {
    split=$1

    # Dynamically name the files for current and history
    file_name_current="navtrain_current_${split}.tgz"
    file_name_history="navtrain_history_${split}.tgz"

    # Download the first file (current split) in the current pane
    download_file_in_pane "https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_${split}.tgz" "$file_name_current"
    
    # Extract and sync files in the current pane (after the download is complete)
    extract_and_sync_in_pane "./tmp_download/$file_name_current" "current_split_${split}"
    
    # Once the first download starts, split the window to handle the second download
    tmux split-window -h -t "$SESSION_NAME"

    # Download and extract the second file (history split) in the new pane
    download_file_in_pane "https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_history_${split}.tgz" "$file_name_history"
    
    # Extract and sync files in the new pane (after the second download is complete)
    extract_and_sync_in_pane "./tmp_download/$file_name_history" "history_split_${split}"
}

# Download and extract the OpenScene dataset in its own tmux pane
tmux new-window -t "$SESSION_NAME" -n "openscene"
download_file_in_pane "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz" "openscene_metadata_trainval.tgz"
tmux send-keys -t "$SESSION_NAME" "echo 'Extracting OpenScene metadata...'; tar -xzf ./tmp_download/openscene_metadata_trainval.tgz -C ./tmp_download; rm ./tmp_download/openscene_metadata_trainval.tgz; mv ./tmp_download/openscene-v1.1/meta_datas trainval_navsim_logs; rm -rf ./tmp_download/openscene-v1.1" C-m

# Create target directory for extracted files
mkdir -p trainval_sensor_blobs/trainval

# Create tmux panes for downloading and processing splits
for split in {1..4}; do
    tmux new-window -t "$SESSION_NAME" -n "split_${split}"  # Create a new window for each split
    process_split_in_pane "$split"
done

tmux attach-session -t "$SESSION_NAME"  # Attach to the tmux sessionw