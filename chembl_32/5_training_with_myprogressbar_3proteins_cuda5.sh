#!/bin/bash

function ProgressBar {
    # Set escape codes for clearing the screen and line
    local ERASE_SCREEN_AFTER="\033[0J"
    local ERASE_LINE_BEFORE="\033[1K"
    
    # Set progress bar width (default is 80)
    local bar_width=50
    character_list=(" " "▏" "▎" "▍" "▌" "▋" "▊" "▉" "█" "█")
    
    # Proccess data
    local progress=$(( ($1*100*100/$2)/100 )) # Calculate progress percentage
    local done=$(( (progress*bar_width/10)/10 )) # Calculate number of filled bar characters
    
    # done is 3.0, 4.0,... integer value with a floating type.
    if [[ $done == *".0" ]]; then
      partial=0
    fi
    
    # Process data
    local partial=$(( (${done//./} % 100) / 10 )) # Ger fraction part of the percentage
    local partial_char="${character_list[partial]}" # Get partial character from character list
    local left=$(( bar_width-done-1 )) # Calculate number of empty bar characters, -1 for partial char
    
    # If progress is 100% and done is equal to bar_width, set left to empty to prevent extra empty space at the end
    if (( progress == 100 && done == bar_width )); then
        left=""
    fi
    
    # Calculate elapsed and remaining time in seconds
    local current_time=$(date +%s) # Get current system time in seconds since epoch
    local elapsed=$((current_time - $3)) # Calculate elapsed time
    local remaining=$(( (elapsed * ($2 - $1)) / $1 )) # Estimate remaining time
    local elapsed_days=$(( elapsed / 86400 ))
    local elapsed_hours=$(( (elapsed % 86400) / 3600 ))
    local elapsed_minutes=$(( (elapsed % 3600) / 60 ))
    local elapsed_seconds=$(( elapsed % 60 ))
    local remaining_days=$(( remaining / 86400 ))
    local remaining_hours=$(( (remaining % 86400) / 3600 ))
    local remaining_minutes=$(( (remaining % 3600) / 60 ))
    local remaining_seconds=$(( remaining % 60 ))
    
    
    # Build progressbar string lengths
    local fill empty
    fill=$(printf "%${done}s") # Create a string of filled bar characters
    empty=$(printf "%${left}s") # Create a string of empty bar characters
    
    local progress_bar="${fill// /${character_list[-1]}}${partial_char}${empty// / }"
    local time_string="[%02dd:%02dh:%02dm:%02ds<%02dd:%02dh:%02dm:%02ds]"
    
    # Print progress message and progress bar
    echo -en "$ERASE_LINE_BEFORE" "$ERASE_SCREEN_AFTER" "\r" # Clear the line and screen
    
    # If any description given print it first, if not only print the bar.
    if [[ -n "$4" ]]; then
        printf "\r${4}\nProgress : %3s%% |%s| %3s%% $time_string" \
    "${progress}" "${progress_bar}" "${progress}" \
    "${elapsed_days}" "${elapsed_hours}" "${elapsed_minutes}" "${elapsed_seconds}" \
    "${remaining_days}" "${remaining_hours}" "${remaining_minutes}" "${remaining_seconds}"
    else
        printf "\rProgress : %3s%% |%s| %3s%% $time_string" \
    "${progress}" "${progress_bar}" "${progress}" \
    "${elapsed_days}" "${elapsed_hours}" "${elapsed_minutes}" "${elapsed_seconds}" \
    "${remaining_days}" "${remaining_hours}" "${remaining_minutes}" "${remaining_seconds}"
    fi
    
}


script_path=$(readlink -f "$0")
script_name=$(basename "$script_path")

# Define the hyperparameter lists for each loop
targetid_list=("CHEMBL4282" "CHEMBL4683" "CHEMBL284")
lr_list=(0.01 0.001 0.0001)
dropout_list=(0.1 0.25)
epoch_list=(20 100 200)
fc2_list=(64 128 256)
bs_list=(64 128)
cuda_selection=5

# Necessary file paths for training and outputing
main_training_file="/home/hayriye/DEEPScreen2.2/bin/main_training.py"
bash_output_file="/home/hayriye/DEEPScreen2.2/result_files/bash_outputs/${targetid_list}.out"
bash_error_file="/home/hayriye/DEEPScreen2.2/result_files/bash_outputs/${targetid_list}.err"
bash_progression_output_file="/home/hayriye/DEEPScreen2.2/result_files/bash_outputs/${targetid_list}_only_progression.out"

# Calculate the number of items in each list
targetid_iterations=${#targetid_list[@]}
lr_iterations=${#lr_list[@]}
dropout_iterations=${#dropout_list[@]}
epoch_iterations=${#epoch_list[@]}
fc2_iterations=${#fc2_list[@]}
bs_iterations=${#bs_list[@]}

# Calculate the total number of iterations
total_iterations=$((targetid_iterations * lr_iterations * dropout_iterations * epoch_iterations * fc2_iterations * bs_iterations))

# Inform terminal stdout and the $bash_output_file
echo -e "\nTraining of Proteins: ${targetid_list[@]} having ${total_iterations} hyperparameter combinations all together with Script: ${script_path} is Started!\n" | tee -a $bash_output_file


# Store current directory
current_dir=$(pwd)
echo $(pwd)

count=0
start_time=$(date +%s)
# Loop through the combinations of values in the hyperparameter lists
for targetid in "${targetid_list[@]}"; do
    
    source_folder="/media/ubuntu/8TB/hayriye/DEEPScreen2.2/training_files/target_training_datasets/$targetid"
    source_ttfolder="/media/ubuntu/8TB/hayriye/DEEPScreen2.2/training_files/target_training_datasets"
    destination_folder="/home/hayriye/DEEPScreen2.2/training_files/target_training_datasets"
    destination_file="$destination_folder/$targetid.zip"

    echo -e "\n"
    # Change directory to the parent directory of media targetid
    cd "$source_ttfolder"
    echo $(pwd)
    
    # Working example
    # tar -caf - $targetid/ | tqdm --bytes --total `du -sb $source_folder/ | cut -f1` --desc Compressing $targetid --position 0 --ncols=150 --leave "True" --ascii "True" > $destination_file
    
    
    #zip -qr - $targetid/ | tqdm --bytes --total `du -sb $source_folder/ | cut -f1` \
    #--desc Compressing $targetid --position 0 --ncols=100 --leave "True" \
    #> $destination_file

    # Change directory to the parent directory of home targetid.zip
    cd "$destination_folder"
    echo $(pwd)
    
    # Working examples
    # unzip CHEMBL4282.zip | tqdm --desc Extracting CHEMBL4282.zip --position 0  --ncols=150 --leave "True" --ascii "True" --unit "file" --total $(unzip -vl CHEMBL4282.zip | wc -l) >/dev/null
    # unzip CHEMBL4282.zip | tqdm --desc "Extracting CHEMBL4282.zip" --position 0 --ncols 150 --leave "True" --unit "file" --total $(unzip -l CHEMBL4282.zip | grep -v "Archive:" | wc -l) >/dev/null
    # tar -vxf $destination_file | tqdm --desc Extracting $targetid.zip --position 0 --ncols=150 --leave "True" --ascii "True" --unit "file" --total $(tar -tvf $targetid.zip | wc -l) >/dev/null
    
    
    #unzip $targetid.zip | tqdm --desc Extracting $targetid.zip --position 0 \
    #--ncols=100 --leave "True" \
    #--unit "file" --total $(unzip -vl $targetid.zip | wc -l) >/dev/null

    #rm $destination_file

    # Change back to the original directory
    cd "$current_dir"
    echo $(pwd)
    
    for lr in "${lr_list[@]}"; do
        for dropout in "${dropout_list[@]}"; do
            for epoch in "${epoch_list[@]}"; do
                for fc2 in "${fc2_list[@]}"; do
                    for bs in "${bs_list[@]}"; do
                        let fc1="2*fc2"
                        let "count+=1"
                        en="${targetid}_CNNModel1_${fc1}_${fc2}_${lr}_${bs}_${dropout}_${epoch}_300x300imgs_36rots"
                        
                        Desc="$count / $total_iterations"
                        Desc="${Desc} targetid:${targetid} lr:${lr} dropout:${dropout} epoch:${epoch} / fc1:${fc1} fc2:${fc2}"
                        Desc="${Desc} bs:$bs \n \t  en: ${en} Training please wait..."
                        
                        ProgressBar "$count" $total_iterations "$start_time" "$Desc" \
                        | tee $bash_progression_output_file
                        
                        
                        python $main_training_file --targetid $targetid --model CNNModel1 \
                        --fc1 $fc1 --fc2 $fc2 --lr $lr --bs $bs --dropout $dropout --epoch $epoch \
                        --en $en --cuda $cuda_selection 1>> $bash_output_file 2>> $bash_error_file
                        
                        #sleep 10
                        
                    done;
                done;
            done;
        done;
    done;
    
    #rm -r "${destination_folder}/${targetid}"
    
done;

# Inform terminal stdout and the $bash_output_file
echo -e "\nTraining of Proteins: ${targetid_list[@]} having ${total_iterations} hyperparameter combinations all together with Script: ${script_path} is Finished!\n" | tee -a $bash_output_file
