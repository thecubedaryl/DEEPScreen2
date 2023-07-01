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

# Application of ProgressBar to DEEPScreen for current proteins

count=0 # Unutma güncelle!!

# 3 x 3 x 2 x 3 x 3 x 2 = 324
for targetid in 'CHEMBL2409', 'CHEMBL5658', 'CHEMBL3066'; do

    zip -r /media/ubuntu/8TB/hayriye/DEEPScreen2.2/training_files/target_training_datasets/CHEMBL4282.zip /media/ubuntu/8TB/hayriye/DEEPScreen2.2/training_files/target_training_datasets/CHEMBL4282
    
    mv /media/ubuntu/8TB/hayriye/DEEPScreen2.2/training_files/target_training_datasets/$targetid.zip /home/hayriye/DEEPScreen2.2/training_files/target_training_datasets/
    
    unzip /home/hayriye/DEEPScreen2.2/training_files/target_training_datasets/$targetid.zip

    for lr in 0.01 0.001 0.0001; do
        for dropout in 0.1 0.25; do
            for epoch in 20 100 200; do
                for fc2 in 64 128 256; do
                    for bs in 64 128; do
                        let fc1="2*fc2"
                        let "count+=1"
                        en="${targetid}_CNNModel1_${fc1}_${fc2}_${lr}_${bs}_${dropout}_${epoch}_300x300imgs_36rots"

                        ProgressBar "$count" 324 "$count/324 targetid:$targetid lr:$lr dropout:$dropout epoch:$epoch / fc1:$fc1 fc2:$fc2 bs:$bs \n \t en: ${en} training please wait..."
                        
                        python /home/hayriye/DEEPScreen2.2/bin/main_training.py --targetid $targetid --model CNNModel1 --fc1 $fc1 --fc2 $fc2 --lr $lr --bs $bs --dropout $dropout --epoch $epoch --en $en --cuda 5 1>> /home/hayriye/DEEPScreen2.2/result_files/bash_outputs/AKT1_cuda1.out 2>> /home/hayriye/DEEPScreen2.2/result_files/bash_outputs/AKT1_cuda2.err

    rm -r /home/hayriye/DEEPScreen2.2/training_files/target_training_datasets/$targetid

done;done;done;done;done;done;

printf '\nFinished!\n'
