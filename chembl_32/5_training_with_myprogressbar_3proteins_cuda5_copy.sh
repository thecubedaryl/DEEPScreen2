#!/bin/bash


function ProgressBar {
    # Set escape codes for clearing the screen and line
    local ERASE_SCREEN_AFTER="\033[0J"
    local ERASE_LINE_BEFORE="\033[1K"
    # Set progress bar width (default is 100)
    local bar_width=80
    # Proccess data
    local progress=$(( ($1*100*100/$2)/100 )) # Calculate progress percentage
    local done=$(( (progress*bar_width/10)/10 )) # Calculate number of filled bar characters
    local left=$(( bar_width-done-1 )) # Calculate number of empty bar characters
    # Build progressbar string lengths
    local fill empty
    fill=$(printf "%${done}s") # Create a string of filled bar characters
    empty=$(printf "%${left}s") # Create a string of empty bar characters
    
    # Output message and progress bar
    echo -en "$ERASE_LINE_BEFORE" "$ERASE_SCREEN_AFTER" "\r" # Clear the line and screen
    printf "\r${3}\nProgress : [${fill// /█}}${empty// / }] ${progress}%%" # Print progress message and progress bar
}


# Store current directory
current_dir=$(pwd)
echo $(pwd)

count=0
total_iterations=300000000

for targetid in "CHEMBL4282" "CHEMBL4683" "CHEMBL284"; do
    
    source_folder="/media/ubuntu/8TB/hayriye/DEEPScreen2.2/training_files/target_training_datasets/$targetid"
    source_ttfolder="/media/ubuntu/8TB/hayriye/DEEPScreen2.2/training_files/target_training_datasets"
    destination_folder="/home/hayriye/DEEPScreen2.2/training_files/target_training_datasets"
    destination_file="$destination_folder/$targetid.zip"

    echo -e "\n"
    # Change directory to the parent directory of media targetid
    cd "$source_ttfolder"
    echo $(pwd)
    
    echo -e "a\nb\nc\nd\ne\n"
    
    # tar -caf $destination_file $targetid
    # tar -caf - $targetid/ |
    #> $destination_file
    # zip -r $destination_file $targetid
    # zip -r - $targetid |
    #> $destination_file

    # Compress the current protein with a tqdm progress bar
    tar -caf - $targetid/ | tqdm --bytes --total `du -sb $source_folder/ | cut -f1` \
    --position 0 --desc Compressing $targetid --ncols=150 --leave "True" --ascii "True" \
    > $destination_file

    # Change directory to the parent directory of home targetid.zip
    cd "$destination_folder"
    echo $(pwd)
    
    echo -e "a\nb\nc\nd\ne\n"
    
    # tar -vxf $targetid.zip
    # unzip -v $targetid.zip
    # unzip -q $targetid.zip | tqdm --unit "file" --total $(unzip -v $targetid.zip | wc -l) > /dev/nul

    # Extract the zip file with a tqdm progress bar
    tar -vxf $destination_file | tqdm --desc Extracting $targetid.zip \
    --position 0 --ncols=150 --leave "True" --ascii "True" \
    --bytes --total $(tar -tvf $targetid.zip | wc -l) >/dev/null

    rm $destination_file
    
    echo -e "a\nb\nc\nd\ne\n"
    
    
    echo -n "♦"
    
    for i in {0..100000000}; do
        let "count+=1"
        
        ProgressBar "$count" $total_iterations "${count}/${total_iterations} targetid:CHEMBL4282 lr:0.01 dropout:0.1 epoch:20 / fc1:64 fc2:128 \n \t en:experiment training please wait..."
    done

        
    
done;

# Change back to the original directory
cd "$current_dir"
echo $(pwd)

printf '\nFinished!\n'


# --bar-format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{unit}/s]"
# It is not very smooth though