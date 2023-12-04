#!/bin/bash
file_to_test="test_actor.py"
test_focus="test"

current_date=$(date +"%y%m%d")
current_time=$(date +"%T")
file_name="${file_to_test%.*}"
log_file="log_${file_name}_${current_date}_${test_focus}.txt"
test_description="
    Test description here.
"
separator="\n\n\n\n\n===================================================================================================="

echo -e "$separator" >> "$log_file"
echo -e "Test started at: $current_time\n" >> "$log_file"
echo -e $test_description >> "$log_file"
python test_actor.py >> "$log_file"