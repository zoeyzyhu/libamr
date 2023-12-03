#!/bin/bash

log_file="log_actor_1203.txt"
current_time=$(date +"%T")
test_description="
    Test if a workder_id represent a unique logical CPU core. \n
    \t- Set num_cpus = 4 \n
    \t- Check if we only get 4 unique worker_ids \n
    :: We get 11 worker_ids. \n\\n
    Replace workder_id with actor_id and check the difference. \n\n
"
separator="===================================================================================================="

echo "\n\n\n\n\n$separator" >> "$log_file"
echo "Test started at: $current_time\n" >> "$log_file"
echo $test_description >> "$log_file"
python test_actor.py >> "$log_file"