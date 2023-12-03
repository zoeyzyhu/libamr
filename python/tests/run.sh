#!/bin/bash

current_time=$(date +"%T")
log_file="log_actor_1203_worker_id.txt"
test_description="
    Test if a workder_id represent a unique logical CPU core. \n
    \t- Set num_cpus = 4 \n
    \t- Check if we only get 4 unique worker_ids \n
    :: We get 11 worker_ids. \n\n
    \t-Replace workder_id with actor_id and check the difference. \n
    :: We get 11 actor_ids. \n\n
    \t-In the decorator for the actor, set fractional num_cpu = 0.1. \n
    :: Still 11 worker_ids. \n\n
"
separator="\n\n\n\n\n===================================================================================================="

echo -e "$separator" >> "$log_file"
echo -e "Test started at: $current_time\n" >> "$log_file"
echo -e $test_description >> "$log_file"
python test_actor.py >> "$log_file"