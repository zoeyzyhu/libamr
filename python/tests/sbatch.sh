#!/bin/bash

#SBATCH --job-name=testr
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem-per-cpu=1g
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f23_class
#SBATCH --partition=standard
#SBATCH --output=testr.log
#SBATCH --nodes=3
#SBATCH --exclusive
### --Give all resources to a single Ray worker runtime.
### --Ray can manage the resources internally.
#SBATCH --ntasks-per-node=1


# Get the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the first node and its IP
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Optional step: if a space character detected in IP, it's likely
# that the IP address contains both IPv6 and IPv4. We only extract
# the ipv4 address.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPv6 address detected. We extract the IPv4 address as $ip."
fi

# Set IP and port for the head node
port=6379 # a common port number used for Redis
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

# Start the Ray head node
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" \
    --port=$port --block &

# Optional; may be useful in Ray versions < 1.0.
sleep 10

# Start the Ray worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --block &
    sleep 5
done

# ===== Run test code below =====

current_time=$(date +"%T")
log_file="log_actor_1203_worker_id.txt"
test_description="
    Test description here. \n
"
separator="\n\n\n\n\n===================================================================================================="

echo -e "$separator" >> "$log_file"
echo -e "Test started at: $current_time\n" >> "$log_file"
echo -e $test_description >> "$log_file"
python test_actor.py >> "$log_file"