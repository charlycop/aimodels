#!/usr/bin/env python3

import subprocess
import wandb
import re
import time
import sys


print(sys.path)

# Initialize wandb
wandb.init(project="marian-nmt", name="de-sq-translation")

# Function to parse Marian output
def parse_marian_output(line):
    epoch_match = re.search(r'Ep\. (\d+) :', line)
    update_match = re.search(r'Up\. (\d+) :', line)
    cost_match = re.search(r'Cost (\S+) :', line)
    time_match = re.search(r'Time (\S+)s', line)
    
    if epoch_match and update_match and cost_match and time_match:
        return {
            "epoch": float(epoch_match.group(1)),
            "update": int(update_match.group(1)),
            "cost": float(cost_match.group(1)),
            "time": float(time_match.group(1))
        }
    return None

# Run the bash script
process = subprocess.Popen(['./train.sh'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

start_time = time.time()
last_output_time = start_time

while True:
    try:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        
        current_time = time.time()
        if line:
            print(line.strip())  # Print the output
            last_output_time = current_time
            
            # Parse Marian output and log to wandb
            data = parse_marian_output(line)
            if data:
                wandb.log(data)
        else:
            # No output for a while, check if process is still running
            if current_time - last_output_time > 60:  # Wait for 60 seconds of no output
                print(f"No output for 60 seconds. Process status: {process.poll()}")
                break
        
        # Check if process has been running for too long
        if current_time - start_time > 3600:  # 1 hour timeout
            print("Process has been running for over an hour. Terminating.")
            process.terminate()
            break
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Wait for the process to complete and get the return code
return_code = process.wait()
print(f"Process finished with return code {return_code}")

# Finish wandb run
wandb.finish()