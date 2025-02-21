import os
import json
import datetime
import yaml
import argparse

class APILogger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.run_dir = None
        if log_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_dir = os.path.join("logs", f"run_{timestamp}")
            os.makedirs(self.run_dir, exist_ok=True)
    
    def log(self, batch_id, data):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                json.dump({"batch": batch_id, "data": data}, f)
                f.write('\n')
        else:
            batch_file = os.path.join(self.run_dir, f"batch_{batch_id:04d}.json")
            with open(batch_file, 'w') as f:
                json.dump(data, f, indent=4)

    def start_new_batch(self):
        return len(os.listdir(self.run_dir)) + 1 if self.run_dir else 0