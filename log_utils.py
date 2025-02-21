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
        """Log data either to a single file or batch files"""
        if self.log_file:
            # Append to single file with newline
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump({"batch": batch_id, "timestamp": datetime.datetime.now().isoformat(), "data": data}, f)
                f.write('\n')
        else:
            # Create new batch file with timestamp to prevent overwrites
            timestamp = datetime.datetime.now().strftime('%H%M%S')
            batch_file = os.path.join(self.run_dir, f"batch_{batch_id:04d}_{timestamp}.json")
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch": batch_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "data": data
                }, f, indent=4)

    def start_new_batch(self):
        """Get next batch ID based on existing files"""
        if self.run_dir:
            existing_files = [f for f in os.listdir(self.run_dir) if f.startswith('batch_')]
            if not existing_files:
                return 1
            # Extract batch numbers from filenames
            batch_numbers = [int(f.split('_')[1]) for f in existing_files]
            return max(batch_numbers) + 1
        return 0