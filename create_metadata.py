import json
import os

# Path to the Nerves folder
folder_path = 'Nerves'

# Get a list of all files in the Nerves folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Open the metadata.jsonl file for appending
with open("metadata.jsonl", "a") as metadata_file:
    for i, file_name in enumerate(files):
        # Define the data to be written with the file name
        data = {
            "file_name": str(i) + os.path.splitext(file_name)[1],
            "additional_feature": "medical image of nerves taken at Langevin Institute"
        }

        # Convert the data to a JSON line and write it to metadata.jsonl
        json_line = json.dumps(data) + "\n"
        metadata_file.write(json_line)

print(f"Metadata for {len(files)} files has been added to metadata.jsonl")