import json
import os
def get_filename(json_file):
    files =[]
    with open(json_file) as f:
        annotations = json.load(f)
    for file in annotations:
        files.append(file["file_name"])
    return files
print(len(get_filename("data_mask-rcnn/test/test.json")))
  
