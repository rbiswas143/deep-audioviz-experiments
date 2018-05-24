import fma_utils
import dataset
import os
import shutil
import json

all_data_path = 'cached/final_2k_fps5_genre_9/full_predictions.json'
export_dir = 'cached/final_2k_fps5_genre_9'

with open(all_data_path, 'rb') as dp:
    all_data = json.load(dp)

paths = [fma_utils.get_audio_path(dataset.audio_path, track['index']) for track in all_data['tracks']]

for path in paths:
    dest_path = os.path.join(export_dir, path)
    parent_dir = os.path.dirname(dest_path)
    os.makedirs(parent_dir, exist_ok=True)
    shutil.copy(path, dest_path)
