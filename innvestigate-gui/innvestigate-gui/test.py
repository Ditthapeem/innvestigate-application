import shutil
import os

# Set the source and destination paths
source_path = 'static/input/images/download.jpeg_1687515571.656064.jpeg'
destination_path = '../../frontend/public/images/input/download.jpeg_1687515571.656064.jpeg'

current_file_path = os.path.abspath(__file__)
folder_path = "static/"
files = os.listdir(folder_path)
num_files = len(files)
print(num_files)

# Move the file
# shutil.move(source_path, destination_path)
