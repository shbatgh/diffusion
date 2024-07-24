import os
from PIL import Image

# Define input and output folders
input_folder = 'OUT'
output_folder = 'OUT'

# Check if output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        # Open the .tif file
        with Image.open(os.path.join(input_folder, filename)) as img:
            # Save it as a .png file in the output folder
            png_filename = os.path.splitext(filename)[0] + '.png'
            img.save(os.path.join(output_folder, png_filename))
        
        # Delete the original .tif file
        os.remove(os.path.join(input_folder, filename))

print("Conversion from .tif to .png completed and old .tif files deleted.")