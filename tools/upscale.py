import os
from PIL import Image

# Define the target size
target_size = (1440, 1440)

# Define the input and output folders
input_folder = 'OUT'

# Ensure the output folder exists
if not os.path.exists(input_folder):
    os.makedirs(input_folder)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        file_path = os.path.join(input_folder, filename)
        
        # Open the image
        with Image.open(file_path) as img:
            # Resize the image
            upscaled_img = img.resize(target_size, Image.ANTIALIAS)
            
            # Save the upscaled image
            upscaled_img.save(file_path)

print(f"All images in {input_folder} have been upscaled to {target_size[0]}x{target_size[1]}")