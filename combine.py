import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import random

# Define the input and output folders
input_folder = 'OUT'
output_folder = 'noisy'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith('.png'):
        input_image_path = os.path.join(input_folder, filename)
        
        # Open the input image
        input_image = Image.open(input_image_path)
        input_image = np.array(input_image)
        input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))

        # Open the noise image
        noise_image_path = f"synthetic/{i}.png"
        noise_i = Image.open(noise_image_path)
        noise_i = np.array(noise_i)
        noise_i = (noise_i - np.min(noise_i)) / (np.max(noise_i) - np.min(noise_i))
        noise_i = noise_i * 2.2

        # Generate Gaussian noise
        mean_noise = 0
        sigma_noise = 150  # Adjust this value for more or less noise
        gaussian_noise = np.random.normal(mean_noise, sigma_noise, input_image.shape)
        noisy_image = input_image + noise_i

        # Normalize the combined image to the range [0, 1]
        noisy_image = (noisy_image - np.min(noisy_image)) / (np.max(noisy_image) - np.min(noisy_image))

        # Scale to the range [0, 255]
        noisy_image = noisy_image * 255

        # Clip the values to be between 0 and 255 and convert to uint8
        noisy_image_array_clipped = np.clip(noisy_image, 0, 255).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_image_array_clipped, 'L')

        # Save the processed image
        output_image_path = os.path.join(output_folder, filename)
        noisy_image.save(output_image_path)

print(f"All images in {input_folder} have been processed and saved to {output_folder}")