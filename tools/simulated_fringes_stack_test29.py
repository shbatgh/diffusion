import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import noise  # Import the noise library
import random

# Function to generate Perlin noise
def generate_perlin_noise(X, Y, seed, scale=100, octaves=6, persistence=0.5, lacunarity=2.0):
    height, width = X.shape
    return np.array([[noise.pnoise2(i / scale, j / scale, octaves=octaves,
                                    persistence=persistence, lacunarity=lacunarity, base=seed)
                      for j in range(width)] for i in range(height)], dtype=np.float32)

# Function to create random surface
def create_random_surface(X, Y, fov, min_std_dev, max_std_dev, num_gaussians, noise_scale):
    surface = np.zeros(X.shape)
    for _ in range(num_gaussians):
        amplitude = np.random.uniform(0.005, 0.015)
        std_dev = np.random.uniform(min_std_dev, max_std_dev)
        center_x = np.random.uniform(0, fov)
        center_y = np.random.uniform(0, fov)
        surface += amplitude * np.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * std_dev ** 2))
    seed = np.random.randint(0, 100)
    perlin_noise = generate_perlin_noise(X, Y, scale=noise_scale, seed = seed)
    surface += perlin_noise * 0.005
    return surface

# Input parameters
fov = 1  # mm
pix = 1440  # number of pixels in each dimension
wavelength = 0.850 / 1000  # mm

input_folder = 'OUT'  # Folder with input images
output_folder = 'IN_WITHOUT_LINES'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a grid (FOV = 1 mm, 1440 x 1440 pixels)
x = np.linspace(0, fov, pix)
y = x
X, Y = np.meshgrid(x, y)

# Parameters for multiple Gaussian surfaces
min_std_dev = 0.005  # Smallest curvature
max_std_dev = 0.5   # Largest curvature
num_gaussians = 5   # Number of Gaussian functions to combine


# Get a list of all .tif files in the input folder
input_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Iterate through each file in the input folder
for file in input_files:
    noise_scale = np.random.uniform(0.0, 1.0)*350 + 50  # Scale for Perlin noise

    input_image_path = os.path.join(input_folder, file)

    # Read the input TIFF image
    input_image = Image.open(input_image_path)
    input_image_array = np.array(input_image)
    input_max = input_image_array.max()
    input_min = input_image_array.min()
    input_avg = input_image_array.mean()
    print(input_max, 'input_max')
    print(input_min, 'input_min')
    # adjusted_input = (input_image_array - input_min) * 200 /(input_max - input_min)

    # # Initial phase
    # Z2 = np.zeros((pix, pix))
    #
    # # Create random surface
    # Z1 = create_random_surface(X, Y, fov, min_std_dev, max_std_dev, num_gaussians, noise_scale)
    #
    # # Generate a random phase step
    # phase_step = np.random.uniform(0, 1.0)  # Adjust the range as needed
    #
    # # Calculate the phase
    # phase1 = np.sin(2 * np.pi * np.abs(Z1 - Z2) / wavelength)
    # phase2 = np.sin(2 * np.pi * np.abs(Z1 - Z2) / wavelength + phase_step)
    # phase_diff = np.abs(np.subtract(phase1, phase2))
    #
    # # Scale to the input data (that is scaled to 0-255)
    # min_val = np.min(phase_diff)
    # max_val = np.max(phase_diff)
    # print(min_val,'min_val')
    # print(max_val,'max_val')
    # # adjusted_phase_diff = (phase_diff - min_val) * 255 / (max_val - min_val)
    # # r = np.random.uniform(0, 1.0)
    # # a = 0.5
    # # b = 0.5 + r*0.1
    # a = 0 * np.random.uniform(0, 1.0)
    # adjusted_phase_diff = (phase_diff - min_val) * (1 - a) / (max_val - min_val) + a
    # # adjusted_phase_diff = (phase_diff - min_val) * 1 / (max_val - min_val)
    #
    # # Multiply with the input image
    # multiplied_image_array = adjusted_phase_diff * input_image_array
    # adjusted_multiplied_image_array = (multiplied_image_array - np.min(multiplied_image_array)) * 255 / (
    #             np.max(multiplied_image_array) - np.min(multiplied_image_array))

    # Generate Gaussian noise
    mean_noise = 0
    sigma_noise = 30  # Adjust this value for more or less noise
    gaussian_noise = np.random.normal(mean_noise, sigma_noise, input_image_array.shape)

    # Add Gaussian noise to the image
    noisy_image_array = input_image_array + gaussian_noise

    ######################################
    # Add line artifacts:
    def add_random_lines_to_image_modified():
        # Load the original image
        width, height = 1440, 1440

        # Create a blank image for drawing lines
        lines_image = Image.new('L', (width, height), 0)

        # Draw random lines with brightness decay
        draw = ImageDraw.Draw(lines_image)
        number_of_lines = random.randint(5, 20)  # Random number of lines
        for _ in range(number_of_lines):
            line_thickness = random.randint(1, 5)
            intensity = random.randint(0, 255)
            decay_factor = intensity // line_thickness*2

            if random.choice([True, False]):  # Randomly choose between horizontal and vertical
                # Horizontal line
                y = random.randint(0, height)
                for i in range(line_thickness):
                    # Calculate brightness for each part of the line
                    line_intensity = max(intensity - abs(i - line_thickness // 2) * decay_factor, 0)
                    draw.line([(0, y + i - line_thickness // 2), (width, y + i - line_thickness // 2)],
                              fill=line_intensity, width=1)
            else:
                # Vertical line
                x = random.randint(0, width)
                for i in range(line_thickness):
                    # Calculate brightness for each part of the line
                    line_intensity = max(intensity - abs(i - line_thickness // 2) * decay_factor, 0)
                    draw.line([(x + i - line_thickness // 2, 0), (x + i - line_thickness // 2, height)],
                              fill=line_intensity, width=1)

        return lines_image


    # Generate the lines image with modified brightness decay
    #lines_image_modified = add_random_lines_to_image_modified()
    lines_image_modified = noisy_image_array
    # To combine with your original image (assuming you have 'noisy_image_array'):
    noisy_image_array_wlines = noisy_image_array + lines_image_modified

    # adjusted_noisy_image_array = (noisy_image_array - np.min(noisy_image_array)) * 255 / (
    #           np.max(noisy_image_array) - np.min(noisy_image_array))

    # Calculate the mean of the noisy image array
    # noisy_image_mean = noisy_image_array.mean()
    #
    # # Scale and shift the noisy image array to match the input image's average
    # scale_factor = input_avg / noisy_image_mean
    # scaled_noisy_image_array = noisy_image_array * scale_factor

    # Clip the values to be between 0 and 255 and convert to uint8
    noisy_image_array_clipped = np.clip(noisy_image_array_wlines, 0, 255).astype(np.uint8)

    # Save the noisy image
    noisy_image_filename = os.path.join(output_folder, file.replace('.png', '.png'))
    noisy_image = Image.fromarray(noisy_image_array_clipped, 'L')
    noisy_image.save(noisy_image_filename)

print(f'Individual noisy images saved in {output_folder}')
