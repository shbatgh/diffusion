import numpy as np
from PIL import Image
import os
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cv2

def create_circular_kernel(radius, diameter):
    """Create a circular kernel with a given radius and diameter."""
    Y, X = np.ogrid[:diameter, :diameter]
    dist_from_center = np.sqrt((X - radius)**2 + (Y-radius)**2)
    kernel = dist_from_center <= radius
    return kernel.astype(np.float32) / kernel.sum()


def random_histogram_variation(image, alpha, beta):
    # Convert image to float32 if it's not already
    image = image.astype(np.float32)

    # Normalize the image to the range 0-1
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Apply random brightness and contrast adjustments
    new_image = alpha * image + beta

    # Clamp values to the range 0-1
    new_image = np.clip(new_image, 0, 1)

    return new_image

def generate_controlled_perlin_noise(width, height, scale=100, octaves=1, persistence=0.5, lacunarity=2.0):
    # Helper functions
    def lerp(a, b, x):
        return a + x * (b - a)

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(h, x, y):
        vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        g = vectors[h % 4]
        return g[:,:,0] * x + g[:,:,1] * y

    # Initialize noise grid
    grid = np.zeros((height, width))

    # Scale adjustment
    x_scale = width / scale
    y_scale = height / scale

    for octave in range(octaves):
        # Adjust scale for octaves
        octave_scale = 1 / (lacunarity ** octave)
        amplitude = persistence ** octave

        # Random gradients for octave
        gradient_x = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        gradient_y = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Scale and amplitude adjustment
        x = np.linspace(0, 1, width, endpoint=False) * x_scale * octave_scale
        y = np.linspace(0, 1, height, endpoint=False) * y_scale * octave_scale
        xv, yv = np.meshgrid(x, y)

        # Compute distance vectors for octave
        dx = xv - xv.astype(int)
        dy = yv - yv.astype(int)

        # Dot products
        n00 = gradient(gradient_x, dx, dy)
        n10 = gradient(gradient_x, dx - 1, dy)
        n01 = gradient(gradient_x, dx, dy - 1)
        n11 = gradient(gradient_x, dx - 1, dy - 1)

        # Fade curves
        u = fade(dx)
        v = fade(dy)

        # Interpolation
        nx0 = lerp(n00, n10, u)
        nx1 = lerp(n01, n11, u)
        nxy = lerp(nx0, nx1, v)

        # Add octave contribution to grid
        grid += nxy * amplitude

    # # Normalize to the full dynamic range (0 to 255)
    #     # normalized_noise = ((grid - grid.min()) / (grid.max() - grid.min()) * 255).astype(np.uint8)
    return grid

# Parameters for noise generation
width, height = 1440, 1440
scale = 100
octaves = 1
persistence = 0.1
lacunarity = 10
kernel_size = 20  # Mean filter kernel size

radius1 = 15  # Radius for circular mean filter
diameter1 = 2 * radius1 + 1  # Diameter of the kernel
radius2 = 1  # Radius for circular mean filter
diameter2 = 2 * radius2 + 1  # Diameter of the kernel

# Folder path for saving images
folder_path = r"OUT"
# Ensure the folder exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Create circular mean filter kernel
circular_kernel_1 = create_circular_kernel(radius1, diameter1)
circular_kernel_2 = create_circular_kernel(radius2, diameter2)

# Generate and save 100 images
for i in range(200):
    # Generate Perlin noise
    noise_image = generate_controlled_perlin_noise(width, height, scale, octaves, persistence, lacunarity).astype(np.float32)

    # Apply circular mean filter using filter2D from OpenCV
    noise_image = cv2.filter2D(noise_image, -1, circular_kernel_1)

    # Normalize to the full dynamic range (0 to 1)
    normalized_noisy_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min()) * 1

    avg = np.mean(normalized_noisy_image)

    # shifting histogram from the middle to the edge.
    # Clipping does not create a problem because we just remove values as they did not exist in structures
    # On the opposite ABS was creating a non-linearity
    new_image = np.clip(normalized_noisy_image*255/avg - 255, 0, 255)

    # Normalize the image to range [0, 1]
    normalized_image = new_image / 255.0
    # Apply the power law transformation
    gamma = np.random.uniform(0.5, 1.0)
    transformed_image = np.power(normalized_image, gamma)
    # Scale back to range [0, 255]
    transformed_image = transformed_image * 255

    normalized_noisy_image = transformed_image.astype(np.uint8)




    # # Apply double square to exoand contrast
    # print(np.min(filtered_image1), 'min')
    # filtered_image1 = filtered_image1**8

    # Apply circular mean filter using filter2D from OpenCV
    # filtered_image2 = cv2.filter2D(filtered_image1, -1, circular_kernel_2)

    # Optional subtract the average value of the entire image and take absolute
    # avg_value = np.mean(filtered_image1)
    # noisy_image_diff = filtered_image1 - avg_value

    # Normalize to the full dynamic range (0 to 1)
    # normalized_noisy_image_diff = ((noisy_image_diff - noisy_image_diff.min()) / (noisy_image_diff.max() - noisy_image_diff.min()) * 1)

    # Clip the values to be between 0 and 1
    # print(np.max(normalized_noise), 'max')
    # print(np.min(normalized_noise), 'min')
    # avg_value = np.mean(normalized_noise)
    # print(avg_value, 'avg_value')
    # print(avg_value, 'avg_value')
    # noisy_image_diff = normalized_noise - avg_value
    # # noisy_image_array_clipped = np.clip(normalized_noise - avg_value, 0, 1)
    #
    # # Squeeze the negative values
    # noisy_image_diff[noisy_image_diff < 0] *= 0.2

    # Apply random histogram variation
    # alpha = np.random.uniform(0.8, 1.5)  # For example, in the range 1.0 to 1.5
    # beta = np.random.uniform(-0.1, 0.1)  # For example, in the range -0.2 to 0.2
    # filtered_image_randhist = random_histogram_variation(filtered_image1, alpha, beta)
    #
    # # Normalize again to the full dynamic range (0 to 255)
    # normalized_noise2 = ((filtered_image_randhist - filtered_image_randhist.min()) / (filtered_image_randhist.max() - filtered_image_randhist.min()) * 255).astype(np.uint8)

    # Image file name as consecutive integers
    image_name = f"{i}.png"
    file_path = os.path.join(folder_path, image_name)

    # Convert the filtered noise to an image and save it
    image = Image.fromarray(normalized_noisy_image)

    # # Ensure the image is in a mode that can be saved as PNG (e.g., 'L' for grayscale)
    # if image.mode != 'L':
    #     image = image.convert('L')

    image.save(file_path)

    print(f"Image {i + 1} saved to {file_path}")