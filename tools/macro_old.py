import numpy as np
import imageio
from skimage import img_as_uint, img_as_float
from skimage import io
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
from PIL import Image

# Load the image stack
input_path = 'test_image.tif'
Image.MAX_IMAGE_PIXELS = None

image_stack = io.imread(input_path)
io.imsave('test_imadfge.tif', image_stack)
# Step 1: Check Slice Intensity and delete low-intensity slices
threshold = 100
filtered_stack = []
for slice in image_stack:
    mean_intensity = np.mean(slice)
    if mean_intensity >= threshold:
        filtered_stack.append(slice)

filtered_stack = np.array(filtered_stack)

# Step 2: Normalization and Subtraction of Average Projection
new_mean = 4000
normalized_stack = []
for slice in filtered_stack:
    current_mean = np.mean(slice)
    factor = new_mean / current_mean
    normalized_slice = slice * factor
    normalized_stack.append(normalized_slice)

normalized_stack = np.array(normalized_stack)

# Create an average intensity projection
average_projection = np.mean(normalized_stack, axis=0)

# Subtract the average projection from each slice
subtracted_stack = []
for slice in normalized_stack:
    subtracted_slice = slice - average_projection
    subtracted_stack.append(subtracted_slice)

subtracted_stack = np.array(subtracted_stack)

# Rescale intensity to 16-bit
rescaled_stack = rescale_intensity(subtracted_stack, out_range='uint16')

# Enhance contrast similar to ImageJ with saturation set to 3.5
saturation = 3.5
p_low, p_high = np.percentile(rescaled_stack, (saturation, 100 - saturation))
contrast_enhanced_stack = rescale_intensity(rescaled_stack, in_range=(p_low, p_high), out_range='uint16')

# Save the processed image stack
output_path = 'processed_stack.tif'
imsave(output_path, img_as_uint(contrast_enhanced_stack))

print("Processing completed and saved to", output_path)