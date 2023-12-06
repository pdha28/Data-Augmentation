import imgaug.augmenters as iaa
import cv2
import os

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Affine(rotate=(-45, 45)),                    # Rotation
    iaa.Fliplr(0.5),                                 # Horizontal flip
    iaa.Flipud(0.5),                                 # Vertical flip
    iaa.Affine(translate_percent={"x": (-0.2, 0.2)}),# Translation
    iaa.Affine(scale=(0.5, 1.5)),                    # Scaling and zooming
    iaa.Multiply((0.5, 1.5), per_channel=0.5),       # Brightness adjustment
    iaa.AdditiveGaussianNoise(scale=(10, 50)),       # Adding Gaussian noise
    iaa.Crop(percent=(0.1, 0.2)),                    # Crop
    iaa.Pad(percent=(0, 0.2)),                       # Pad
    iaa.AddToHueAndSaturation(value=(-30, 30)),      # Color jitter
    iaa.ElasticTransformation(alpha=50, sigma=5),    # Elastic deformation
    iaa.ShearX((-20, 20)),                           # Shearing
])

# Load an example image (replace with your own image path)
image_path = "C:/Users/Admin/Downloads/Avatar-Teaser-Poster.jpg"
image = cv2.imread(image_path)
images_aug = [seq(image=image) for _ in range(10)]  # Augment 10 variations

# Save augmented images to a directory
output_dir = "F:/Lab/test"
os.makedirs(output_dir, exist_ok=True)
for i, augmented_image in enumerate(images_aug):
    cv2.imwrite(os.path.join(output_dir, f"augmented_image_{i}.jpg"), augmented_image)
