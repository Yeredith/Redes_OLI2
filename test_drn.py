import tensorflow as tf
import numpy as np
import cv2
import math
from model_drn import DRN

def save_img(image, path):
    # Convert and save the image as a PNG file
    image = tf.math.multiply(image, 255.)
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    image = tf.image.encode_png(image, 3)
    tf.io.write_file(path, image)
    return 'Done!'

def calculate_psnr(target, ref):
    target_data = tf.image.convert_image_dtype(target, dtype=tf.float32)
    ref_data = tf.image.convert_image_dtype(ref, dtype=tf.float32)

    mse = tf.reduce_mean(tf.square(target_data - ref_data))
    psnr = 20 * tf.math.log(1.0 / tf.math.sqrt(mse)) / tf.math.log(10.0)

    return psnr.numpy()


def calculate_ssim(target, ref):
    target_data = tf.image.convert_image_dtype(target, dtype=tf.float32)
    ref_data = tf.image.convert_image_dtype(ref, dtype=tf.float32)

    ssim_value = tf.image.ssim(target_data, ref_data, max_val=1.0)
    return ssim_value.numpy()

def calculate_epi(original, restored):
    # Laplacian filter
    H = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]])

    # Highpass filter the input and restored images
    deltas = cv2.filter2D(original.numpy(), -1, H)
    deltascap = cv2.filter2D(restored.numpy(), -1, H)

    # Calculate mean values
    meandeltas = np.mean(deltas)
    meandeltascap = np.mean(deltascap)

    # Compute EPI
    p1 = deltas - meandeltas
    p2 = deltascap - meandeltascap
    num = np.sum(p1 * p2)
    den = np.sqrt(np.sum(p1**2) * np.sum(p2**2))
    epi = num / den if den != 0 else 0

    return epi

def Inference(weights_path, scale=4):
    sr_model = DRN(input_shape=(None, None, 3), model='DRN-S', scale=scale, nColor=3, training=False, dual=False)
    sr_model.load_weights(weights_path, by_name=True)

    psnr_sum = 0.0
    ssim_sum = 0.0
    epi_sum = 0.0

    def inference(lr_path, sr_path, hr_path, sr_model=sr_model):
        nonlocal psnr_sum, ssim_sum, epi_sum

        # Read the LR image
        image = tf.io.read_file(lr_path)
        image = tf.io.decode_image(image, 3, expand_animations=False)

        # Get the size of the HR image
        hr_size = tf.io.read_file(hr_path)
        hr_size = tf.io.decode_image(hr_size, 3, expand_animations=False)
        hr_size = tf.shape(hr_size)[:2]

        # Expand dimensions, normalize, and predict with the SR model
        image = tf.expand_dims(image, axis=0)
        image = tf.math.divide(tf.cast(image, tf.float32), 255.)
        sr_image = sr_model.predict(image)[0]

        # Resize the SR image to HR size (optional)
        image = tf.image.resize(sr_image, hr_size, method=tf.image.ResizeMethod.BICUBIC)

        # Save the SR image
        save_img(image, sr_path)

        # Read the HR image
        hr_image = tf.io.read_file(hr_path)
        hr_image = tf.io.decode_image(hr_image, 3, expand_animations=False)
        hr_image = tf.cast(hr_image, tf.float32) / 255.0

        # Calculate metrics
        psnr_value = calculate_psnr(image, hr_image)
        ssim_value = calculate_ssim(image, hr_image)
        epi_value = calculate_epi(hr_image, image)


        print(f'Image: {lr_path}')
        print(f'PSNR: {psnr_value:.2f} dB')
        print(f'SSIM: {ssim_value:.4f}')
        print(f'EPI: {epi_value:.4f}')

        # Accumulate values
        psnr_sum += psnr_value
        ssim_sum += ssim_value
        epi_sum += epi_value

    def get_averages():
        num_images = 20  # Adjust this based on the total number of images
        avg_psnr = psnr_sum / num_images
        avg_ssim = ssim_sum / num_images
        avg_epi = epi_sum / num_images

        print(f'\nAverage PSNR: {avg_psnr:.2f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')
        print(f'Average EPI: {avg_epi:.4f}')

    return inference, get_averages

# Specify the path to the pre-trained weights
weights_path = 'models/DRN_SS/weight-400-0.0255.h5'
scale = 4

# Create an inference function with the specified weights and scale
inference, get_averages = Inference(weights_path, scale)

# Paths for LR, SR, and HR images
path_test = './Redes_OLI2/test_OLI/test_lr_png/'
path_output = './Redes_OLI2/output/'
path_reference = './Redes_OLI2/test_OLI/test_hr_png/'

# Iterate through images and perform inference
for i in range(1, 21):
    name = 'zh' + str(i) + '_RGB.png'
    lr_path = path_test + name
    sr_path = path_output + name
    hr_path = path_reference + name
    
    # Perform inference and save the SR image
    inference(lr_path, sr_path, hr_path)

# Calculate and print average values
get_averages()
