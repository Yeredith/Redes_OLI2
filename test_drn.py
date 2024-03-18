import os
import tensorflow as tf
import numpy as np
import cv2
import math
from Utils.model_drn import DRN  

def save_img(image, path):
    # Convertir y guardar la imagen como un archivo PNG
    image = tf.math.multiply(image, 255.)
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    image = tf.image.encode_png(image, 3)
    tf.io.write_file(path, image)
    return '¡Hecho!'

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
    # Filtro Laplaciano
    H = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]])

    # Filtrar paso alto las imágenes de entrada y restauradas
    deltas = cv2.filter2D(original.numpy(), -1, H)
    deltascap = cv2.filter2D(restored.numpy(), -1, H)

    # Calcular valores medios
    meandeltas = np.mean(deltas)
    meandeltascap = np.mean(deltascap)

    # Calcular EPI
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

        # Leer la imagen LR
        image = tf.io.read_file(lr_path)
        image = tf.io.decode_image(image, 3, expand_animations=False)

        # Obtener el tamaño de la imagen HR
        hr_size = tf.io.read_file(hr_path)
        hr_size = tf.io.decode_image(hr_size, 3, expand_animations=False)
        hr_size = tf.shape(hr_size)[:2]

        # Expandir dimensiones, normalizar y predecir con el modelo SR
        image = tf.expand_dims(image, axis=0)
        image = tf.math.divide(tf.cast(image, tf.float32), 255.)
        sr_image = sr_model.predict(image)[0]

        # Redimensionar la imagen SR al tamaño HR (opcional)
        image = tf.image.resize(sr_image, hr_size, method=tf.image.ResizeMethod.BICUBIC)

        # Guardar la imagen SR
        save_img(image, sr_path)

        # Leer la imagen HR
        hr_image = tf.io.read_file(hr_path)
        hr_image = tf.io.decode_image(hr_image, 3, expand_animations=False)
        hr_image = tf.cast(hr_image, tf.float32) / 255.0

        # Calcular métricas
        psnr_value = calculate_psnr(image, hr_image)
        ssim_value = calculate_ssim(image, hr_image)
        epi_value = calculate_epi(hr_image, image)


        print(f'Imagen: {lr_path}')
        print(f'PSNR: {psnr_value:.2f} dB')
        print(f'SSIM: {ssim_value:.4f}')
        print(f'EPI: {epi_value:.4f}')

        # Acumular valores
        psnr_sum += psnr_value
        ssim_sum += ssim_value
        epi_sum += epi_value

    def get_averages(num_images):
        avg_psnr = psnr_sum / num_images
        avg_ssim = ssim_sum / num_images
        avg_epi = epi_sum / num_images

        print(f'\nPSNR Promedio: {avg_psnr:.2f} dB')
        print(f'SSIM Promedio: {avg_ssim:.4f}')
        print(f'EPI Promedio: {avg_epi:.4f}')

        # Guardar los resultados en un archivo de texto
        with open('resultados.txt', 'a') as f:
            f.write('\nResultados Promedio:\n')
            f.write(f'PSNR Promedio: {avg_psnr:.2f} dB\n')
            f.write(f'SSIM Promedio: {avg_ssim:.4f}\n')
            f.write(f'EPI Promedio: {avg_epi:.4f}\n')

        return inference, avg_psnr, avg_ssim, avg_epi

    return inference, get_averages

# Especificar la ruta a los pesos pre-entrenados
weights_path = os.path.join(os.getcwd(), 'models_drn', 'DRN_S', 'weight-400.h5') #modificar el nombre del archivo h5
scale = 4

# Crear una función de inferencia con los pesos y la escala especificados
inference, get_averages = Inference(weights_path, scale)

# Rutas para imágenes LR, SR y HR
#path_test = os.path.join(os.getcwd(), 'test_OLI', 'test_lr_png')
#path_output = os.path.join(os.getcwd(), 'output_drn')
#path_reference = os.path.join(os.getcwd(),'test_OLI', 'test_hr_png')

#Imagenes Zurich RGB
path_test_RGB = os.path.join(os.getcwd(), 'test','RGB_x4','data')
path_output_RGB = os.path.join(os.getcwd(), 'output_drn','RGB')
path_reference_RGB = os.path.join(os.getcwd(),'test','RGB_x4','reference')
num_images_RGB = len(os.listdir(path_reference_RGB))
# Iterar a través de las imágenes y realizar inferencias
for i in range(1, num_images_RGB+1):
    name = 'zh' + str(i) + '_RGB.png'
    lr_path = os.path.join(path_test_RGB, name)
    sr_path = os.path.join(path_output_RGB, name)
    hr_path = os.path.join(path_reference_RGB, name)
    
    # Realizar inferencias y guardar la imagen SR
    inference(lr_path, sr_path, hr_path)

# Calcular e imprimir valores promedio
get_averages(num_images_RGB)


#Imagenes Zurich B4
path_test_B4 = os.path.join(os.getcwd(), 'test','B4_x4','data')
path_output_B4 = os.path.join(os.getcwd(), 'output_drn','B4')
path_reference_B4 = os.path.join(os.getcwd(),'test','B4_x4','reference')
num_images_B4 = len(os.listdir(path_reference_B4))
# Iterar a través de las imágenes y realizar inferencias
for i in range(1, num_images_B4+1):
    name = 'zh' + str(i) + '_B4.png'
    lr_path = os.path.join(path_test_B4, name)
    sr_path = os.path.join(path_output_B4, name)
    hr_path = os.path.join(path_reference_B4, name)
    
    # Realizar inferencias y guardar la imagen SR
    inference(lr_path, sr_path, hr_path)

# Calcular e imprimir valores promedio
get_averages(num_images_B4)
