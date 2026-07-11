from tensorflow.keras import models
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import functools
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Fija la semilla aleatoria global para garantizar la reproducibilidad del experimento.

    Establece la semilla para los módulos Python ``random``, NumPy y TensorFlow,
    además de la variable de entorno ``PYTHONHASHSEED``.

    Args:
        seed (int): Valor entero de la semilla a aplicar globalmente.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Configuración de Aumento de Datos
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Configuración centralizada para todas las operaciones de aumento de datos.

    Cada operación puede habilitarse o deshabilitarse de forma independiente
    mediante su bandera booleana correspondiente. Los parámetros adicionales
    permiten ajustar la intensidad de cada transformación.

    Attributes:
        horizontal_flip (bool): Aplica volteo horizontal aleatorio con p=0.5.
            Por defecto es True.
        rotation (bool): Aplica rotación aleatoria continua.
            Por defecto es True.
        rotation_max_degrees (float): Máximo de grados (±) para la rotación.
            Por defecto es 15.0.
        random_resized_crop (bool): Recorte aleatorio redimensionado al tamaño
            original. Por defecto es False.
        rrc_scale (tuple): Rango (mín, máx) de la fracción del área de imagen
            para el recorte. Por defecto es (0.8, 1.0).
        rrc_ratio (tuple): Rango (mín, máx) de la relación de aspecto del recorte.
            Por defecto es (0.75, 1.333).
        scaling (bool): Escala aleatoria para simular una cámara más cerca o más
            lejos. Por defecto es False.
        scale_factor (float): Factor máximo de escala (> 1.0). El rango efectivo
            es [1/scale_factor, scale_factor]. Por defecto es 1.2.
        perspective (bool): Aplica una transformación de perspectiva aleatoria.
            Por defecto es False.
        perspective_scale_min (float): Distorsión de perspectiva mínima como
            fracción de la dimensión de la imagen. Por defecto es 0.05.
        perspective_scale_max (float): Distorsión de perspectiva máxima como
            fracción de la dimensión de la imagen. Por defecto es 0.1.
        translation (bool): Aplica traslación aleatoria horizontal y vertical.
            Por defecto es False.
        translate_x_max_pct (float): Desplazamiento máximo (±) en el eje X como
            porcentaje del ancho de la imagen. Por defecto es 0.1.
        translate_y_max_pct (float): Desplazamiento máximo (±) en el eje Y como
            porcentaje del alto de la imagen. Por defecto es 0.1.
        brightness (bool): Adición aleatoria de brillo. Por defecto es True.
        brightness_limit (float): Límite de variación del brillo como fracción
            del rango de píxeles [0, 255]. Por defecto es 0.2.
        contrast (bool): Ajuste aleatorio del factor de contraste.
            Por defecto es True.
        contrast_limit (float): Límite de variación del factor de contraste (±).
            Por defecto es 0.2.
        gamma (bool): Aplica corrección Gamma aleatoria. Por defecto es False.
        gamma_limit (float): Límite de variación del exponente Gamma (±).
            Por defecto es 0.2.
        hue_saturation (bool): Ajusta aleatoriamente el matiz y la saturación en
            el espacio HSV. Por defecto es False.
        hue_shift_limit (int): Límite de desplazamiento del matiz (±) en grados
            del espacio HSV de OpenCV [0, 179]. Por defecto es 20.
        sat_shift_limit (int): Límite de desplazamiento de la saturación (±) en
            el rango [0, 255]. Por defecto es 30.
        gaussian_blur (bool): Aplica desenfoque Gaussiano aleatorio.
            Por defecto es False.
        blur_limit (tuple): Rango (mín, máx) del tamaño del kernel de desenfoque
            Gaussiano. Por defecto es (3, 7).
        motion_blur (bool): Aplica desenfoque de movimiento en dirección aleatoria.
            Por defecto es False.
        motion_blur_limit (tuple): Rango (mín, máx) del tamaño del kernel de
            movimiento. Por defecto es (3, 7).
        gaussian_noise (bool): Añade ruido Gaussiano aditivo aleatorio.
            Por defecto es False.
        noise_std_limit (float): Límite superior para la desviación estándar del
            ruido como fracción del rango de píxeles [0, 255]. Por defecto es 0.05.
        random_erasing (bool): Borra aleatoriamente una región rectangular de la
            imagen. Por defecto es False.
        erasing_scale (tuple): Rango (mín, máx) de la fracción del área de imagen
            a borrar. Por defecto es (0.02, 0.2).
        erasing_ratio (tuple): Rango (mín, máx) de la relación de aspecto de la
            región borrada. Por defecto es (0.3, 3.3).
    """

    # --- Transformaciones geométricas ---
    horizontal_flip: bool = True

    rotation: bool = True
    rotation_max_degrees: float = 15.0

    random_resized_crop: bool = False
    rrc_scale: tuple = (0.8, 1.0)
    rrc_ratio: tuple = (0.75, 1.333)

    scaling: bool = False
    scale_factor: float = 1.2

    perspective: bool = False
    perspective_scale_min: float = 0.05
    perspective_scale_max: float = 0.1

    translation: bool = False
    translate_x_max_pct: float = 0.1
    translate_y_max_pct: float = 0.1

    # --- Transformaciones fotométricas ---
    brightness: bool = True
    brightness_limit: float = 0.2

    contrast: bool = True
    contrast_limit: float = 0.2

    gamma: bool = False
    gamma_limit: float = 0.2

    hue_saturation: bool = False
    hue_shift_limit: int = 20
    sat_shift_limit: int = 30

    # --- Desenfoque y ruido ---
    gaussian_blur: bool = False
    blur_limit: tuple = (3, 7)

    motion_blur: bool = False
    motion_blur_limit: tuple = (3, 7)

    gaussian_noise: bool = False
    noise_std_limit: float = 0.05

    # --- Oclusión ---
    random_erasing: bool = False
    erasing_scale: tuple = (0.02, 0.2)
    erasing_ratio: tuple = (0.3, 3.3)


# ---------------------------------------------------------------------------
# Utilidades de visualización internas
# ---------------------------------------------------------------------------

def _apply_plot_style() -> None:
    """Aplica un estilo de gráfico consistente para las figuras de la tesis."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.size": 10,
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# Carga y preprocesamiento de imágenes
# ---------------------------------------------------------------------------

def load_and_preprocess_image(
    image_path: str,
    label: int,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[tf.Tensor, int]:
    """Carga una imagen desde su ruta, la decodifica, redimensiona y preprocesa.

    Esta función lee la imagen desde el disco en formato JPEG, la decodifica,
    la escala al tamaño objetivo y aplica el preprocesamiento específico de
    la arquitectura ResNet (centrado de canales y reordenamiento a BGR).

    Args:
        image_path (str): Ruta absoluta o relativa al archivo de imagen.
        label (int): Identificador de la clase asignada.
        target_size (tuple[int, int], optional): Dimensiones de salida (ancho, alto).
            Por defecto es (224, 224).

    Returns:
        tuple[tf.Tensor, int]: Tupla que contiene el tensor de la imagen
            preprocesada y la etiqueta correspondiente.
    """
    # Leer el archivo
    img_raw = tf.io.read_file(image_path)
    # Decodificar el archivo como JPEG con 3 canales de color (RGB)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    # Redimensionar al tamaño objetivo
    img = tf.image.resize(img, target_size)
    # Aplicar preprocesamiento específico de ResNet152 (espera valores de 0 a 255)
    img = preprocess_input(img)

    return img, label


# ---------------------------------------------------------------------------
# Funciones auxiliares privadas — Augmentaciones NumPy
# ---------------------------------------------------------------------------

def _np_rotation(img: np.ndarray, max_degrees: float) -> np.ndarray:
    """Aplica una rotación aleatoria a la imagen.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        max_degrees (float): Rotación máxima en ± grados.

    Returns:
        np.ndarray: Array HxWx3 float32 rotado, mismo tamaño que la entrada.
    """
    angle = np.random.uniform(-max_degrees, max_degrees)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return rotated.astype(np.float32)


def _np_scaling(img: np.ndarray, scale_factor: float) -> np.ndarray:
    """Escala la imagen aleatoriamente para simular una cámara más cerca o más lejos.

    El factor de escala se muestrea uniformemente en [1/scale_factor, scale_factor].
    Para zoom in (escala > 1) se recorta el centro; para zoom out (escala < 1) se
    añaden bordes reflejados para conservar el tamaño original.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        scale_factor (float): Factor máximo de escala (debe ser > 1.0).

    Returns:
        np.ndarray: Array HxWx3 float32 escalado al mismo tamaño que la entrada.
    """
    safe_factor = max(scale_factor, 1.0001)
    scale = np.random.uniform(1.0 / safe_factor, safe_factor)
    h, w = img.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    scaled = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale >= 1.0:
        # Zoom in: recortar el centro al tamaño original
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        result = scaled[start_h:start_h + h, start_w:start_w + w]
    else:
        # Zoom out: rellenar los bordes con reflexión
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        result = cv2.copyMakeBorder(
            scaled, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101,
        )

    return result.astype(np.float32)


def _np_perspective(
    img: np.ndarray,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    """Aplica una transformación de perspectiva aleatoria.

    Desplaza aleatoriamente las cuatro esquinas de la imagen dentro de los límites
    definidos por scale_min y scale_max, mapeando la región distorsionada de vuelta
    al tamaño original de la imagen.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        scale_min (float): Distorsión mínima como fracción de la dimensión.
        scale_max (float): Distorsión máxima como fracción de la dimensión.

    Returns:
        np.ndarray: Array HxWx3 float32 con perspectiva aplicada.
    """
    h, w = img.shape[:2]
    scale = np.random.uniform(scale_min, scale_max)
    dx = scale * w
    dy = scale * h

    # Desplazamiento aleatorio de cada esquina del rectángulo fuente
    pts_src = np.float32([
        [np.random.uniform(0.0, dx),       np.random.uniform(0.0, dy)],
        [w - 1 - np.random.uniform(0.0, dx), np.random.uniform(0.0, dy)],
        [np.random.uniform(0.0, dx),       h - 1 - np.random.uniform(0.0, dy)],
        [w - 1 - np.random.uniform(0.0, dx), h - 1 - np.random.uniform(0.0, dy)],
    ])
    pts_dst = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped.astype(np.float32)


def _np_translation(
    img: np.ndarray,
    x_max_pct: float,
    y_max_pct: float,
) -> np.ndarray:
    """Desplaza la imagen aleatoriamente en los ejes horizontal y vertical.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        x_max_pct (float): Desplazamiento máximo (±) en X como porcentaje del ancho.
        y_max_pct (float): Desplazamiento máximo (±) en Y como porcentaje del alto.

    Returns:
        np.ndarray: Array HxWx3 float32 desplazado.
    """
    h, w = img.shape[:2]
    tx = np.random.uniform(-x_max_pct, x_max_pct) * w
    ty = np.random.uniform(-y_max_pct, y_max_pct) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return shifted.astype(np.float32)


def _np_gamma(img: np.ndarray, gamma_limit: float) -> np.ndarray:
    """Aplica corrección Gamma aleatoria.

    Muestrea un exponente gamma en [1 - gamma_limit, 1 + gamma_limit] y aplica la
    transformación de potencia sobre los valores normalizados al rango [0, 1].
    Un gamma < 1 aclara la imagen; un gamma > 1 la oscurece.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        gamma_limit (float): Límite de variación del exponente gamma (±).

    Returns:
        np.ndarray: Array HxWx3 float32 con corrección gamma aplicada.
    """
    gamma = np.random.uniform(1.0 - gamma_limit, 1.0 + gamma_limit)
    gamma = max(0.1, gamma)
    img_norm = np.clip(img / 255.0, 0.0, 1.0)
    corrected = np.power(img_norm, gamma) * 255.0
    return np.clip(corrected, 0.0, 255.0).astype(np.float32)


def _np_hue_saturation(
    img: np.ndarray,
    hue_limit: int,
    sat_limit: int,
) -> np.ndarray:
    """Ajusta aleatoriamente el matiz (hue) y la saturación en el espacio HSV.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255] en formato RGB.
        hue_limit (int): Límite de desplazamiento del matiz (±) en el espacio HSV
            de OpenCV, donde el matiz está en [0, 179].
        sat_limit (int): Límite de desplazamiento de la saturación (±) en [0, 255].

    Returns:
        np.ndarray: Array HxWx3 float32 en formato RGB con matiz y saturación ajustados.
    """
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.int32)

    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + np.random.randint(-hue_limit, hue_limit + 1)) % 180
    img_hsv[:, :, 1] = np.clip(
        img_hsv[:, :, 1] + np.random.randint(-sat_limit, sat_limit + 1), 0, 255
    )

    img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img_rgb.astype(np.float32)


def _np_gaussian_blur(img: np.ndarray, blur_limit: tuple) -> np.ndarray:
    """Aplica desenfoque Gaussiano con tamaño de kernel aleatorio.

    El tamaño del kernel se ajusta siempre a un valor impar dentro del rango.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        blur_limit (tuple): Tupla (mín, máx) que define el rango del tamaño del kernel.

    Returns:
        np.ndarray: Array HxWx3 float32 con desenfoque gaussiano aplicado.
    """
    min_k, max_k = blur_limit
    k = np.random.randint(min_k, max_k + 1)
    if k % 2 == 0:
        k += 1  # el kernel de Gauss debe ser de tamaño impar
    k = max(3, k)

    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (k, k), 0)
    return blurred.astype(np.float32)


def _np_motion_blur(img: np.ndarray, blur_limit: tuple) -> np.ndarray:
    """Aplica desenfoque de movimiento con dirección aleatoria.

    Genera un kernel de línea en dirección horizontal y lo rota aleatoriamente
    para simular el desenfoque en cualquier ángulo.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        blur_limit (tuple): Tupla (mín, máx) del tamaño del kernel de movimiento.

    Returns:
        np.ndarray: Array HxWx3 float32 con desenfoque de movimiento aplicado.
    """
    min_k, max_k = blur_limit
    k = np.random.randint(min_k, max_k + 1)
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    # Kernel de línea horizontal
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0

    # Rotar el kernel para simular movimiento en dirección aleatoria
    angle = float(np.random.uniform(0, 180))
    M_rot = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M_rot, (k, k))
    total = kernel.sum()
    if total > 1e-10:
        kernel /= total

    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    blurred = cv2.filter2D(img_uint8, -1, kernel)
    return blurred.astype(np.float32)


def _np_gaussian_noise(img: np.ndarray, noise_std_limit: float) -> np.ndarray:
    """Añade ruido Gaussiano aditivo aleatorio a la imagen.

    La desviación estándar del ruido se muestrea uniformemente en
    [0, noise_std_limit * 255].

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        noise_std_limit (float): Límite superior de la desviación estándar como
            fracción del rango de píxeles. Por ejemplo, 0.05 → std ≤ 12.75.

    Returns:
        np.ndarray: Array HxWx3 float32 con ruido añadido, recortado a [0, 255].
    """
    std = np.random.uniform(0.0, noise_std_limit) * 255.0
    noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 255.0).astype(np.float32)


def _np_random_erasing(
    img: np.ndarray,
    erasing_scale: tuple,
    erasing_ratio: tuple,
) -> np.ndarray:
    """Borra aleatoriamente una región rectangular de la imagen.

    Intenta encontrar una región válida en hasta 10 intentos. Si no encuentra
    una región que cumpla las restricciones, retorna la imagen sin modificar.
    La región borrada se rellena con un valor de píxel constante aleatorio.

    Args:
        img (np.ndarray): Array HxWx3 float32 con valores en [0, 255].
        erasing_scale (tuple): Rango (mín, máx) de la fracción del área a borrar.
        erasing_ratio (tuple): Rango (mín, máx) de la relación de aspecto
            (alto/ancho) de la región borrada.

    Returns:
        np.ndarray: Array HxWx3 float32 con la región borrada.
    """
    h, w = img.shape[:2]
    area = h * w
    result = img.copy()

    for _ in range(10):
        erase_area = np.random.uniform(erasing_scale[0], erasing_scale[1]) * area
        aspect_ratio = np.random.uniform(erasing_ratio[0], erasing_ratio[1])

        erase_h = int(round(np.sqrt(erase_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(erase_area / aspect_ratio)))

        if 0 < erase_h < h and 0 < erase_w < w:
            top = np.random.randint(0, h - erase_h)
            left = np.random.randint(0, w - erase_w)
            fill_value = np.random.uniform(0.0, 255.0)
            result[top:top + erase_h, left:left + erase_w] = fill_value
            break

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Función auxiliar privada — Recorte Aleatorio Redimensionado (TF-native)
# ---------------------------------------------------------------------------

def _tf_random_resized_crop(
    image: tf.Tensor,
    scale: tuple,
    ratio: tuple,
) -> tf.Tensor:
    """Aplica un recorte aleatorio redimensionado utilizando operaciones TensorFlow.

    Muestrea un área de recorte dentro del rango ``scale`` (fracción del área
    total) y una relación de aspecto dentro del rango ``ratio``, luego redimensiona
    el recorte al tamaño original de la imagen.

    Args:
        image (tf.Tensor): Tensor HxWxC float32.
        scale (tuple): Rango (mín, máx) de la fracción del área de imagen para el recorte.
        ratio (tuple): Rango (mín, máx) de la relación de aspecto del recorte.

    Returns:
        tf.Tensor: Tensor HxWxC float32 recortado y redimensionado al tamaño original.
    """
    shape = tf.shape(image)
    orig_h = shape[0]
    orig_w = shape[1]
    area = tf.cast(orig_h * orig_w, tf.float32)

    target_area = tf.random.uniform([], scale[0], scale[1]) * area
    log_ratio_min = tf.math.log(tf.constant(ratio[0], dtype=tf.float32))
    log_ratio_max = tf.math.log(tf.constant(ratio[1], dtype=tf.float32))
    aspect_ratio = tf.exp(tf.random.uniform([], log_ratio_min, log_ratio_max))

    crop_w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
    crop_h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

    # Acotar al tamaño de la imagen
    crop_w = tf.maximum(tf.minimum(crop_w, orig_w), 1)
    crop_h = tf.maximum(tf.minimum(crop_h, orig_h), 1)

    offset_h = tf.random.uniform([], 0, tf.maximum(orig_h - crop_h + 1, 1), dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, tf.maximum(orig_w - crop_w + 1, 1), dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_h, crop_w)
    image = tf.image.resize(image, [orig_h, orig_w])
    return image


# ---------------------------------------------------------------------------
# Función auxiliar privada — Bloque de augmentaciones NumPy vía tf.py_function
# ---------------------------------------------------------------------------

def _apply_numpy_augmentations(
    image: tf.Tensor,
    config: AugmentationConfig,
) -> tf.Tensor:
    """Aplica todas las augmentaciones basadas en NumPy/OpenCV mediante tf.py_function.

    Agrupa todas las operaciones que requieren NumPy u OpenCV en una única llamada
    ``tf.py_function`` para minimizar el número de cruces entre el grafo de
    TensorFlow y el intérprete Python.

    Args:
        image (tf.Tensor): Tensor HxWxC float32 con valores en [0, 255].
        config (AugmentationConfig): Configuración de las augmentaciones.

    Returns:
        tf.Tensor: Tensor HxWxC float32 con las augmentaciones aplicadas.
    """
    original_shape = image.shape

    def _fn(img_tensor: tf.Tensor) -> np.ndarray:
        img = img_tensor.numpy()

        if config.rotation:
            img = _np_rotation(img, config.rotation_max_degrees)
        if config.scaling:
            img = _np_scaling(img, config.scale_factor)
        if config.perspective:
            img = _np_perspective(img, config.perspective_scale_min, config.perspective_scale_max)
        if config.translation:
            img = _np_translation(img, config.translate_x_max_pct, config.translate_y_max_pct)
        if config.gamma:
            img = _np_gamma(img, config.gamma_limit)
        if config.hue_saturation:
            img = _np_hue_saturation(img, config.hue_shift_limit, config.sat_shift_limit)
        if config.gaussian_blur:
            img = _np_gaussian_blur(img, config.blur_limit)
        if config.motion_blur:
            img = _np_motion_blur(img, config.motion_blur_limit)
        if config.gaussian_noise:
            img = _np_gaussian_noise(img, config.noise_std_limit)
        if config.random_erasing:
            img = _np_random_erasing(img, config.erasing_scale, config.erasing_ratio)

        return img.astype(np.float32)

    result = tf.py_function(_fn, [image], tf.float32)
    result.set_shape(original_shape)
    return result


# ---------------------------------------------------------------------------
# Aumento de datos
# ---------------------------------------------------------------------------

def augment_image(
    image: tf.Tensor,
    label: int,
    config: AugmentationConfig | None = None,
) -> tuple[tf.Tensor, int]:
    """Aplica aumento de datos configurable a una imagen en escala [0, 255].

    Combina operaciones nativas de TensorFlow con operaciones NumPy/OpenCV
    encapsuladas en ``tf.py_function``. Las augmentaciones geométricas se aplican
    primero, seguidas de las fotométricas y finalmente las de desenfoque, ruido
    y borrado.

    El orden de aplicación de las augmentaciones es el siguiente:

        1.  Volteo Horizontal (TF-native, p=0.5)
        2.  Recorte Aleatorio Redimensionado (TF-native)
        3.  Brillo (TF-native)
        4.  Contraste (TF-native)
        5.  Rotación aleatoria (NumPy/OpenCV)
        6.  Escalado (NumPy/OpenCV)
        7.  Transformación de Perspectiva (NumPy/OpenCV)
        8.  Traslación (NumPy/OpenCV)
        9.  Corrección Gamma (NumPy)
        10. Matiz / Saturación (NumPy/OpenCV)
        11. Desenfoque Gaussiano (NumPy/OpenCV)
        12. Desenfoque de Movimiento (NumPy/OpenCV)
        13. Ruido Gaussiano (NumPy)
        14. Borrado Aleatorio (NumPy)

    Args:
        image (tf.Tensor): Tensor HxWxC float32 con valores en [0, 255].
        label (int): Identificador numérico de la clase (sin modificar).
        config (AugmentationConfig, optional): Configuración de las augmentaciones.
            Si es None, se utilizan los valores por defecto de AugmentationConfig.

    Returns:
        tuple[tf.Tensor, int]: Tensor de la imagen aumentada y la etiqueta sin cambios.
    """
    if config is None:
        config = AugmentationConfig()

    # --- 1. Volteo Horizontal (TF-native, p=0.5) ---
    if config.horizontal_flip:
        image = tf.image.random_flip_left_right(image)

    # --- 2. Recorte Aleatorio Redimensionado (TF-native) ---
    if config.random_resized_crop:
        image = _tf_random_resized_crop(image, config.rrc_scale, config.rrc_ratio)

    # --- 3. Brillo (TF-native) ---
    if config.brightness:
        max_delta = config.brightness_limit * 255.0
        image = tf.image.random_brightness(image, max_delta=max_delta)
        image = tf.clip_by_value(image, 0.0, 255.0)

    # --- 4. Contraste (TF-native) ---
    if config.contrast:
        lower = max(0.0, 1.0 - config.contrast_limit)
        upper = 1.0 + config.contrast_limit
        image = tf.image.random_contrast(image, lower=lower, upper=upper)
        image = tf.clip_by_value(image, 0.0, 255.0)

    # --- 5-14. Augmentaciones NumPy via tf.py_function ---
    numpy_augs_enabled = any([
        config.rotation,
        config.scaling,
        config.perspective,
        config.translation,
        config.gamma,
        config.hue_saturation,
        config.gaussian_blur,
        config.motion_blur,
        config.gaussian_noise,
        config.random_erasing,
    ])

    if numpy_augs_enabled:
        image = _apply_numpy_augmentations(image, config)

    return image, label


# ---------------------------------------------------------------------------
# Visualización de augmentaciones
# ---------------------------------------------------------------------------

def visualize_augmentations(
    image_path: str | Path,
    config: AugmentationConfig | None = None,
    n_augmented: int = 7,
    target_size: tuple[int, int] = (224, 224),
    label_text: str = "",
) -> None:
    """Muestra la imagen original y N versiones aumentadas en una cuadrícula.

    Carga la imagen en escala [0, 255], aplica las augmentaciones definidas en
    ``config`` y presenta los resultados en una figura de matplotlib. Esta función
    permite verificar visualmente el efecto de cada combinación de parámetros
    de AugmentationConfig sin necesidad de ejecutar el pipeline de entrenamiento.

    Args:
        image_path (str | Path): Ruta al archivo de imagen.
        config (AugmentationConfig, optional): Configuración de augmentación.
            Si es None, se utilizan los valores por defecto de AugmentationConfig.
        n_augmented (int, optional): Número de variantes aumentadas a mostrar.
            Por defecto es 7.
        target_size (tuple[int, int], optional): Dimensiones de redimensionado de
            la imagen. Por defecto es (224, 224).
        label_text (str, optional): Texto descriptivo para el título de la figura.
    """
    _apply_plot_style()

    if config is None:
        config = AugmentationConfig()

    # Cargar y redimensionar la imagen sin normalización (valores en [0, 255])
    img_raw = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32)

    total = 1 + n_augmented
    cols = min(4, total)
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    # Mostrar imagen original
    axes[0].imshow(np.clip(img.numpy(), 0, 255).astype(np.uint8))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # Mostrar variantes aumentadas
    for i in range(1, 1 + n_augmented):
        aug_img, _ = augment_image(img, 0, config=config)
        axes[i].imshow(np.clip(aug_img.numpy(), 0, 255).astype(np.uint8))
        axes[i].set_title(f"Aumentada {i}", fontsize=10)
        axes[i].axis("off")

    # Ocultar ejes vacíos
    for ax in axes[1 + n_augmented:]:
        ax.axis("off")

    title = "Demostración de Aumento de Datos"
    if label_text:
        title += f" — {label_text}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Generador de datos
# ---------------------------------------------------------------------------

def get_data_generator(
    df: pd.DataFrame,
    dataset_path: str | Path | None = None,
    level: str = "fine",
    batch_size: int = 32,
    target_size: tuple[int, int] = (224, 224),
    augment: bool = True,
    shuffle: bool = True,
    train_fraction: float = 1.0,
    seed: int | None = None,
    aug_config: AugmentationConfig | None = None,
) -> tf.data.Dataset:
    """Crea un generador de datos optimizado en base a un DataFrame de metadatos.

    Genera un ``tf.data.Dataset`` aplicando carga asíncrona, aumento de datos
    opcional (ejecutado sobre imágenes en escala [0, 255] **antes** de la
    normalización ResNet), preprocesamiento, loteo y pre-búsqueda.

    Admite el uso de un subconjunto del split de entrenamiento mediante el
    parámetro ``train_fraction``.

    El orden de operaciones del pipeline es:

        1. Muestreo del DataFrame (si ``train_fraction`` < 1.0)
        2. Carga y redimensionado de la imagen (valores en [0, 255])
        3. Mezcla aleatoria (si ``shuffle=True``)
        4. Aumento de datos (si ``augment=True``, sobre valores en [0, 255])
        5. Normalización ResNet (``preprocess_input``)
        6. Agrupación por lotes y pre-carga (prefetch)

    Args:
        df (pd.DataFrame): DataFrame con los metadatos de las imágenes.
        dataset_path (str | Path, optional): Ruta raíz del dataset, necesaria si las
            rutas son relativas y no existe la columna ``Absolute Path``.
        level (str, optional): Nivel jerárquico de clasificación ('macro', 'coarse'
            o 'fine'). Por defecto es 'fine'.
        batch_size (int, optional): Tamaño del lote. Por defecto es 32.
        target_size (tuple[int, int], optional): Dimensiones de la imagen para la red.
            Por defecto es (224, 224).
        augment (bool, optional): Indica si se aplica aumento de datos.
            Por defecto es True.
        shuffle (bool, optional): Mezclar el orden de las muestras en cada época.
            Por defecto es True.
        train_fraction (float, optional): Fracción del DataFrame a utilizar.
            1.0 emplea el 100% de las imágenes; 0.5 emplea el 50%.
            Por defecto es 1.0.
        seed (int, optional): Semilla para la mezcla aleatoria y el muestreo por
            fracción. Garantiza la reproducibilidad del experimento. Por defecto
            es None.
        aug_config (AugmentationConfig, optional): Configuración de las augmentaciones.
            Si es None y ``augment=True``, se usan los valores por defecto de
            AugmentationConfig.

    Returns:
        tf.data.Dataset: Dataset de TensorFlow listo para entrenamiento o evaluación.
    """
    # Aplicar fracción de entrenamiento (muestreo estratificado por semilla)
    if train_fraction < 1.0:
        df = df.sample(frac=train_fraction, random_state=seed).reset_index(drop=True)

    # Determinar columna de rutas
    if "Absolute Path" in df.columns:
        paths = df["Absolute Path"].astype(str).tolist()
    elif dataset_path is not None:
        paths = [str(Path(dataset_path) / p) for p in df["Image Path"]]
    else:
        raise ValueError(
            "El DataFrame debe contener 'Absolute Path' o se debe proveer dataset_path."
        )

    # Determinar columna de etiquetas según el nivel taxonómico
    level = level.lower()
    if level == "fine":
        label_col = "Fine ID"
    elif level == "coarse":
        label_col = "Coarse ID"
    elif level == "macro":
        label_col = "Macro ID"
    else:
        raise ValueError("level debe ser uno de: 'macro', 'coarse' o 'fine'.")

    labels = df[label_col].astype(np.int32).tolist()

    # Crear dataset de base
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=seed)

    # --- Paso 1: Cargar y redimensionar sin normalizar (valores en [0, 255]) ---
    def _load_and_resize(path: tf.Tensor, label: tf.Tensor):
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, target_size)
        return tf.cast(img, tf.float32), label

    dataset = dataset.map(
        _load_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # --- Paso 2: Aumento de datos sobre [0, 255] (ANTES de la normalización) ---
    if augment:
        augment_fn = functools.partial(augment_image, config=aug_config)
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # --- Paso 3: Normalización ResNet (preprocess_input) ---
    def _apply_normalization(img: tf.Tensor, label: tf.Tensor):
        return preprocess_input(img), label

    dataset = dataset.map(
        _apply_normalization,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Agrupar por lotes y pre-cargar en memoria de forma optimizada
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def compute_class_weights(
    df: pd.DataFrame,
    level: str = "fine",
) -> tuple[dict[int, float], float]:
    """
    Calcular balanced class weights y class imbalance ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame.

    level : str
        Taxonomic level:
        "macro", "coarse" o "fine".

    Returns
    -------
    class_weight_dict : dict[int, float]

    imbalance_ratio : float
    """

    level = level.lower()

    label_columns = {
        "macro": "Macro ID",
        "coarse": "Coarse ID",
        "fine": "Fine ID",
    }

    try:
        label_col = label_columns[level]
    except KeyError:
        raise ValueError(
            "TAXONOMIC LEVEL debe ser 'macro', 'coarse' o 'fine'."
        )

    labels = df[label_col].astype(np.int32)

    counts = labels.value_counts()

    imbalance_ratio = (
        counts.max() /
        counts.min()
    )

    classes = np.sort(labels.unique())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    class_weight_dict = {
        int(c): float(w)
        for c, w in zip(classes, weights)
    }

    return class_weight_dict, imbalance_ratio


# ---------------------------------------------------------------------------
# Construcción del modelo
# ---------------------------------------------------------------------------

def build_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
    fine_tune_at: int | None = None,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
) -> keras.Model:
    """Construye y compila un modelo de red neuronal convolucional basado en ResNet152.

    Carga el extractor de características preentrenado en ImageNet. Permite congelar
    completamente el extractor para transferencia de aprendizaje clásica, o congelar
    hasta una capa determinada para ajuste fino (fine-tuning). Añade una cabeza de
    clasificación personalizada con Regularización Dropout y una capa Softmax final.

    Args:
        num_classes (int): Cantidad de clases en la capa de salida.
        input_shape (tuple[int, int, int], optional): Dimensiones de entrada (alto, ancho, canales).
            Por defecto es (224, 224, 3).
        learning_rate (float, optional): Tasa de aprendizaje inicial para el optimizador AdamW.
            Por defecto es 1e-4.
        fine_tune_at (int, optional): Índice de capa a partir del cual se descongelarán
            las capas del extractor de características para el ajuste fino.
            Si es None, se congela por completo el extractor.
        optimizer_name (str, optional): Nombre del optimizador a usar. 
            'adamw', o 'adam'.
        weight_decay (float, optional): Tasa de decaimiento de peso para optimizadores
            que lo soportan (AdamW, SGD). Por defecto es 1e-4.
        label_smoothing (float, optional): Suavizado de etiquetas. 0.0 = estándar,
            0.1 = suave. Por defecto es 0.1.

    Returns:
        keras.Model: Modelo de Keras compilado y listo para entrenar.
    """
    # Cargar el extractor de características preentrenado sin la cabeza de clasificación
    base_model = keras.applications.ResNet152(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    if fine_tune_at is None:
        # Transferencia de aprendizaje básica: congelar todo el extractor
        base_model.trainable = False
    else:
        # Ajuste fino: descongelar capas superiores a partir de fine_tune_at
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # Definir la arquitectura usando la API funcional de Keras
    inputs = keras.Input(shape=input_shape)

    # Importante: training=False asegura que las capas de BatchNormalization corran
    # en modo de inferencia y no destruyan los pesos aprendidos de ImageNet
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)

    # Crear el optimizador
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    elif optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            "Optimizador no soportado. Utilice 'adam' o 'adamw'."
        )

    # Compilar el modelo
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Entrenamiento del modelo
# ---------------------------------------------------------------------------

def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = 20,
    callbacks: list | None = None,
    class_weight: dict[int, float] | None = None,
) -> keras.callbacks.History:
    """Entrena el modelo utilizando los datasets proporcionados y callbacks configurados.

    Args:
        model (keras.Model): Modelo compilado de Keras.
        train_dataset (tf.data.Dataset): Dataset para el entrenamiento.
        val_dataset (tf.data.Dataset): Dataset para la validación durante el entrenamiento.
        epochs (int, optional): Número máximo de épocas. Por defecto es 20.
        callbacks (list, optional): Lista de callbacks personalizados de Keras. Si es None,
            se configuran callbacks por defecto (EarlyStopping y ReduceLROnPlateau).

    Returns:
        keras.callbacks.History: Historial del entrenamiento con las pérdidas y métricas.
    """
    if callbacks is None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    return history


# ---------------------------------------------------------------------------
# Evaluación del modelo
# ---------------------------------------------------------------------------

def evaluate_model(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
) -> dict:
    """Evalúa el modelo en un conjunto de prueba y genera métricas de clasificación.

    Calcula la exactitud global, precisión, exhaustividad (recall) y medida F1
    tanto macro como ponderada (weighted). Retorna además los arrays de etiquetas reales
    y predicciones para análisis posteriores.

    Args:
        model (keras.Model): Modelo entrenado de Keras.
        test_dataset (tf.data.Dataset): Dataset de prueba.
        class_names (list[str]): Nombres de las clases en orden correspondiente a sus índices.

    Returns:
        dict: Diccionario que contiene las métricas calculadas y los vectores y_true e y_pred.
    """
    y_true = []
    y_pred = []

    # Iterar sobre el dataset para extraer predicciones y etiquetas reales de forma alinedad
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calcular métricas globales
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1-Score (Macro)": f1_macro,
        "Precision (Weighted)": precision_weighted,
        "Recall (Weighted)": recall_weighted,
        "F1-Score (Weighted)": f1_weighted,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return metrics


# ---------------------------------------------------------------------------
# Gráficos de resultados
# ---------------------------------------------------------------------------

def plot_training_history(
    history: keras.callbacks.History,
    save_path: str | Path | None = None,
) -> None:
    """Grafica la evolución de la pérdida y exactitud durante el entrenamiento.

    Muestra dos subgráficos alineados: uno para la pérdida (loss) en entrenamiento
    y validación, y otro para la exactitud (accuracy).

    Args:
        history (keras.callbacks.History): Historial retornado por model.fit().
        save_path (str | Path, optional): Ruta donde se guardará la imagen.
            Si es None, se muestra la imagen en pantalla sin guardar.
    """
    _apply_plot_style()

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico de Exactitud (Accuracy)
    ax1.plot(epochs_range, acc, label="Entrenamiento", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validación", color="#ff7f0e", linewidth=2)
    ax1.set_title("Exactitud del Modelo (Accuracy)")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Exactitud")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Gráfico de Pérdida (Loss)
    ax2.plot(epochs_range, loss, label="Entrenamiento", color="#1f77b4", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validación", color="#ff7f0e", linewidth=2)
    ax2.set_title("Pérdida del Modelo (Loss)")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Pérdida")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Gráfico de historial guardado en: {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    max_classes_to_show: int = 20,
) -> None:
    """Genera y visualiza la matriz de confusión del conjunto de pruebas.

    Si el número de clases supera `max_classes_to_show`, se muestra una submatriz
    con las clases que presentan mayor cantidad de errores para mantener la legibilidad.

    Args:
        y_true (np.ndarray): Etiquetas reales.
        y_pred (np.ndarray): Predicciones del modelo.
        class_names (list[str]): Lista ordenada con nombres de las clases.
        save_path (str | Path, optional): Ruta donde guardar el gráfico generado.
        max_classes_to_show (int, optional): Límite superior de clases a graficar en detalle.
            Por defecto es 20.
    """
    _apply_plot_style()

    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    cm = sk_confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)

    if num_classes > max_classes_to_show:
        # Calcular el número de errores cometidos por clase
        class_errors = np.sum(cm, axis=1) - np.diag(cm)
        # Tomar los índices de las clases con más errores
        worst_class_indices = np.argsort(class_errors)[::-1][:max_classes_to_show]
        # Ordenar los índices para que la matriz sea consistente jerárquicamente
        worst_class_indices = sorted(worst_class_indices)

        # Filtrar la matriz de confusión y nombres de clases
        cm = cm[np.ix_(worst_class_indices, worst_class_indices)]
        displayed_classes = [class_names[i] for i in worst_class_indices]
        title = (
            f"Matriz de Confusión (Top {max_classes_to_show} Clases con más Errores)"
        )
        figsize = (12, 10)
    else:
        displayed_classes = class_names
        title = "Matriz de Confusión Completa"
        figsize = (min(16, num_classes * 0.8), min(14, num_classes * 0.7))

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=displayed_classes,
        yticklabels=displayed_classes,
        cbar=True,
    )
    plt.title(title, pad=20)
    plt.ylabel("Clase Real")
    plt.xlabel("Clase Predicha")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Matriz de confusión guardada en: {save_path}")

    plt.show()
