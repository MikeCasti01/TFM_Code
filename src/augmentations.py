import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración de Aumento de Datos
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Configuración centralizada para todas las operaciones de aumento de datos.

    Cada operación puede habilitarse o deshabilitarse de forma independiente
    mediante su bandera booleana correspondiente. Los parámetros adicionales
    permiten ajustar la intensidad de cada transformación.
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

    # --- Borrado aleatorio ---
    random_erasing: bool = False
    erasing_scale: tuple = (0.02, 0.2)
    erasing_ratio: tuple = (0.3, 3.3)

    # --- Estrategia de grupos ---
    use_group_strategy: bool = False


# ---------------------------------------------------------------------------
# Funciones auxiliares matemáticas en pure TensorFlow
# ---------------------------------------------------------------------------

def build_rotation_matrix(angle_deg: tf.Tensor, h: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
    """Construye una matriz de transformación proyectiva para rotación alrededor del centro.

    Ángulo positivo representa rotación antihoraria.
    """
    angle_rad = angle_deg * (np.pi / 180.0)
    cos_val = tf.cos(angle_rad)
    sin_val = tf.sin(angle_rad)
    
    h_f = tf.cast(h, tf.float32)
    w_f = tf.cast(w, tf.float32)
    
    x_c = (w_f - 1.0) / 2.0
    y_c = (h_f - 1.0) / 2.0
    
    a0 = cos_val
    a1 = -sin_val
    a2 = x_c - x_c * cos_val + y_c * sin_val
    a3 = sin_val
    a4 = cos_val
    a5 = y_c - x_c * sin_val - y_c * cos_val
    a6 = 0.0
    a7 = 0.0
    
    return tf.stack([a0, a1, a2, a3, a4, a5, a6, a7], axis=0)


def get_projective_transform(src_pts: tf.Tensor, dst_pts: tf.Tensor) -> tf.Tensor:
    """Calcula los 8 parámetros del homograma proyectivo mapeando dst -> src.

    Utiliza tf.linalg.solve para resolver el sistema de ecuaciones.
    """
    equations = []
    values = []
    for i in range(4):
        x, y = dst_pts[i][0], dst_pts[i][1]
        xp, yp = src_pts[i][0], src_pts[i][1]
        
        row1 = tf.stack([x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp])
        row2 = tf.stack([0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp])
        
        equations.append(row1)
        equations.append(row2)
        values.append(xp)
        values.append(yp)
        
    M = tf.stack(equations, axis=0)
    V = tf.stack(values, axis=0)
    
    A = tf.linalg.solve(M, V[:, None])
    return tf.squeeze(A, axis=-1)


# ---------------------------------------------------------------------------
# Implementaciones de transformaciones en pure TensorFlow (Stateless)
# ---------------------------------------------------------------------------

def tf_rotation(image: tf.Tensor, max_degrees: float, seed: tf.Tensor) -> tf.Tensor:
    angle = tf.random.stateless_uniform([], seed=seed, minval=-max_degrees, maxval=max_degrees)
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    
    transform = build_rotation_matrix(angle, h, w)
    
    transformed = tf.raw_ops.ImageProjectiveTransformV2(
        images=image[None, :, :, :],
        transforms=transform[None, :],
        output_shape=shape[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT"
    )
    return transformed[0]


def tf_scaling(image: tf.Tensor, scale_factor: float, seed: tf.Tensor) -> tf.Tensor:
    safe_factor = max(scale_factor, 1.0001)
    scale = tf.random.stateless_uniform([], seed=seed, minval=1.0 / safe_factor, maxval=safe_factor)
    
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    
    x_c = (w - 1.0) / 2.0
    y_c = (h - 1.0) / 2.0
    
    inv_scale = 1.0 / scale
    a0 = inv_scale
    a1 = 0.0
    a2 = x_c * (1.0 - inv_scale)
    a3 = 0.0
    a4 = inv_scale
    a5 = y_c * (1.0 - inv_scale)
    a6 = 0.0
    a7 = 0.0
    
    transform = tf.stack([a0, a1, a2, a3, a4, a5, a6, a7], axis=0)
    
    transformed = tf.raw_ops.ImageProjectiveTransformV2(
        images=image[None, :, :, :],
        transforms=transform[None, :],
        output_shape=shape[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT"
    )
    return transformed[0]


def tf_perspective(image: tf.Tensor, scale_min: float, scale_max: float, seed: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    
    seed_scale = tf.stack([seed[0], seed[1] + 1])
    seed_tl_x = tf.stack([seed[0], seed[1] + 2])
    seed_tl_y = tf.stack([seed[0], seed[1] + 3])
    seed_tr_x = tf.stack([seed[0], seed[1] + 4])
    seed_tr_y = tf.stack([seed[0], seed[1] + 5])
    seed_bl_x = tf.stack([seed[0], seed[1] + 6])
    seed_bl_y = tf.stack([seed[0], seed[1] + 7])
    seed_br_x = tf.stack([seed[0], seed[1] + 8])
    seed_br_y = tf.stack([seed[0], seed[1] + 9])
    
    scale = tf.random.stateless_uniform([], seed=seed_scale, minval=scale_min, maxval=scale_max)
    dx = scale * w
    dy = scale * h
    
    src_tl = tf.stack([tf.random.stateless_uniform([], seed=seed_tl_x, minval=0.0, maxval=dx),
                       tf.random.stateless_uniform([], seed=seed_tl_y, minval=0.0, maxval=dy)])
    src_tr = tf.stack([w - 1.0 - tf.random.stateless_uniform([], seed=seed_tr_x, minval=0.0, maxval=dx),
                       tf.random.stateless_uniform([], seed=seed_tr_y, minval=0.0, maxval=dy)])
    src_bl = tf.stack([tf.random.stateless_uniform([], seed=seed_bl_x, minval=0.0, maxval=dx),
                       h - 1.0 - tf.random.stateless_uniform([], seed=seed_bl_y, minval=0.0, maxval=dy)])
    src_br = tf.stack([w - 1.0 - tf.random.stateless_uniform([], seed=seed_br_x, minval=0.0, maxval=dx),
                       h - 1.0 - tf.random.stateless_uniform([], seed=seed_br_y, minval=0.0, maxval=dy)])
              
    src_pts = tf.stack([src_tl, src_tr, src_bl, src_br], axis=0)
    
    dst_pts = tf.stack([
        tf.stack([0.0, 0.0]),
        tf.stack([w - 1.0, 0.0]),
        tf.stack([0.0, h - 1.0]),
        tf.stack([w - 1.0, h - 1.0])
    ], axis=0)
    
    transform = get_projective_transform(src_pts, dst_pts)
    
    transformed = tf.raw_ops.ImageProjectiveTransformV2(
        images=image[None, :, :, :],
        transforms=transform[None, :],
        output_shape=shape[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT"
    )
    return transformed[0]


def tf_translation(image: tf.Tensor, x_max_pct: float, y_max_pct: float, seed: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    
    seed_x = tf.stack([seed[0], seed[1] + 1])
    seed_y = tf.stack([seed[0], seed[1] + 2])
    
    tx = tf.random.stateless_uniform([], seed=seed_x, minval=-x_max_pct, maxval=x_max_pct) * w
    ty = tf.random.stateless_uniform([], seed=seed_y, minval=-y_max_pct, maxval=y_max_pct) * h
    
    transform = tf.stack([1.0, 0.0, -tx, 0.0, 1.0, -ty, 0.0, 0.0], axis=0)
    
    transformed = tf.raw_ops.ImageProjectiveTransformV2(
        images=image[None, :, :, :],
        transforms=transform[None, :],
        output_shape=shape[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT"
    )
    return transformed[0]


def _tf_random_resized_crop(image: tf.Tensor, scale: tuple, ratio: tuple, seed: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(image)
    orig_h = shape[0]
    orig_w = shape[1]
    area = tf.cast(orig_h * orig_w, tf.float32)
    
    seed_area = tf.stack([seed[0], seed[1] + 1])
    seed_ratio = tf.stack([seed[0], seed[1] + 2])
    seed_offset_h = tf.stack([seed[0], seed[1] + 3])
    seed_offset_w = tf.stack([seed[0], seed[1] + 4])
    
    target_area = tf.random.stateless_uniform([], seed=seed_area, minval=scale[0], maxval=scale[1]) * area
    log_ratio_min = tf.math.log(tf.constant(ratio[0], dtype=tf.float32))
    log_ratio_max = tf.math.log(tf.constant(ratio[1], dtype=tf.float32))
    aspect_ratio = tf.exp(tf.random.stateless_uniform([], seed=seed_ratio, minval=log_ratio_min, maxval=log_ratio_max))
    
    crop_w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
    crop_h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)
    
    crop_w = tf.maximum(tf.minimum(crop_w, orig_w), 1)
    crop_h = tf.maximum(tf.minimum(crop_h, orig_h), 1)
    
    offset_h = tf.random.stateless_uniform([], seed=seed_offset_h, minval=0, maxval=tf.maximum(orig_h - crop_h + 1, 1), dtype=tf.int32)
    offset_w = tf.random.stateless_uniform([], seed=seed_offset_w, minval=0, maxval=tf.maximum(orig_w - crop_w + 1, 1), dtype=tf.int32)
    
    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_h, crop_w)
    image = tf.image.resize(image, [orig_h, orig_w])
    return image


def tf_brightness(image: tf.Tensor, limit: float, seed: tf.Tensor) -> tf.Tensor:
    max_delta = limit * 255.0
    delta = tf.random.stateless_uniform([], seed=seed, minval=-max_delta, maxval=max_delta)
    image = image + delta
    return tf.clip_by_value(image, 0.0, 255.0)


def tf_contrast(image: tf.Tensor, limit: float, seed: tf.Tensor) -> tf.Tensor:
    lower = tf.maximum(0.0, 1.0 - limit)
    upper = 1.0 + limit
    factor = tf.random.stateless_uniform([], seed=seed, minval=lower, maxval=upper)
    mean = tf.reduce_mean(image, axis=[0, 1], keepdims=True)
    image = (image - mean) * factor + mean
    return tf.clip_by_value(image, 0.0, 255.0)


def tf_gamma(image: tf.Tensor, limit: float, seed: tf.Tensor) -> tf.Tensor:
    gamma = tf.random.stateless_uniform([], seed=seed, minval=1.0 - limit, maxval=1.0 + limit)
    gamma = tf.maximum(0.1, gamma)
    img_norm = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    corrected = tf.pow(img_norm, gamma) * 255.0
    return tf.clip_by_value(corrected, 0.0, 255.0)


def tf_hue_saturation(image: tf.Tensor, hue_limit: float, sat_limit: float, seed: tf.Tensor) -> tf.Tensor:
    img_norm = image / 255.0
    hsv = tf.image.rgb_to_hsv(img_norm)
    h, s, v = tf.unstack(hsv, axis=-1)
    
    seed_h = tf.stack([seed[0], seed[1] + 1])
    seed_s = tf.stack([seed[0], seed[1] + 2])
    
    hue_shift = tf.random.stateless_uniform([], seed=seed_h, minval=-hue_limit / 180.0, maxval=hue_limit / 180.0)
    h_new = tf.math.floormod(h + hue_shift, 1.0)
    
    sat_shift = tf.random.stateless_uniform([], seed=seed_s, minval=-sat_limit / 255.0, maxval=sat_limit / 255.0)
    s_new = tf.clip_by_value(s + sat_shift, 0.0, 1.0)
    
    hsv_new = tf.stack([h_new, s_new, v], axis=-1)
    rgb_new = tf.image.hsv_to_rgb(hsv_new) * 255.0
    return tf.clip_by_value(rgb_new, 0.0, 255.0)


def tf_gaussian_blur(image: tf.Tensor, blur_limit: tuple, seed: tf.Tensor) -> tf.Tensor:
    seed_k = tf.stack([seed[0], seed[1] + 1])
    k = tf.random.stateless_uniform([], seed=seed_k, minval=blur_limit[0], maxval=blur_limit[1] + 1, dtype=tf.int32)
    k = tf.cond(tf.equal(k % 2, 0), lambda: k + 1, lambda: k)
    k = tf.maximum(3, k)
    
    k_float = tf.cast(k, tf.float32)
    sigma = 0.3 * ((k_float - 1.0) * 0.5 - 1.0) + 0.8
    
    pad = k // 2
    img_padded = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    
    x = tf.range(-pad, pad + 1, dtype=tf.float32)
    g = tf.exp(-0.5 * tf.square(x) / tf.square(sigma))
    g = g / tf.reduce_sum(g)
    
    kernel2d = g[:, None] * g[None, :]
    
    channels = tf.shape(image)[-1]
    kernel = tf.tile(kernel2d[:, :, None, None], [1, 1, channels, 1])
    
    blurred = tf.nn.depthwise_conv2d(
        img_padded[None, :, :, :],
        kernel,
        strides=[1, 1, 1, 1],
        padding="VALID"
    )
    return blurred[0]


def tf_motion_blur(image: tf.Tensor, blur_limit: tuple, seed: tf.Tensor) -> tf.Tensor:
    seed_k = tf.stack([seed[0], seed[1] + 1])
    seed_angle = tf.stack([seed[0], seed[1] + 2])
    
    k = tf.random.stateless_uniform([], seed=seed_k, minval=blur_limit[0], maxval=blur_limit[1] + 1, dtype=tf.int32)
    k = tf.cond(tf.equal(k % 2, 0), lambda: k + 1, lambda: k)
    k = tf.maximum(3, k)
    
    angle = tf.random.stateless_uniform([], seed=seed_angle, minval=0.0, maxval=180.0)
    
    pad = k // 2
    img_padded = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    
    kernel_line = tf.one_hot(tf.repeat(k // 2, k), k, dtype=tf.float32)
    
    kernel_expanded = kernel_line[:, :, None, None]
    kernel_input = tf.transpose(kernel_expanded, [3, 0, 1, 2])
    
    rot_mat = build_rotation_matrix(angle, k, k)
    
    rotated_kernel = tf.raw_ops.ImageProjectiveTransformV2(
        images=kernel_input,
        transforms=rot_mat[None, :],
        output_shape=tf.stack([k, k]),
        interpolation="BILINEAR",
        fill_mode="CONSTANT"
    )
    
    kernel_2d = rotated_kernel[0, :, :, 0]
    total = tf.reduce_sum(kernel_2d)
    kernel_2d = tf.cond(total > 1e-10, lambda: kernel_2d / total, lambda: kernel_2d)
    
    channels = tf.shape(image)[-1]
    kernel = tf.tile(kernel_2d[:, :, None, None], [1, 1, channels, 1])
    
    blurred = tf.nn.depthwise_conv2d(
        img_padded[None, :, :, :],
        kernel,
        strides=[1, 1, 1, 1],
        padding="VALID"
    )
    return blurred[0]


def tf_gaussian_noise(image: tf.Tensor, std_limit: float, seed: tf.Tensor) -> tf.Tensor:
    seed_std = tf.stack([seed[0], seed[1] + 1])
    seed_noise = tf.stack([seed[0], seed[1] + 2])
    
    std = tf.random.stateless_uniform([], seed=seed_std, minval=0.0, maxval=std_limit * 255.0)
    noise = tf.random.stateless_normal(shape=tf.shape(image), seed=seed_noise, mean=0.0, stddev=std)
    return tf.clip_by_value(image + noise, 0.0, 255.0)


def tf_random_erasing(image: tf.Tensor, erasing_scale: tuple, erasing_ratio: tuple, seed: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    area = h * w
    
    def cond(img, success, i):
        return tf.logical_and(tf.logical_not(success), i < 10)
        
    def body(img, success, i):
        iter_seed = tf.stack([seed[0], seed[1] + i], axis=0)
        
        s_area = tf.random.stateless_uniform([], seed=iter_seed, minval=erasing_scale[0], maxval=erasing_scale[1])
        s_ratio = tf.random.stateless_uniform([], seed=tf.stack([iter_seed[0] + 1, iter_seed[1]], axis=0), minval=erasing_ratio[0], maxval=erasing_ratio[1])
        
        erase_area = s_area * area
        aspect_ratio = s_ratio
        
        erase_h = tf.cast(tf.round(tf.sqrt(erase_area * aspect_ratio)), tf.int32)
        erase_w = tf.cast(tf.round(tf.sqrt(erase_area / aspect_ratio)), tf.int32)
        
        h_int = tf.cast(h, tf.int32)
        w_int = tf.cast(w, tf.int32)
        
        valid = tf.logical_and(
            tf.logical_and(erase_h > 0, erase_h < h_int),
            tf.logical_and(erase_w > 0, erase_w < w_int)
        )
        
        def apply_erasing():
            top = tf.random.stateless_uniform([], seed=tf.stack([iter_seed[0] + 2, iter_seed[1]], axis=0), minval=0, maxval=h_int - erase_h, dtype=tf.int32)
            left = tf.random.stateless_uniform([], seed=tf.stack([iter_seed[0] + 3, iter_seed[1]], axis=0), minval=0, maxval=w_int - erase_w, dtype=tf.int32)
            fill_val = tf.random.stateless_uniform([], seed=tf.stack([iter_seed[0] + 4, iter_seed[1]], axis=0), minval=0.0, maxval=255.0)
            
            rows = tf.range(h_int)[:, None]
            cols = tf.range(w_int)[None, :]
            
            in_erase_box = tf.logical_and(
                tf.logical_and(rows >= top, rows < top + erase_h),
                tf.logical_and(cols >= left, cols < left + erase_w)
            )
            
            mask = tf.cast(in_erase_box[:, :, None], tf.float32)
            new_img = img * (1.0 - mask) + fill_val * mask
            return new_img, tf.constant(True)
            
        new_img, next_success = tf.cond(valid, apply_erasing, lambda: (img, tf.constant(False)))
        return new_img, next_success, i + 1
        
    final_img, _, _ = tf.while_loop(
        cond,
        body,
        loop_vars=(image, tf.constant(False), tf.constant(0)),
        shape_invariants=(image.get_shape(), tf.TensorShape([]), tf.TensorShape([]))
    )
    return final_img


# ---------------------------------------------------------------------------
# Selección y ejecución de aumento de datos
# ---------------------------------------------------------------------------

def apply_geom_augmentation(image: tf.Tensor, geom_id: int, config: AugmentationConfig, seed: tf.Tensor) -> tf.Tensor:
    if geom_id == 0:
        flip_prob = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0)
        return tf.cond(flip_prob < 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
    elif geom_id == 1:
        return tf_rotation(image, config.rotation_max_degrees, seed)
    elif geom_id == 2:
        return _tf_random_resized_crop(image, config.rrc_scale, config.rrc_ratio, seed)
    elif geom_id == 3:
        return tf_scaling(image, config.scale_factor, seed)
    elif geom_id == 4:
        return tf_perspective(image, config.perspective_scale_min, config.perspective_scale_max, seed)
    elif geom_id == 5:
        return tf_translation(image, config.translate_x_max_pct, config.translate_y_max_pct, seed)
    return image


def apply_photo_augmentation(image: tf.Tensor, photo_id: int, config: AugmentationConfig, seed: tf.Tensor) -> tf.Tensor:
    if photo_id == 0:
        return tf_brightness(image, config.brightness_limit, seed)
    elif photo_id == 1:
        return tf_contrast(image, config.contrast_limit, seed)
    elif photo_id == 2:
        return tf_gamma(image, config.gamma_limit, seed)
    elif photo_id == 3:
        return tf_hue_saturation(image, float(config.hue_shift_limit), float(config.sat_shift_limit), seed)
    return image


def apply_blur_noise_augmentation(image: tf.Tensor, bn_id: int, config: AugmentationConfig, seed: tf.Tensor) -> tf.Tensor:
    if bn_id == 0:
        return tf_gaussian_blur(image, config.blur_limit, seed)
    elif bn_id == 1:
        return tf_motion_blur(image, config.motion_blur_limit, seed)
    elif bn_id == 2:
        return tf_gaussian_noise(image, config.noise_std_limit, seed)
    return image


def apply_erasing_augmentation(image: tf.Tensor, config: AugmentationConfig, seed: tf.Tensor) -> tf.Tensor:
    return tf_random_erasing(image, config.erasing_scale, config.erasing_ratio, seed)


def augment_image(
    image: tf.Tensor,
    label: int,
    config: AugmentationConfig | None = None,
    master_seed: int = 42,
    sample_index: int | tf.Tensor = 0,
) -> tuple[tf.Tensor, int]:
    """Aplica aumento de datos configurable y completamente determinista a una imagen en escala [0, 255].
    """
    if config is None:
        config = AugmentationConfig()

    # Convertir seed a tensor
    seed_base = tf.stack([
        tf.cast(master_seed, tf.int32),
        tf.cast(sample_index, tf.int32)
    ])

    # 1. Identificar augmentaciones habilitadas por grupo en Python
    enabled_geom = []
    if config.horizontal_flip: enabled_geom.append(0)
    if config.rotation: enabled_geom.append(1)
    if config.random_resized_crop: enabled_geom.append(2)
    if config.scaling: enabled_geom.append(3)
    if config.perspective: enabled_geom.append(4)
    if config.translation: enabled_geom.append(5)

    enabled_photo = []
    if config.brightness: enabled_photo.append(0)
    if config.contrast: enabled_photo.append(1)
    if config.gamma: enabled_photo.append(2)
    if config.hue_saturation: enabled_photo.append(3)

    enabled_blur_noise = []
    if config.gaussian_blur: enabled_blur_noise.append(0)
    if config.motion_blur: enabled_blur_noise.append(1)
    if config.gaussian_noise: enabled_blur_noise.append(2)

    enabled_erasing = []
    if config.random_erasing: enabled_erasing.append(0)

    # 2. STEP 1: Uniformemente elegir EXACTAMENTE un grupo
    # Los 4 grupos son:
    # 0: Geométrico, 1: Fotométrico, 2: Desenfoque / Ruido, 3: Borrado
    seed_group = tf.stack([seed_base[0], seed_base[1] + 100])
    group_idx = tf.random.stateless_uniform([], seed=seed_group, minval=0, maxval=4, dtype=tf.int32)

    # 3. Definir funciones de rama para cada grupo
    def branch_geom():
        num_enabled = len(enabled_geom)
        if num_enabled == 0:
            return image
        if num_enabled == 1:
            return apply_geom_augmentation(image, enabled_geom[0], config, tf.stack([seed_base[0], seed_base[1] + 200]))
        
        seed_geom_sel = tf.stack([seed_base[0], seed_base[1] + 200])
        aug_idx = tf.random.stateless_uniform([], seed=seed_geom_sel, minval=0, maxval=num_enabled, dtype=tf.int32)
        
        branch_fns = {}
        for idx, geom_id in enumerate(enabled_geom):
            branch_fns[idx] = lambda g_id=geom_id: apply_geom_augmentation(image, g_id, config, tf.stack([seed_base[0], seed_base[1] + 300 + g_id]))
            
        return tf.switch_case(aug_idx, branch_fns)

    def branch_photo():
        num_enabled = len(enabled_photo)
        if num_enabled == 0:
            return image
        if num_enabled == 1:
            return apply_photo_augmentation(image, enabled_photo[0], config, tf.stack([seed_base[0], seed_base[1] + 400]))
            
        seed_photo_sel = tf.stack([seed_base[0], seed_base[1] + 400])
        aug_idx = tf.random.stateless_uniform([], seed=seed_photo_sel, minval=0, maxval=num_enabled, dtype=tf.int32)
        
        branch_fns = {}
        for idx, photo_id in enumerate(enabled_photo):
            branch_fns[idx] = lambda p_id=photo_id: apply_photo_augmentation(image, p_id, config, tf.stack([seed_base[0], seed_base[1] + 500 + p_id]))
            
        return tf.switch_case(aug_idx, branch_fns)

    def branch_blur_noise():
        num_enabled = len(enabled_blur_noise)
        if num_enabled == 0:
            return image
        if num_enabled == 1:
            return apply_blur_noise_augmentation(image, enabled_blur_noise[0], config, tf.stack([seed_base[0], seed_base[1] + 600]))
            
        seed_bn_sel = tf.stack([seed_base[0], seed_base[1] + 600])
        aug_idx = tf.random.stateless_uniform([], seed=seed_bn_sel, minval=0, maxval=num_enabled, dtype=tf.int32)
        
        branch_fns = {}
        for idx, bn_id in enumerate(enabled_blur_noise):
            branch_fns[idx] = lambda b_id=bn_id: apply_blur_noise_augmentation(image, b_id, config, tf.stack([seed_base[0], seed_base[1] + 700 + b_id]))
            
        return tf.switch_case(aug_idx, branch_fns)

    def branch_erasing():
        if len(enabled_erasing) == 0:
            return image
        return apply_erasing_augmentation(image, config, tf.stack([seed_base[0], seed_base[1] + 800]))

    # Ejecutar la rama seleccionada
    augmented_image = tf.switch_case(
        group_idx,
        branch_fns={
            0: branch_geom,
            1: branch_photo,
            2: branch_blur_noise,
            3: branch_erasing
        }
    )

    return augmented_image, label


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
    """
    if config is None:
        config = AugmentationConfig()

    # Cargar y redimensionar la imagen sin normalización
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
        aug_img, _ = augment_image(img, 0, config=config, master_seed=42, sample_index=i)
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
