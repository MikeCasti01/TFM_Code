"""Módulo para el análisis y generación de explicaciones visuales mediante Grad-CAM.

Este módulo implementa el cálculo de mapas de activación (heatmaps) utilizando
Grad-CAM, compatible con TensorFlow 2.20 y Keras 3.13. Permite analizar el
comportamiento del modelo sobre las clases débiles identificadas.

Compatibilidad:
--------------
- Python 3.12
- TensorFlow 2.20
- Keras 3.13
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Búsqueda y construcción del modelo Grad-CAM
# ---------------------------------------------------------------------------

def _find_last_conv_layer(model_or_layer: keras.layers.Layer) -> Tuple[Optional[keras.layers.Layer], Optional[keras.layers.Layer]]:
    """Busca recursivamente la última capa convolucional y su modelo padre.

    Args:
        model_or_layer (keras.layers.Layer): El modelo o capa a inspeccionar.

    Returns:
        Tuple[Optional[keras.layers.Layer], Optional[keras.layers.Layer]]:
            (ultima_capa_conv, modelo_padre)
    """
    last_conv: Optional[keras.layers.Layer] = None
    parent_model: Optional[keras.layers.Layer] = None

    def recurse(l: keras.layers.Layer, parent: Optional[keras.layers.Layer] = None) -> None:
        nonlocal last_conv, parent_model
        class_name = l.__class__.__name__
        
        # Identificar capas convolucionales de 2D
        if "Conv" in class_name or isinstance(l, (layers.Conv2D, layers.DepthwiseConv2D)):
            last_conv = l
            parent_model = parent
            
        if hasattr(l, "layers") and l.layers:
            for sub_layer in l.layers:
                recurse(sub_layer, l)

    recurse(model_or_layer)
    return last_conv, parent_model


def _find_conv_layer_by_name(model_or_layer: keras.layers.Layer, name: str) -> Tuple[Optional[keras.layers.Layer], Optional[keras.layers.Layer]]:
    """Busca recursivamente una capa convolucional por su nombre.

    Args:
        model_or_layer (keras.layers.Layer): El modelo o capa a inspeccionar.
        name (str): Nombre de la capa convolucional objetivo.

    Returns:
        Tuple[Optional[keras.layers.Layer], Optional[keras.layers.Layer]]:
            (capa_conv, modelo_padre)
    """
    found_layer: Optional[keras.layers.Layer] = None
    parent_model: Optional[keras.layers.Layer] = None

    def recurse(l: keras.layers.Layer, parent: Optional[keras.layers.Layer] = None) -> None:
        nonlocal found_layer, parent_model
        if l.name == name:
            found_layer = l
            parent_model = parent
            return
            
        if hasattr(l, "layers") and l.layers:
            for sub_layer in l.layers:
                recurse(sub_layer, l)
                if found_layer is not None:
                    return

    recurse(model_or_layer)
    return found_layer, parent_model


def build_gradcam_model(
    model: keras.Model,
    last_conv_layer_name: Optional[str] = None,
) -> keras.Model:
    """Construye un modelo Grad-CAM a partir de un modelo funcional de Keras.

    Identifica la capa convolucional objetivo (por nombre o autodetectando la última)
    y retorna un nuevo modelo de Keras que genera como salida tanto las activaciones
    de dicha capa como la predicción final de la red. Soporta modelos aninados.

    Args:
        model (keras.Model): Modelo funcional entrenado.
        last_conv_layer_name (Optional[str], optional): Nombre de la capa
            convolucional a analizar. Si es None, se autodetectará la última.

    Returns:
        keras.Model: Modelo Grad-CAM con salidas: [conv_layer_output, final_output].

    Raises:
        ValueError: Si no se encuentra ninguna capa convolucional.
    """
    conv_layer: Optional[keras.layers.Layer] = None
    parent_model: Optional[keras.layers.Layer] = None

    if last_conv_layer_name is not None:
        conv_layer, parent_model = _find_conv_layer_by_name(model, last_conv_layer_name)
    else:
        conv_layer, parent_model = _find_last_conv_layer(model)

    if conv_layer is None:
        raise ValueError(
            f"No se encontró la capa convolucional en el modelo. "
            f"Búsqueda por nombre: {last_conv_layer_name}."
        )

    logger.info(
        "Capa convolucional seleccionada para Grad-CAM: '%s' (Clase: %s)",
        conv_layer.name,
        conv_layer.__class__.__name__,
    )

    if parent_model is None or parent_model == model:
        # La capa convolucional está en el nivel superior del modelo
        return keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
    else:
        # La capa está anidada dentro de un submodelo (por ejemplo, el backbone)
        logger.info(
            "Capa anidada dentro del submodelo: '%s'. Reconstruyendo flujo funcional...",
            parent_model.name,
        )
        
        # 1. Crear un modelo intermedio para el submodelo anidado
        intermediate_parent = keras.Model(
            inputs=parent_model.inputs,
            outputs=[conv_layer.output, parent_model.output],
        )

        # 2. Localizar el índice del submodelo en las capas del modelo principal
        parent_idx = -1
        for i, layer in enumerate(model.layers):
            if layer == parent_model:
                parent_idx = i
                break

        if parent_idx == -1:
            raise ValueError(
                f"El submodelo '{parent_model.name}' no se encontró en "
                f"las capas del modelo principal."
            )

        # 3. Reconstruir el grafo conectando el modelo intermedio con las capas posteriores
        if isinstance(model.input_shape, list):
            input_shape = model.input_shape[0][1:]
        else:
            input_shape = model.input_shape[1:]

        inputs = keras.Input(shape=input_shape)
        conv_outputs, parent_outputs = intermediate_parent(inputs)

        # Aplicar secuencialmente las capas posteriores del modelo principal
        x = parent_outputs
        for layer in model.layers[parent_idx + 1:]:
            x = layer(x)

        gradcam_model = keras.Model(inputs=inputs, outputs=[conv_outputs, x])
        return gradcam_model


# ---------------------------------------------------------------------------
# Cálculos de mapas Grad-CAM
# ---------------------------------------------------------------------------

def compute_gradcam_batch(
    gradcam_model: keras.Model,
    images: np.ndarray,
    target_classes: Union[List[int], np.ndarray],
) -> np.ndarray:
    """Calcula mapas Grad-CAM normalizados en lote (batch) de forma eficiente.

    Args:
        gradcam_model (keras.Model): Modelo Grad-CAM de doble salida.
        images (np.ndarray): Lote de imágenes preprocesadas, shape (B, H, W, C).
        target_classes (Union[List[int], np.ndarray]): IDs de clases a explicar para cada imagen.

    Returns:
        np.ndarray: Lote de mapas de calor interpolados al tamaño de entrada,
            shape (B, H, W) con valores normalizados en el rango [0, 1].
    """
    images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    target_classes_tensor = tf.convert_to_tensor(target_classes, dtype=tf.int32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(images_tensor)
        # Extraer los scores de las clases objetivo
        loss = tf.gather(predictions, target_classes_tensor, batch_dims=1)

    # Gradientes de los scores w.r.t las salidas convolucionales
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "No se pudieron calcular los gradientes. Asegúrese de que el "
            "modelo se ejecute con TensorFlow y las capas estén conectadas."
        )

    # Promedio global de los gradientes por canal
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # Shape: (B, C)

    # Combinación lineal ponderada de los mapas de características
    weighted_conv = pooled_grads[:, tf.newaxis, tf.newaxis, :] * conv_outputs
    heatmaps = tf.reduce_sum(weighted_conv, axis=-1)  # Shape: (B, conv_H, conv_W)

    # ReLU: Conservar solo características con impacto positivo en la clase
    heatmaps = tf.maximum(heatmaps, 0.0)

    # Convertir a NumPy para normalización y redimensionado
    heatmaps_np = heatmaps.numpy()
    batch_size, img_H, img_W, _ = images.shape

    normalized_heatmaps = []
    for i in range(len(heatmaps_np)):
        h = heatmaps_np[i].astype(np.float32)
        h_max = h.max()
        h_min = h.min()

        # Evitar división por cero
        if h_max > h_min:
            h = (h - h_min) / (h_max - h_min + 1e-8)
        else:
            h = np.zeros_like(h)

        # Redimensionar al tamaño espacial de las imágenes de entrada
        h_resized = cv2.resize(h, (img_W, img_H), interpolation=cv2.INTER_LINEAR)
        normalized_heatmaps.append(h_resized)

    return np.array(normalized_heatmaps)


def compute_gradcam(
    gradcam_model: keras.Model,
    image: np.ndarray,
    target_class: int,
) -> np.ndarray:
    """Calcula el mapa Grad-CAM normalizado para una única imagen.

    Args:
        gradcam_model (keras.Model): Modelo Grad-CAM de doble salida.
        image (np.ndarray): Imagen preprocesada, shape (H, W, C) o (1, H, W, C).
        target_class (int): ID de la clase a explicar.

    Returns:
        np.ndarray: Mapa de calor normalizado, shape (H, W).
    """
    if len(image.shape) == 3:
        images = np.expand_dims(image, axis=0)
    else:
        images = image

    heatmaps = compute_gradcam_batch(gradcam_model, images, [target_class])
    return heatmaps[0]


def average_heatmaps(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula el mapa promedio (mean) y el mapa de desviación estándar (std).

    Args:
        heatmaps (np.ndarray): Lote de mapas de calor, shape (N, H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean_heatmap, std_heatmap) de shape (H, W).
    """
    if len(heatmaps) == 0:
        raise ValueError("El lote de mapas de calor está vacío.")
    mean_h = np.mean(heatmaps, axis=0)
    std_h = np.std(heatmaps, axis=0)
    return mean_h, std_h


# ---------------------------------------------------------------------------
# Visualizaciones con matplotlib
# ---------------------------------------------------------------------------

def plot_gradcam_comparison(
    image: np.ndarray,
    heatmap: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.5,
    title: Optional[str] = None,
    display_plot: bool = False,
) -> None:
    """Genera una figura de calidad de publicación con imagen cruda, heatmap y overlay.

    Args:
        image (np.ndarray): Imagen original RGB sin preprocesamiento inverso, shape (H, W, C).
        heatmap (np.ndarray): Mapa de calor normalizado [0, 1], shape (H, W).
        save_path (Optional[Union[str, Path]]): Ruta para guardar la figura.
        alpha (float): Transparencia del mapa sobre la imagen.
        title (Optional[str]): Título principal de la figura.
        display_plot (bool): Si es True, muestra la imagen en el notebook.
    """
    # Garantizar que la imagen sea uint8 en rango [0, 255]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = np.uint8(255 * image)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image

    # Colorear heatmap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Superponer
    overlay = cv2.addWeighted(img_uint8, 1.0 - alpha, heatmap_color, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Imagen original
    axes[0].imshow(img_uint8)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap solo
    im_h = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    fig.colorbar(im_h, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (alpha={alpha})")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
        logger.info("Gráfico Grad-CAM guardado en: %s", save_path)

    if display_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_confused_pair_gradcam(
    image: np.ndarray,
    heatmap_true: np.ndarray,
    heatmap_pred: np.ndarray,
    true_class_name: str,
    pred_class_name: str,
    save_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.5,
    display_plot: bool = False,
) -> None:
    """Genera comparación lado a lado del Grad-CAM de la clase real frente a la predicha.

    Útil para diagnosticar muestras con clasificaciones erróneas recurrentes.

    Args:
        image (np.ndarray): Imagen original RGB sin preprocesar, shape (H, W, C).
        heatmap_true (np.ndarray): Heatmap explicando la clase real, shape (H, W).
        heatmap_pred (np.ndarray): Heatmap explicando la clase predicha, shape (H, W).
        true_class_name (str): Nombre de la clase real.
        pred_class_name (str): Nombre de la clase predicha.
        save_path (Optional[Union[str, Path]]): Ruta de guardado del gráfico.
        alpha (float): Transparencia del mapa de calor.
        display_plot (bool): Si es True, muestra la imagen en el notebook.
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = np.uint8(255 * image)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image

    def get_overlay(h: np.ndarray) -> np.ndarray:
        h_uint8 = np.uint8(255 * h)
        h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img_uint8, 1.0 - alpha, h_color, alpha, 0)

    overlay_true = get_overlay(heatmap_true)
    overlay_pred = get_overlay(heatmap_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(img_uint8)
    axes[0].set_title(f"Original Image\n(True: {true_class_name} | Pred: {pred_class_name})")
    axes[0].axis("off")

    axes[1].imshow(overlay_true)
    axes[1].set_title(f"Grad-CAM: {true_class_name} (True)")
    axes[1].axis("off")

    axes[2].imshow(overlay_pred)
    axes[2].set_title(f"Grad-CAM: {pred_class_name} (Predicted)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
        logger.info("Gráfico Confused Pair guardado en: %s", save_path)

    if display_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_raw_image(path: str | Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Carga y redimensiona una imagen en formato RGB sin normalizar."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("No se pudo cargar la imagen en: %s. Generando imagen vacía.", path)
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, target_size)


def _preprocess_image(raw_img: np.ndarray, backbone_name: str) -> np.ndarray:
    """Aplica preprocesamiento específico del backbone sobre la imagen RGB."""
    img_float = raw_img.astype(np.float32)
    if backbone_name == "ResNet152":
        from tensorflow.keras.applications.resnet import preprocess_input
    elif backbone_name == "MobileNetV3Large":
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    elif backbone_name == "EfficientNetV2S":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    else:
        raise ValueError(f"Backbone no soportado: {backbone_name}")
    return preprocess_input(img_float)


# ---------------------------------------------------------------------------
# Orquestación de diagnósticos
# ---------------------------------------------------------------------------

def _select_samples(
    indices: np.ndarray,
    strategy: str,
    num_samples: int,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
    random_seed: int = 42,
) -> np.ndarray:
    """Selecciona de manera determinista un número de muestras a partir de los índices dados,
    siguiendo la estrategia seleccionada ('first', 'highest_confidence', 'lowest_confidence', 'random').
    """
    if len(indices) == 0:
        return indices

    # Evitar seleccionar más de lo disponible
    n_select = min(num_samples, len(indices))

    if strategy == "first":
        selected_indices = indices[:n_select]
    elif strategy == "highest_confidence":
        # Confianza de la predicción: la probabilidad de la clase predicha por el modelo
        confidences = y_pred_proba[indices, y_pred[indices]]
        # Ordenar descendente
        sorted_order = np.argsort(confidences)[::-1]
        selected_indices = indices[sorted_order[:n_select]]
    elif strategy == "lowest_confidence":
        # Confianza de la predicción: la probabilidad de la clase predicha por el modelo
        confidences = y_pred_proba[indices, y_pred[indices]]
        # Ordenar ascendente
        sorted_order = np.argsort(confidences)
        selected_indices = indices[sorted_order[:n_select]]
    elif strategy == "random":
        # Selección aleatoria determinista
        rng = np.random.default_rng(random_seed)
        shuffled_indices = indices.copy()
        rng.shuffle(shuffled_indices)
        selected_indices = shuffled_indices[:n_select]
    else:
        raise ValueError(f"Estrategia de selección no soportada: {strategy}")

    return selected_indices


# ---------------------------------------------------------------------------
# Orquestación de diagnósticos
# ---------------------------------------------------------------------------

def run_gradcam_diagnostics(
    model: keras.Model,
    test_metadata: pd.DataFrame,
    class_names: list[str],
    weak_classes: list[str],
    misclassified: dict[str, dict[str, list[str]]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: str | Path,
    backbone_name: str = "ResNet152",
    target_size: Tuple[int, int] = (224, 224),
    display_plots: bool = False,
    num_examples_per_case: int = 1,
    sample_selection_strategy: str = "first",
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Ejecuta el pipeline completo de diagnósticos de Grad-CAM para clases débiles.

    Para cada clase débil realiza:
    1. Promedio Grad-CAM y Desviación Estándar de muestras clasificadas correctamente.
    2. Comparación de mapa Grad-CAM de muestras correctas vs erróneas.
    3. Visualización contrastiva (True vs Predicted) de muestras erróneas con la
       clase más confundida.
    4. Generación de métricas de consistencia de Grad-CAM guardadas en summary.json.

    Args:
        model (keras.Model): Modelo de clasificación cargado.
        test_metadata (pd.DataFrame): Metadatos alineados del test split.
        class_names (list[str]): Nombres ordenados de las clases.
        weak_classes (list[str]): Lista de clases débiles identificadas.
        misclassified (dict): Muestras mal clasificadas del módulo anterior.
        y_true (np.ndarray): Etiquetas reales del conjunto de pruebas.
        y_pred (np.ndarray): Predicciones argmax del conjunto de pruebas.
        y_pred_proba (np.ndarray): Probabilidades de predicción.
        output_dir (str | Path): Directorio de salida.
        backbone_name (str): Nombre del backbone usado.
        target_size (Tuple[int, int]): Dimensiones espaciales del modelo.
        display_plots (bool): Si es True, muestra los gráficos en el notebook. Por defecto False.
        num_examples_per_case (int): Número de ejemplos a analizar para cada caso. Por defecto 1.
        sample_selection_strategy (str): Estrategia de selección de muestras. Por defecto "first".
        random_seed (int): Semilla para reproducibilidad de la selección aleatoria. Por defecto 42.

    Returns:
        list[dict[str, Any]]: Resumen de resultados diagnósticos por clase.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construir el modelo Grad-CAM una sola vez
    logger.info("Construyendo el modelo Grad-CAM...")
    gradcam_model = build_gradcam_model(model)

    diagnostics_summary: list[dict[str, Any]] = []

    for weak_class in weak_classes:
        c_idx = class_names.index(weak_class)
        logger.info("=== Iniciando diagnóstico Grad-CAM para clase débil: '%s' ===", weak_class)

        # 1. Identificar índices correctos y erróneos
        correct_idxs = np.where((y_true == c_idx) & (y_pred == c_idx))[0]
        error_idxs = np.where((y_true == c_idx) & (y_pred != c_idx))[0]

        logger.info(
            "Clase '%s' -> Muestras correctas: %d | Errores: %d",
            weak_class,
            len(correct_idxs),
            len(error_idxs),
        )

        avg_confidence = 0.0
        consistency = 0.0
        avg_std = 0.0
        mean_h: Optional[np.ndarray] = None
        std_h: Optional[np.ndarray] = None

        # 2. Análisis promedio (Correct Samples)
        if len(correct_idxs) > 0:
            avg_confidence = float(np.mean(y_pred_proba[correct_idxs, c_idx]))
            
            # Cargar imágenes correctas
            correct_paths = test_metadata.iloc[correct_idxs]["Absolute Path"].tolist()
            
            # Procesar en lotes pequeños para seguridad en memoria
            correct_heatmaps_list = []
            chunk_size = 32
            for i in range(0, len(correct_paths), chunk_size):
                chunk_paths = correct_paths[i:i + chunk_size]
                chunk_raw = np.array([_load_raw_image(p, target_size) for p in chunk_paths])
                chunk_preprocessed = np.array([_preprocess_image(img, backbone_name) for img in chunk_raw])
                chunk_targets = [c_idx] * len(chunk_paths)
                
                chunk_heatmaps = compute_gradcam_batch(gradcam_model, chunk_preprocessed, chunk_targets)
                correct_heatmaps_list.append(chunk_heatmaps)
                
            correct_heatmaps = np.concatenate(correct_heatmaps_list, axis=0)
            mean_h, std_h = average_heatmaps(correct_heatmaps)
            avg_std = float(np.mean(std_h))

            # Calcular consistencia del Grad-CAM (similitud de coseno media con el promedio)
            flat_heatmaps = correct_heatmaps.reshape(len(correct_heatmaps), -1)
            flat_mean = mean_h.reshape(-1)
            norm_mean = np.linalg.norm(flat_mean)
            if norm_mean > 0:
                norms = np.linalg.norm(flat_heatmaps, axis=1)
                norms = np.where(norms == 0, 1.0, norms)
                cosine_sims = np.dot(flat_heatmaps, flat_mean) / (norms * norm_mean)
                consistency = float(np.mean(cosine_sims))
            else:
                consistency = 0.0

            # Guardar visualización del promedio y desviación
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            im0 = axes[0].imshow(mean_h, cmap="jet", vmin=0.0, vmax=1.0)
            axes[0].set_title(f"Mean Grad-CAM\n(Consistency: {consistency:.4f})")
            axes[0].axis("off")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(std_h, cmap="hot")
            axes[1].set_title(f"Std Grad-CAM\n(Avg Std: {avg_std:.4f})")
            axes[1].axis("off")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            fig.suptitle(f"Grad-CAM Consistency Analysis: {weak_class}", fontsize=12)
            plt.tight_layout()
            avg_plot_path = output_dir / f"average_gradcam_{weak_class.replace(' ', '_')}.png"
            plt.savefig(str(avg_plot_path), bbox_inches="tight", dpi=150)
            if display_plots:
                plt.show()
            else:
                plt.close(fig)
            logger.info("Gráfico promedio guardado en: %s", avg_plot_path)
        else:
            logger.warning("No hay muestras correctas para la clase débil: '%s'", weak_class)

        # 3. Comparación Correcto vs Incorrecto
        if len(correct_idxs) > 0 and len(error_idxs) > 0:
            selected_correct = _select_samples(
                correct_idxs,
                sample_selection_strategy,
                num_examples_per_case,
                y_pred_proba,
                y_pred,
                random_seed,
            )
            selected_error = _select_samples(
                error_idxs,
                sample_selection_strategy,
                num_examples_per_case,
                y_pred_proba,
                y_pred,
                random_seed,
            )

            num_plots = min(len(selected_correct), len(selected_error))
            for i in range(num_plots):
                c_idx_sel = selected_correct[i]
                e_idx_sel = selected_error[i]

                c_path = test_metadata.iloc[c_idx_sel]["Absolute Path"]
                e_path = test_metadata.iloc[e_idx_sel]["Absolute Path"]
                
                c_raw = _load_raw_image(c_path, target_size)
                e_raw = _load_raw_image(e_path, target_size)
                
                c_prep = np.expand_dims(_preprocess_image(c_raw, backbone_name), axis=0)
                e_prep = np.expand_dims(_preprocess_image(e_raw, backbone_name), axis=0)
                
                c_heatmap = compute_gradcam(gradcam_model, c_prep, c_idx)
                e_heatmap = compute_gradcam(gradcam_model, e_prep, c_idx)
                
                alpha = 0.5
                def overlay_h(img, h):
                    h_uint8 = np.uint8(255 * h)
                    color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
                    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    return cv2.addWeighted(img, 1.0 - alpha, color, alpha, 0)
                    
                c_overlay = overlay_h(c_raw, c_heatmap)
                e_overlay = overlay_h(e_raw, e_heatmap)
                
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                
                axes[0, 0].imshow(c_raw)
                axes[0, 0].set_title(f"Correct Sample (True: {weak_class})\nConf: {y_pred_proba[c_idx_sel, c_idx]:.4f}")
                axes[0, 0].axis("off")
                
                axes[0, 1].imshow(c_overlay)
                axes[0, 1].set_title("Grad-CAM overlay (True Class)")
                axes[0, 1].axis("off")
                
                pred_class_label = class_names[y_pred[e_idx_sel]]
                axes[1, 0].imshow(e_raw)
                axes[1, 0].set_title(f"Misclassified Sample (True: {weak_class} | Pred: {pred_class_label})\nConf: {y_pred_proba[e_idx_sel, y_pred[e_idx_sel]]:.4f}")
                axes[1, 0].axis("off")
                
                axes[1, 1].imshow(e_overlay)
                axes[1, 1].set_title("Grad-CAM overlay (True Class)")
                axes[1, 1].axis("off")
                
                plt.tight_layout()
                vs_plot_path = output_dir / f"correct_vs_misclassified_{weak_class.replace(' ', '_')}_sample_{i}.png"
                plt.savefig(str(vs_plot_path), bbox_inches="tight", dpi=150)
                if display_plots:
                    plt.show()
                else:
                    plt.close(fig)
                logger.info("Gráfico comparativo guardado en: %s", vs_plot_path)

        # 4. Visualización de pareja confundida (Confused-Pair)
        class_errors = misclassified.get(weak_class, {})
        if class_errors:
            confused_class = max(class_errors, key=lambda k: len(class_errors[k]))
            confused_idx = class_names.index(confused_class)
            
            # Obtener todos los índices correspondientes a este confused pair
            confused_pair_idxs = np.where((y_true == c_idx) & (y_pred == confused_idx))[0]
            if len(confused_pair_idxs) > 0:
                selected_confused = _select_samples(
                    confused_pair_idxs,
                    sample_selection_strategy,
                    num_examples_per_case,
                    y_pred_proba,
                    y_pred,
                    random_seed,
                )
                
                for i, idx in enumerate(selected_confused):
                    pair_path = test_metadata.iloc[idx]["Absolute Path"]
                    pair_raw = _load_raw_image(pair_path, target_size)
                    pair_prep = np.expand_dims(_preprocess_image(pair_raw, backbone_name), axis=0)
                    
                    h_true = compute_gradcam(gradcam_model, pair_prep, c_idx)
                    h_pred = compute_gradcam(gradcam_model, pair_prep, confused_idx)
                    
                    pair_plot_path = output_dir / f"confused_pair_{weak_class.replace(' ', '_')}_vs_{confused_class.replace(' ', '_')}_sample_{i}.png"
                    plot_confused_pair_gradcam(
                        image=pair_raw,
                        heatmap_true=h_true,
                        heatmap_pred=h_pred,
                        true_class_name=weak_class,
                        pred_class_name=confused_class,
                        save_path=pair_plot_path,
                        alpha=0.5,
                        display_plot=display_plots,
                    )
            else:
                logger.info("No se encontraron índices para la pareja confundida: %s vs %s", weak_class, confused_class)
        else:
            logger.info("Clase '%s' no registra confusiones o errores de clasificación.", weak_class)

        # 5. Agregar al resumen general
        diagnostics_summary.append({
            "class": weak_class,
            "average confidence": round(avg_confidence, 4),
            "Grad-CAM consistency": round(consistency, 4),
            "heatmap std": round(avg_std, 4),
        })

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics_summary, f, indent=4, ensure_ascii=False)
        
    logger.info("Resumen de diagnósticos Grad-CAM guardado en: %s", summary_path)
    return diagnostics_summary
