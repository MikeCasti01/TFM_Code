"""Módulo para el análisis y explicación mediante SHAP (SHapley Additive exPlanations).

Este módulo calcula los valores SHAP y genera explicaciones contrastivas
para el análisis del comportamiento del modelo sobre las clases débiles.
Está diseñado para ser compatible con TensorFlow 2.20 y Keras 3.13.

Compatibilidad:
--------------
- Python 3.12
- TensorFlow 2.20
- Keras 3.13
- SHAP >= 0.40
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import shap

try:
    from skimage.segmentation import slic, quickshift, felzenszwalb, mark_boundaries
except ImportError as e:
    logging.getLogger(__name__).warning("skimage (scikit-image) no está instalado. Superpixel SHAP no estará disponible localmente, pero funcionará en Kaggle.")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Construcción del explicador SHAP
# ---------------------------------------------------------------------------

def build_shap_explainer(
    model: keras.Model,
    background_dataset: Any,
    explainer_type: str = "gradient",
) -> Any:
    """Construye y configura un explicador SHAP compatible con Keras 3 y TF 2.20.

    Si se proporciona un tf.data.Dataset para background_dataset, se extraen
    automáticamente muestras representativas (ej. 100 muestras) y se convierten
    a un arreglo NumPy como es requerido por SHAP.

    Args:
        model (keras.Model): Modelo funcional entrenado.
        background_dataset (Any): Datos de fondo para el explicador. Puede ser
            un tf.data.Dataset o un np.ndarray de imágenes de entrenamiento.
        explainer_type (str): Tipo de explicador. Valores soportados: "gradient".

    Returns:
        Any: Instancia del explicador de SHAP listo para usar.

    Raises:
        ValueError: Si el tipo de explicador no es soportado.
    """
    logger.info("Preparando datos de fondo (background data) para SHAP...")
    
    # Conversión del tf.data.Dataset si es necesario
    if isinstance(background_dataset, tf.data.Dataset):
        bg_images = []
        # Extraer un lote representativo
        for imgs, _ in background_dataset.take(5):
            bg_images.append(imgs.numpy())
        background_data = np.concatenate(bg_images, axis=0)[:100]
        logger.info(
            "tf.data.Dataset convertido a NumPy array. Muestras de fondo: %d",
            len(background_data),
        )
    else:
        background_data = np.asarray(background_dataset)
        logger.info("Arreglo NumPy de fondo recibido. Muestras: %d", len(background_data))

    explainer_type_lower = explainer_type.lower()
    if explainer_type_lower == "gradient":
        # GradientExplainer es el más robusto para TensorFlow 2.x eager y Keras 3
        logger.info("Inicializando shap.GradientExplainer...")
        explainer = shap.GradientExplainer(model, background_data)
    elif explainer_type_lower == "deep":
        logger.warning(
            "DeepExplainer puede reportar advertencias en Keras 3/TF 2.x "
            "debido a la compatibilidad de grafos. Usar con precaución."
        )
        explainer = shap.DeepExplainer(model, background_data)
    else:
        raise ValueError(
            f"Tipo de explicador SHAP no soportado: '{explainer_type}'. "
            f"Opciones válidas: 'gradient'."
        )

    return explainer


# ---------------------------------------------------------------------------
# Cálculos de valores SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(
    explainer: Any,
    images: np.ndarray,
    target_classes: Union[List[int], np.ndarray],
) -> Dict[int, np.ndarray]:
    """Calcula los valores SHAP para múltiples clases de salida deseadas.

    Args:
        explainer: Instancia del explicador de SHAP.
        images (np.ndarray): Imágenes a explicar, shape (M, H, W, C).
        target_classes (Union[List[int], np.ndarray]): IDs de clases a extraer.

    Returns:
        Dict[int, np.ndarray]: Diccionario mapeando class_id (int) -> arreglo
            de valores SHAP de shape (M, H, W, C).
    """
    logger.info("Ejecutando SHAP sobre %d muestras...", len(images))
    shap_vals = explainer.shap_values(images)

    results: Dict[int, np.ndarray] = {}
    unique_targets = np.unique(target_classes)

    for cls_id in unique_targets:
        cls_id_int = int(cls_id)
        if isinstance(shap_vals, list):
            # En la API clásica, shap_values es una lista de longitud num_classes
            if cls_id_int >= len(shap_vals):
                raise ValueError(
                    f"El ID de clase {cls_id_int} está fuera de los "
                    f"límites del número de salidas del modelo ({len(shap_vals)})."
                )
            results[cls_id_int] = shap_vals[cls_id_int]
        elif isinstance(shap_vals, np.ndarray):
            # En APIs modernas, puede retornar un arreglo multi-dimensional
            # Shape esperado: (M, H, W, C, num_classes)
            if len(shap_vals.shape) == 5:
                results[cls_id_int] = shap_vals[..., cls_id_int]
            elif len(shap_vals.shape) == 4:
                # Caso de salida única
                results[cls_id_int] = shap_vals
            else:
                raise ValueError(
                    f"Forma inesperada del array de valores SHAP: {shap_vals.shape}."
                )
        else:
            raise TypeError(f"Formato desconocido para shap_values: {type(shap_vals)}")

    return results


def compute_contrastive_shap(
    shap_values_true: np.ndarray,
    shap_values_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula el mapa de contraste restando la importancia predicha de la real.

    El mapa de contraste (diferencia) resalta los rasgos visuales que justifican
    la clase real frente a la clase que causó la confusión.

    Args:
        shap_values_true (np.ndarray): Valores SHAP de la clase real, shape (H, W, C) o (M, H, W, C).
        shap_values_pred (np.ndarray): Valores SHAP de la clase predicha.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (shap_values_true, shap_values_pred, diff_map) de la misma forma que la entrada.
    """
    diff_map = shap_values_true - shap_values_pred
    return shap_values_true, shap_values_pred, diff_map


# ---------------------------------------------------------------------------
# Visualizaciones con matplotlib
# ---------------------------------------------------------------------------

def plot_contrastive_shap(
    image: np.ndarray,
    shap_true: np.ndarray,
    shap_pred: np.ndarray,
    diff_map: np.ndarray,
    true_class_name: str,
    pred_class_name: str,
    save_path: Optional[Union[str, Path]] = None,
    display_plot: bool = False,
) -> None:
    """Genera una figura de calidad científica comparando SHAP real, predicho y diferencia.

    Args:
        image (np.ndarray): Imagen original RGB en [0, 255] o [0, 1].
        shap_true (np.ndarray): Valores SHAP 3D para la clase real.
        shap_pred (np.ndarray): Valores SHAP 3D para la clase predicha.
        diff_map (np.ndarray): Valores de la diferencia SHAP (shap_true - shap_pred).
        true_class_name (str): Nombre de la clase real.
        pred_class_name (str): Nombre de la clase predicha (confundida).
        save_path (Optional[Union[str, Path]]): Ruta de guardado.
        display_plot (bool): Si es True, muestra la imagen en pantalla.
    """
    # Garantizar rango [0, 255] en la imagen
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = np.uint8(255 * image)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image

    # Sumar a través de los canales RGB para obtener mapas 2D significativos
    shap_t_2d = np.sum(shap_true, axis=-1)
    shap_p_2d = np.sum(shap_pred, axis=-1)
    diff_2d = np.sum(diff_map, axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Imagen original
    axes[0].imshow(img_uint8)
    axes[0].set_title(f"Original Image\n(True: {true_class_name} | Pred: {pred_class_name})")
    axes[0].axis("off")

    def _plot_map(ax: Any, data: np.ndarray, title: str) -> None:
        # Centrar el mapa de color en cero usando un colormap divergente
        max_val = np.max(np.abs(data))
        if max_val == 0:
            max_val = 1e-8
        im = ax.imshow(data, cmap="seismic", vmin=-max_val, vmax=max_val)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. SHAP True Class
    _plot_map(axes[1], shap_t_2d, f"SHAP: {true_class_name}\n(True Class)")

    # 3. SHAP Predicted Class
    _plot_map(axes[2], shap_p_2d, f"SHAP: {pred_class_name}\n(Predicted Class)")

    # 4. Difference Map
    _plot_map(axes[3], diff_2d, "Difference Map\n(True - Pred)")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
        logger.info("Gráfico contrastivo SHAP guardado en: %s", save_path)

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


def plot_superpixel_shap(
    image: np.ndarray,
    segments: np.ndarray,
    shap_true: np.ndarray,
    shap_pred: np.ndarray,
    true_class_name: str,
    pred_class_name: str,
    save_path: Optional[Union[str, Path]] = None,
    display_plot: bool = False,
) -> None:
    """Genera una figura de calidad de publicación comparando imagen, segmentos y atribuciones SHAP.
    
    Args:
        image (np.ndarray): Imagen original RGB, shape (H, W, C).
        segments (np.ndarray): Mapa de etiquetas de superpíxeles, shape (H, W).
        shap_true (np.ndarray): Mapa 2D de importancia mapeada a los píxeles para la clase real.
        shap_pred (np.ndarray): Mapa 2D de importancia mapeada a los píxeles para la clase predicha.
        true_class_name (str): Nombre de la clase real.
        pred_class_name (str): Nombre de la clase predicha.
        save_path (Optional[Union[str, Path]]): Ruta de guardado del gráfico.
        display_plot (bool): Si es True, muestra la imagen en el notebook.
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = np.uint8(255 * image)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image

    # Calcular la diferencia (atribución contrastiva)
    diff_map = shap_true - shap_pred

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Imagen original
    axes[0].imshow(img_uint8)
    axes[0].set_title(f"Original Image\n(True: {true_class_name} | Pred: {pred_class_name})")
    axes[0].axis("off")

    # 2. Límites de segmentación
    img_boundaries = mark_boundaries(img_uint8, segments, color=(1, 1, 0)) # límites en amarillo
    axes[1].imshow(img_boundaries)
    axes[1].set_title("Superpixel Segments")
    axes[1].axis("off")

    # Centrar la escala de colores en cero para visualización de importancia
    max_val = np.max(np.abs(diff_map))
    if max_val == 0:
        max_val = 1e-8

    # 3. Atribución SHAP sobre los segmentos (Diferencia: True - Pred)
    im_shap = axes[2].imshow(diff_map, cmap="seismic", vmin=-max_val, vmax=max_val)
    axes[2].set_title("SHAP Attribution (True - Pred)")
    axes[2].axis("off")
    fig.colorbar(im_shap, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Superposición (Overlay) de SHAP sobre la imagen
    axes[3].imshow(img_uint8)
    axes[3].imshow(diff_map, cmap="seismic", vmin=-max_val, vmax=max_val, alpha=0.5)
    axes[3].set_title("SHAP Overlay (alpha=0.5)")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
        logger.info("Gráfico superpixel SHAP guardado en: %s", save_path)

    if display_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Orquestación de diagnósticos
# ---------------------------------------------------------------------------

def run_shap_diagnostics(
    model: keras.Model,
    test_metadata: pd.DataFrame,
    class_names: list[str],
    weak_classes: list[str],
    misclassified: dict[str, dict[str, list[str]]],
    background_dataset: Any,
    output_dir: str | Path,
    backbone_name: str = "ResNet152",
    target_size: Tuple[int, int] = (224, 224),
    num_samples_per_class: int = 3,
    explainer_type: str = "gradient",
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    y_pred_proba: Optional[np.ndarray] = None,
    display_plots: bool = False,
    use_superpixel_shap: bool = False,
    segmentation_method: str = "slic",
    sample_selection_strategy: str = "first",
    random_seed: int = 42,
) -> dict[str, Any]:
    """Ejecuta el análisis SHAP contrastivo para las clases débiles.

    Para cada clase débil, toma hasta `num_samples_per_class` de muestras mal
    clasificadas de la clase más confundida, calcula los valores SHAP de la
    clase real y predicha, calcula la diferencia, grafica las explicaciones
    contrastivas y guarda los arreglos NumPy correspondientes.

    Args:
        model (keras.Model): Modelo funcional entrenado.
        test_metadata (pd.DataFrame): Metadatos alineados del test split.
        class_names (list[str]): Nombres ordenados de las clases.
        weak_classes (list[str]): Lista de clases débiles identificadas.
        misclassified (dict): Muestras mal clasificadas del módulo anterior.
        background_dataset (Any): Datos de entrenamiento de fondo para SHAP.
        output_dir (str | Path): Directorio de salida.
        backbone_name (str): Nombre del backbone usado.
        target_size (Tuple[int, int]): Dimensiones espaciales del modelo.
        num_samples_per_class (int): Número de muestras erróneas a analizar por clase.
        explainer_type (str): Tipo de explicador. Usualmente 'gradient'.
        y_true (np.ndarray, optional): Etiquetas reales del conjunto de pruebas.
        y_pred (np.ndarray, optional): Predicciones argmax del conjunto de pruebas.
        y_pred_proba (np.ndarray, optional): Probabilidades de predicción.
        display_plots (bool): Si es True, muestra los gráficos en el notebook. Por defecto False.
        use_superpixel_shap (bool): Si es True, ejecuta además la explicación basada en superpíxeles. Por defecto False.
        segmentation_method (str): Algoritmo de segmentación de superpíxeles ("slic", "quickshift", "felzenszwalb"). Por defecto "slic".
        sample_selection_strategy (str): Estrategia de selección de muestras ("first", "highest_confidence", "lowest_confidence", "random"). Por defecto "first".
        random_seed (int): Semilla para la reproducibilidad de la selección aleatoria. Por defecto 42.

    Returns:
        dict[str, Any]: Estructura de metadatos general de SHAP.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Construir el explicador SHAP (esto puede tardar unos segundos)
    logger.info("Construyendo el explicador SHAP (%s)...", explainer_type)
    explainer = build_shap_explainer(model, background_dataset, explainer_type)

    shap_metadata: dict[str, Any] = {
        "explainer_type": explainer_type,
        "backbone_name": backbone_name,
        "use_superpixel_shap": use_superpixel_shap,
        "segmentation_method": segmentation_method if use_superpixel_shap else None,
        "samples": [],
    }

    for weak_class in weak_classes:
        c_idx = class_names.index(weak_class)
        class_errors = misclassified.get(weak_class, {})

        if not class_errors:
            logger.info("La clase débil '%s' no tiene errores registrados.", weak_class)
            continue

        # Obtener la clase más confundida
        confused_class = max(class_errors, key=lambda k: len(class_errors[k]))
        confused_idx = class_names.index(confused_class)
        
        # Determinar de forma determinista qué muestras analizar si se proveen las predicciones
        if y_true is not None and y_pred is not None and y_pred_proba is not None:
            confused_pair_idxs = np.where((y_true == c_idx) & (y_pred == confused_idx))[0]
            selected_idxs = _select_samples(
                confused_pair_idxs,
                sample_selection_strategy,
                num_samples_per_class,
                y_pred_proba,
                y_pred,
                random_seed,
            )
            confused_paths = [str(test_metadata.iloc[idx]["Absolute Path"]) for idx in selected_idxs]
        else:
            logger.warning("No se proporcionaron y_true, y_pred o y_pred_proba. Usando estrategia 'first' por defecto.")
            confused_paths = class_errors[confused_class][:num_samples_per_class]

        logger.info(
            "Clase débil '%s' -> Más confundida con '%s'. Analizando %d muestras...",
            weak_class,
            confused_class,
            len(confused_paths),
        )

        for i, path_str in enumerate(confused_paths):
            logger.info("Analizando muestra SHAP %d/%d: %s", i + 1, len(confused_paths), path_str)
            raw_img = _load_raw_image(path_str, target_size)
            preprocessed_img = _preprocess_image(raw_img, backbone_name)
            
            # Preparar lote de una sola imagen para SHAP
            img_batch = np.expand_dims(preprocessed_img, axis=0)

            # 2. Calcular valores SHAP tradicionales
            target_ids = [c_idx, confused_idx]
            shap_results = compute_shap_values(explainer, img_batch, target_ids)

            # Extraer para la única muestra del lote
            shap_true = shap_results[c_idx][0]
            shap_pred = shap_results[confused_idx][0]

            # 3. Calcular mapa contrastivo
            _, _, diff_map = compute_contrastive_shap(shap_true, shap_pred)

            # 4. Guardar visualización
            class_clean = weak_class.replace(" ", "_")
            confused_clean = confused_class.replace(" ", "_")
            
            plot_filename = f"shap_contrastive_{class_clean}_vs_{confused_clean}_sample_{i}.png"
            plot_path = output_dir / plot_filename
            
            plot_contrastive_shap(
                image=raw_img,
                shap_true=shap_true,
                shap_pred=shap_pred,
                diff_map=diff_map,
                true_class_name=weak_class,
                pred_class_name=confused_class,
                save_path=plot_path,
                display_plot=display_plots,
            )

            # 5. Guardar arreglos NumPy binarios (.npz)
            array_filename = f"shap_arrays_{class_clean}_vs_{confused_clean}_sample_{i}.npz"
            array_path = output_dir / array_filename
            np.savez_compressed(
                array_path,
                shap_true=shap_true,
                shap_pred=shap_pred,
                diff_map=diff_map,
            )
            logger.info("Arreglos SHAP guardados en: %s", array_path)

            # 6. Opcional: Superpixel-based SHAP
            sp_plot_filename = None
            sp_array_filename = None
            if use_superpixel_shap:
                logger.info("Ejecutando análisis SHAP basado en superpíxeles...")
                
                # Segmentar
                if segmentation_method == "slic":
                    segments = slic(raw_img, n_segments=50, compactness=10, sigma=1)
                elif segmentation_method == "quickshift":
                    segments = quickshift(raw_img, kernel_size=4, max_dist=200, ratio=0.2)
                elif segmentation_method == "felzenszwalb":
                    segments = felzenszwalb(raw_img, scale=100, sigma=0.5, min_size=50)
                else:
                    raise ValueError(f"Método de segmentación no soportado: {segmentation_method}")
                
                unique_segs = np.unique(segments)
                num_segments = len(unique_segs)
                
                # Inicializar masker de SHAP con difuminado para el fondo
                image_masker = shap.maskers.Image("blur(20,20)", raw_img.shape)
                
                # Función que toma máscaras de segmentos y devuelve predicciones del modelo
                def predict_fn(masks):
                    batch_images = []
                    for mask in masks:
                        pixel_mask = np.zeros(target_size, dtype=bool)
                        for j, seg_label in enumerate(unique_segs):
                            if mask[j]:
                                pixel_mask[segments == seg_label] = True
                        
                        masked_img = image_masker(pixel_mask.flatten(), raw_img)[0]
                        batch_images.append(_preprocess_image(masked_img, backbone_name))
                    return model.predict(np.array(batch_images), verbose=0)
                
                # Crear KernelExplainer para superpíxeles
                background_mask = np.zeros((1, num_segments))
                superpixel_explainer = shap.KernelExplainer(predict_fn, background_mask)
                
                # Evaluar
                active_mask = np.ones((1, num_segments))
                sp_shap_values = superpixel_explainer.shap_values(active_mask, nsamples=100)
                
                # Extraer robustamente
                if isinstance(sp_shap_values, list):
                    shap_true_seg = sp_shap_values[c_idx][0]
                    shap_pred_seg = sp_shap_values[confused_idx][0]
                else:
                    shap_true_seg = sp_shap_values[0, :, c_idx]
                    shap_pred_seg = sp_shap_values[0, :, confused_idx]
                
                # Mapear de segmento a píxel
                shap_true_pixel = np.zeros(target_size[:2])
                shap_pred_pixel = np.zeros(target_size[:2])
                for j, seg_label in enumerate(unique_segs):
                    shap_true_pixel[segments == seg_label] = shap_true_seg[j]
                    shap_pred_pixel[segments == seg_label] = shap_pred_seg[j]
                
                sp_plot_filename = f"shap_superpixel_{class_clean}_vs_{confused_clean}_sample_{i}.png"
                sp_plot_path = output_dir / sp_plot_filename
                
                plot_superpixel_shap(
                    image=raw_img,
                    segments=segments,
                    shap_true=shap_true_pixel,
                    shap_pred=shap_pred_pixel,
                    true_class_name=weak_class,
                    pred_class_name=confused_class,
                    save_path=sp_plot_path,
                    display_plot=display_plots,
                )
                
                sp_array_filename = f"shap_superpixel_arrays_{class_clean}_vs_{confused_clean}_sample_{i}.npz"
                sp_array_path = output_dir / sp_array_filename
                np.savez_compressed(
                    sp_array_path,
                    shap_true=shap_true_pixel,
                    shap_pred=shap_pred_pixel,
                    segments=segments,
                )
                logger.info("Arreglos SHAP superpixel guardados en: %s", sp_array_path)

            # 7. Registrar en metadatos
            sample_entry = {
                "true_class": weak_class,
                "predicted_class": confused_class,
                "image_path": str(path_str),
                "npz_file": str(array_filename),
                "plot_file": str(plot_filename),
            }
            if use_superpixel_shap:
                sample_entry["superpixel_npz_file"] = str(sp_array_filename)
                sample_entry["superpixel_plot_file"] = str(sp_plot_filename)
            shap_metadata["samples"].append(sample_entry)

    # Guardar metadatos como JSON
    metadata_path = output_dir / "shap_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(shap_metadata, f, indent=4, ensure_ascii=False)

    logger.info("Metadatos globales SHAP guardados en: %s", metadata_path)
    return shap_metadata
