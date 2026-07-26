"""Módulo de análisis de explicaciones SHAP para modelos de clasificación de imágenes.

Utiliza `shap.PartitionExplainer` con `shap.maskers.Image("blur(64,64)", ...)` sobre
imágenes RGB sin preprocesamiento inicial (rango [0, 255]). Realiza análisis
contrastivo comparando atribuciones SHAP para la clase verdadera y la clase predicha.

Compatibilidad:
- Python 3.12
- TensorFlow 2.20
- Keras 3.13
- SHAP >= 0.40
- scikit-learn
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carga y preprocesamiento de imágenes
# ---------------------------------------------------------------------------

def _load_raw_image(
    path: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Carga y redimensiona una imagen en formato RGB sin normalizar (rango 0-255)."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("No se pudo cargar la imagen en: %s. Generando imagen vacía.", path)
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, target_size)


def build_shap_predict_fn(
    model: keras.Model,
    backbone_name: str = "ResNet152",
) -> Callable[[np.ndarray], np.ndarray]:
    """Crea una función wrapper de predicción para SHAP.

    Entrada:
        images (np.ndarray): Lote de imágenes RGB en el rango [0, 255] con forma (B, H, W, 3).

    Salida:
        np.ndarray: Probabilidades softmax del modelo con forma (B, num_classes).
    """
    def predict_fn(images: np.ndarray) -> np.ndarray:
        images_float = images.astype(np.float32)
        if backbone_name == "ResNet152":
            from tensorflow.keras.applications.resnet import preprocess_input
        elif backbone_name == "MobileNetV3Large":
            from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        else:
            raise ValueError(f"Backbone no soportado para SHAP: {backbone_name}")

        preprocessed = preprocess_input(images_float)
        preds = model.predict(preprocessed, verbose=0)
        return preds

    return predict_fn


# ---------------------------------------------------------------------------
# Extracción de valores SHAP
# ---------------------------------------------------------------------------

def _extract_shap_map(shap_output: Any, target_class_idx: int) -> np.ndarray:
    """Extrae y agrega el mapa de atribución espacial 2D (H, W) para la clase objetivo.

    Preserva valores de atribución positivos (soporte) y negativos (oposición).
    No aplica ReLU.
    """
    if hasattr(shap_output, "values"):
        vals = shap_output.values
    else:
        vals = shap_output

    if isinstance(vals, list):
        class_vals = vals[target_class_idx]
        if class_vals.ndim == 4:  # (1, H, W, C)
            class_vals = class_vals[0]
        if class_vals.ndim == 3:  # (H, W, C)
            return np.mean(class_vals, axis=-1)
        return class_vals

    if isinstance(vals, np.ndarray):
        if vals.ndim == 5:  # (batch=1, H, W, C, num_classes)
            class_vals = vals[0, :, :, :, target_class_idx]
            return np.mean(class_vals, axis=-1)
        elif vals.ndim == 4:  # (batch=1, H, W, num_classes)
            return vals[0, :, :, target_class_idx]
        elif vals.ndim == 3:  # (H, W, num_classes)
            if vals.shape[-1] > target_class_idx:
                return vals[:, :, target_class_idx]
            return np.mean(vals, axis=-1)

    raise ValueError(
        f"Estructura de valores SHAP no soportada con forma/tipo: {getattr(vals, 'shape', type(vals))}"
    )


# ---------------------------------------------------------------------------
# Visualización SHAP en 4 paneles
# ---------------------------------------------------------------------------

def plot_shap_contrastive_comparison(
    raw_image: np.ndarray,
    shap_true: np.ndarray,
    shap_pred: np.ndarray,
    true_class_name: str,
    pred_class_name: str,
    true_confidence: float,
    pred_confidence: float,
    save_path: str | Path | None = None,
    display_plot: bool = False,
    overlay_shap: bool = False,
    overlay_alpha: float = 0.5,
) -> None:
    """Genera una figura de 4 paneles con explicaciones contrastivas de SHAP.

    Paneles:
    1. Imagen RGB original con títulos de clase y confianza.
    2. Atribución SHAP para la clase verdadera (mapa divergente bwr/seismic).
    3. Atribución SHAP para la clase predicha (mapa divergente bwr/seismic).
    4. Mapa contrastivo: SHAP(predicha) - SHAP(verdadera).

    Args:
        raw_image (np.ndarray): Imagen RGB original (H, W, 3).
        shap_true (np.ndarray): Mapa de atribución SHAP 2D (H, W) para la clase verdadera.
        shap_pred (np.ndarray): Mapa de atribución SHAP 2D (H, W) para la clase predicha.
        true_class_name (str): Nombre de la clase verdadera.
        pred_class_name (str): Nombre de la clase predicha por el modelo.
        true_confidence (float): Probabilidad softmax predicha para la clase verdadera.
        pred_confidence (float): Probabilidad softmax predicha para la clase predicha.
        save_path (str | Path | None): Ruta opcional de guardado para la figura.
        display_plot (bool): Si es True, muestra la figura en el entorno interactivo.
        overlay_shap (bool): Si es True, superpone los mapas SHAP sobre la imagen RGB original.
        overlay_alpha (float): Transparencia del mapa SHAP superpuesto en el rango [0.0, 1.0].
    """
    if not (0.0 <= overlay_alpha <= 1.0):
        raise ValueError(
            f"overlay_alpha debe estar en el rango [0.0, 1.0], pero se recibió {overlay_alpha}."
        )

    if raw_image.dtype != np.uint8:
        if raw_image.max() <= 1.0:
            img_uint8 = np.uint8(255 * raw_image)
        else:
            img_uint8 = raw_image.astype(np.uint8)
    else:
        img_uint8 = raw_image

    contrastive_map = shap_pred - shap_true

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Imagen original
    axes[0].imshow(img_uint8)
    axes[0].set_title(
        f"Original Image\n"
        f"True: {true_class_name} ({true_confidence:.4f})\n"
        f"Pred: {pred_class_name} ({pred_confidence:.4f})"
    )
    axes[0].axis("off")

    def plot_attribution(ax: plt.Axes, map_data: np.ndarray, title: str) -> None:
        abs_max = max(abs(np.min(map_data)), abs(np.max(map_data)), 1e-6)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

        if overlay_shap:
            ax.imshow(img_uint8)
            im = ax.imshow(map_data, cmap="seismic", norm=norm, alpha=overlay_alpha)
        else:
            im = ax.imshow(map_data, cmap="seismic", norm=norm)

        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: SHAP clase verdadera
    plot_attribution(
        axes[1],
        shap_true,
        f"SHAP (True Class: {true_class_name})\nRed=Support | Blue=Oppose",
    )

    # Panel 3: SHAP clase predicha
    plot_attribution(
        axes[2],
        shap_pred,
        f"SHAP (Pred Class: {pred_class_name})\nRed=Support | Blue=Oppose",
    )

    # Panel 4: Mapa contrastivo
    plot_attribution(
        axes[3],
        contrastive_map,
        f"Contrastive: SHAP(Pred) - SHAP(True)\nRed=Favors Pred | Blue=Favors True",
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
        logger.info("Gráfico SHAP guardado en: %s", save_path)

    if display_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entrada principal del análisis SHAP
# ---------------------------------------------------------------------------

def run_shap_analysis(
    model: keras.Model,
    test_metadata: pd.DataFrame,
    class_names: list[str],
    weak_class: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: str | Path,
    sample_type: str = "correct",
    overlay_shap: bool = False,
    overlay_alpha: float = 0.5,
    num_examples: int = 3,
    random_seed: int = 42,
    offset: int = 0,
    max_evals: int = 300,
    backbone_name: str = "ResNet152",
    target_size: tuple[int, int] = (224, 224),
    display_plots: bool = False,
) -> dict[str, Any]:
    """Ejecuta el análisis XAI con SHAP PartitionExplainer para una clase débil especificada.

    Soporta la selección explícita entre muestras correctamente clasificadas ('correct')
    y muestras mal clasificadas ('misclassified'), así como la superposición visual
    opcional del mapa divergente SHAP sobre la imagen original.

    Args:
        model (keras.Model): Modelo Keras entrenado.
        test_metadata (pd.DataFrame): DataFrame de metadatos alineados del test split.
        class_names (list[str]): Lista ordenada de nombres de clases.
        weak_class (str): Nombre de la clase débil a analizar (clase real).
        y_true (np.ndarray): Etiquetas reales (n_samples,).
        y_pred (np.ndarray): Predicciones argmax (n_samples,).
        y_pred_proba (np.ndarray): Probabilidades predichas (n_samples, num_classes).
        output_dir (str | Path): Directorio de salida para guardar imágenes y resúmenes.
        sample_type (str): Tipo de muestra a analizar ('correct' o 'misclassified'). Por defecto 'correct'.
        overlay_shap (bool): Si es True, superpone el mapa divergente SHAP sobre la imagen RGB. Por defecto False.
        overlay_alpha (float): Opacidad del mapa SHAP superpuesto [0.0, 1.0]. Por defecto 0.5.
        num_examples (int): Cantidad máxima de ejemplos a analizar. Por defecto 3.
        random_seed (int): Semilla para la selección determinista. Por defecto 42.
        offset (int): Posición inicial tras la mezcla determinista. Por defecto 0.
        max_evals (int): Presupuesto máximo de evaluaciones de SHAP. Por defecto 300.
        backbone_name (str): Nombre del backbone utilizado. Por defecto 'ResNet152'.
        target_size (tuple[int, int]): Dimensiones espaciales del modelo. Por defecto (224, 224).
        display_plots (bool): Si es True, muestra los gráficos en el notebook.

    Returns:
        dict[str, Any]: Estructura de resultados detallada del análisis SHAP.
    """
    if weak_class not in class_names:
        raise ValueError(
            f"La clase débil '{weak_class}' no existe en class_names."
        )

    if sample_type not in {"correct", "misclassified"}:
        raise ValueError(
            f"sample_type no soportado: '{sample_type}'. Debe ser 'correct' o 'misclassified'."
        )

    if not (0.0 <= overlay_alpha <= 1.0):
        raise ValueError(
            f"overlay_alpha debe estar en el rango [0.0, 1.0], pero se recibió {overlay_alpha}."
        )

    if max_evals <= 0:
        raise ValueError(f"max_evals debe ser > 0, pero se recibió {max_evals}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c_idx = class_names.index(weak_class)

    # Filtrado según el tipo de muestra especificado
    if sample_type == "correct":
        candidate_idxs = np.where((y_true == c_idx) & (y_pred == c_idx))[0]
    else:  # "misclassified"
        candidate_idxs = np.where((y_true == c_idx) & (y_pred != c_idx))[0]

    logger.info(
        "=== Iniciando análisis SHAP (%s) | Clase: '%s' | Candidatos disponibles: %d | Offset: %d ===",
        sample_type,
        weak_class,
        len(candidate_idxs),
        offset,
    )

    if len(candidate_idxs) == 0:
        logger.warning(
            "No hay candidatos de tipo '%s' para la clase débil '%s'.",
            sample_type,
            weak_class,
        )
        return {
            "weak_class": weak_class,
            "sample_type": sample_type,
            "selected_samples": 0,
            "status": "empty",
            "samples": [],
        }

    # Mezcla determinista
    rng = np.random.default_rng(random_seed)
    shuffled_idxs = candidate_idxs.copy()
    rng.shuffle(shuffled_idxs)

    if offset >= len(shuffled_idxs):
        logger.warning(
            "El offset %d sobrepasa el número de muestras disponibles (%d) para la clase '%s' (%s).",
            offset,
            len(shuffled_idxs),
            weak_class,
            sample_type,
        )
        return {
            "weak_class": weak_class,
            "sample_type": sample_type,
            "selected_samples": 0,
            "status": "offset_exceeded",
            "samples": [],
        }

    selected_idxs = shuffled_idxs[offset : offset + num_examples]

    # Construir wrapper y PartitionExplainer de SHAP
    predict_fn = build_shap_predict_fn(model, backbone_name=backbone_name)
    masker = shap.maskers.Image("blur(64,64)", target_size + (3,))
    explainer = shap.PartitionExplainer(predict_fn, masker)

    samples_metadata: list[dict[str, Any]] = []

    for i, idx in enumerate(selected_idxs):
        img_path = test_metadata.iloc[idx]["Absolute Path"]
        raw_img = _load_raw_image(img_path, target_size=target_size)

        true_cls_idx = int(y_true[idx])
        pred_cls_idx = int(y_pred[idx])

        true_cls_name = class_names[true_cls_idx]
        pred_cls_name = class_names[pred_cls_idx]

        true_conf = float(y_pred_proba[idx, true_cls_idx])
        pred_conf = float(y_pred_proba[idx, pred_cls_idx])

        # SHAP requiere batch input: (1, H, W, 3)
        batch_input = np.expand_dims(raw_img, axis=0)

        logger.info(
            "Calculando SHAP para muestra %d/%d (%s, Fila %d)...",
            i + 1,
            len(selected_idxs),
            sample_type,
            idx,
        )
        shap_res = explainer(batch_input, max_evals=max_evals)

        shap_true_map = _extract_shap_map(shap_res, true_cls_idx)
        shap_pred_map = _extract_shap_map(shap_res, pred_cls_idx)

        # Nombre y ruta de guardado sin colisión entre modos
        save_filename = (
            f"shap_{weak_class.replace(' ', '_')}_{sample_type}_sample_{offset + i:03d}.png"
        )
        save_file_path = output_dir / save_filename

        plot_shap_contrastive_comparison(
            raw_image=raw_img,
            shap_true=shap_true_map,
            shap_pred=shap_pred_map,
            true_class_name=true_cls_name,
            pred_class_name=pred_cls_name,
            true_confidence=true_conf,
            pred_confidence=pred_conf,
            save_path=save_file_path,
            display_plot=display_plots,
            overlay_shap=overlay_shap,
            overlay_alpha=overlay_alpha,
        )

        sample_info = {
            "sample_index": int(idx),
            "image_path": str(img_path),
            "true_class": true_cls_name,
            "predicted_class": pred_cls_name,
            "true_class_confidence": round(true_conf, 4),
            "predicted_class_confidence": round(pred_conf, 4),
            "shap_plot_path": str(save_file_path),
            "true_class_idx": true_cls_idx,
            "predicted_class_idx": pred_cls_idx,
            "sample_type": sample_type,
        }
        samples_metadata.append(sample_info)

    summary_result: dict[str, Any] = {
        "weak_class": weak_class,
        "sample_type": sample_type,
        "selected_samples": len(selected_idxs),
        "random_seed": random_seed,
        "offset": offset,
        "max_evals": max_evals,
        "overlay_shap": overlay_shap,
        "overlay_alpha": overlay_alpha,
        "backbone_name": backbone_name,
        "output_dir": str(output_dir),
        "samples": samples_metadata,
    }

    # Guardar metadata JSON específica del tipo de muestra
    json_path = output_dir / f"shap_summary_{weak_class.replace(' ', '_')}_{sample_type}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=4, ensure_ascii=False)

    logger.info("Análisis SHAP (%s) completado. Resultados guardados en: %s", sample_type, json_path)
    return summary_result
