"""Módulo de análisis de rendimiento por clase para la preparación del análisis XAI.

Este módulo identifica las clases débiles del modelo entrenado a partir de un
único paso de inferencia. Está diseñado para ser importado desde el notebook
XAI (``xai-analysis.ipynb``) sin necesidad de reentrenar ni relanzar
``model.predict`` más de una vez.

Workflow previsto
-----------------
::

    model = keras.models.load_model(checkpoint_path)
    cm, y_true, y_pred, y_proba = compute_confusion_matrix(model, test_ds, class_names)

    metrics_df = compute_per_class_metrics(y_true, y_pred, class_names)
    weak = select_weak_classes(metrics_df, n_classes=15, metric="recall")
    confusions = get_top_confusions(cm, class_names, weak, top_k=3)
    misclassified = get_misclassified_samples(y_true, y_pred, test_metadata, weak)

Compatibilidad
--------------
- Python  3.12
- TensorFlow 2.20
- Keras 3.13
- scikit-learn >= 1.0
- pandas >= 1.5
- numpy >= 1.24
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from tensorflow import keras

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------

_LEVEL_LABEL_MAP: dict[str, str] = {
    "macro": "Macro ID",
    "coarse": "Coarse ID",
    "fine": "Fine ID",
}

_LEVEL_NAME_MAP: dict[str, str] = {
    "macro": "Macro",
    "coarse": "Coarse",
    "fine": "Fine",
}

_SUPPORTED_METRICS: frozenset[str] = frozenset(
    {
        "recall",
        "precision",
        "f1",
        "f2",
        "mcc",
        "accuracy",
        "support",
        "train support",
        "train_support",
        "train images",
        "train_images",
    }
)

_METRIC_COLUMN_MAP: dict[str, str] = {
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "f2": "F2",
    "mcc": "MCC",
    "accuracy": "Accuracy",
    "support": "Support",
    "train support": "Train Support",
    "train_support": "Train Support",
    "train images": "Train Support",
    "train_images": "Train Support",
}


# ---------------------------------------------------------------------------
# Función principal de inferencia
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ejecuta inferencia sobre el conjunto de prueba y retorna la matriz de confusión.

    Esta función es el **único punto de inferencia** del módulo. Llama a
    ``model.predict`` una sola vez y devuelve todos los arrays necesarios para
    que las funciones siguientes operen sin relanzar la inferencia.

    El ``test_dataset`` debe haberse creado con ``shuffle=False`` para garantizar
    que el orden de las predicciones coincida con el de las etiquetas reales.

    Args:
        model (keras.Model): Modelo Keras entrenado y cargado desde checkpoint.
        test_dataset (tf.data.Dataset): Dataset de prueba sin mezcla (``shuffle=False``).
            Cada elemento debe ser una tupla ``(imagen_batch, etiqueta_batch)``.
        class_names (list[str]): Lista ordenada de nombres de clases. Su longitud
            debe coincidir con el número de neuronas de salida del modelo.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tupla de cuatro arrays:

        - ``confusion_matrix`` (np.ndarray): Matriz de confusión de forma
          ``(num_classes, num_classes)``. Elemento ``[i, j]`` es el número de
          muestras de la clase ``i`` clasificadas como clase ``j``.
        - ``y_true`` (np.ndarray): Etiquetas reales, shape ``(n_samples,)``.
        - ``y_pred`` (np.ndarray): Predicciones argmax, shape ``(n_samples,)``.
        - ``y_pred_proba`` (np.ndarray): Probabilidades predichas por el modelo,
          shape ``(n_samples, num_classes)``.

    Raises:
        ValueError: Si las longitudes de ``y_pred`` e ``y_true`` no coinciden,
            lo que indica que ``test_dataset`` fue creado con ``shuffle=True``.
        ValueError: Si ``y_pred_proba.shape[1]`` no coincide con ``len(class_names)``.
    """
    logger.info("Ejecutando inferencia sobre test_dataset (una sola vez)...")
    y_pred_proba: np.ndarray = model.predict(test_dataset, verbose=0)

    num_classes = len(class_names)
    if y_pred_proba.shape[1] != num_classes:
        raise ValueError(
            f"El modelo produce {y_pred_proba.shape[1]} clases de salida, "
            f"pero class_names contiene {num_classes} clases."
        )

    y_pred: np.ndarray = np.argmax(y_pred_proba, axis=1)

    # Extraer etiquetas reales concatenando todos los batches
    y_true: np.ndarray = np.concatenate(
        [labels.numpy() for _, labels in test_dataset],
        axis=0,
    )

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Desajuste de tamaños: y_pred tiene {len(y_pred)} muestras pero "
            f"y_true tiene {len(y_true)}. Verifique que test_dataset se creó "
            "con shuffle=False."
        )

    cm: np.ndarray = sk_confusion_matrix(y_true, y_pred)

    logger.info(
        "Inferencia completada. Muestras: %d | Clases: %d | Forma CM: %s",
        len(y_true),
        num_classes,
        cm.shape,
    )
    return cm, y_true, y_pred, y_pred_proba


# ---------------------------------------------------------------------------
# Métricas por clase
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    train_metadata: Optional[pd.DataFrame] = None,
    taxonomic_level: str = "coarse",
) -> pd.DataFrame:
    """Calcula métricas de clasificación para cada clase individualmente.

    Utiliza ``precision_recall_fscore_support`` con ``average=None`` para obtener
    métricas por clase, ``fbeta_score`` con beta=2 para F2-score por clase,
    one-vs-rest binary ``matthews_corrcoef`` para MCC por clase, y ``accuracy_score``
    global como referencia. Opcionalmente cuenta el soporte de entrenamiento desde ``train_metadata``.

    El DataFrame resultante está ordenado de forma **ascendente por Recall** para que las
    clases más débiles aparezcan primero, facilitando su identificación directa.

    Args:
        y_true (np.ndarray): Etiquetas reales, shape ``(n_samples,)``.
        y_pred (np.ndarray): Predicciones argmax del modelo, shape ``(n_samples,)``.
        class_names (list[str]): Lista ordenada de nombres de clases.
        train_metadata (Optional[pd.DataFrame]): DataFrame opcional del conjunto de entrenamiento
            para computar el número de imágenes de entrenamiento por clase.
        taxonomic_level (str): Nivel taxonómico ('coarse', 'fine', 'macro') para extraer la columna
            de clase en ``train_metadata``. Por defecto es 'coarse'.

    Returns:
        pd.DataFrame: DataFrame con las siguientes columnas:

        - ``Class``: Nombre de la clase.
        - ``Class ID``: Índice entero de la clase.
        - ``Precision``: Precisión (float).
        - ``Recall``: Exhaustividad / recall (float).
        - ``F1``: Medida F1 (float).
        - ``F2``: Medida F2 (beta=2) (float).
        - ``MCC``: Coeficiente de Correlación de Matthews One-vs-Rest (float).
        - ``Support``: Número de muestras reales de la clase en test (int).
        - ``Train Support``: Número de muestras reales de la clase en entrenamiento (int).
        - ``Accuracy``: Exactitud global del modelo (misma para todas las filas).

        Ordenado ascendentemente por ``Recall``.

    Raises:
        ValueError: Si ``len(class_names)`` no coincide con el número de clases
            únicas inferido de ``y_true``.
    """
    num_classes = len(class_names)
    unique_labels = np.unique(y_true)

    if len(unique_labels) > num_classes:
        raise ValueError(
            f"y_true contiene {len(unique_labels)} clases únicas pero "
            f"class_names tiene {num_classes} entradas."
        )

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    f2_arr = fbeta_score(
        y_true,
        y_pred,
        beta=2.0,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    # Coeficiente MCC por clase (One-vs-Rest)
    mcc_arr = np.zeros(num_classes, dtype=np.float64)
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(np.int32)
        y_pred_binary = (y_pred == i).astype(np.int32)
        if len(np.unique(y_true_binary)) <= 1 and len(np.unique(y_pred_binary)) <= 1:
            mcc_val = 0.0
        else:
            try:
                mcc_val = float(matthews_corrcoef(y_true_binary, y_pred_binary))
            except Exception:
                mcc_val = 0.0
            if np.isnan(mcc_val):
                mcc_val = 0.0
        mcc_arr[i] = mcc_val

    # Recuento de imágenes de entrenamiento por clase si train_metadata está disponible
    train_support_arr = np.zeros(num_classes, dtype=np.int64)
    if train_metadata is not None:
        level_lower = taxonomic_level.lower()
        name_col = _LEVEL_NAME_MAP.get(level_lower, "Coarse")
        if name_col in train_metadata.columns:
            counts = train_metadata[name_col].value_counts().to_dict()
            train_support_arr = np.array(
                [int(counts.get(name, 0)) for name in class_names], dtype=np.int64
            )
        else:
            logger.warning(
                "La columna '%s' no está en train_metadata. Columnas: %s.",
                name_col,
                list(train_metadata.columns),
            )

    global_accuracy: float = accuracy_score(y_true, y_pred)

    df = pd.DataFrame(
        {
            "Class": class_names,
            "Class ID": list(range(num_classes)),
            "Precision": precision_arr.astype(np.float64),
            "Recall": recall_arr.astype(np.float64),
            "F1": f1_arr.astype(np.float64),
            "F2": f2_arr.astype(np.float64),
            "MCC": mcc_arr.astype(np.float64),
            "Support": support_arr.astype(np.int64),
            "Train Support": train_support_arr.astype(np.int64),
            "Accuracy": global_accuracy,
        }
    )

    df = df.sort_values("Recall", ascending=True).reset_index(drop=True)

    logger.info(
        "Métricas por clase calculadas. Clases con Recall = 0: %d",
        int((df["Recall"] == 0.0).sum()),
    )
    return df


# ---------------------------------------------------------------------------
# Selección de clases débiles
# ---------------------------------------------------------------------------

def select_weak_classes(
    per_class_metrics: pd.DataFrame,
    n_classes: int = 15,
    metric: str = "recall",
) -> list[str]:
    """Retorna las N clases con peor rendimiento según la métrica especificada.

    Ordena el DataFrame de métricas de forma ascendente según ``metric`` y
    devuelve los nombres de las primeras ``n_classes`` filas.

    Args:
        per_class_metrics (pd.DataFrame): DataFrame retornado por
            :func:`compute_per_class_metrics`. Debe contener la columna ``Class``
            y la columna indicada en ``metric``.
        n_classes (int, optional): Número de clases débiles a seleccionar.
            Por defecto es 15.
        metric (str, optional): Nombre de la métrica a usar como criterio de
            debilidad. Valores permitidos: ``'recall'``, ``'precision'``,
            ``'f1'``, ``'f2'``, ``'mcc'``, ``'accuracy'``, ``'support'``, ``'train support'``.
            Por defecto es ``'recall'``.

    Returns:
        list[str]: Lista de ``n_classes`` nombres de clases ordenados de menor
            a mayor valor de la métrica especificada.

    Raises:
        ValueError: Si ``metric`` no es uno de los valores permitidos.
        ValueError: Si ``n_classes`` supera el número total de clases disponibles.
        KeyError: Si la columna correspondiente a ``metric`` no existe en ``per_class_metrics``.
    """
    metric_lower = metric.lower().strip()
    if metric_lower not in _SUPPORTED_METRICS:
        raise ValueError(
            f"metric='{metric}' no está soportada. "
            f"Opciones válidas: {sorted(_SUPPORTED_METRICS)}."
        )

    total_classes = len(per_class_metrics)
    if n_classes > total_classes:
        raise ValueError(
            f"n_classes={n_classes} supera el número de clases disponibles "
            f"({total_classes})."
        )

    col_name = _METRIC_COLUMN_MAP.get(metric_lower, metric_lower.capitalize())
    if col_name not in per_class_metrics.columns:
        raise KeyError(
            f"La columna '{col_name}' no existe en per_class_metrics. "
            f"Columnas disponibles: {list(per_class_metrics.columns)}."
        )

    sorted_df = per_class_metrics.sort_values(col_name, ascending=True)
    weak_classes: list[str] = sorted_df["Class"].iloc[:n_classes].tolist()

    logger.info(
        "Clases débiles seleccionadas por %s (N=%d): %s",
        col_name,
        n_classes,
        weak_classes,
    )
    return weak_classes


# ---------------------------------------------------------------------------
# Top confusiones por clase débil
# ---------------------------------------------------------------------------

def get_top_confusions(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    weak_classes: list[str],
    top_k: int = 3,
) -> dict[str, list[tuple[str, int]]]:
    """Identifica las clases con las que se confunde más cada clase débil.

    Para cada clase débil, extrae la fila correspondiente de la matriz de
    confusión, excluye la diagonal (aciertos), y retorna las ``top_k`` clases
    con mayor número de confusiones.

    Args:
        confusion_matrix (np.ndarray): Matriz de confusión de forma
            ``(num_classes, num_classes)`` retornada por :func:`compute_confusion_matrix`.
        class_names (list[str]): Lista ordenada de nombres de clases.
        weak_classes (list[str]): Lista de nombres de clases débiles, como la
            retornada por :func:`select_weak_classes`.
        top_k (int, optional): Número máximo de confusiones a reportar por clase.
            Por defecto es 3.

    Returns:
        dict[str, list[tuple[str, int]]]: Diccionario donde cada clave es el
            nombre de una clase débil y el valor es una lista de hasta ``top_k``
            tuplas ``(nombre_clase_confundida, recuento)``, ordenadas de mayor a
            menor recuento.

            Ejemplo::

                {
                    "Apple": [("Pear", 12), ("Mango", 5), ("Banana", 3)],
                    "Milk": [("Yogurt", 8), ("Cream", 2)],
                }

    Raises:
        ValueError: Si algún nombre en ``weak_classes`` no está en ``class_names``.
        ValueError: Si ``confusion_matrix.shape`` no es ``(n, n)`` donde
            ``n == len(class_names)``.
    """
    num_classes = len(class_names)
    if confusion_matrix.shape != (num_classes, num_classes):
        raise ValueError(
            f"confusion_matrix.shape={confusion_matrix.shape} no coincide con "
            f"({num_classes}, {num_classes}) clases esperadas."
        )

    name_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
    missing = [c for c in weak_classes if c not in name_to_idx]
    if missing:
        raise ValueError(
            f"Las siguientes clases no están en class_names: {missing}."
        )

    top_confusions: dict[str, list[tuple[str, int]]] = {}

    for class_name in weak_classes:
        class_idx = name_to_idx[class_name]
        row = confusion_matrix[class_idx].copy().astype(np.int64)

        # Excluir la diagonal (aciertos)
        row[class_idx] = 0

        # Obtener los índices de las top_k confusiones descendentemente
        top_indices = np.argsort(row)[::-1][:top_k]
        top_pairs: list[tuple[str, int]] = [
            (class_names[idx], int(row[idx]))
            for idx in top_indices
            if row[idx] > 0
        ]
        top_confusions[class_name] = top_pairs

        logger.debug(
            "Clase '%s': confusiones top-%d -> %s",
            class_name,
            top_k,
            top_pairs,
        )

    logger.info(
        "Top-%d confusiones calculadas para %d clases débiles.",
        top_k,
        len(weak_classes),
    )
    return top_confusions


# ---------------------------------------------------------------------------
# Extracción de muestras mal clasificadas
# ---------------------------------------------------------------------------

def get_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_metadata: pd.DataFrame,
    weak_classes: list[str],
    taxonomic_level: str = "coarse",
) -> dict[str, dict[str, list[str]]]:
    """Retorna las rutas absolutas de las muestras mal clasificadas de cada clase débil.

    Cruza los arrays de predicción con el DataFrame de metadatos del split de
    prueba (cargado desde ``dataset_metadata.parquet``) para recuperar las rutas
    absolutas de las imágenes. No relanza inferencia; opera exclusivamente sobre
    ``y_true`` e ``y_pred`` ya computados.

    El DataFrame ``test_metadata`` debe estar en el mismo orden que los arrays
    ``y_true`` e ``y_pred`` (subconjunto del split de prueba, sin mezcla posterior).

    Args:
        y_true (np.ndarray): Etiquetas reales, shape ``(n_samples,)``.
        y_pred (np.ndarray): Predicciones argmax del modelo, shape ``(n_samples,)``.
        test_metadata (pd.DataFrame): DataFrame de metadatos del split de prueba,
            cargado desde ``dataset_metadata.parquet`` filtrado por
            ``Split == 'test'``. Debe contener ``'Absolute Path'``, la columna de
            nombre de clase (p. ej. ``'Coarse'``) y la columna de ID de clase
            (p. ej. ``'Coarse ID'``).
        weak_classes (list[str]): Lista de nombres de clases débiles, como la
            retornada por :func:`select_weak_classes`.
        taxonomic_level (str, optional): Nivel taxonómico de clasificación.
            Acepta ``'macro'``, ``'coarse'`` o ``'fine'``. Por defecto es ``'coarse'``.

    Returns:
        dict[str, dict[str, list[str]]]: Diccionario anidado::

            {
                "true_class_name": {
                    "predicted_class_name": [
                        "/ruta/absoluta/imagen1.jpg",
                        "/ruta/absoluta/imagen2.jpg",
                    ]
                }
            }

        Solo se incluyen pares ``(clase_real, clase_predicha)`` con errores reales.

    Raises:
        ValueError: Si ``taxonomic_level`` no es uno de los valores permitidos.
        ValueError: Si ``len(y_true) != len(test_metadata)``.
        KeyError: Si el DataFrame no contiene las columnas requeridas.
    """
    level_lower = taxonomic_level.lower()
    if level_lower not in _LEVEL_LABEL_MAP:
        raise ValueError(
            f"taxonomic_level='{taxonomic_level}' no está soportado. "
            f"Opciones válidas: {list(_LEVEL_LABEL_MAP.keys())}."
        )

    id_col: str = _LEVEL_LABEL_MAP[level_lower]
    name_col: str = _LEVEL_NAME_MAP[level_lower]

    # Validar columnas requeridas
    required_cols = {"Absolute Path", id_col, name_col}
    missing_cols = required_cols - set(test_metadata.columns)
    if missing_cols:
        raise KeyError(
            f"El DataFrame test_metadata no contiene las columnas: {missing_cols}."
        )

    if len(y_true) != len(test_metadata):
        raise ValueError(
            f"y_true tiene {len(y_true)} muestras pero test_metadata tiene "
            f"{len(test_metadata)} filas. Verifique el alineamiento."
        )

    # Construir mapeo id → nombre de clase para decodificar y_pred
    id_to_name: dict[int, str] = (
        test_metadata[[id_col, name_col]]
        .drop_duplicates()
        .set_index(id_col)[name_col]
        .to_dict()
    )

    # Resetear índice para garantizar indexación posicional correcta
    meta_reset = test_metadata.reset_index(drop=True)

    misclassified: dict[str, dict[str, list[str]]] = {}

    for true_class_name in weak_classes:
        # Máscara: clase real == clase débil Y predicción incorrecta
        class_mask = np.array(
            [id_to_name.get(int(label), "") == true_class_name for label in y_true]
        )
        error_mask = (y_true != y_pred)
        combined_mask = class_mask & error_mask

        if not combined_mask.any():
            logger.debug("Clase '%s': sin errores de clasificación.", true_class_name)
            continue

        class_errors: dict[str, list[str]] = {}
        for idx in np.where(combined_mask)[0]:
            pred_class_name = id_to_name.get(int(y_pred[idx]), f"ID_{y_pred[idx]}")
            abs_path = str(meta_reset.at[idx, "Absolute Path"])
            class_errors.setdefault(pred_class_name, []).append(abs_path)

        misclassified[true_class_name] = class_errors
        logger.debug(
            "Clase '%s': %d errores en %d clases predichas.",
            true_class_name,
            int(combined_mask.sum()),
            len(class_errors),
        )

    logger.info(
        "Muestras mal clasificadas extraídas para %d/%d clases débiles.",
        len(misclassified),
        len(weak_classes),
    )
    return misclassified


# ---------------------------------------------------------------------------
# Selección unificada de muestras para XAI
# ---------------------------------------------------------------------------

def select_xai_sample_indices(
    indices: np.ndarray,
    strategy: str = "first",
    num_samples: int = 1,
    y_pred_proba: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    random_seed: int = 42,
    offset: int = 0,
) -> np.ndarray:
    """Selecciona de manera determinista y unificada un subconjunto de índices de muestras
    para análisis XAI (Grad-CAM, SHAP, etc.).

    Es la fuente única de verdad para la selección de muestras entre distintas técnicas XAI.
    Estrategias soportadas:
    - 'first': Preserva el orden original de los candidatos.
    - 'highest_confidence': Ordena los candidatos descendentemente según la probabilidad
      predicha por el modelo para la clase predicha (y_pred_proba[idx, y_pred[idx]]).
    - 'lowest_confidence': Ordena los candidatos ascendentemente según la probabilidad
      predicha por el modelo para la clase predicha.
    - 'random': Realiza un barajado aleatorio determinista utilizando np.random.default_rng(random_seed).

    Tras determinar el orden de los candidatos, aplica el parámetro `offset` y selecciona
    hasta `num_samples` elementos: `ordered_indices[offset : offset + num_samples]`.

    Args:
        indices (np.ndarray): Array 1D de índices candidatos del dataset original.
        strategy (str): Estrategia de selección ('first', 'highest_confidence', 'lowest_confidence', 'random').
            Por defecto 'first'.
        num_samples (int): Cantidad máxima de muestras a seleccionar. Por defecto 1.
        y_pred_proba (np.ndarray | None): Matriz de probabilidades predichas (N, num_classes).
            Requerido para estrategias basadas en confianza.
        y_pred (np.ndarray | None): Vector de clases predichas (N,).
            Requerido para estrategias basadas en confianza.
        random_seed (int): Semilla para la estrategia 'random'. Por defecto 42.
        offset (int): Posición inicial de selección tras el ordenamiento. Por defecto 0.

    Returns:
        np.ndarray: Array 1D con los índices seleccionados del dataset original.

    Raises:
        ValueError: Si `strategy` no es una estrategia válida.
        ValueError: Si `strategy` requiere `y_pred_proba` o `y_pred` y no son provistos.
        ValueError: Si `offset` es negativo.
    """
    if len(indices) == 0:
        return np.array([], dtype=indices.dtype if hasattr(indices, "dtype") else int)

    if offset < 0:
        raise ValueError(f"offset debe ser >= 0, pero se recibió {offset}.")

    if num_samples <= 0:
        return np.array([], dtype=indices.dtype if hasattr(indices, "dtype") else int)

    strategy_lower = strategy.lower().strip()
    valid_strategies = {"first", "highest_confidence", "lowest_confidence", "random"}
    if strategy_lower not in valid_strategies:
        raise ValueError(
            f"Estrategia de selección '{strategy}' no soportada. "
            f"Opciones válidas: {sorted(valid_strategies)}."
        )

    if strategy_lower in ("highest_confidence", "lowest_confidence"):
        if y_pred_proba is None or y_pred is None:
            raise ValueError(
                f"La estrategia '{strategy}' requiere proporcionar y_pred_proba e y_pred."
            )
        # Confianza de la predicción: probabilidad de la clase predicha por el modelo
        confidences = y_pred_proba[indices, y_pred[indices]]
        if strategy_lower == "highest_confidence":
            # Ordenar descendente (usando stable sort para determinismo total ante empates)
            sorted_order = np.argsort(-confidences, kind="stable")
        else:  # lowest_confidence
            # Ordenar ascendente
            sorted_order = np.argsort(confidences, kind="stable")
        ordered_indices = indices[sorted_order]

    elif strategy_lower == "random":
        rng = np.random.default_rng(random_seed)
        shuffled_indices = indices.copy()
        rng.shuffle(shuffled_indices)
        ordered_indices = shuffled_indices

    elif strategy_lower == "first":
        ordered_indices = indices.copy()

    # Aplicar offset y recortar a num_samples
    selected_indices = ordered_indices[offset : offset + num_samples]
    return selected_indices

