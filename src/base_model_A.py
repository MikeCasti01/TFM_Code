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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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


from src.augmentations import AugmentationConfig, augment_image, visualize_augmentations

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
    backbone_name: str = "ResNet152",
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

    # Aplicar fracción de entrenamiento (muestreo estratificado por semilla)
    if train_fraction < 1.0:
        df, _ = train_test_split(
            df,
            train_size=train_fraction,
            stratify=df[label_col],
            random_state=seed,
        )
        df = df.reset_index(drop=True)

    # Determinar columna de rutas
    if "Absolute Path" in df.columns:
        paths = df["Absolute Path"].astype(str).tolist()
    elif dataset_path is not None:
        paths = [str(Path(dataset_path) / p) for p in df["Image Path"]]
    else:
        raise ValueError(
            "El DataFrame debe contener 'Absolute Path' o se debe proveer dataset_path."
        )

    labels = df[label_col].astype(np.int32).tolist()

    # Seleccionar preprocesamiento según el backbone
    if backbone_name == "ResNet152":
        from tensorflow.keras.applications.resnet import preprocess_input
    elif backbone_name == "MobileNetV3Large":
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    elif backbone_name == "EfficientNetV2S":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    else:
        raise ValueError(f"Backbone no soportado: {backbone_name}")

    # Crear dataset de base
    indices = list(range(len(paths)))
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, indices))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, len(paths)), seed=seed)

    # --- Paso 1: Cargar y redimensionar sin normalizar (valores en [0, 255]) ---
    def _load_and_resize(path: tf.Tensor, label: tf.Tensor, idx: tf.Tensor):
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, target_size)
        return tf.cast(img, tf.float32), label, idx

    dataset = dataset.map(
        _load_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # --- Paso 2: Aumento de datos sobre [0, 255] (ANTES de la normalización) ---
    if augment:
        def _augment_wrapper(img: tf.Tensor, label: tf.Tensor, idx: tf.Tensor):
            aug_img, _ = augment_image(
                img,
                label,
                config=aug_config,
                master_seed=seed,
                sample_index=idx
            )
            return aug_img, label, idx
        dataset = dataset.map(_augment_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # --- Paso 3: Normalización ResNet (preprocess_input) ---
    def _apply_normalization(img: tf.Tensor, label: tf.Tensor, idx: tf.Tensor):
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
    fine_tune_at: str | int | None = None,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    backbone_name: str = "ResNet152",
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
        learning_rate (float, optional): Tasa de aprendizaje inicial para el optimizador.
            Por defecto es 1e-4.
        fine_tune_at (str, optional): capa a partir del cual se descongelarán
            las capas del extractor de características para el ajuste fino.            
        optimizer_name (str, optional): Nombre del optimizador a usar.
            'adamw', o 'adam'.
        weight_decay (float, optional): Tasa de decaimiento de peso para optimizadores
            que lo soportan (AdamW). Por defecto es 1e-4.

    Returns:
        keras.Model: Modelo de Keras compilado y listo para entrenar.
    """
    # Cargar el extractor de características preentrenado sin la cabeza de clasificación
    if backbone_name == "ResNet152":
        base_model = keras.applications.ResNet152(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
    elif backbone_name == "MobileNetV3Large":
        base_model = keras.applications.MobileNetV3Large(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
    elif backbone_name == "EfficientNetV2S":
        base_model = keras.applications.EfficientNetV2S(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Backbone no soportado: {backbone_name}")

    if fine_tune_at is None:
        # Transferencia de aprendizaje básica: congelar todo el extractor
        base_model.trainable = False
    else:
        # Ajuste fino: descongelar capas superiores a partir de fine_tune_at
        base_model.trainable = True
        
        # Si se recibe un string, buscar la capa por nombre y obtener su índice
        if isinstance(fine_tune_at, str):
            layer_index = None
            for idx, layer in enumerate(base_model.layers):
                if layer.name == fine_tune_at:
                    layer_index = idx
                    break
            if layer_index is None:
                raise ValueError(f"La capa '{fine_tune_at}' no se encontró en el backbone.")
            fine_tune_at_idx = layer_index
        else:
            fine_tune_at_idx = fine_tune_at

        for layer in base_model.layers[:fine_tune_at_idx]:
            layer.trainable = False

    # Definir la arquitectura usando la API funcional de Keras
    inputs = keras.Input(shape=input_shape)

    # Mantenemos training=False incluso durante el fine-tuning
    # para que BatchNormalization no actualice sus estadísticas móviles (buenas prácticas).
    bn_training_mode = False
    x = base_model(inputs, training=bn_training_mode)

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
        loss=keras.losses.SparseCategoricalCrossentropy(),
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
    initial_epoch: int = 0,
) -> keras.callbacks.History:
    """Entrena el modelo utilizando los datasets proporcionados y callbacks configurados.

    Diseñado para ser agnóstico a la etapa de entrenamiento: puede usarse tanto
    para la Etapa 1 (cabeza de clasificación) como para la Etapa 2 (ajuste fino)
    sin ninguna lógica interna específica. El notebook es responsable de orquestar
    las etapas, recompilar el modelo con la tasa de aprendizaje correcta y
    proporcionar los callbacks adecuados para cada etapa.

    Args:
        model (keras.Model): Modelo compilado de Keras.
        train_dataset (tf.data.Dataset): Dataset para el entrenamiento.
        val_dataset (tf.data.Dataset): Dataset para la validación durante el entrenamiento.
        epochs (int, optional): Número máximo de épocas totales. Por defecto es 20.
        callbacks (list, optional): Lista de callbacks personalizados de Keras. Si es None,
            se configuran callbacks por defecto (EarlyStopping y ReduceLROnPlateau).
        class_weight (dict[int, float], optional): Diccionario de pesos por clase para
            compensar el desbalanceo. Si es None, no se aplica ponderación.
        initial_epoch (int, optional): Época desde la que se comienza el entrenamiento.
            Útil para continuar el historial entre etapas (p. ej., Stage 2 comienza
            desde ``STAGE1_EPOCHS`` para mantener el historial acumulado).
            Por defecto es 0.

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
        initial_epoch=initial_epoch,
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
    y_pred_probs = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)
    
    if len(y_pred) != len(y_true):
        raise ValueError(f"Mismatch in evaluation counts: {len(y_pred)} vs {len(y_true)}. Check shuffle=False in test_dataset.")

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

    try:
        auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float('nan')

    metrics = {
        "Accuracy": accuracy,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1-Score (Macro)": f1_macro,
        "Precision (Weighted)": precision_weighted,
        "Recall (Weighted)": recall_weighted,
        "F1-Score (Weighted)": f1_weighted,
        "AUC (Macro)": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
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


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
) -> None:
    """Calcula y grafica la curva ROC (One-vs-Rest) para las clases del dataset.

    Args:
        y_true (np.ndarray): Etiquetas reales.
        y_pred_probs (np.ndarray): Probabilidades predichas por el modelo.
        class_names (list[str]): Nombres de las clases.
        save_path (str | Path, optional): Ruta donde guardar el gráfico.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    _apply_plot_style()
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    if y_true_bin.shape[1] == 1:
        # Binario
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC Multi-clase (One-vs-Rest)')
    # Para muchas clases, la leyenda puede ser grande. Ocultarla o colocarla afuera si es muy larga.
    if num_classes <= 20:
        plt.legend(loc="lower right", fontsize=8)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Curva ROC guardada en: {save_path}")
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


# ---------------------------------------------------------------------------
# Validación cruzada K-Fold estratificada
# ---------------------------------------------------------------------------

def compute_kfold_splits(
    df: pd.DataFrame,
    level: str = "fine",
    n_splits: int = 5,
    seed: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Genera los índices de entrenamiento y validación para K-Fold estratificado.

    Utiliza ``StratifiedKFold`` para dividir el DataFrame en ``n_splits`` pliegues
    (folds) conservando las proporciones de clase de cada split. Cada fold produce
    un subconjunto de entrenamiento y uno de validación completamente independientes.

    Args:
        df (pd.DataFrame): DataFrame con los metadatos de las imágenes. Debe
            contener la columna de etiquetas correspondiente al nivel taxonómico
            indicado ('Fine ID', 'Coarse ID' o 'Macro ID').
        level (str, optional): Nivel taxonómico de clasificación. Acepta
            'macro', 'coarse' o 'fine'. Por defecto es 'fine'.
        n_splits (int, optional): Número de pliegues. Por defecto es 5.
        seed (int, optional): Semilla aleatoria para garantizar la reproducibilidad
            de la partición. Por defecto es None.

    Returns:
        list[tuple[pd.DataFrame, pd.DataFrame]]: Lista de ``n_splits`` tuplas.
            Cada tupla contiene ``(train_df, val_df)`` con los metadatos
            correspondientes a ese pliegue, reiniciando el índice.

    Raises:
        ValueError: Si ``level`` no es uno de 'macro', 'coarse' o 'fine'.
    """
    level = level.lower()
    label_columns = {"macro": "Macro ID", "coarse": "Coarse ID", "fine": "Fine ID"}

    if level not in label_columns:
        raise ValueError("level debe ser uno de: 'macro', 'coarse' o 'fine'.")

    label_col = label_columns[level]
    labels = df[label_col].astype(np.int32).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []

    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        splits.append((train_df, val_df))

    return splits


def aggregate_kfold_metrics(
    fold_metrics: list[dict],
) -> dict[str, dict[str, float]]:
    """Agrega las métricas de todos los pliegues calculando media y desviación estándar.

    Itera sobre la lista de diccionarios de métricas producidos por ``evaluate_model``
    (uno por pliegue) y calcula la media y la desviación estándar de cada métrica
    escalar. Los campos no escalares (p. ej., 'y_true', 'y_pred') se ignoran.

    Args:
        fold_metrics (list[dict]): Lista de diccionarios de métricas, uno por pliegue.
            Cada diccionario debe contener claves escalares con valores float
            compatibles con NumPy.

    Returns:
        dict[str, dict[str, float]]: Diccionario anidado con la siguiente estructura::

            {
                "Accuracy": {"mean": 0.92, "std": 0.01},
                "F1-Score (Macro)": {"mean": 0.90, "std": 0.02},
                ...
            }
    """
    # Identificar claves escalares (excluir arrays como y_true/y_pred)
    scalar_keys = [
        k for k, v in fold_metrics[0].items()
        if not isinstance(v, np.ndarray)
    ]

    aggregated: dict[str, dict[str, float]] = {}

    for key in scalar_keys:
        values = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return aggregated


def print_kfold_report(
    fold_metrics: list[dict],
    aggregated: dict[str, dict[str, float]],
) -> None:
    """Imprime un reporte formateado con los resultados de la validación cruzada.

    Muestra las métricas de cada pliegue individualmente y, al final, el resumen
    estadístico (media ± desviación estándar) para las métricas principales.

    Args:
        fold_metrics (list[dict]): Lista de diccionarios de métricas por pliegue,
            tal como devuelve ``evaluate_model``.
        aggregated (dict[str, dict[str, float]]): Diccionario de métricas agregadas,
            tal como devuelve ``aggregate_kfold_metrics``.
    """
    # Métricas principales a mostrar en el resumen final
    summary_keys = [
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1-Score (Macro)",
        "Precision (Weighted)",
        "Recall (Weighted)",
        "F1-Score (Weighted)",
        "AUC (Macro)",
    ]

    # Imprimir resultados por pliegue
    for fold_idx, metrics in enumerate(fold_metrics, start=1):
        print(f"\nFold {fold_idx}")
        print(f"  Accuracy             : {metrics.get('Accuracy', float('nan')):.4f}")
        print(f"  Precision (Macro)    : {metrics.get('Precision (Macro)', float('nan')):.4f}")
        print(f"  Recall (Macro)       : {metrics.get('Recall (Macro)', float('nan')):.4f}")
        print(f"  F1-Score (Macro)     : {metrics.get('F1-Score (Macro)', float('nan')):.4f}")
        print(f"  Precision (Weighted) : {metrics.get('Precision (Weighted)', float('nan')):.4f}")
        print(f"  Recall (Weighted)    : {metrics.get('Recall (Weighted)', float('nan')):.4f}")
        print(f"  F1-Score (Weighted)  : {metrics.get('F1-Score (Weighted)', float('nan')):.4f}")

    # Imprimir resumen de validación cruzada
    print("\n" + "=" * 56)
    print("Resultados de Validación Cruzada")
    print("=" * 56)

    for key in summary_keys:
        if key in aggregated:
            mean = aggregated[key]["mean"]
            std = aggregated[key]["std"]
            print(f"\n{key}")
            print(f"  Media : {mean:.4f}")
            print(f"  Desv. : {std:.4f}")


def print_evaluation_report(metrics: dict) -> None:
    """Imprime un reporte legible con todas las métricas de evaluación del modelo.

    Muestra en consola las métricas escalares retornadas por ``evaluate_model``,
    incluyendo exactitud, precisión, recall, F1 y AUC macro-OvR. Los arrays no
    escalares (``y_true``, ``y_pred``, ``y_pred_probs``) se omiten
    automáticamente. Diseñada para ser invocada tanto en el flujo sin K-Fold como
    en cualquier flujo de evaluación individual, garantizando que ``AUC (Macro)``
    siempre forme parte del reporte impreso.

    Args:
        metrics (dict): Diccionario retornado por :func:`evaluate_model`. Se
            esperan las claves ``'Accuracy'``, ``'Precision (Macro)'``,
            ``'Recall (Macro)'``, ``'F1-Score (Macro)'``, ``'Precision (Weighted)'``,
            ``'Recall (Weighted)'``, ``'F1-Score (Weighted)'`` y ``'AUC (Macro)'``.
            Las claves ausentes se muestran como ``nan``.
    """
    print("\n" + "=" * 56)
    print("Reporte de Evaluación del Modelo")
    print("=" * 56)
    print(f"Exactitud (Accuracy)        : {metrics.get('Accuracy', float('nan')):.4f}")
    print(f"Precisión (Macro)           : {metrics.get('Precision (Macro)', float('nan')):.4f}")
    print(f"Sensibilidad/Recall (Macro) : {metrics.get('Recall (Macro)', float('nan')):.4f}")
    print(f"F1-Score (Macro)            : {metrics.get('F1-Score (Macro)', float('nan')):.4f}")
    print(f"Precisión (Weighted)        : {metrics.get('Precision (Weighted)', float('nan')):.4f}")
    print(f"Sensibilidad/Recall (Wgt.)  : {metrics.get('Recall (Weighted)', float('nan')):.4f}")
    print(f"F1-Score (Weighted)         : {metrics.get('F1-Score (Weighted)', float('nan')):.4f}")
    print(f"AUC (Macro OvR)             : {metrics.get('AUC (Macro)', float('nan')):.4f}")
    print("=" * 56)


# ---------------------------------------------------------------------------
# Curvas ROC Globales — Micro-average y Macro-average
# ---------------------------------------------------------------------------

def plot_roc_micro_average(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    display_plot: bool = True,
) -> float:
    """Calcula y grafica la curva ROC micro-average para clasificación multi-clase.

    La estrategia micro-average agrega las contribuciones de todas las clases
    concatenando los vectores OvR binarios antes de calcular la curva ROC única.
    Esta representación pondera igualmente cada par (muestra, clase), siendo
    especialmente útil cuando las clases están desbalanceadas.

    El valor AUC micro-average se incluye en la leyenda de la figura. La figura
    generada tiene calidad de publicación (DPI 150, ejes etiquetados, cuadrícula
    sutil).

    Compatibilidad: scikit-learn >= 1.0, Python 3.12, TF 2.20, Keras 3.13.

    Args:
        y_true (np.ndarray): Array 1-D de etiquetas enteras reales con valores en
            ``[0, num_classes - 1]``. Shape: ``(n_samples,)``.
        y_pred_probs (np.ndarray): Matriz de probabilidades predichas por el modelo.
            Shape: ``(n_samples, num_classes)``.
        class_names (list[str]): Lista ordenada de nombres de clases. Su longitud
            determina ``num_classes``.
        save_path (str | Path, optional): Ruta completa donde se guardará la figura
            (p. ej. ``CACHE_PATH / 'XAI' / 'roc_micro.png'``). Si es ``None``, la
            figura no se guarda en disco. Por defecto es ``None``.
        display_plot (bool, optional): Si es ``True``, llama a ``plt.show()`` para
            mostrar la figura. Por defecto es ``True``.

    Returns:
        float: Valor del AUC micro-average.

    Raises:
        ValueError: Si ``y_pred_probs.shape[1]`` no coincide con ``len(class_names)``.
    """
    import logging
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    logger = logging.getLogger(__name__)

    num_classes = len(class_names)
    if y_pred_probs.shape[1] != num_classes:
        raise ValueError(
            f"y_pred_probs tiene {y_pred_probs.shape[1]} columnas "
            f"pero class_names tiene {num_classes} elementos."
        )

    # --- Estilo de publicación (independiente de _apply_plot_style) ---
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (8, 7),
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "font.size": 11,
        "axes.grid": True,
    })

    # Binarizar etiquetas: shape (n_samples, num_classes)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_true_bin.shape[1] == 1:
        # Caso binario: label_binarize produce (n, 1); expandir a (n, 2)
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Micro-average: tratar todas las columnas OvR como un único problema binario
    fpr_micro, tpr_micro, _ = roc_curve(
        y_true_bin.ravel(),
        y_pred_probs.ravel(),
    )
    auc_micro = auc(fpr_micro, tpr_micro)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        fpr_micro,
        tpr_micro,
        color="#E63946",
        lw=2.5,
        label=f"Micro-average ROC (AUC = {auc_micro:.4f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        color="#6C757D",
        lw=1.5,
        linestyle="--",
        label="Clasificador aleatorio (AUC = 0.5000)",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", labelpad=8)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", labelpad=8)
    ax.set_title(
        f"Curva ROC Global — Micro-average\n"
        f"({num_classes} clases, One-vs-Rest)",
        pad=12,
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Curva ROC micro-average guardada en: %s", save_path)

    if display_plot:
        plt.show()
    else:
        plt.close(fig)

    return float(auc_micro)


def plot_roc_macro_average(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    display_plot: bool = True,
) -> float:
    """Calcula y grafica la curva ROC macro-average para clasificación multi-clase.

    La estrategia macro-average calcula la curva ROC de cada clase de forma
    independiente (OvR), interpola todas las curvas en un eje FPR común de 1000
    puntos y promedia las TPR resultantes. Cada clase tiene el mismo peso
    independientemente de su soporte, lo que la hace sensible a clases con pocos
    ejemplos.

    El valor AUC macro-average se incluye en la leyenda de la figura. La figura
    generada tiene calidad de publicación (DPI 150, ejes etiquetados, cuadrícula
    sutil).

    Compatibilidad: scikit-learn >= 1.0, Python 3.12, TF 2.20, Keras 3.13.

    Args:
        y_true (np.ndarray): Array 1-D de etiquetas enteras reales con valores en
            ``[0, num_classes - 1]``. Shape: ``(n_samples,)``.
        y_pred_probs (np.ndarray): Matriz de probabilidades predichas por el modelo.
            Shape: ``(n_samples, num_classes)``.
        class_names (list[str]): Lista ordenada de nombres de clases. Su longitud
            determina ``num_classes``.
        save_path (str | Path, optional): Ruta completa donde se guardará la figura
            (p. ej. ``CACHE_PATH / 'XAI' / 'roc_macro.png'``). Si es ``None``, la
            figura no se guarda en disco. Por defecto es ``None``.
        display_plot (bool, optional): Si es ``True``, llama a ``plt.show()`` para
            mostrar la figura. Por defecto es ``True``.

    Returns:
        float: Valor del AUC macro-average calculado sobre las TPR interpoladas.

    Raises:
        ValueError: Si ``y_pred_probs.shape[1]`` no coincide con ``len(class_names)``.
    """
    import logging
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    logger = logging.getLogger(__name__)

    num_classes = len(class_names)
    if y_pred_probs.shape[1] != num_classes:
        raise ValueError(
            f"y_pred_probs tiene {y_pred_probs.shape[1]} columnas "
            f"pero class_names tiene {num_classes} elementos."
        )

    # --- Estilo de publicación (independiente de _apply_plot_style) ---
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (8, 7),
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "font.size": 11,
        "axes.grid": True,
    })

    # Binarizar etiquetas
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Eje FPR común para la interpolación de todas las curvas OvR
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    tpr_per_class = np.zeros((num_classes, len(fpr_grid)))
    auc_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        auc_per_class[i] = auc(fpr_i, tpr_i)
        # Interpolar TPR en el eje FPR común y anclar en (0, 0)
        tpr_per_class[i] = np.interp(fpr_grid, fpr_i, tpr_i)
        tpr_per_class[i][0] = 0.0

    # Macro-average: media aritmética de las TPR interpoladas y anclar en (1, 1)
    mean_tpr = np.mean(tpr_per_class, axis=0)
    mean_tpr[-1] = 1.0
    auc_macro = auc(fpr_grid, mean_tpr)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        fpr_grid,
        mean_tpr,
        color="#2A9D8F",
        lw=2.5,
        label=f"Macro-average ROC (AUC = {auc_macro:.4f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        color="#6C757D",
        lw=1.5,
        linestyle="--",
        label="Clasificador aleatorio (AUC = 0.5000)",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", labelpad=8)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", labelpad=8)
    ax.set_title(
        f"Curva ROC Global — Macro-average\n"
        f"({num_classes} clases, OvR, interpolación en {len(fpr_grid)} puntos)",
        pad=12,
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Curva ROC macro-average guardada en: %s", save_path)

    if display_plot:
        plt.show()
    else:
        plt.close(fig)

    return float(auc_macro)
