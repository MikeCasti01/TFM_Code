from collections.abc import Mapping
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


def augment_image(image: tf.Tensor, label: int) -> tuple[tf.Tensor, int]:
    """Aplica aumento de datos limitado únicamente a rotaciones y reflejos (flips).

    Esta función realiza aumentación geométrica simple de manera aleatoria:
    volteo horizontal, volteo vertical y rotaciones en ángulos rectos (0, 90, 180, 270).
    Esto es ideal para reconocimiento de productos donde la orientación puede variar.

    Args:
        image (tf.Tensor): Tensor de la imagen preprocesada.
        label (int): Identificador numérico de la clase.

    Returns:
        tuple[tf.Tensor, int]: Tensor de la imagen con aumentación aplicada y la etiqueta.
    """
    # Flips aleatorios (vertical y horizontal)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Rotación aleatoria utilizando múltiplos de 90 grados
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    return image, label


def get_data_generator(
    df: pd.DataFrame,
    dataset_path: str | Path | None = None,
    level: str = "fine",
    batch_size: int = 32,
    target_size: tuple[int, int] = (224, 224),
    augment: bool = True,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Crea un generador de datos optimizado en base a un DataFrame de metadatos.

    Genera un objeto tf.data.Dataset aplicando carga asíncrona, mapeo paralela,
    aumento de datos (opcional para entrenamiento), loteo (batching) y pre-búsqueda
    (prefetching) para maximizar el uso de la GPU.

    Args:
        df (pd.DataFrame): DataFrame que contiene los metadatos de las imágenes.
        dataset_path (str | Path, optional): Ruta raíz del dataset si las rutas en
            el DataFrame son relativas y no se incluye la columna "Absolute Path".
        level (str, optional): Nivel jerárquico de clasificación ('macro', 'coarse'
            o 'fine'). Por defecto es 'fine'.
        batch_size (int, optional): Tamaño del lote para el entrenamiento.
            Por defecto es 32.
        target_size (tuple[int, int], optional): Dimensiones de la imagen para la red.
            Por defecto es (224, 224).
        augment (bool, optional): Indica si se debe aplicar aumento de datos.
            Por defecto es True.
        shuffle (bool, optional): Mezclar el orden de las muestras en cada época.
            Por defecto es True.

    Returns:
        tf.data.Dataset: Dataset de TensorFlow listo para el entrenamiento o evaluación.
    """
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
        # Añadir un buffer de mezcla del tamaño completo o un tamaño razonable
        dataset = dataset.shuffle(buffer_size=len(paths))

    # Cargar y preprocesar imágenes en paralelo
    dataset = dataset.map(
        lambda p, l: load_and_preprocess_image(p, l, target_size=target_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Aplicar aumento de datos si está habilitado
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Agrupar por lotes y pre-cargar en memoria de forma optimizada
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def build_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
    fine_tune_at: int | None = None,
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
        learning_rate (float, optional): Tasa de aprendizaje inicial para el optimizador Adam.
            Por defecto es 1e-4.
        fine_tune_at (int, optional): Índice de capa a partir del cual se descongelarán
            las capas del extractor de características para el ajuste fino.
            Si es None, se congela por completo el extractor.

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
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # Compilar el modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = 20,
    callbacks: list | None = None,
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
    )

    return history


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
