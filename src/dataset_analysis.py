from collections.abc import Mapping
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Mapeo de clases macro a identificadores numéricos
MACRO_CLASSES = {
    "Fruit": 0,
    "Packages": 1,
    "Vegetables": 2,
}


def _get_group_columns(level: str) -> list[str]:
    """Devuelve las columnas de agrupación correspondientes al nivel solicitado.

    Args:
        level (str): Nivel taxonómico. Debe ser uno de: "macro", "coarse" o "fine".

    Returns:
        list[str]: Lista de columnas para operaciones de agrupación.
    """
    groups = {
        "macro": ["Split", "Macro ID", "Macro"],
        "coarse": ["Split", "Macro ID", "Macro", "Coarse ID", "Coarse"],
        "fine": ["Split", "Macro ID", "Macro", "Coarse ID", "Coarse", "Fine ID", "Fine"],
    }

    try:
        return groups[level.lower()]
    except KeyError:
        raise ValueError("level debe ser uno de: 'macro', 'coarse' o 'fine'.")


def load_dataset_metadata(
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
    dataset_path: str | Path | None = None,
) -> pd.DataFrame:
    """Carga los metadatos del Grocery Store Dataset.

    Esta función procesa los archivos oficiales de splits (train.txt, val.txt, test.txt)
    junto con classes.csv y devuelve un DataFrame con una fila por imagen.

    Args:
        split_files (Mapping[str, str | Path]): Diccionario asociando el nombre
            del split (train/val/test) con su respectiva ruta de archivo.
        classes_file (str | Path): Ruta al archivo classes.csv.
        dataset_path (str | Path, optional): Ruta raíz del dataset. Si se proporciona,
            se incluirá una columna "Absolute Path" con la ruta absoluta de cada imagen.

    Returns:
        pd.DataFrame: DataFrame ordenado con una fila por imagen y la jerarquía
            taxonómica completa.
    """
    classes_file = Path(classes_file)
    if not classes_file.exists():
        raise FileNotFoundError(f"Archivo de clases no encontrado: {classes_file}")

    # Cargar classes.csv
    classes = pd.read_csv(
        classes_file,
        usecols=[
            "Class Name (str)",
            "Class ID (int)",
            "Coarse Class Name (str)",
            "Coarse Class ID (int)",
        ],
    ).rename(
        columns={
            "Class Name (str)": "Fine",
            "Class ID (int)": "Fine ID",
            "Coarse Class Name (str)": "Coarse",
            "Coarse Class ID (int)": "Coarse ID",
        }
    )

    records = []
    # Procesar cada archivo de split
    for split_name, split_file in split_files.items():
        split_file = Path(split_file)
        if not split_file.exists():
            raise FileNotFoundError(f"Archivo de split no encontrado: {split_file}")

        with split_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                # Ejemplo de línea: train/Fruit/Apple/Golden-Delicious/img001.jpg,0,0
                tokens = [token.strip() for token in line.split(",")]
                image_path = tokens[0]
                fine_id = int(tokens[1])
                coarse_id = int(tokens[2])
                macro = Path(image_path).parts[1]

                if macro not in MACRO_CLASSES:
                    raise ValueError(
                        f"Clase macro desconocida '{macro}' encontrada en '{image_path}'."
                    )

                record = {
                    "Split": split_name,
                    "Image Path": image_path,
                    "Macro ID": MACRO_CLASSES[macro],
                    "Macro": macro,
                    "Coarse ID": coarse_id,
                    "Fine ID": fine_id,
                }

                if dataset_path is not None:
                    record["Absolute Path"] = Path(dataset_path) / image_path

                records.append(record)

    # Crear DataFrame
    metadata = pd.DataFrame(records)

    # Combinar con la información de clases
    metadata = metadata.merge(classes, on=["Fine ID", "Coarse ID"], how="left")

    # Definir orden de las columnas
    columns = ["Split", "Image Path"]
    if dataset_path is not None:
        columns.append("Absolute Path")

    columns.extend(
        [
            "Macro ID",
            "Macro",
            "Coarse ID",
            "Coarse",
            "Fine ID",
            "Fine",
        ]
    )

    metadata = metadata[columns]

    # Ordenar lógicamente
    metadata = metadata.sort_values(
        ["Split", "Macro ID", "Coarse ID", "Fine ID"]
    ).reset_index(drop=True)

    return metadata


def _apply_plot_style() -> None:
    """Aplica un estilo de gráfico consistente para las figuras de la tesis."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.figsize": (12, 8),
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.size": 12,
        "axes.grid": False,
    })


def _create_figure(rows: int, cols: int, figsize: tuple | None = None):
    """Crea una figura de matplotlib utilizando el estilo de la tesis.

    Args:
        rows (int): Número de filas de la cuadrícula de gráficos.
        cols (int): Número de columnas de la cuadrícula de gráficos.
        figsize (tuple, optional): Tamaño personalizado de la figura.

    Returns:
        tuple: Tupla (fig, axes) con la figura y los ejes de matplotlib.
    """
    _apply_plot_style()

    if figsize is None:
        figsize = (4 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    return fig, axes


def _get_image_label(row: pd.Series) -> str:
    """Construye la etiqueta de texto para mostrar debajo de la imagen.

    Args:
        row (pd.Series): Fila del DataFrame de metadatos de la imagen.

    Returns:
        str: Etiqueta de la imagen en formato Macro/Coarse/Fine.
    """
    return f"{row['Macro']}/{row['Coarse']}/{row['Fine']}"


def plot_image_grid(metadata: pd.DataFrame, title: str | None = None) -> None:
    """Dibuja una cuadrícula con las imágenes contenidas en el dataframe.

    Args:
        metadata (pd.DataFrame): DataFrame generado por load_dataset_metadata()
            o alguna versión filtrada de este.
        title (str, optional): Título superior de la figura.
    """
    if metadata.empty:
        raise ValueError("No hay imágenes para mostrar.")

    cols = min(3, len(metadata))
    rows = (len(metadata) + cols - 1) // cols
    fig, axes = _create_figure(rows, cols)

    for ax, (_, row) in zip(axes, metadata.iterrows()):
        image = Image.open(row["Absolute Path"])
        ax.imshow(image)
        ax.set_title(_get_image_label(row), fontsize=9)
        ax.axis("off")

    # Ocultar ejes vacíos que no contienen imágenes
    for ax in axes[len(metadata):]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=18)

    plt.tight_layout()
    plt.show()


def load_image_properties(
    dataset_path: str | Path,
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
) -> pd.DataFrame:
    """Carga los metadatos e calcula sus propiedades físicas y de almacenamiento.

    Args:
        dataset_path (str | Path): Carpeta raíz del Grocery Store Dataset.
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.

    Returns:
        pd.DataFrame: DataFrame con metadatos extendidos (ancho, alto, canales,
            formato, tamaño de archivo, etc.).
    """
    metadata = load_dataset_metadata(
        split_files=split_files,
        classes_file=classes_file,
        dataset_path=dataset_path,
    ).copy()

    widths = []
    heights = []
    aspect_ratios = []
    resolutions = []
    channels = []
    modes = []
    formats = []
    file_sizes = []

    for img_path_str in metadata["Absolute Path"]:
        img_path = Path(img_path_str)
        file_sizes.append(img_path.stat().st_size)

        with Image.open(img_path) as image:
            width, height = image.size
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)
            resolutions.append(width * height)
            channels.append(len(image.getbands()))
            modes.append(image.mode)
            formats.append(image.format)

    metadata["Width"] = widths
    metadata["Height"] = heights
    metadata["Aspect Ratio"] = aspect_ratios
    metadata["Resolution"] = resolutions
    metadata["Channels"] = channels
    metadata["Mode"] = modes
    metadata["Format"] = formats
    metadata["File Size"] = file_sizes

    return metadata


def _sort_images(
    df: pd.DataFrame,
    by: str,
    ascending: bool = True,
    n_images: int | None = None,
) -> pd.DataFrame:
    """Ordena las imágenes según una columna específica del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de propiedades de imágenes.
        by (str): Nombre de la columna para ordenar.
        ascending (bool, optional): Orden ascendente si es True, descendente si es False.
        n_images (int, optional): Número máximo de imágenes a retornar.

    Returns:
        pd.DataFrame: DataFrame ordenado.
    """
    if by not in df.columns:
        raise ValueError(f"Columna desconocida '{by}'.")

    result = df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
    if n_images is not None:
        result = result.head(n_images)

    return result


def _filter_images(
    df: pd.DataFrame,
    mode: str | None = None,
    channels: int | None = None,
    image_format: str | None = None,
    min_resolution: int | None = None,
    max_resolution: int | None = None,
) -> pd.DataFrame:
    """Filtra las imágenes basándose en una o más propiedades de la imagen.

    Args:
        df (pd.DataFrame): DataFrame de propiedades de imágenes.
        mode (str, optional): Modo de color de la imagen (ej. 'RGB', 'L').
        channels (int, optional): Cantidad de canales de color.
        image_format (str, optional): Formato de almacenamiento (ej. 'JPEG', 'PNG').
        min_resolution (int, optional): Resolución mínima (píxeles totales).
        max_resolution (int, optional): Resolución máxima (píxeles totales).

    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    result = df.copy()

    if mode is not None:
        result = result[result["Mode"] == mode]

    if channels is not None:
        result = result[result["Channels"] == channels]

    if image_format is not None:
        result = result[result["Format"] == image_format]

    if min_resolution is not None:
        result = result[result["Resolution"] >= min_resolution]

    if max_resolution is not None:
        result = result[result["Resolution"] <= max_resolution]

    return result.reset_index(drop=True)


def count_images_per_fine_class(
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
) -> pd.DataFrame:
    """Cuenta el número de imágenes pertenecientes a cada clase fina.

    Args:
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.

    Returns:
        pd.DataFrame: Conteo de imágenes por clase fina y split.
    """
    metadata = load_dataset_metadata(
        split_files=split_files,
        classes_file=classes_file,
    )

    group_cols = _get_group_columns("fine")
    result = (
        metadata.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "Images"})
        .sort_values(["Split", "Macro ID", "Coarse ID", "Fine ID"])
        .reset_index(drop=True)
    )

    return result


def count_images_per_coarse_class(
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
) -> pd.DataFrame:
    """Cuenta el número de imágenes pertenecientes a cada clase intermedia (coarse).

    Args:
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.

    Returns:
        pd.DataFrame: Conteo de imágenes por clase intermedia y split.
    """
    metadata = load_dataset_metadata(
        split_files=split_files,
        classes_file=classes_file,
    )

    group_cols = _get_group_columns("coarse")
    result = (
        metadata.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "Images"})
        .sort_values(["Split", "Macro ID", "Coarse ID"])
        .reset_index(drop=True)
    )

    return result


def count_images_per_macro_class(
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
) -> pd.DataFrame:
    """Cuenta el número de imágenes pertenecientes a cada clase macro.

    Args:
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.

    Returns:
        pd.DataFrame: Conteo de imágenes por clase macro y split.
    """
    metadata = load_dataset_metadata(
        split_files=split_files,
        classes_file=classes_file,
    )

    group_cols = _get_group_columns("macro")
    result = (
        metadata.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "Images"})
        .sort_values(["Split", "Macro ID"])
        .reset_index(drop=True)
    )

    return result


def class_distribution(
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
    level: str = "fine",
    normalize: bool = True,
    by_split: bool = True,
) -> pd.DataFrame:
    """Calcula la distribución (porcentaje o conteo) para el nivel taxonómico seleccionado.

    Args:
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.
        level (str, optional): Nivel taxonómico ('macro', 'coarse' o 'fine').
        normalize (bool, optional): Retorna porcentajes si es True, conteos si es False.
        by_split (bool, optional): Separa la distribución por split si es True.

    Returns:
        pd.DataFrame: Distribución de las imágenes.
    """
    metadata = load_dataset_metadata(
        split_files=split_files,
        classes_file=classes_file,
    )

    group_cols = _get_group_columns(level)
    if not by_split:
        group_cols = [col for col in group_cols if col != "Split"]

    result = (
        metadata.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "Images"})
    )

    if normalize:
        if by_split:
            result["Percentage"] = (
                result.groupby("Split")["Images"]
                .transform(lambda x: 100 * x / x.sum())
            )
        else:
            total = result["Images"].sum()
            result["Percentage"] = 100 * result["Images"] / total

    id_columns = [col for col in group_cols if col.endswith("ID")]
    result = result.sort_values(id_columns).reset_index(drop=True)

    return result


def dataset_balance(
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
    level: str = "fine",
    by_split: bool = True,
) -> pd.DataFrame:
    """Calcula métricas de balance (clases, total, media, desviación, desbalance) del dataset.

    Args:
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.
        level (str, optional): Nivel taxonómico ('macro', 'coarse' o 'fine').
        by_split (bool, optional): Calcula métricas por split si es True.

    Returns:
        pd.DataFrame: Métricas de balance del dataset.
    """
    distribution = class_distribution(
        split_files=split_files,
        classes_file=classes_file,
        level=level,
        normalize=False,
        by_split=by_split,
    )

    if by_split:
        result = (
            distribution.groupby("Split")
            .agg(
                Classes=("Images", "count"),
                Total_Images=("Images", "sum"),
                Mean_Images=("Images", "mean"),
                Std_Images=("Images", "std"),
                Min_Images=("Images", "min"),
                Max_Images=("Images", "max"),
            )
            .reset_index()
        )
    else:
        result = pd.DataFrame(
            {
                "Classes": [distribution["Images"].count()],
                "Total Images": [distribution["Images"].sum()],
                "Mean Images": [distribution["Images"].mean()],
                "Std Images": [distribution["Images"].std()],
                "Min Images": [distribution["Images"].min()],
                "Max Images": [distribution["Images"].max()],
            }
        )

    result["Imbalance Ratio"] = result["Max Images" if not by_split else "Max_Images"] / \
                               result["Min Images" if not by_split else "Min_Images"]

    if by_split:
        result = result.rename(
            columns={
                "Total_Images": "Total Images",
                "Mean_Images": "Mean Images",
                "Std_Images": "Std Images",
                "Min_Images": "Min Images",
                "Max_Images": "Max Images",
            }
        )

    return result


def split_distribution(
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
) -> pd.DataFrame:
    """Calcula la cantidad y el porcentaje de imágenes en cada split del dataset.

    Args:
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.

    Returns:
        pd.DataFrame: Distribución de imágenes por split.
    """
    metadata = load_dataset_metadata(
        split_files=split_files,
        classes_file=classes_file,
    )

    result = (
        metadata.groupby("Split", as_index=False)
        .size()
        .rename(columns={"size": "Images"})
    )

    total_images = result["Images"].sum()
    result["Percentage"] = (100 * result["Images"] / total_images).round(2)

    split_order = ["train", "val", "test"]
    result["Split"] = pd.Categorical(
        result["Split"],
        categories=split_order,
        ordered=True,
    )

    result = result.sort_values("Split").reset_index(drop=True)
    return result


def show_random_images(
    dataset_path: str | Path,
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
    n_images: int = 9,
    split: str = "train",
    random_state: int | None = None,
) -> None:
    """Muestra una cuadrícula de imágenes seleccionadas aleatoriamente de un split específico.

    Args:
        dataset_path (str | Path): Carpeta raíz del Grocery Store Dataset.
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.
        n_images (int, optional): Cantidad de imágenes a mostrar.
        split (str, optional): Split de origen ('train', 'val', 'test').
        random_state (int, optional): Semilla aleatoria para reproducibilidad.
    """
    metadata = load_dataset_metadata(
        split_files=split_files,
        classes_file=classes_file,
        dataset_path=dataset_path,
    )

    metadata = metadata[metadata["Split"] == split]
    metadata = metadata.sample(
        n=min(n_images, len(metadata)),
        random_state=random_state,
    )

    plot_image_grid(
        metadata=metadata,
        title=f"Imágenes aleatorias ({split})",
    )


def show_random_images_from_class(
    dataset_path: str | Path,
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
    class_name: str,
    level: str = "fine",
    split: str = "train",
    n_images: int = 9,
    random_state: int | None = None,
) -> None:
    """Muestra imágenes aleatorias pertenecientes a una clase y nivel taxonómico específicos.

    Args:
        dataset_path (str | Path): Carpeta raíz del Grocery Store Dataset.
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.
        class_name (str): Nombre de la clase.
        level (str, optional): Nivel taxonómico ('macro', 'coarse' o 'fine').
        split (str, optional): Split de origen ('train', 'val', 'test').
        n_images (int, optional): Cantidad de imágenes a mostrar.
        random_state (int, optional): Semilla aleatoria para reproducibilidad.
    """
    metadata = load_dataset_metadata(
        split_files=split_files,
        classes_file=classes_file,
        dataset_path=dataset_path,
    )

    metadata = metadata[metadata["Split"] == split]
    level = level.lower()

    column = {
        "macro": "Macro",
        "coarse": "Coarse",
        "fine": "Fine",
    }.get(level)

    if column is None:
        raise ValueError("level debe ser uno de: 'macro', 'coarse' o 'fine'.")

    metadata = metadata[metadata[column] == class_name]
    if metadata.empty:
        raise ValueError(f"No se encontraron imágenes para la clase '{class_name}'.")

    metadata = metadata.sample(
        n=min(n_images, len(metadata)),
        random_state=random_state,
    )

    plot_image_grid(
        metadata=metadata,
        title=f"Clase {level.capitalize()}: {class_name} ({split})",
    )


def show_class_examples(
    dataset_path: str | Path,
    split_files: Mapping[str, str | Path],
    classes_file: str | Path,
    level: str = "fine",
    split: str = "train",
    examples_per_class: int = 3,
    max_classes: int | None = None,
    random_state: int | None = None,
) -> None:
    """Muestra ejemplos visuales representativos para cada clase taxonómica.

    Args:
        dataset_path (str | Path): Carpeta raíz del Grocery Store Dataset.
        split_files (Mapping[str, str | Path]): Diccionario con los archivos de split.
        classes_file (str | Path): Ruta al archivo classes.csv.
        level (str, optional): Nivel taxonómico ('macro', 'coarse' o 'fine').
        split (str, optional): Split de origen.
        examples_per_class (int, optional): Cantidad de ejemplos por clase.
        max_classes (int, optional): Límite superior de clases a mostrar.
        random_state (int, optional): Semilla aleatoria.
    """
    metadata = load_dataset_metadata(
        dataset_path=dataset_path,
        split_files=split_files,
        classes_file=classes_file,
    )

    metadata = metadata[metadata["Split"] == split]
    level = level.lower()

    column = {
        "macro": "Macro",
        "coarse": "Coarse",
        "fine": "Fine",
    }.get(level)

    if column is None:
        raise ValueError("level debe ser uno de: 'macro', 'coarse' o 'fine'.")

    class_names = sorted(metadata[column].unique())
    if max_classes is not None:
        class_names = class_names[:max_classes]

    for class_name in class_names:
        subset = metadata[metadata[column] == class_name]
        subset = subset.sample(
            n=min(examples_per_class, len(subset)),
            random_state=random_state,
        )

        cols = examples_per_class
        rows = 1

        fig, axes = _create_figure(
            rows=rows,
            cols=cols,
            figsize=(4 * cols, 4),
        )

        for ax, (_, row) in zip(axes, subset.iterrows()):
            image = Image.open(row["Absolute Path"])
            ax.imshow(image)
            ax.set_title(_get_image_label(row), fontsize=9)
            ax.axis("off")

        # Ocultar ejes vacíos
        for ax in axes[len(subset):]:
            ax.axis("off")

        fig.suptitle(
            f"Clase {level.capitalize()}: {class_name}",
            fontsize=16,
        )
        plt.tight_layout()
        plt.show()


def plot_fine_grained_distribution(
    df: pd.DataFrame,
    split: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Dibuja la distribución del número de imágenes para cada clase fina (fine-grained).

    Args:
        df (pd.DataFrame): DataFrame obtenido de count_images_per_fine_class().
        split (str, optional): Nombre del split para filtrar ('train', 'val', 'test').
        figsize (tuple, optional): Dimensiones de la figura de matplotlib.
    """
    data = df.copy()

    if split is None:
        data = (
            data.groupby(["Macro", "Coarse", "Fine"], as_index=False)["Images"]
            .sum()
            .sort_values(["Macro", "Coarse", "Fine"])
        )
        title = "Distribución de clases de grano fino (fine-grained)"
    else:
        data = data[data["Split"] == split].sort_values(["Macro", "Coarse", "Fine"])
        title = f"Distribución de clases de grano fino (fine-grained) ({split})"

    fig, ax = plt.subplots(figsize=figsize or (20, 8))
    _apply_plot_style()

    ax.bar(range(len(data)), data["Images"])
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["Fine"], rotation=90)
    ax.set_ylabel("Imágenes")
    ax.set_xlabel("Clase fina")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_coarse_distribution(
    df: pd.DataFrame,
    split: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Dibuja la distribución del número de imágenes para cada clase intermedia (coarse).

    Args:
        df (pd.DataFrame): DataFrame obtenido de count_images_per_coarse_class().
        split (str, optional): Nombre del split para filtrar ('train', 'val', 'test').
        figsize (tuple, optional): Dimensiones de la figura de matplotlib.
    """
    data = df.copy()

    if split is None:
        data = (
            data.groupby(["Macro", "Coarse"], as_index=False)["Images"]
            .sum()
        )
        title = "Distribución de clases intermedias (coarse)"
    else:
        data = data[data["Split"] == split]
        title = f"Distribución de clases intermedias (coarse) ({split})"

    fig, ax = plt.subplots(figsize=figsize or (14, 6))
    _apply_plot_style()

    ax.bar(data["Coarse"], data["Images"])
    ax.set_ylabel("Imágenes")
    ax.set_xlabel("Clase intermedia")
    ax.set_title(title)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_macro_distribution(
    df: pd.DataFrame,
    split: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Dibuja la distribución del número de imágenes para cada clase macro.

    Args:
        df (pd.DataFrame): DataFrame obtenido de count_images_per_macro_class().
        split (str, optional): Nombre del split para filtrar ('train', 'val', 'test').
        figsize (tuple, optional): Dimensiones de la figura de matplotlib.
    """
    data = df.copy()

    if split is None:
        data = (
            data.groupby("Macro", as_index=False)["Images"]
            .sum()
        )
        title = "Distribución de clases macro"
    else:
        data = data[data["Split"] == split]
        title = f"Distribución de clases macro ({split})"

    fig, ax = plt.subplots(figsize=figsize or (7, 5))
    _apply_plot_style()

    ax.bar(data["Macro"], data["Images"])
    ax.set_ylabel("Imágenes")
    ax.set_xlabel("Clase macro")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_image_size_distribution(
    df: pd.DataFrame,
    split: str | None = None,
    bins: int = 30,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Dibuja dos histogramas con la distribución del ancho y alto de las imágenes en píxeles.

    Args:
        df (pd.DataFrame): DataFrame obtenido de load_image_properties().
        split (str, optional): Split para filtrar las imágenes.
        bins (int, optional): Número de particiones (bins) de los histogramas.
        figsize (tuple, optional): Dimensiones de la figura de matplotlib.
    """
    data = df.copy()

    if split is not None:
        data = data[data["Split"] == split]
        if data.empty:
            raise ValueError(f"No se encontraron imágenes para el split '{split}'.")
        title = f"Distribución del tamaño de imágenes ({split})"
    else:
        title = "Distribución del tamaño de imágenes"

    fig, axes = _create_figure(rows=1, cols=2, figsize=figsize or (14, 5))

    # Gráfico para el Ancho
    axes[0].hist(data["Width"], bins=bins)
    width_mean = data["Width"].mean()
    axes[0].axvline(
        width_mean,
        linestyle="--",
        linewidth=2,
        label=f"Media = {width_mean:.1f}px",
    )
    axes[0].set_title("Ancho (Width)")
    axes[0].set_xlabel("Píxeles")
    axes[0].set_ylabel("Imágenes")
    axes[0].legend()

    # Gráfico para el Alto
    axes[1].hist(data["Height"], bins=bins)
    height_mean = data["Height"].mean()
    axes[1].axvline(
        height_mean,
        linestyle="--",
        linewidth=2,
        label=f"Media = {height_mean:.1f}px",
    )
    axes[1].set_title("Alto (Height)")
    axes[1].set_xlabel("Píxeles")
    axes[1].set_ylabel("Imágenes")
    axes[1].legend()

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_aspect_ratio_distribution(
    df: pd.DataFrame,
    split: str | None = None,
    bins: int = 30,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Dibuja la distribución de la relación de aspecto de las imágenes.

    Args:
        df (pd.DataFrame): DataFrame obtenido de load_image_properties().
        split (str, optional): Split para filtrar las imágenes.
        bins (int, optional): Número de particiones (bins) del histograma.
        figsize (tuple, optional): Dimensiones de la figura de matplotlib.
    """
    data = df.copy()

    if split is not None:
        data = data[data["Split"] == split]
        if data.empty:
            raise ValueError(f"No se encontraron imágenes para el split '{split}'.")
        title = f"Distribución de la relación de aspecto ({split})"
    else:
        title = "Distribución de la relación de aspecto"

    fig, axes = _create_figure(rows=1, cols=1, figsize=figsize or (8, 5))
    ax = axes[0]

    ax.hist(data["Aspect Ratio"], bins=bins)
    aspect_mean = data["Aspect Ratio"].mean()

    ax.axvline(1.0, linestyle="--", linewidth=2, label="Imagen cuadrada (1:1)")
    ax.axvline(
        aspect_mean,
        linestyle=":",
        linewidth=2,
        label=f"Media = {aspect_mean:.3f}",
    )
    ax.set_title("Relación de Aspecto (Aspect Ratio)")
    ax.set_xlabel("Proporción Ancho / Alto")
    ax.set_ylabel("Imágenes")
    ax.legend()

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_image_dimension_distribution(
    df: pd.DataFrame,
    split: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Dibuja un diagrama de barras con la distribución de las dimensiones únicas (Ancho x Alto).

    Args:
        df (pd.DataFrame): DataFrame obtenido de load_image_properties().
        split (str, optional): Split para filtrar las imágenes.
        figsize (tuple, optional): Dimensiones de la figura de matplotlib.
    """
    data = df.copy()

    if split is not None:
        data = data[data["Split"] == split]
        if data.empty:
            raise ValueError(f"No se encontraron imágenes para el split '{split}'.")
        title = f"Distribución de dimensiones físicas ({split})"
    else:
        title = "Distribución de dimensiones físicas"

    distribution = (
        data.assign(
            Dimension=data["Width"].astype(str) + " × " + data["Height"].astype(str)
        )
        .groupby(["Width", "Height", "Dimension"], as_index=False)
        .size()
        .rename(columns={"size": "Images"})
        .sort_values(["Width", "Height"])
        .reset_index(drop=True)
    )

    fig, axes = _create_figure(rows=1, cols=1, figsize=figsize or (9, 5))
    ax = axes[0]

    bars = ax.bar(distribution["Dimension"], distribution["Images"])

    for bar, (_, row) in zip(bars, distribution.iterrows()):
        ax.text(
            x=bar.get_x() + bar.get_width() / 2,
            y=bar.get_height(),
            s=f"{row['Images']:,}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_title(title)
    ax.set_xlabel("Dimensiones de la imagen (Ancho × Alto)")
    ax.set_ylabel("Número de imágenes")
    ax.set_ylim(0, distribution["Images"].max() * 1.12)

    plt.tight_layout()
    plt.show()


def plot_split_distribution(
    df: pd.DataFrame,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Dibuja el reparto de imágenes entre los diferentes splits (train, val, test).

    Args:
        df (pd.DataFrame): DataFrame obtenido de split_distribution().
        figsize (tuple, optional): Dimensiones de la figura de matplotlib.
    """
    required_columns = {"Split", "Images", "Percentage"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "El DataFrame de entrada debe contener las columnas 'Split', 'Images' y 'Percentage'."
        )

    split_order = ["train", "val", "test"]
    missing = set(split_order) - set(df["Split"])
    if missing:
        raise ValueError(f"Faltan los siguientes splits en el dataset: {sorted(missing)}")

    data = (
        df.copy()
        .set_index("Split")
        .loc[split_order]
        .reset_index()
    )

    fig, axes = _create_figure(rows=1, cols=1, figsize=figsize or (8, 5))
    ax = axes[0]

    bars = ax.bar(data["Split"], data["Images"])

    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(
            x=bar.get_x() + bar.get_width() / 2,
            y=bar.get_height(),
            s=f"{row['Images']:,}\n({row['Percentage']:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_title("Distribución por splits del dataset")
    ax.set_xlabel("Split del Dataset")
    ax.set_ylabel("Número de imágenes")
    ax.set_ylim(0, data["Images"].max() * 1.12)

    plt.tight_layout()
    plt.show()


def find_grayscale_images(df: pd.DataFrame) -> pd.DataFrame:
    """Encuentra todas las imágenes en escala de grises en el dataset (modo PIL 'L').

    Args:
        df (pd.DataFrame): DataFrame de propiedades obtenido de load_image_properties().

    Returns:
        pd.DataFrame: Subconjunto del DataFrame original con imágenes en grises.
    """
    return _filter_images(df=df, mode="L")


def find_non_rgb_images(df: pd.DataFrame) -> pd.DataFrame:
    """Encuentra las imágenes cuyo modo de color sea diferente de RGB.

    Args:
        df (pd.DataFrame): DataFrame de propiedades obtenido de load_image_properties().

    Returns:
        pd.DataFrame: Subconjunto del DataFrame con imágenes que no son RGB.
    """
    return df[df["Mode"] != "RGB"].reset_index(drop=True)


def find_low_resolution_images(
    df: pd.DataFrame,
    min_resolution: int = 224 * 224,
) -> pd.DataFrame:
    """Encuentra las imágenes cuya resolución (ancho x alto) esté por debajo de un umbral dado.

    Args:
        df (pd.DataFrame): DataFrame obtenido de load_image_properties().
        min_resolution (int, optional): Umbral mínimo de píxeles (ancho * alto).

    Returns:
        pd.DataFrame: Subconjunto con imágenes de baja resolución.
    """
    return _filter_images(df=df, max_resolution=min_resolution - 1)


def show_largest_images(df: pd.DataFrame, n_images: int = 9) -> None:
    """Muestra las imágenes de mayor resolución (ancho x alto) del dataset.

    Args:
        df (pd.DataFrame): DataFrame de propiedades del dataset.
        n_images (int, optional): Cantidad máxima de imágenes a mostrar.
    """
    images = _sort_images(
        df=df,
        by="Resolution",
        ascending=False,
        n_images=n_images,
    )
    plot_image_grid(metadata=images, title=f"{n_images} Imágenes más grandes")


def show_smallest_images(df: pd.DataFrame, n_images: int = 9) -> None:
    """Muestra las imágenes de menor resolución (ancho x alto) del dataset.

    Args:
        df (pd.DataFrame): DataFrame de propiedades del dataset.
        n_images (int, optional): Cantidad máxima de imágenes a mostrar.
    """
    images = _sort_images(
        df=df,
        by="Resolution",
        ascending=True,
        n_images=n_images,
    )
    plot_image_grid(metadata=images, title=f"{n_images} Imágenes más pequeñas")


def show_extreme_aspect_ratios(df: pd.DataFrame, n_images: int = 9) -> None:
    """Muestra las imágenes con mayor desviación con respecto a una relación de aspecto cuadrada.

    Args:
        df (pd.DataFrame): DataFrame de propiedades del dataset.
        n_images (int, optional): Cantidad máxima de imágenes a mostrar.
    """
    images = df.copy()
    images["Aspect Ratio Deviation"] = (images["Aspect Ratio"] - 1.0).abs()
    images = _sort_images(
        df=images,
        by="Aspect Ratio Deviation",
        ascending=False,
        n_images=n_images,
    )
    plot_image_grid(metadata=images, title=f"{n_images} Relaciones de aspecto más extremas")


def save_parquet(df: pd.DataFrame, output_file: str | Path) -> None:
    """Guarda un DataFrame en formato Parquet, convirtiendo objetos de ruta a texto.

    Args:
        df (pd.DataFrame): DataFrame que se va a guardar.
        output_file (str | Path): Ruta de destino del archivo Parquet.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    for column in df.columns:
        if df[column].map(lambda value: isinstance(value, Path)).any():
            df[column] = df[column].astype(str)

    df.to_parquet(output_file, index=False)


def load_parquet(input_file: str | Path) -> pd.DataFrame:
    """Carga y devuelve un DataFrame desde un archivo Parquet.

    Args:
        input_file (str | Path): Ruta del archivo Parquet origen.

    Returns:
        pd.DataFrame: DataFrame de metadatos cargados.
    """
    return pd.read_parquet(input_file)
