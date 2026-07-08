"""
Filesystem utilities for the Master's Thesis project.

This module contains helper functions for inspecting the project
structure and datasets inside Kaggle notebooks.
"""

from collections import Counter
from pathlib import Path


def print_directory_tree(root_path, max_depth=None):
    """
    Print the directory tree.

    Parameters
    ----------
    root_path : str or Path
        Root directory.
    max_depth : int, optional
        Maximum depth to display.
    """
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")

    print(root.resolve())

    def _walk(directory, prefix="", depth=0):
        if max_depth is not None and depth >= max_depth:
            return

        children = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))

        for index, child in enumerate(children):
            connector = "└── " if index == len(children) - 1 else "├── "

            print(f"{prefix}{connector}{child.name}")

            if child.is_dir():
                extension = "    " if index == len(children) - 1 else "│   "
                _walk(
                    child,
                    prefix + extension,
                    depth + 1,
                )

    _walk(root)


def count_files_by_extension(root_path):
    """
    Count files grouped by extension.

    Parameters
    ----------
    root_path : str or Path

    Returns
    -------
    dict
        Dictionary containing extension counts.
    """
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")

    counter = Counter()

    for file in root.rglob("*"):

        if file.is_file():

            extension = file.suffix.lower()

            if extension == "":
                extension = "[no extension]"

            counter[extension] += 1

    return dict(sorted(counter.items()))


def print_file_extension_summary(root_path):
    """
    Print the number of files for every extension.
    """
    counts = count_files_by_extension(root_path)

    total = sum(counts.values())

    print("\nFiles grouped by extension\n")

    for extension, amount in counts.items():
        print(f"{extension:<15}{amount}")

    print("-" * 28)
    print(f"{'TOTAL':<15}{total}")


def count_jpeg_images(root_path):
    """
    Count JPEG images.

    Parameters
    ----------
    root_path : str or Path

    Returns
    -------
    int
    """
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")

    extensions = {".jpg", ".jpeg"}

    return sum(
        1
        for file in root.rglob("*")
        if file.is_file()
        and file.suffix.lower() in extensions
    )


def print_jpeg_summary(root_path):
    """
    Print the number of JPEG images.
    """
    total = count_jpeg_images(root_path)

    print(f"\nJPEG images found: {total}")