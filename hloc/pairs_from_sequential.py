import argparse
import collections.abc as collections
from pathlib import Path
from typing import List, Optional, Tuple, Union

from . import logger

DEFAULT_OVERLAP = 10
DEFAULT_QUADRATIC_OVERLAP = False


def parse_image_list(path: Path) -> List[str]:
    images = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip("\n")
            if len(line) == 0 or line[0] == "#":
                continue
            name, *_ = line.split()
            images.append(name)
    if len(images) == 0:
        raise ValueError(f"Could not find any image in {path}.")
    return images


def parse_image_lists(paths: Path) -> List[str]:
    files = list(Path(paths.parent).glob(paths.name))
    if len(files) == 0:
        raise ValueError(f"Could not find any image list matching {paths}.")

    images = []
    for lfile in files:
        images += parse_image_list(lfile)
    return images


def list_h5_names(path: Path) -> List[str]:
    import h5py

    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def parse_names(
    image_list: Optional[Union[Path, List[str]]] = None,
    features: Optional[Path] = None,
) -> List[str]:
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            names = parse_image_lists(image_list)
        elif isinstance(image_list, collections.Iterable):
            names = list(image_list)
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        names = list_h5_names(features)
    else:
        raise ValueError("Provide either a list of images or a feature file.")

    # COLMAP defines the sequence using ascending image names.
    names = sorted(names)
    if len(names) != len(set(names)):
        raise ValueError("Image names must be unique for sequential matching.")
    return names


def get_offsets(
    overlap: int,
    max_offset: int,
    quadratic_overlap: bool,
) -> List[int]:
    if overlap < 1:
        raise ValueError("`overlap` must be >= 1.")

    offsets = set(range(1, min(overlap, max_offset) + 1))
    if quadratic_overlap:
        offset = 1
        for _ in range(overlap):
            if offset > max_offset:
                break
            offsets.add(offset)
            offset *= 2
    return sorted(offsets)


def pairs_from_names(
    names: List[str],
    overlap: int = DEFAULT_OVERLAP,
    quadratic_overlap: bool = DEFAULT_QUADRATIC_OVERLAP,
) -> List[Tuple[str, str]]:
    names = sorted(names)
    if len(names) != len(set(names)):
        raise ValueError("Image names must be unique for sequential matching.")
    if len(names) < 2:
        return []

    offsets = get_offsets(
        overlap=overlap,
        max_offset=len(names) - 1,
        quadratic_overlap=quadratic_overlap,
    )

    pairs = []
    for i, name0 in enumerate(names[:-1]):
        for offset in offsets:
            j = i + offset
            if j >= len(names):
                break
            pairs.append((name0, names[j]))
    return pairs


def main(
    output: Path,
    image_list: Optional[Union[Path, List[str]]] = None,
    features: Optional[Path] = None,
    overlap: int = DEFAULT_OVERLAP,
    quadratic_overlap: bool = DEFAULT_QUADRATIC_OVERLAP,
):
    names = parse_names(image_list=image_list, features=features)
    pairs = pairs_from_names(
        names,
        overlap=overlap,
        quadratic_overlap=quadratic_overlap,
    )

    logger.info(
        "Found %d pairs from %d images with overlap=%d and quadratic_overlap=%s.",
        len(pairs),
        len(names),
        overlap,
        quadratic_overlap,
    )
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--overlap", default=DEFAULT_OVERLAP, type=int)
    parser.add_argument(
        "--quadratic_overlap",
        dest="quadratic_overlap",
        action="store_true",
        help="Also match images at exponentially increasing offsets.",
    )
    parser.add_argument(
        "--no_quadratic_overlap",
        dest="quadratic_overlap",
        action="store_false",
        help="Only match within the linear overlap window.",
    )
    parser.set_defaults(quadratic_overlap=DEFAULT_QUADRATIC_OVERLAP)
    args = parser.parse_args()
    main(**args.__dict__)
