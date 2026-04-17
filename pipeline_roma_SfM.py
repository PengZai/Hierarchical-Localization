import argparse
import os
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import pycolmap

from hloc import (
    logger,
    match_dense,
    pairs_from_exhaustive,
    reconstruction,
    visualization,
)
from hloc.utils.io import get_sparse_points_per_image, write_point_cloud_pcd


def ensure_runtime_cache_dirs():
    runtime_dirs = {
        "MPLCONFIGDIR": Path("/tmp/matplotlib"),
        "TORCH_HOME": Path("/tmp/torch-cache"),
        "XDG_CACHE_HOME": Path("/tmp"),
    }
    for env_var, path in runtime_dirs.items():
        os.environ.setdefault(env_var, str(path))
        path.mkdir(parents=True, exist_ok=True)


def run(args):
    ensure_runtime_cache_dirs()

    dataset = args.dataset
    images = dataset / "mapping"
    if not images.exists():
        raise FileNotFoundError(f"Could not find mapping images at {images}")

    outputs = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)

    sfm_pairs = outputs / "pairs-exhaustive.txt"
    sfm_dir = outputs / "sfm_roma"

    logger.info("Configs for dense feature matchers:\n%s", pformat(match_dense.confs))

    matcher_conf = match_dense.confs[args.matcher]

    image_list = sorted(path.relative_to(images).as_posix() for path in images.iterdir())
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)

    features_path, matches_path = match_dense.main(
        matcher_conf,
        sfm_pairs,
        images,
        outputs,
        max_kps=args.max_kps,
        overwrite=args.overwrite,
    )

    model = reconstruction.main(
        sfm_dir,
        images,
        sfm_pairs,
        features_path,
        matches_path,
        camera_mode=getattr(pycolmap.CameraMode, args.camera_mode),
    )

    if model is None:
        logger.error("RoMa SfM reconstruction failed.")
        return None

    logger.info("Finished RoMa SfM with statistics:\n%s", model.summary())

    if args.export_sparse_pcd:
        pcd_dir = args.sparse_pcd_dir or (outputs / "sparse_pcd_roma")
        sparse_points = get_sparse_points_per_image(
            model,
            image_dir=images,
            min_track_length=args.min_track_length,
            with_color=True,
        )
        for image_name, data in sparse_points.items():
            write_point_cloud_pcd(
                pcd_dir / Path(image_name).with_suffix(".pcd"),
                data["xyz"],
                data["colors"],
            )
        logger.info("Wrote %d sparse colored PCD files to %s", len(sparse_points), pcd_dir)

    if args.visualize:
        visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)
        visualization.visualize_sfm_2d(model, images, color_by="track_length", n=5)
        visualization.visualize_sfm_2d(model, images, color_by="depth", n=5)
        plt.show()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/sacre_coeur",
        help="Path to the dataset root, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/sacre_coeur_roma",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="roma",
        choices=["roma"],
        help="Dense matcher configuration, default: %(default)s",
    )
    parser.add_argument(
        "--max_kps",
        type=int,
        default=8192,
        help="Maximum number of keypoints per image kept from dense matches, default: %(default)s",
    )
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
        help="COLMAP camera mode passed to reconstruction, default: %(default)s",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute dense features and matches even if outputs already exist.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the reconstructed model with matplotlib visualizations.",
    )
    parser.add_argument(
        "--export_sparse_pcd",
        action="store_true",
        help="Export one colored sparse point cloud .pcd per image after reconstruction.",
    )
    parser.add_argument(
        "--sparse_pcd_dir",
        type=Path,
        default=None,
        help="Output directory for sparse colored .pcd files, default: <outputs>/sparse_pcd_roma",
    )
    parser.add_argument(
        "--min_track_length",
        type=int,
        default=2,
        help="Minimum 3D track length kept for sparse point export, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)
