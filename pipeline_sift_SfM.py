import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pycolmap

from hloc import (
    extract_features,
    logger,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
    visualization,
)
from hloc.utils.io import get_sparse_points_per_image, write_point_cloud_pcd


def run(args):
    dataset = args.dataset
    images = dataset / "mapping"
    if not images.exists():
        raise FileNotFoundError(f"Could not find mapping images at {images}")

    outputs = args.outputs
    outputs.mkdir(parents=True, exist_ok=True)

    sfm_pairs = outputs / "pairs-exhaustive.txt"
    sfm_dir = outputs / "sfm_sift"

    feature_conf = extract_features.confs[args.features]
    matcher_conf = match_features.confs[args.matcher]

    image_list = sorted(path.relative_to(images).as_posix() for path in images.iterdir())
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)

    feature_path = extract_features.main(
        feature_conf,
        images,
        outputs,
        overwrite=args.overwrite,
    )
    match_path = match_features.main(
        matcher_conf,
        sfm_pairs,
        feature_conf["output"],
        outputs,
        overwrite=args.overwrite,
    )

    model = reconstruction.main(
        sfm_dir,
        images,
        sfm_pairs,
        feature_path,
        match_path,
        camera_mode=getattr(pycolmap.CameraMode, args.camera_mode),
    )

    if model is None:
        logger.error("SIFT SfM reconstruction failed.")
        return None

    logger.info("Finished SIFT SfM with statistics:\n%s", model.summary())

    if args.export_sparse_pcd:
        pcd_dir = args.sparse_pcd_dir or (outputs / "sparse_pcd_sift")
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
        default="outputs/sacre_coeur_sift",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="sift",
        choices=["sift", "sosnet"],
        help="Feature extractor configuration, default: %(default)s",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="NN-ratio",
        choices=["NN-ratio", "NN-mutual", "adalam"],
        help="Feature matcher configuration, default: %(default)s",
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
        help="Recompute features and matches even if outputs already exist.",
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
        help="Output directory for sparse colored .pcd files, default: <outputs>/sparse_pcd_sift",
    )
    parser.add_argument(
        "--min_track_length",
        type=int,
        default=2,
        help="Minimum 3D track length kept for sparse point export, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)
