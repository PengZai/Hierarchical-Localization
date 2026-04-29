import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pycolmap

from hloc import (
    extract_features,
    logger,
    match_features,
    pairs_from_exhaustive,
    pairs_from_sequential,
    reconstruction,
    visualization,
)
from hloc.utils.io import get_sparse_points_per_image, write_point_cloud_pcd
from hloc.triangulation import (
    get_mapper_option_help,
    parse_mapper_option_args,
    parse_option_args,
)

PAIRING_CHOICES = ("exhaustive", "sequential")


def build_sfm_pairs(outputs: Path, image_list, args) -> Path:
    if args.pairing == "exhaustive":
        sfm_pairs = outputs / "pairs-exhaustive.txt"
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
        return sfm_pairs
    if args.pairing == "sequential":
        pairing_mode = "quad" if args.quadratic_overlap else "linear"
        sfm_pairs = (
            outputs
            / f"pairs-sequential-overlap{args.sequential_overlap}-{pairing_mode}.txt"
        )
        pairs_from_sequential.main(
            sfm_pairs,
            image_list=image_list,
            overlap=args.sequential_overlap,
            quadratic_overlap=args.quadratic_overlap,
        )
        return sfm_pairs
    raise ValueError(f"Unknown pairing strategy: {args.pairing}")


def ensure_runtime_cache_dirs():
    # LoMa downloads weights through torch.hub, which fails here if it tries to
    # use the default read-only cache location under the home directory.
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

    sfm_dir = outputs / "sfm_loma"

    feature_conf = extract_features.confs[args.features]
    matcher_conf = match_features.confs[args.matcher]

    image_list = sorted(path.relative_to(images).as_posix() for path in images.iterdir())
    sfm_pairs = build_sfm_pairs(outputs, image_list, args)

    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf,
        sfm_pairs,
        feature_conf["output"],
        outputs,
        overwrite=args.overwrite,
    )

    if args.visualize_matches:
        match_viz_dir = args.match_viz_dir or (outputs / "match_visualizations")
        keep_match_figures_open = args.visualize
        visualization.visualize_match_summary(
            match_path,
            sfm_pairs,
            output_path=match_viz_dir / "summary.png",
            close=not keep_match_figures_open,
        )
        visualization.visualize_feature_matches(
            feature_path,
            match_path,
            sfm_pairs,
            images,
            output_dir=match_viz_dir / "pairs",
            num_pairs=args.match_viz_pairs,
            max_plot_matches=args.match_viz_max_lines,
            close=not keep_match_figures_open,
        )
        logger.info("Saved match visualizations to %s", match_viz_dir)

    model = reconstruction.main(
        sfm_dir,
        images,
        sfm_pairs,
        feature_path,
        match_path,
        camera_mode=getattr(pycolmap.CameraMode, args.camera_mode),
        image_options=args.image_options,
        mapper_options=args.mapper_options,
    )

    if model is None:
        if args.visualize and args.visualize_matches:
            plt.show()
        return None

    if args.export_sparse_pcd:
        pcd_dir = args.sparse_pcd_dir or (outputs / "sparse_pcd_loma")
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
        default="datasets/botanic_garden",
        help="Path to the dataset root, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/botanic_garden_loma",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="loma_aachen",
        choices=["loma_aachen", "loma_inloc"],
        help="LoMa feature extractor configuration, default: %(default)s",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="loma",
        choices=["loma"],
        help="Feature matcher configuration, default: %(default)s",
    )
    parser.add_argument(
        "--pairing",
        type=str,
        default="exhaustive",
        choices=PAIRING_CHOICES,
        help="Image pair generation strategy, default: %(default)s",
    )
    parser.add_argument(
        "--sequential_overlap",
        "--overlap",
        dest="sequential_overlap",
        type=int,
        default=pairs_from_sequential.DEFAULT_OVERLAP,
        help="Sequential pairing overlap window. Only used with --pairing sequential, default: %(default)s",
    )
    parser.add_argument(
        "--quadratic_overlap",
        dest="quadratic_overlap",
        action="store_true",
        help="Also pair sequential images at exponentially increasing offsets.",
    )
    parser.add_argument(
        "--no_quadratic_overlap",
        dest="quadratic_overlap",
        action="store_false",
        help="Only pair sequential images within the linear overlap window.",
    )
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
        help="COLMAP camera mode passed to reconstruction, default: %(default)s",
    )
    parser.add_argument(
        "--image_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(pycolmap.ImageReaderOptions().todict()),
    )
    parser.add_argument(
        "--mapper_options",
        nargs="+",
        default=[],
        help=(
            "List of key=value mapper options. Accepts flat mapper keys like "
            "init_min_tri_angle=2.0 or dotted pipeline keys like "
            f"mapper.init_min_tri_angle=2.0 from {get_mapper_option_help()}"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute matches even if outputs already exist.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the reconstructed model with matplotlib visualizations.",
    )
    parser.add_argument(
        "--visualize_matches",
        action="store_true",
        help="Save match diagnostic plots and, with --visualize, keep them open.",
    )
    parser.add_argument(
        "--match_viz_pairs",
        type=int,
        default=6,
        help="Number of representative image pairs visualized for matches, default: %(default)s",
    )
    parser.add_argument(
        "--match_viz_max_lines",
        type=int,
        default=300,
        help="Maximum matched lines shown per pair visualization, default: %(default)s",
    )
    parser.add_argument(
        "--match_viz_dir",
        type=Path,
        default=None,
        help="Directory for match diagnostic plots, default: <outputs>/match_visualizations",
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
        help="Output directory for sparse colored .pcd files, default: <outputs>/sparse_pcd_loma",
    )
    parser.add_argument(
        "--min_track_length",
        type=int,
        default=2,
        help="Minimum 3D track length kept for sparse point export, default: %(default)s",
    )
    parser.set_defaults(
        quadratic_overlap=pairs_from_sequential.DEFAULT_QUADRATIC_OVERLAP
    )
    args = parser.parse_args()
    args.image_options = parse_option_args(
        args.image_options, pycolmap.ImageReaderOptions()
    )
    args.mapper_options = parse_mapper_option_args(args.mapper_options)
    run(args)
