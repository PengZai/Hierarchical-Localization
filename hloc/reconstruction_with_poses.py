import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap

from . import logger
from .reconstruction import create_empty_db, get_image_ids, import_images
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)


def parse_image_poses(path: Path) -> Dict[str, pycolmap.Rigid3d]:
    poses = {}
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            data = line.split()
            if len(data) != 8:
                raise ValueError(
                    f"Invalid pose line {line_num} in {path}: expected "
                    "image_name qw qx qy qz tx ty tz"
                )
            name = data[0]
            qw, qx, qy, qz, tx, ty, tz = map(float, data[1:])
            rotation = pycolmap.Rotation3d([qx, qy, qz, qw])
            poses[name] = pycolmap.Rigid3d(rotation, [tx, ty, tz])

    if len(poses) == 0:
        raise ValueError(f"No poses found in {path}.")

    logger.info("Imported %d image poses from %s.", len(poses), path.name)
    return poses


def create_reference_reconstruction(
    database_path: Path,
    poses: Dict[str, pycolmap.Rigid3d],
) -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction()
    with pycolmap.Database.open(database_path) as db:
        cameras = db.read_all_cameras()
        images = db.read_all_images()

    for camera in cameras:
        reconstruction.add_camera_with_trivial_rig(camera)

    missing_poses = [image.name for image in images if image.name not in poses]
    if missing_poses:
        preview = ", ".join(missing_poses[:5])
        if len(missing_poses) > 5:
            preview += ", ..."
        raise ValueError(
            "Missing poses for imported images: "
            f"{preview} ({len(missing_poses)} missing)."
        )

    image_names = {image.name for image in images}
    extra_poses = sorted(set(poses.keys()) - image_names)
    if extra_poses:
        logger.warning(
            "Ignoring %d pose(s) that do not match imported images: %s",
            len(extra_poses),
            ", ".join(extra_poses[:5]) + (", ..." if len(extra_poses) > 5 else ""),
        )

    for image in images:
        reconstruction.add_image_with_trivial_frame(image, poses[image.name])

    logger.info(
        "Created a reference reconstruction with %d registered images.",
        reconstruction.num_reg_images(),
    )
    return reconstruction


def run_reconstruction_with_poses(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    reference: pycolmap.Reconstruction,
    verbose: bool = False,
    triangulation_options: Optional[Dict[str, Any]] = None,
    bundle_adjustment_options: Optional[Dict[str, Any]] = None,
    refine: bool = True,
) -> pycolmap.Reconstruction:
    sfm_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running triangulation from the provided initial camera poses...")
    with OutputCapture(verbose):
        reconstruction = pycolmap.triangulate_points(
            reference,
            database_path,
            image_dir,
            sfm_dir,
            options=triangulation_options or {},
        )
        if refine:
            logger.info(
                "Refining the initialized reconstruction with bundle adjustment..."
            )
            pycolmap.bundle_adjustment(
                reconstruction,
                pycolmap.BundleAdjustmentOptions(bundle_adjustment_options or {}),
            )
        else:
            logger.info("Skipping bundle adjustment refinement.")
    return reconstruction


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    poses: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
    triangulation_options: Optional[Dict[str, Any]] = None,
    bundle_adjustment_options: Optional[Dict[str, Any]] = None,
    refine: bool = True,
) -> pycolmap.Reconstruction:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches
    assert poses.exists(), poses

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    logger.info(f"Writing COLMAP logs to {sfm_dir / 'colmap.LOG.*'}")
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / "colmap.LOG.")

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)

    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features)
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)

    pose_dict = parse_image_poses(poses)
    reference = create_reference_reconstruction(database, pose_dict)
    reconstruction = run_reconstruction_with_poses(
        sfm_dir=sfm_dir,
        database_path=database,
        image_dir=image_dir,
        reference=reference,
        verbose=verbose,
        triangulation_options=triangulation_options,
        bundle_adjustment_options=bundle_adjustment_options,
        refine=refine,
    )
    logger.info(
        f"Reconstruction statistics:\n{reconstruction.summary()}"
        + f"\n\tnum_input_images = {len(image_ids)}"
    )
    return reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
    )
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip_bundle_adjustment",
        action="store_false",
        dest="refine",
        help="Skip bundle adjustment after triangulation.",
    )
    parser.add_argument(
        "--image_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(pycolmap.ImageReaderOptions().todict()),
    )
    parser.add_argument(
        "--triangulation_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(
            pycolmap.IncrementalTriangulatorOptions().todict()
        ),
    )
    parser.add_argument(
        "--bundle_adjustment_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(
            pycolmap.BundleAdjustmentOptions().todict()
        ),
    )
    args = parser.parse_args().__dict__

    args["camera_mode"] = getattr(pycolmap.CameraMode, args["camera_mode"])
    image_options = parse_option_args(
        args.pop("image_options"), pycolmap.ImageReaderOptions()
    )
    triangulation_options = parse_option_args(
        args.pop("triangulation_options"), pycolmap.IncrementalTriangulatorOptions()
    )
    bundle_adjustment_options = parse_option_args(
        args.pop("bundle_adjustment_options"), pycolmap.BundleAdjustmentOptions()
    )

    main(
        **args,
        image_options=image_options,
        triangulation_options=triangulation_options,
        bundle_adjustment_options=bundle_adjustment_options,
    )
