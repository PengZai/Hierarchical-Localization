from pathlib import Path
from typing import Mapping, Tuple

import cv2
import h5py
import numpy as np
import pycolmap

from .parsers import names_to_pair, names_to_pair_old


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def get_keypoints(
    path: Path, name: str, return_uncertainty: bool = False
) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        dset = hfile[name]["keypoints"]
        p = dset.__array__()
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f"Could not find pair {(name0, name1)}... "
        "Maybe you matched with a different list of pairs? "
    )


def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]["matches0"].__array__()
        scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores


def write_poses(
    poses: Mapping[str, pycolmap.Rigid3d], path: str, prepend_camera_name: bool
):
    with open(path, "w") as f:
        for query, t in poses.items():
            qvec = " ".join(map(str, t.rotation.quat[[3, 0, 1, 2]]))
            tvec = " ".join(map(str, t.translation))
            name = query.split("/")[-1]
            if prepend_camera_name:
                name = query.split("/")[-2] + "/" + name
            f.write(f"{name} {qvec} {tvec}\n")


def get_sparse_points_per_image(
    reconstruction,
    image_dir: Path = None,
    image_names=None,
    min_track_length: int = 2,
    with_color: bool = False,
    with_depth: bool = False,
):
    if not isinstance(reconstruction, pycolmap.Reconstruction):
        reconstruction = pycolmap.Reconstruction(reconstruction)
    if with_color:
        if image_dir is None:
            raise ValueError("image_dir is required when with_color=True.")
        image_dir = Path(image_dir)

    if image_names is None:
        images = reconstruction.images.values()
    else:
        image_name_set = set(image_names)
        images = [
            image for image in reconstruction.images.values() if image.name in image_name_set
        ]

    sparse_points = {}
    for image in images:
        if with_color:
            image_array = read_image(image_dir / image.name)
        if with_color or with_depth:
            camera = reconstruction.cameras[image.camera_id]
            depth = np.zeros((camera.height, camera.width), dtype=np.float32)
        points = []
        colors = []
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            point3D = reconstruction.points3D[point2D.point3D_id]
            if point3D.track.length() < min_track_length:
                continue
            point3D_in_cam = None
            if with_color or with_depth:
                point3D_in_cam = image.cam_from_world() * point3D.xyz
                if point3D_in_cam[2] <= 0:
                    continue
                xy = camera.img_from_cam(point3D_in_cam)
                if xy is None:
                    continue
                u, v = np.round(xy).astype(int)
                if not (0 <= u < camera.width and 0 <= v < camera.height):
                    continue
            points.append(point3D.xyz)
            if with_color:
                colors.append(image_array[v, u])
            if with_depth and (depth[v, u] == 0.0 or point3D_in_cam[2] < depth[v, u]):
                depth[v, u] = point3D_in_cam[2]

        xyz = (
            np.stack(points, axis=0).astype(np.float32)
            if len(points) > 0
            else np.empty((0, 3), dtype=np.float32)
        )
        if not (with_color or with_depth):
            sparse_points[image.name] = xyz
            continue

        item = {"xyz": xyz}
        if with_color:
            item["colors"] = (
                np.stack(colors, axis=0).astype(np.float32) / 255.0
                if len(colors) > 0
                else np.empty((0, 3), dtype=np.float32)
            )
        if with_depth:
            item["depth"] = depth.astype(np.float32, copy=False)
        sparse_points[image.name] = item

    return sparse_points


def _pack_rgb_to_float(colors: np.ndarray) -> np.ndarray:
    colors_uint32 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint32)
    packed = (
        (colors_uint32[:, 0] << 16)
        | (colors_uint32[:, 1] << 8)
        | colors_uint32[:, 2]
    )
    return packed.view(np.float32)


def write_point_cloud_pcd(path: Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
    if colors.shape != xyz.shape:
        raise ValueError(
            f"colors must have the same shape as xyz, got {colors.shape} vs {xyz.shape}"
        )

    rgb = _pack_rgb_to_float(colors)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {xyz.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {xyz.shape[0]}\n")
        f.write("DATA ascii\n")
        for point, color in zip(xyz, rgb):
            f.write(f"{point[0]} {point[1]} {point[2]} {color}\n")
