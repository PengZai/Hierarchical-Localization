import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from matplotlib import cm

from .utils.io import (
    get_dense_pair_matches,
    get_feature_pair_matches,
    get_matches,
    read_image,
)
from .utils.parsers import parse_retrieval
from .utils.viz import (
    add_text,
    cm_RdGn,
    plot_images,
    plot_keypoints,
    plot_matches,
    save_plot,
)


def visualize_sfm_2d(
    reconstruction, image_dir, color_by="visibility", selected=[], n=1, seed=0, dpi=75
):
    assert image_dir.exists()
    if not isinstance(reconstruction, pycolmap.Reconstruction):
        reconstruction = pycolmap.Reconstruction(reconstruction)

    if not selected:
        image_ids = list(reconstruction.reg_image_ids())
        selected = random.Random(seed).sample(image_ids, min(n, len(image_ids)))

    for i in selected:
        image = reconstruction.images[i]
        keypoints = np.array([p.xy for p in image.points2D])
        visible = np.array([p.has_point3D() for p in image.points2D])

        if color_by == "visibility":
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            text = f"visible: {np.count_nonzero(visible)}/{len(visible)}"
        elif color_by == "track_length":
            tl = np.array(
                [
                    (
                        reconstruction.points3D[p.point3D_id].track.length()
                        if p.has_point3D()
                        else 1
                    )
                    for p in image.points2D
                ]
            )
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f"max/median track length: {max_}/{med_}"
        elif color_by == "depth":
            p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
            z = np.array(
                [
                    (image.cam_from_world() * reconstruction.points3D[j].xyz)[-1]
                    for j in p3ids
                ]
            )
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f"visible: {np.count_nonzero(visible)}/{len(visible)}"
            keypoints = keypoints[visible]
        else:
            raise NotImplementedError(f"Coloring not implemented: {color_by}.")

        name = image.name
        plot_images([read_image(image_dir / name)], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        add_text(0, text)
        add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")


def visualize_loc(
    results,
    image_dir,
    reconstruction=None,
    db_image_dir=None,
    selected=[],
    n=1,
    seed=0,
    prefix=None,
    **kwargs,
):
    assert image_dir.exists()

    with open(str(results) + "_logs.pkl", "rb") as f:
        logs = pickle.load(f)

    if not selected:
        queries = list(logs["loc"].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        selected = random.Random(seed).sample(queries, min(n, len(queries)))

    if reconstruction is not None:
        if not isinstance(reconstruction, pycolmap.Reconstruction):
            reconstruction = pycolmap.Reconstruction(reconstruction)

    for qname in selected:
        loc = logs["loc"][qname]
        visualize_loc_from_log(
            image_dir, qname, loc, reconstruction, db_image_dir, **kwargs
        )


def visualize_loc_from_log(
    image_dir,
    query_name,
    loc,
    reconstruction=None,
    db_image_dir=None,
    top_k_db=2,
    dpi=75,
):
    q_image = read_image(image_dir / query_name)
    if loc.get("covisibility_clustering", False):
        # select the first, largest cluster if the localization failed
        loc = loc["log_clusters"][loc["best_cluster"] or 0]

    inliers = np.array(loc["PnP_ret"]["inlier_mask"])
    mkp_q = loc["keypoints_query"]
    n = len(loc["db"])
    if reconstruction is not None:
        # for each pair of query keypoint and its matched 3D point,
        # we need to find its corresponding keypoint in each database image
        # that observes it. We also count the number of inliers in each.
        kp_idxs, kp_to_3D_to_db = loc["keypoint_index_to_db"]
        counts = np.zeros(n)
        dbs_kp_q_db = [[] for _ in range(n)]
        inliers_dbs = [[] for _ in range(n)]
        for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers, kp_to_3D_to_db)):
            track = reconstruction.points3D[p3D_id].track
            track = {el.image_id: el.point2D_idx for el in track.elements}
            for db_idx in db_idxs:
                counts[db_idx] += inl
                kp_db = track[loc["db"][db_idx]]
                dbs_kp_q_db[db_idx].append((i, kp_db))
                inliers_dbs[db_idx].append(inl)
    else:
        # for inloc the database keypoints are already in the logs
        assert "keypoints_db" in loc
        assert "indices_db" in loc
        counts = np.array([np.sum(loc["indices_db"][inliers] == i) for i in range(n)])

    # display the database images with the most inlier matches
    db_sort = np.argsort(-counts)
    for db_idx in db_sort[:top_k_db]:
        if reconstruction is not None:
            db = reconstruction.images[loc["db"][db_idx]]
            db_name = db.name
            db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
            kp_q = mkp_q[db_kp_q_db[:, 0]]
            kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
            inliers_db = inliers_dbs[db_idx]
        else:
            db_name = loc["db"][db_idx]
            kp_q = mkp_q[loc["indices_db"] == db_idx]
            kp_db = loc["keypoints_db"][loc["indices_db"] == db_idx]
            inliers_db = inliers[loc["indices_db"] == db_idx]

        db_image = read_image((db_image_dir or image_dir) / db_name)
        color = cm_RdGn(inliers_db).tolist()
        text = f"inliers: {sum(inliers_db)}/{len(inliers_db)}"

        plot_images([q_image, db_image], dpi=dpi)
        plot_matches(kp_q, kp_db, color, a=0.1)
        add_text(0, text)
        opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va="bottom")
        add_text(0, query_name, **opts)
        add_text(1, db_name, **opts)


def _get_unique_pairs(pairs_path):
    pairs_dict = parse_retrieval(pairs_path)
    unique_pairs = []
    seen = set()
    for name0, names1 in pairs_dict.items():
        for name1 in names1:
            key = tuple(sorted((name0, name1)))
            if key in seen:
                continue
            seen.add(key)
            unique_pairs.append((name0, name1))
    return unique_pairs


def summarize_match_assignments(matches_path, pairs_path):
    pair_stats = []
    image_totals = {}
    image_degrees = {}
    for name0, name1 in _get_unique_pairs(pairs_path):
        _, scores = get_matches(matches_path, name0, name1)
        valid_matches = len(scores)
        pair_stats.append(
            {
                "pair": (name0, name1),
                "valid_matches": valid_matches,
                "mean_score": float(np.mean(scores)) if valid_matches else 0.0,
            }
        )
        for name in (name0, name1):
            image_totals[name] = image_totals.get(name, 0) + valid_matches
            image_degrees[name] = image_degrees.get(name, 0) + 1

    image_stats = [
        {
            "name": name,
            "total_valid_matches": image_totals[name],
            "connected_pairs": image_degrees[name],
            "avg_valid_matches": image_totals[name] / image_degrees[name],
        }
        for name in sorted(image_totals)
    ]
    return pair_stats, image_stats


def summarize_dense_matches(matches_path, pairs_path):
    return summarize_match_assignments(matches_path, pairs_path)


def visualize_dense_match_summary(
    matches_path,
    pairs_path,
    output_path=None,
    dpi=100,
    close=False,
):
    pair_stats, image_stats = summarize_match_assignments(matches_path, pairs_path)
    pair_counts = np.array([s["valid_matches"] for s in pair_stats], dtype=np.float32)
    image_totals = np.array(
        [s["total_valid_matches"] for s in image_stats], dtype=np.float32
    )
    image_avgs = np.array([s["avg_valid_matches"] for s in image_stats], dtype=np.float32)

    plt.figure(figsize=(15, 4), dpi=dpi)
    axes = [plt.subplot(1, 3, i + 1) for i in range(3)]
    stats = [
        (pair_counts, "Valid matches / pair"),
        (image_totals, "Total valid matches / image"),
        (image_avgs, "Avg valid matches / connected pair / image"),
    ]
    for ax, (values, title) in zip(axes, stats):
        ax.hist(values, bins=min(20, max(len(values), 1)), color="#2c7fb8", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("count")
        ax.set_ylabel("frequency")
        if len(values) > 0:
            ax.axvline(values.mean(), color="#d95f0e", linestyle="--", linewidth=1.5)
            ax.text(
                0.02,
                0.95,
                f"mean={values.mean():.1f}\nmedian={np.median(values):.1f}\nmin={values.min():.1f}\nmax={values.max():.1f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
    plt.suptitle("Match summary")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_plot(output_path)
    if close:
        plt.close(plt.gcf())


def visualize_match_summary(
    matches_path,
    pairs_path,
    output_path=None,
    dpi=100,
    close=False,
):
    visualize_dense_match_summary(
        matches_path,
        pairs_path,
        output_path=output_path,
        dpi=dpi,
        close=close,
    )


def _select_representative_pairs(pair_stats, num_pairs):
    if num_pairs <= 0:
        return []
    sorted_stats = sorted(pair_stats, key=lambda stat: stat["valid_matches"], reverse=True)
    if len(sorted_stats) <= num_pairs:
        return sorted_stats
    spread = np.linspace(0, len(sorted_stats) - 1, num_pairs).round().astype(int)
    selected = []
    seen = set()
    for idx in spread:
        pair = sorted_stats[idx]["pair"]
        if pair in seen:
            continue
        seen.add(pair)
        selected.append(sorted_stats[idx])
    return selected


def _scores_to_colors(scores):
    if len(scores) == 0:
        return []
    score_min, score_max = float(np.min(scores)), float(np.max(scores))
    if score_max > score_min:
        normalized = (scores - score_min) / (score_max - score_min)
    else:
        normalized = np.ones_like(scores)
    return cm_RdGn(normalized).tolist()


def visualize_dense_matches(
    matches_path,
    pairs_path,
    image_dir,
    output_dir=None,
    num_pairs=6,
    max_plot_matches=300,
    dpi=100,
    close=False,
):
    pair_stats, _ = summarize_match_assignments(matches_path, pairs_path)
    selected = _select_representative_pairs(pair_stats, num_pairs)
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for rank, stat in enumerate(selected):
        name0, name1 = stat["pair"]
        image0 = read_image(Path(image_dir) / name0)
        image1 = read_image(Path(image_dir) / name1)
        matched0, matched1, scores = get_dense_pair_matches(matches_path, name0, name1)

        if len(scores) > max_plot_matches:
            order = np.argsort(scores)[::-1][:max_plot_matches]
            matched0, matched1, scores = matched0[order], matched1[order], scores[order]

        plot_images(
            [image0, image1],
            titles=[Path(name0).name, Path(name1).name],
            dpi=dpi,
        )
        if len(scores) > 0:
            plot_matches(
                matched0,
                matched1,
                color=_scores_to_colors(scores),
                lw=0.8,
                ps=2,
                a=0.5,
            )

        add_text(
            0,
            f"valid matches: {stat['valid_matches']} | shown: {len(scores)} | mean score: {stat['mean_score']:.3f}",
        )
        opts = dict(pos=(0.01, 0.01), fs=6, lcolor=None, va="bottom")
        add_text(0, name0, **opts)
        add_text(1, name1, **opts)

        if output_dir is not None:
            stem0 = Path(name0).stem.replace("/", "_")
            stem1 = Path(name1).stem.replace("/", "_")
            filename = f"{rank:02d}_{stem0}__{stem1}.png"
            save_plot(output_dir / filename)
        if close:
            plt.close(plt.gcf())


def visualize_feature_matches(
    feature_path_q,
    matches_path,
    pairs_path,
    image_dir,
    output_dir=None,
    num_pairs=6,
    max_plot_matches=300,
    dpi=100,
    close=False,
    feature_path_r=None,
):
    pair_stats, _ = summarize_match_assignments(matches_path, pairs_path)
    selected = _select_representative_pairs(pair_stats, num_pairs)
    feature_path_q = Path(feature_path_q)
    feature_path_r = feature_path_q if feature_path_r is None else Path(feature_path_r)
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for rank, stat in enumerate(selected):
        name0, name1 = stat["pair"]
        image0 = read_image(Path(image_dir) / name0)
        image1 = read_image(Path(image_dir) / name1)
        matched0, matched1, scores = get_feature_pair_matches(
            feature_path_q,
            feature_path_r,
            matches_path,
            name0,
            name1,
        )

        if len(scores) > max_plot_matches:
            order = np.argsort(scores)[::-1][:max_plot_matches]
            matched0, matched1, scores = matched0[order], matched1[order], scores[order]

        plot_images(
            [image0, image1],
            titles=[Path(name0).name, Path(name1).name],
            dpi=dpi,
        )
        if len(scores) > 0:
            plot_matches(
                matched0,
                matched1,
                color=_scores_to_colors(scores),
                lw=0.8,
                ps=2,
                a=0.5,
            )

        add_text(
            0,
            f"valid matches: {stat['valid_matches']} | shown: {len(scores)} | mean score: {stat['mean_score']:.3f}",
        )
        opts = dict(pos=(0.01, 0.01), fs=6, lcolor=None, va="bottom")
        add_text(0, name0, **opts)
        add_text(1, name1, **opts)

        if output_dir is not None:
            stem0 = Path(name0).stem.replace("/", "_")
            stem1 = Path(name1).stem.replace("/", "_")
            filename = f"{rank:02d}_{stem0}__{stem1}.png"
            save_plot(output_dir / filename)
        if close:
            plt.close(plt.gcf())
