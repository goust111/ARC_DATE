from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_task_019_2",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location
scale_sprite = _ARC_UTILS.scale_sprite

BG = 0
INPUT_COLORS = (1, 2, 3)
AREA_SET = (2, 3, 4)
GEN_RETRIES = 300

SHAPES_BY_AREA = {
    2: (
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ),
    3: (
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ),
    4: (
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1], [1, 1], [0, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=int),
    ),
}


def _extract_clusters(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=INPUT_COLORS,
        return_format="dict",
    )
    assert len(comps) == 3

    seen_colors = set()
    infos = []
    for comp in comps:
        color = int(comp["color"])
        assert color in INPUT_COLORS
        assert color not in seen_colors
        seen_colors.add(color)
        patch = crop_bbox(grid, comp["bbox"])
        area = int(comp["area"])
        assert area in AREA_SET
        infos.append(
            {
                "color": color,
                "area": area,
                "bbox": comp["bbox"],
                "patch": patch,
            }
        )

    areas = sorted(info["area"] for info in infos)
    assert areas == list(AREA_SET)
    return infos


def _compose_output(
    infos: list[dict],
    *,
    area_order: str = "desc",
    anchor: str = "center",
    scale_mode: str = "self",
) -> np.ndarray:
    if scale_mode == "self":
        scales = {id(info): int(info["area"]) for info in infos}
    elif scale_mode == "global_max":
        k = max(int(info["area"]) for info in infos)
        scales = {id(info): k for info in infos}
    else:
        raise ValueError("unknown scale_mode")

    scaled_infos = []
    side = 0
    for info in infos:
        scaled = scale_sprite(info["patch"], scales[id(info)])
        scaled_infos.append({**info, "scaled": scaled})
        side = max(side, int(scaled.shape[0]), int(scaled.shape[1]))

    reverse = area_order == "desc"
    assert area_order in {"desc", "asc"}
    scaled_infos.sort(key=lambda item: int(item["area"]), reverse=reverse)

    out = np.full((side, side), BG, dtype=int)
    for info in scaled_infos:
        patch = info["scaled"]
        h, w = patch.shape
        if anchor == "center":
            r0 = (side - h) // 2
            c0 = (side - w) // 2
        elif anchor == "topleft":
            r0 = 0
            c0 = 0
        else:
            raise ValueError("unknown anchor")
        paste(out, patch, r0, c0, transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each colored cluster by its own pixel count, then center-overlay the enlarged results.

    Input structure:
    - The grid has black background and exactly three disconnected 4-connected monochrome clusters.
    - The three clusters use blue, red, and green, one component per color.
    - Their pixel counts are all different, specifically 2, 3, and 4, and each cluster is read inside its own tight bounding box.

    Rule:
    - For each non-black cluster, crop its bounding box and count its colored pixels N.
    - Scale that entire cropped patch by factor N, replicating both colored pixels and black bbox cells into N x N blocks.
    - Create a square black canvas whose side is the largest height or width among the scaled patches.
    - Draw the scaled patches center-aligned on that canvas from largest N down to smallest N, with later non-black pixels overwriting earlier ones.

    Key insight:
    - Each object self-counts to determine both its own enlargement factor and its depth in the final overlay, so counting cannot be separated from the stacking rule.
    """
    g = np.asarray(grid, dtype=int)
    infos = _extract_clusters(g)
    return _compose_output(infos, area_order="desc", anchor="center", scale_mode="self")


def _colorize(shape: np.ndarray, color: int) -> np.ndarray:
    return np.where(shape != 0, int(color), BG).astype(int)


def _sample_cluster_patches(rng: np.random.Generator) -> list[np.ndarray]:
    areas = list(AREA_SET)
    rng.shuffle(areas)
    patches = []
    for color, area in zip(INPUT_COLORS, areas):
        shape_bank = SHAPES_BY_AREA[int(area)]
        shape = np.array(shape_bank[int(rng.integers(len(shape_bank)))], dtype=int, copy=True)
        patches.append(_colorize(shape, color))
    rng.shuffle(patches)
    return patches


def _place_patches(
    patches: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray | None:
    H = int(rng.integers(13, 19))
    W = int(rng.integers(13, 19))
    grid = np.full((H, W), BG, dtype=int)

    order = list(range(len(patches)))
    rng.shuffle(order)
    for idx in order:
        patch = patches[idx]
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=BG,
                border=1,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, int(r0), int(c0), transparent=BG)
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    infos = _extract_clusters(inp)
    correct = _compose_output(infos, area_order="desc", anchor="center", scale_mode="self")

    if not np.array_equal(main(inp), correct):
        return False

    wrong_order = _compose_output(infos, area_order="asc", anchor="center", scale_mode="self")
    wrong_anchor = _compose_output(infos, area_order="desc", anchor="topleft", scale_mode="self")
    wrong_scale = _compose_output(infos, area_order="desc", anchor="center", scale_mode="global_max")

    if np.array_equal(correct, wrong_order):
        return False
    if np.array_equal(correct, wrong_anchor):
        return False
    if np.array_equal(correct, wrong_scale):
        return False

    counts = sorted(int(info["area"]) for info in infos)
    if counts != list(AREA_SET):
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    patches = _sample_cluster_patches(rng)
    inp = _place_patches(patches, rng)
    if inp is None:
        return None
    if not _semantic_checks(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid self-scaling overlay input")
