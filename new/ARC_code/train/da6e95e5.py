from itertools import product
from collections import defaultdict

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import random


QUADS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _record_components(g: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for idx, comp in enumerate(
        find_components(g, bg=bg, connectivity=8, monochromatic=True)
    ):
        r0, c0, r1, c1 = comp["bbox"]
        comps.append(
            {
                **comp,
                "id": idx,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "cr2": r0 + r1,
                "cc2": c0 + c1,
            }
        )
    return comps


def _bbox(comp: dict) -> tuple[int, int, int, int]:
    return comp["bbox"]


def _aligned_markers(markers: tuple[dict, dict, dict, dict]) -> bool:
    bnw, bne, bsw, bse = map(_bbox, markers)
    return (
        (bnw[0], bnw[2]) == (bne[0], bne[2])
        and (bsw[0], bsw[2]) == (bse[0], bse[2])
        and (bnw[1], bnw[3]) == (bsw[1], bsw[3])
        and (bne[1], bne[3]) == (bse[1], bse[3])
        and bnw[2] < bsw[0]
        and bne[2] < bse[0]
        and bnw[3] < bne[1]
        and bsw[3] < bse[1]
    )


def _is_cluster(center: dict, markers: tuple[dict, dict, dict, dict]) -> bool:
    nw, ne, sw, _ = markers
    if not _aligned_markers(markers):
        return False

    r0, c0, r1, c1 = _bbox(center)
    outer_r0 = min(_bbox(marker)[0] for marker in markers)
    outer_c0 = min(_bbox(marker)[1] for marker in markers)
    outer_r1 = max(_bbox(marker)[2] for marker in markers)
    outer_c1 = max(_bbox(marker)[3] for marker in markers)

    return (
        outer_r0 <= r0 <= r1 <= outer_r1
        and outer_c0 <= c0 <= c1 <= outer_c1
        and nw["cr2"] < center["cr2"] < sw["cr2"]
        and nw["cc2"] < center["cc2"] < ne["cc2"]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the central object from the marker cluster whose markers have the same color.

    Input structure
    The grid has a dominant background and several clusters; each cluster has four
    same-color corner marker objects around one central monochromatic object.
    Rule
    Find every central object enclosed by four aligned corner markers of one color,
    then choose the unique one whose object color equals the marker color.
    Key insight
    The answer is not the largest or most frequent object; it is the center of the
    self-colored cluster, cropped to that object's bounding box with background kept.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = _record_components(g, bg)

    targets = {}
    for center in comps:
        by_marker_role = defaultdict(lambda: defaultdict(list))
        for marker in comps:
            if marker["id"] == center["id"]:
                continue
            if marker["cr2"] == center["cr2"] or marker["cc2"] == center["cc2"]:
                continue
            quad = (
                0 if marker["cr2"] < center["cr2"] else 1,
                0 if marker["cc2"] < center["cc2"] else 1,
            )
            key = (marker["color"], marker["area"], marker["h"], marker["w"])
            by_marker_role[key][quad].append(marker)

        for key, markers_by_quad in by_marker_role.items():
            if not all(quad in markers_by_quad for quad in QUADS):
                continue
            for markers in product(*(markers_by_quad[quad] for quad in QUADS)):
                if len({marker["id"] for marker in markers}) != 4:
                    continue
                if _is_cluster(center, markers) and center["color"] == key[0]:
                    targets[center["id"]] = center

    assert len(targets) == 1
    target = next(iter(targets.values()))
    r0, c0, r1, c1 = _bbox(target)
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all((out == bg) | (out == target["color"]))
    return out

# --------------- generator (att2) ---------------

# Canvas, palette, and placement.
GRID_SHAPE = (30, 30)
COLOR_COUNT = 10
PLACEMENT_BORDER = 1

# Cluster structure and shared center bbox ranges.
MARKER_SIZES = (2, 3)
CENTER_BOUNDS_BY_MARKER = {
    2: ((2, 4), (2, 5)),
    3: ((3, 4), (3, 5)),
}
CLUSTER_COUNT_RANGE = (4, 6)
CENTER_OFFSET = 2

# Center sprite area and retry bounds.
MIN_CENTER_AREA = 3
MAX_CENTER_AREA = 14
GEN_ATTEMPTS = 300


def _gen_l_marker(marker_size: int, color: int, bg: int) -> np.ndarray:
    marker = np.full((marker_size, marker_size), bg, dtype=int)
    marker[0, :] = color
    marker[:, 0] = color
    return marker


def _gen_center_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if random.randrange(2):
        r, c, end_r, end_c = 0, 0, height - 1, width - 1
    else:
        r, c, end_r, end_c = 0, width - 1, height - 1, 0

    mask[r, c] = True
    while (r, c) != (end_r, end_c):
        if r < end_r:
            r += 1
        elif r > end_r:
            r -= 1
        if c < end_c:
            c += 1
        elif c > end_c:
            c -= 1
        mask[r, c] = True

    while int(mask.sum()) < area:
        candidates = set()
        for rr, cc in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = int(rr) + dr, int(cc) + dc
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        candidates.add((nr, nc))
        nr, nc = random.choice(tuple(candidates))
        mask[nr, nc] = True
    return mask


def _gen_center_patch(height: int, width: int, color: int, bg: int) -> np.ndarray:
    min_area = max(MIN_CENTER_AREA, height, width)
    max_area = min(MAX_CENTER_AREA, height * width - 1)
    area = random.randint(min_area, max_area)
    patch = np.full((height, width), bg, dtype=int)
    patch[_gen_center_mask(height, width, area)] = color
    return patch


def _gen_cluster_patch(
    marker_size: int,
    center_height: int,
    center_width: int,
    marker_color: int,
    center_color: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    overlap = marker_size - 2
    center_r0 = CENTER_OFFSET
    center_c0 = CENTER_OFFSET
    bottom_r0 = center_r0 + center_height - overlap
    right_c0 = center_c0 + center_width - overlap
    patch = np.full((bottom_r0 + marker_size, right_c0 + marker_size), bg, dtype=int)

    marker = _gen_l_marker(marker_size, marker_color, bg)
    paste(patch, marker, 0, 0, transparent=bg)
    paste(patch, np.fliplr(marker), 0, right_c0, transparent=bg)
    paste(patch, np.flipud(marker), bottom_r0, 0, transparent=bg)
    paste(patch, np.flipud(np.fliplr(marker)), bottom_r0, right_c0, transparent=bg)

    center = _gen_center_patch(center_height, center_width, center_color, bg)
    paste(patch, center, center_r0, center_c0, transparent=bg)
    return patch, center


def _place_cluster_patches(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    grid = np.full(GRID_SHAPE, bg, dtype=int)
    occupied = np.zeros(GRID_SHAPE, dtype=int)
    n_clusters = len(patches)
    order = list(range(n_clusters))
    random.shuffle(order)
    try:
        for idx in order:
            patch = patches[idx]
            h, w = patch.shape
            rect = np.ones((h, w), dtype=int)
            r0, c0 = random_free_location(
                occupied, rect, transparent=0, border=PLACEMENT_BORDER
            )
            occupied[r0 : r0 + h, c0 : c0 + w] = 1
            paste(grid, patch, r0, c0, transparent=bg)
    except ValueError:
        return None
    return grid


def _solves_to_expected(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(COLOR_COUNT)
    marker_size = random.choice(MARKER_SIZES)
    height_bounds, width_bounds = CENTER_BOUNDS_BY_MARKER[marker_size]
    center_height = random.randint(height_bounds[0], height_bounds[1])
    center_width = random.randint(width_bounds[0], width_bounds[1])

    n_clusters = random.randint(CLUSTER_COUNT_RANGE[0], CLUSTER_COUNT_RANGE[1])
    palette = [color for color in range(COLOR_COUNT) if color != bg]
    colors = random.sample(palette, n_clusters + 1)
    marker_colors = colors[:n_clusters]
    extra_center_color = colors[-1]
    target_idx = random.randrange(n_clusters)
    extra_idx = random.choice([idx for idx in range(n_clusters) if idx != target_idx])

    patches = []
    expected = None
    for idx, marker_color in enumerate(marker_colors):
        if idx == target_idx:
            center_color = marker_color
        elif idx == extra_idx:
            center_color = extra_center_color
        else:
            center_color = random.choice(
                [extra_center_color]
                + [color for color in marker_colors if color != marker_color]
            )
        patch, center = _gen_cluster_patch(
            marker_size,
            center_height,
            center_width,
            marker_color,
            center_color,
            bg,
        )
        patches.append(patch)
        if idx == target_idx:
            expected = center

    grid = _place_cluster_patches(patches, bg)
    if expected is not None and grid is not None and _solves_to_expected(grid, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid da6e95e5 input")
