from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_opus_13_001_9",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
neighbors4 = _ARC_UTILS.neighbors4

BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DIR_SEQUENCE = ("U", "D", "L", "R")
MAX_HALF_SIZE = 11
MAX_GENERATION_ATTEMPTS = 400


def _bbox_area(comp: dict) -> int:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _shape_components(grid: np.ndarray) -> list[dict]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BLACK, RED)]
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=colors,
        return_format="dict",
    )
    comps = [comp for comp in comps if int(comp["color"]) not in (BLACK, RED)]
    comps.sort(
        key=lambda comp: (
            _bbox_area(comp),
            int(comp["area"]),
            -int(comp["bbox"][0]),
            -int(comp["bbox"][1]),
        ),
        reverse=True,
    )
    return comps


def _ring_summary(colors: list[int]) -> np.ndarray:
    k = len(colors)
    size = 2 * k - 1
    out = np.zeros((size, size), dtype=int)
    for layer, color in enumerate(colors):
        out[layer : size - layer, layer : size - layer] = int(color)
    return out


def _ray_dirs(grid: np.ndarray, marker: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, marker)
    h, w = grid.shape
    dirs: list[tuple[int, int]] = []
    for name in DIR_SEQUENCE:
        dr, dc = DIRS[name]
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) not in (BLACK, RED):
            dirs.append((dr, dc))
    return dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress nested colored shapes into concentric square layers, then reveal only the layers struck by center-outward rays.

    Input structure:
    - The grid has black background plus 3-5 monochromatic 4-connected shapes nested from large to small.
    - Each shape has a distinct non-black, non-red color and a unique bounding-box area.
    - The smallest shape surrounds a single red marker pixel placed at its bounding-box center.
    - Among the red marker's four orthogonal neighbors, exactly 1-3 positions are occupied by the smallest shape; the occupied directions encode ray directions.

    Rule:
    - Detect every non-black, non-red connected component and sort them by bounding-box area from largest to smallest.
    - Build a (2K-1) x (2K-1) concentric-square summary where the largest shape's color is the outermost layer, the next-largest is the next layer, and the smallest shape's color is the center pixel.
    - Read the red marker's occupied orthogonal neighbors and cast straight cardinal rays from the summary center along those directions.
    - Keep only the summary pixels touched by at least one ray, preserving each touched pixel's layer color; set all untouched pixels to black. The center pixel is always kept.

    Key insight:
    - The output combines a global ordering cue and a local control cue: bounding-box areas decide which color belongs to each square layer, while the red marker's neighboring occupied directions decide which of those layers remain visible.
    """
    g = np.asarray(grid, dtype=int)
    markers = np.argwhere(g == RED)
    assert len(markers) == 1, "expected exactly one red marker"
    marker = tuple(map(int, markers[0]))

    comps = _shape_components(g)
    assert comps, "expected nested colored components"
    areas = [_bbox_area(comp) for comp in comps]
    assert len(set(areas)) == len(areas), "bounding-box areas must be unique"

    colors = [int(comp["color"]) for comp in comps]
    base = _ring_summary(colors)
    out = np.zeros_like(base)
    center = len(colors) - 1

    out[center, center] = int(colors[-1])
    dirs = _ray_dirs(g, marker)
    assert 1 <= len(dirs) <= 3, "expected 1-3 ray directions"
    for dr, dc in dirs:
        r, c = center, center
        while 0 <= r < base.shape[0] and 0 <= c < base.shape[1]:
            out[r, c] = int(base[r, c])
            r += dr
            c += dc

    return out


def _sample_sizes(k: int) -> list[tuple[int, int]]:
    inner_limit = MAX_HALF_SIZE - 2 * (k - 1)
    inner_hr = random.randint(2, inner_limit)
    inner_hc = random.randint(2, inner_limit)
    sizes = [(inner_hr, inner_hc)]
    for layer in range(1, k):
        remaining_after = k - 1 - layer
        prev_hr, prev_hc = sizes[-1]
        hr = random.randint(prev_hr + 2, MAX_HALF_SIZE - 2 * remaining_after)
        hc = random.randint(prev_hc + 2, MAX_HALF_SIZE - 2 * remaining_after)
        sizes.append((hr, hc))
    return sizes


def _add_inward_bar(mask: np.ndarray, side: str, limit: int) -> None:
    h, w = mask.shape
    if limit <= 0:
        return

    if side == "U" and w >= 5:
        col_choices = [c for c in range(1, w - 1) if c != w // 2]
        if not col_choices:
            return
        c = random.choice(col_choices)
        depth = random.randint(1, limit)
        mask[: depth + 1, c] = True
    elif side == "D" and w >= 5:
        col_choices = [c for c in range(1, w - 1) if c != w // 2]
        if not col_choices:
            return
        c = random.choice(col_choices)
        depth = random.randint(1, limit)
        mask[h - depth - 1 :, c] = True
    elif side == "L" and h >= 5:
        row_choices = [r for r in range(1, h - 1) if r != h // 2]
        if not row_choices:
            return
        r = random.choice(row_choices)
        depth = random.randint(1, limit)
        mask[r, : depth + 1] = True
    elif side == "R" and h >= 5:
        row_choices = [r for r in range(1, h - 1) if r != h // 2]
        if not row_choices:
            return
        r = random.choice(row_choices)
        depth = random.randint(1, limit)
        mask[r, w - depth - 1 :] = True


def _inner_mask(hr: int, hc: int, active_dirs: set[str]) -> np.ndarray:
    h, w = 2 * hr + 1, 2 * hc + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    if "U" in active_dirs:
        mask[:hr, hc] = True
    if "D" in active_dirs:
        mask[hr + 1 :, hc] = True
    if "L" in active_dirs:
        mask[hr, :hc] = True
    if "R" in active_dirs:
        mask[hr, hc + 1 :] = True

    extra_count = random.randint(0, 2)
    bar_limit = max(0, min(hr, hc) - 1)
    for _ in range(extra_count):
        _add_inward_bar(mask, random.choice(DIR_SEQUENCE), bar_limit)

    mask[hr, hc] = False
    return mask


def _outer_mask(hr: int, hc: int, inner_hr: int, inner_hc: int) -> np.ndarray:
    h, w = 2 * hr + 1, 2 * hc + 1
    max_top = hr - inner_hr - 1
    max_left = hc - inner_hc - 1
    if max_top < 1 or max_left < 1:
        raise ValueError("outer layer too small for moat")

    top = random.randint(1, max_top)
    bottom = random.randint(1, max_top)
    left = random.randint(1, max_left)
    right = random.randint(1, max_left)

    mask = np.zeros((h, w), dtype=bool)
    mask[:top, :] = True
    mask[h - bottom :, :] = True
    mask[:, :left] = True
    mask[:, w - right :] = True

    vertical_limit = max(0, max_top - 1)
    horizontal_limit = max(0, max_left - 1)
    for _ in range(random.randint(0, 2)):
        side = random.choice(DIR_SEQUENCE)
        limit = vertical_limit if side in ("U", "D") else horizontal_limit
        _add_inward_bar(mask, side, limit)

    return mask


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _attempt_grid() -> np.ndarray | None:
    k = random.randint(3, 5)
    colors = random.sample(PALETTE, k)
    direction_count = random.choices([1, 2, 3], weights=[3, 2, 3], k=1)[0]
    active_dirs = set(random.sample(DIR_SEQUENCE, direction_count))
    sizes = _sample_sizes(k)
    areas = [(2 * hr + 1) * (2 * hc + 1) for hr, hc in sizes]
    if len(set(areas)) != len(areas):
        return None

    outer_hr, outer_hc = sizes[-1]
    margin_top = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_right = random.randint(1, 2)
    height = 2 * outer_hr + 1 + margin_top + margin_bottom
    width = 2 * outer_hc + 1 + margin_left + margin_right
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    center_r = margin_top + outer_hr
    center_c = margin_left + outer_hc

    for idx in range(k - 1, -1, -1):
        hr, hc = sizes[idx]
        top = center_r - hr
        left = center_c - hc
        if idx == 0:
            mask = _inner_mask(hr, hc, active_dirs)
        else:
            inner_hr, inner_hc = sizes[idx - 1]
            mask = _outer_mask(hr, hc, inner_hr, inner_hc)
        _place_mask(grid, mask, top, left, colors[idx])

    grid[center_r, center_c] = RED

    try:
        comps = _shape_components(grid)
        if len(comps) != k:
            return None
        if len({int(comp["color"]) for comp in comps}) != k:
            return None

        smallest = comps[-1]
        r0, c0, r1, c1 = map(int, smallest["bbox"])
        if ((r0 + r1) // 2, (c0 + c1) // 2) != (center_r, center_c):
            return None

        dirs = _ray_dirs(grid, (center_r, center_c))
        if len(dirs) != direction_count:
            return None

        if int(get_bg(grid)) != BLACK:
            return None

        out = main(grid)
        if out.shape != (2 * k - 1, 2 * k - 1):
            return None
        if max(grid.shape) > 30 or max(out.shape) > 30:
            return None
        if int(np.count_nonzero(out)) <= k:
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-ray input")
