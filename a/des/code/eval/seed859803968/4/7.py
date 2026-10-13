from __future__ import annotations

from collections import Counter

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    corner_counts = Counter(corners).most_common(1)
    if corner_counts and corner_counts[0][1] >= 2:
        return int(corner_counts[0][0])
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(mask) - 1)))
    return runs


def _extract_container(grid: np.ndarray, bg: int, c0: int, c1: int) -> dict:
    cols = grid[:, c0 : c1 + 1]
    row_mask = np.any(cols != bg, axis=1)
    rows = np.flatnonzero(row_mask)
    assert rows.size > 0, "container columns must contain foreground pixels"

    r0, r1 = int(rows[0]), int(rows[-1])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = map(int, patch.shape)
    assert height >= 3 and width >= 4, "container must be a visible U shape"

    left = patch[:, 0]
    right = patch[:, -1]
    bottom = patch[-1, :]
    wall_candidates = (set(map(int, left)) & set(map(int, right)) & set(map(int, bottom))) - {int(bg)}
    assert wall_candidates, "failed to identify container wall color"
    wall = int(sorted(wall_candidates)[0])

    assert bool(np.all(left == wall))
    assert bool(np.all(right == wall))
    assert bool(np.all(bottom == wall))

    inner = patch[:, 1:-1]
    liquid_mask = (inner != bg) & (inner != wall)
    liquid_rows = np.flatnonzero(np.any(liquid_mask, axis=1))
    assert liquid_rows.size > 0, "container must contain liquid"

    top_liquid = int(liquid_rows[0])
    liquid_height = int(liquid_rows.size)
    expected_rows = np.arange(top_liquid, top_liquid + liquid_height)
    assert np.array_equal(liquid_rows, expected_rows), "liquid rows must be contiguous"

    liquid_colors = set(map(int, inner[liquid_mask].reshape(-1)))
    assert len(liquid_colors) == 1, "each container must use one liquid color"
    liquid_color = int(next(iter(liquid_colors)))

    assert liquid_rows[-1] == height - 2, "liquid must rest on the container floor"
    assert bool(np.all(inner[top_liquid : height - 1] == liquid_color)), "liquid must fill a rectangle"
    if top_liquid > 0:
        assert bool(np.all(inner[:top_liquid] == bg)), "space above the liquid must be empty"

    crop = patch[top_liquid:, :].copy()
    return {
        "c0": int(c0),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "wall": wall,
        "liquid_color": liquid_color,
        "liquid_height": liquid_height,
        "crop": crop,
    }


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    non_bg_cols = np.any(g != bg, axis=0)
    groups = _runs(non_bg_cols)
    assert groups, "expected at least one container"

    containers = [_extract_container(g, bg, c0, c1) for c0, c1 in groups]
    containers.sort(key=lambda item: int(item["liquid_height"]))

    out_h = max(int(item["crop"].shape[0]) for item in containers)
    out_w = sum(int(item["crop"].shape[1]) for item in containers) + len(containers) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for item in containers:
        crop = item["crop"]
        h, w = map(int, crop.shape)
        out[out_h - h :, col : col + w] = crop
        col += w + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each U-shaped container down to its liquid level, then reorder the cropped
    slices by liquid height from low to high while keeping ties left-to-right.
    """
    return _transform(grid)


_RNG = np.random.default_rng()

BG = 8
WALL = 5
LIQUID_COLORS = (1, 2, 3, 4, 6, 7, 9)

GRID_MAX = 30
GEN_ATTEMPTS = 300
COUNT_RANGE = (3, 5)
HEIGHT_RANGE = (4, 11)
TOP_MARGIN_RANGE = (1, 4)
BOTTOM_MARGIN_RANGE = (1, 3)
SIDE_MARGIN_RANGE = (1, 3)
GAP_RANGE = (1, 3)
FILL_TO_TOP_PROB = 0.35
FORCE_TIE_PROB = 0.5


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _build_container(height: int, width: int, liquid_height: int, liquid_color: int) -> np.ndarray:
    container = np.full((height, width), BG, dtype=int)
    container[:, 0] = WALL
    container[:, -1] = WALL
    container[-1, :] = WALL
    container[height - 1 - liquid_height : height - 1, 1:-1] = int(liquid_color)
    return container


def _expected_output(width: int, containers: list[dict]) -> np.ndarray:
    ordered = sorted(
        enumerate(containers),
        key=lambda pair: (int(pair[1]["liquid_height"]), int(pair[0])),
    )
    crops = []
    for _, item in ordered:
        full = _build_container(
            int(item["height"]),
            int(width),
            int(item["liquid_height"]),
            int(item["liquid_color"]),
        )
        top = int(item["height"]) - 1 - int(item["liquid_height"])
        crops.append(full[top:, :])

    out_h = max(crop.shape[0] for crop in crops)
    out_w = len(crops) * width + len(crops) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for crop in crops:
        h = int(crop.shape[0])
        out[out_h - h :, col : col + width] = crop
        col += width + 1
    return out


def _sample_liquid_heights(heights: list[int], force_tie: bool) -> list[int] | None:
    n = len(heights)
    for _ in range(200):
        values = [_randint(1, h - 1) for h in heights]

        if bool(_RNG.random() < FILL_TO_TOP_PROB):
            idx = _randint(0, n - 1)
            values[idx] = heights[idx] - 1

        if force_tie:
            i, j = map(int, _RNG.choice(n, size=2, replace=False))
            common = _randint(1, min(heights[i], heights[j]) - 1)
            values[i] = common
            values[j] = common
            if len(set(values)) == n:
                continue
        else:
            if len(set(values)) != n:
                continue

        if all(values[k] <= values[k + 1] for k in range(n - 1)):
            continue
        return values
    return None


def _generate_attempt() -> np.ndarray | None:
    count = _randint(*COUNT_RANGE)
    width_ranges = {
        3: (4, 7),
        4: (4, 6),
        5: (4, 4),
    }
    width_lo, width_hi = width_ranges[count]
    width = _randint(width_lo, width_hi)

    height_pool = list(range(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    heights = sorted(_RNG.choice(height_pool, size=count, replace=False).tolist())
    _RNG.shuffle(heights)

    liquid_heights = _sample_liquid_heights(heights, force_tie=bool(_RNG.random() < FORCE_TIE_PROB))
    if liquid_heights is None:
        return None

    colors = _RNG.choice(LIQUID_COLORS, size=count, replace=False).tolist()
    gaps = [_randint(*GAP_RANGE) for _ in range(count - 1)]
    left_margin = _randint(*SIDE_MARGIN_RANGE)
    right_margin = _randint(*SIDE_MARGIN_RANGE)
    top_margin = _randint(*TOP_MARGIN_RANGE)
    bottom_margin = _randint(*BOTTOM_MARGIN_RANGE)

    grid_h = top_margin + max(heights) + bottom_margin
    grid_w = left_margin + right_margin + count * width + sum(gaps)
    if grid_h > GRID_MAX or grid_w > GRID_MAX:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    base_row = grid_h - bottom_margin - 1

    containers: list[dict] = []
    col = left_margin
    for idx in range(count):
        height = int(heights[idx])
        liquid_height = int(liquid_heights[idx])
        liquid_color = int(colors[idx])

        top = base_row - height + 1
        if top < 0:
            return None

        patch = _build_container(height, width, liquid_height, liquid_color)
        grid[top : top + height, col : col + width] = patch
        containers.append(
            {
                "height": height,
                "liquid_height": liquid_height,
                "liquid_color": liquid_color,
                "left": col,
            }
        )
        col += width + (gaps[idx] if idx < len(gaps) else 0)

    expected = _expected_output(width, containers)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-sorting input")
