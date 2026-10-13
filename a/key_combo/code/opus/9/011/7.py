import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components as (N, 2) arrays in (r, c) coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    yellow_coords = np.argwhere(g == YELLOW)
    assert yellow_coords.size > 0, "expected yellow guide pixels"
    guide_box = bbox_coords(yellow_coords)

    bar_components = components_mask(g == GREY)
    assert len(bar_components) >= 3, "expected at least three grey bars"

    bars = []
    for coords in bar_components:
        r0, c0, r1, c1 = bbox_coords(coords)
        assert r0 == 0, "grey bars must start from the top edge"
        assert c0 == c1, "grey objects must be width-1 vertical bars"
        assert coords.shape[0] == r1 - r0 + 1, "grey bars must be solid"
        bars.append(
            {
                "bbox": (r0, c0, r1, c1),
                "col": int(c0),
                "height": int(r1 - r0 + 1),
            }
        )
    bars.sort(key=lambda bar: bar["col"])

    bar_cols = {bar["col"] for bar in bars}
    assert all(int(c) not in bar_cols for _r, c in yellow_coords), "guide pixels may not share a bar column"

    enclosed = [idx for idx, bar in enumerate(bars) if _bbox_contains(guide_box, bar["bbox"])]
    assert len(enclosed) == 1, "guide bbox must enclose exactly one grey bar"
    seed_idx = enclosed[0]
    assert 0 < seed_idx < len(bars) - 1, "seed bar must have walls on both sides"

    seed = bars[seed_idx]
    left_wall = bars[seed_idx - 1]
    right_wall = bars[seed_idx + 1]
    assert left_wall["col"] < seed["col"] < right_wall["col"]

    return {
        "grid": g,
        "guide_box": guide_box,
        "bars": bars,
        "seed_idx": int(seed_idx),
        "seed": seed,
        "left_wall": left_wall,
        "right_wall": right_wall,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the uniquely guide-boxed grey bar and extend asymmetric top-down fills until side walls stop them.

    Input structure:
    - The background is black.
    - Several grey 1-pixel-wide vertical bars start at the top edge and have different heights.
    - Yellow guide pixels collectively define one bounding box, and that box contains exactly one grey bar.

    Rule:
    - Compute the bounding box of all yellow guide pixels and find the unique grey bar fully contained in it.
    - Let that seed bar have height h and column c0. For each column d steps left of c0, fill teal from row 0 down to min(h + 2d, H) until the nearest grey bar on that side; do not color the wall column or go past it.
    - For each column d steps right of c0, fill pink from row 0 down to max(h - 2d, 0) until either the nearest grey bar is reached or the height drops to 0; again, do not color the wall column.
    - Keep every grey bar and every yellow guide pixel unchanged by redrawing them above the new fills.

    Key insight:
    - The guide bbox identifies the seed, while the neighboring grey bars act as hard stopping walls, so both the seed height and obstacle positions are necessary to determine the final shape.
    """
    info = _parse_instance(grid)
    g = info["grid"]
    h, _w = g.shape
    seed = info["seed"]
    seed_col = seed["col"]
    seed_height = seed["height"]
    left_wall_col = info["left_wall"]["col"]
    right_wall_col = info["right_wall"]["col"]

    out = g.copy()

    for c in range(seed_col - 1, left_wall_col, -1):
        d = seed_col - c
        fill_h = min(seed_height + 2 * d, h)
        out[:fill_h, c] = TEAL

    for c in range(seed_col + 1, right_wall_col):
        d = c - seed_col
        fill_h = seed_height - 2 * d
        if fill_h <= 0:
            break
        out[:fill_h, c] = PINK

    out[g == GREY] = GREY
    out[g == YELLOW] = YELLOW
    return out


GRID_H_RANGE = (15, 20)
BAR_COUNT_RANGE = (4, 6)
BAR_HEIGHT_RANGE = (3, 10)
GRID_W_MAX = 28
BAR_DELTA_RANGE = (4, 6)  # column distance between consecutive bars
MARGIN_RANGE = (1, 3)
GUIDE_EXTRA_COUNT_RANGE = (1, 6)
GEN_ATTEMPTS = 400


def _build_columns(n_bars: int, seed_idx: int, seed_height: int, right_mode: str) -> tuple[int, list[int]] | None:
    for _ in range(120):
        left_margin = random.randint(*MARGIN_RANGE)
        right_margin = random.randint(*MARGIN_RANGE)
        deltas = [random.randint(*BAR_DELTA_RANGE) for _ in range(n_bars - 1)]
        deltas[seed_idx - 1] = random.randint(4, 6)

        if right_mode == "wall_first":
            deltas[seed_idx] = 4
            if seed_height < 9:
                return None
        else:
            natural_stop_d = (seed_height + 1) // 2
            deltas[seed_idx] = random.randint(max(4, natural_stop_d + 1), 6)

        width = left_margin + 1 + sum(deltas) + right_margin
        if width > GRID_W_MAX:
            continue

        cols = [left_margin]
        for delta in deltas:
            cols.append(cols[-1] + delta)
        return width, cols
    return None


def _guide_pixels(
    shape: tuple[int, int],
    bar_cols: set[int],
    seed_col: int,
    seed_height: int,
    left_wall_col: int,
    right_wall_col: int,
) -> list[tuple[int, int]]:
    h, _w = shape
    left_extent = random.randint(1, seed_col - left_wall_col - 1)
    right_extent = random.randint(1, right_wall_col - seed_col - 1)
    left = seed_col - left_extent
    right = seed_col + right_extent
    bottom = random.randint(seed_height - 1, h - 1)

    points = {
        (0, left),
        (0, right),
        (bottom, left),
        (bottom, right),
    }

    candidates = []
    for r in range(0, bottom + 1):
        for c in range(left, right + 1):
            if c in bar_cols:
                continue
            candidates.append((r, c))

    extra_count = random.randint(*GUIDE_EXTRA_COUNT_RANGE)
    random.shuffle(candidates)
    for pt in candidates:
        points.add(pt)
        if len(points) >= 4 + extra_count:
            break

    guide_box = bbox_coords(np.array(sorted(points), dtype=int))
    assert guide_box[0] == 0
    assert guide_box[1] <= seed_col <= guide_box[3]
    return sorted(points)


def _construct_example() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    n_bars = random.randint(*BAR_COUNT_RANGE)
    seed_idx = random.randint(1, n_bars - 2)
    right_mode = random.choice(["wall_first", "decay_first"])

    if right_mode == "wall_first":
        seed_height = random.randint(max(9, BAR_HEIGHT_RANGE[0]), min(BAR_HEIGHT_RANGE[1], h))
    else:
        seed_height = random.randint(BAR_HEIGHT_RANGE[0], min(BAR_HEIGHT_RANGE[1], h))

    layout = _build_columns(n_bars, seed_idx, seed_height, right_mode)
    if layout is None:
        return None
    w, cols = layout

    bar_heights = [random.randint(BAR_HEIGHT_RANGE[0], min(BAR_HEIGHT_RANGE[1], h)) for _ in range(n_bars)]
    bar_heights[seed_idx] = seed_height
    if len(set(bar_heights)) == 1:
        tweak = 1 if seed_height > BAR_HEIGHT_RANGE[0] else 2
        bar_heights[0] = max(BAR_HEIGHT_RANGE[0], seed_height - tweak)

    grid = np.zeros((h, w), dtype=int)
    for c, bar_h in zip(cols, bar_heights, strict=True):
        grid[:bar_h, c] = GREY

    seed_col = cols[seed_idx]
    left_wall_col = cols[seed_idx - 1]
    right_wall_col = cols[seed_idx + 1]
    guide_pts = _guide_pixels((h, w), set(cols), seed_col, seed_height, left_wall_col, right_wall_col)
    for r, c in guide_pts:
        grid[r, c] = YELLOW

    return grid


def _expected_semantic_output(grid: np.ndarray) -> np.ndarray:
    info = _parse_instance(grid)
    g = info["grid"]
    h, w = g.shape
    seed_col = info["seed"]["col"]
    seed_height = info["seed"]["height"]
    left_wall_col = info["left_wall"]["col"]
    right_wall_col = info["right_wall"]["col"]

    expected = g.copy()
    for r in range(h):
        for c in range(w):
            if c == seed_col or c == left_wall_col or c == right_wall_col:
                continue
            if left_wall_col < c < seed_col:
                d = seed_col - c
                if r < min(seed_height + 2 * d, h):
                    expected[r, c] = TEAL
            elif seed_col < c < right_wall_col:
                d = c - seed_col
                if r < max(seed_height - 2 * d, 0):
                    expected[r, c] = PINK
    expected[g == GREY] = GREY
    expected[g == YELLOW] = YELLOW
    return expected


def _validate_semantics(grid: np.ndarray, out: np.ndarray) -> None:
    info = _parse_instance(grid)
    seed_col = info["seed"]["col"]
    seed_height = info["seed"]["height"]
    left_wall_col = info["left_wall"]["col"]
    right_wall_col = info["right_wall"]["col"]

    expected = _expected_semantic_output(grid)
    assert np.array_equal(out, expected), "output must match the described extension and wall-truncation rule"
    assert np.array_equal(out[grid == GREY], grid[grid == GREY]), "grey bars must be preserved"
    assert np.array_equal(out[grid == YELLOW], grid[grid == YELLOW]), "yellow guides must be preserved"
    assert np.any(out == TEAL), "left extension must produce teal cells"
    assert seed_height - 2 > 0, "seed height must allow at least one right extension column"

    max_right_d = right_wall_col - seed_col - 1
    positive_ds = [d for d in range(1, max_right_d + 1) if seed_height - 2 * d > 0]
    assert positive_ds, "there must be at least one positive right-side extension distance"

    if seed_height - 2 * (right_wall_col - seed_col) > 0:
        assert positive_ds[-1] == max_right_d, "the right wall should truncate an otherwise positive extension"
    else:
        assert positive_ds[-1] == (seed_height - 1) // 2, "natural right stop should come from the height-decay formula"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _construct_example()
        if grid is None:
            continue
        out = main(grid)
        _validate_semantics(grid, out)
        return grid
    raise RuntimeError("failed to generate a valid guided-bar extension puzzle")
