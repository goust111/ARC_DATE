from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _cross_cells(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each same-color five-pixel cluster into a centered cross at its coordinate-wise median.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    out = np.full((height, width), bg, dtype=int)
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 5, "each foreground color must occur exactly 5 times"

        center_r = int(np.sort(coords[:, 0])[2])
        center_c = int(np.sort(coords[:, 1])[2])
        assert 0 < center_r < height - 1, "cross center must admit vertical arms"
        assert 0 < center_c < width - 1, "cross center must admit horizontal arms"

        for r, c in _cross_cells((center_r, center_c)):
            assert 0 <= r < height and 0 <= c < width
            assert int(out[r, c]) in {bg, color}, "different color crosses may not overlap"
            out[r, c] = color

    return out


# --------------- generator ---------------

GROUP_COUNT_MIN = 2
GROUP_COUNT_MAX = 5
GRID_SIDE_MAX = 30
GRID_SIDE_BASE = 10
GRID_SIDE_PER_GROUP = 3
CENTER_MARGIN = 3
CENTER_CLEARANCE = 8
OFFSET_RETRY_LIMIT = 200
GENERATION_RETRY_LIMIT = 400
SCATTER_MODES = ("compact", "wide", "diagonal", "axis")
PLUS_OFFSETS = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}


def _sample_centers(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(CENTER_MARGIN, height - CENTER_MARGIN)
        for c in range(CENTER_MARGIN, width - CENTER_MARGIN)
    ]
    random.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for r, c in candidates:
        conflict = False
        for rr, cc in centers:
            if abs(r - rr) < CENTER_CLEARANCE and abs(c - cc) < CENTER_CLEARANCE:
                conflict = True
                break
        if conflict:
            continue
        centers.append((r, c))
        if len(centers) == count:
            return centers
    return None


def _sample_axis_values(reach: int) -> list[int] | None:
    values = list(range(-reach, reach + 1))
    for _ in range(OFFSET_RETRY_LIMIT):
        chosen = [random.choice(values) for _ in range(5)]
        chosen_sorted = sorted(chosen)
        if chosen_sorted[2] != 0:
            continue
        if chosen_sorted[0] == 0 or chosen_sorted[-1] == 0:
            continue
        if chosen_sorted[-1] - chosen_sorted[0] < 2:
            continue
        return chosen
    return None


def _sample_offsets(
    mode: str,
    forbidden_cells: set[tuple[int, int]],
    center: tuple[int, int],
    used_input: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    reach = 2 if mode == "compact" else random.choice((2, 3))
    if mode == "wide":
        reach = 3

    require_missing_center = random.random() < 0.5

    for _ in range(OFFSET_RETRY_LIMIT):
        rows = _sample_axis_values(reach)
        cols = _sample_axis_values(reach)
        if rows is None or cols is None:
            return None

        random.shuffle(cols)
        offsets = list(zip(rows, cols))
        if len(set(offsets)) != 5:
            continue
        if set(offsets) == PLUS_OFFSETS:
            continue
        if require_missing_center and (0, 0) in offsets:
            continue

        cross_hits = sum((dr, dc) in PLUS_OFFSETS for dr, dc in offsets)
        if cross_hits > 2:
            continue

        bbox_h = max(dr for dr, _ in offsets) - min(dr for dr, _ in offsets) + 1
        bbox_w = max(dc for _, dc in offsets) - min(dc for _, dc in offsets) + 1
        if bbox_h < 3 or bbox_w < 3:
            continue

        radii = [max(abs(dr), abs(dc)) for dr, dc in offsets]
        if max(radii) < 2:
            continue

        if mode == "compact" and max(radii) > 2:
            continue
        if mode == "wide" and 3 not in radii:
            continue
        if mode == "diagonal" and sum(dr != 0 and dc != 0 for dr, dc in offsets) < 3:
            continue
        if mode == "axis" and sum(dr == 0 or dc == 0 for dr, dc in offsets) < 3:
            continue

        points = tuple((center[0] + dr, center[1] + dc) for dr, dc in offsets)
        if any(point in used_input or point in forbidden_cells for point in points):
            continue
        return points

    return None


def _build_expected_output(
    shape: tuple[int, int],
    bg: int,
    groups: list[tuple[int, tuple[int, int]]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color, center in groups:
        for r, c in _cross_cells(center):
            assert int(out[r, c]) in {bg, color}
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(GROUP_COUNT_MIN, GROUP_COUNT_MAX)
    min_side = GRID_SIDE_BASE + GRID_SIDE_PER_GROUP * group_count
    if min_side > GRID_SIDE_MAX:
        return None

    height = random.randint(min_side, GRID_SIDE_MAX)
    width = random.randint(min_side, GRID_SIDE_MAX)
    bg = 0
    colors = random.sample(range(1, 10), group_count)
    centers = _sample_centers(height, width, group_count)
    if centers is None:
        return None

    cross_by_center = {center: set(_cross_cells(center)) for center in centers}
    all_crosses = set().union(*cross_by_center.values())
    expected = _build_expected_output((height, width), bg, list(zip(colors, centers)))

    inp = np.full((height, width), bg, dtype=int)
    used_input: set[tuple[int, int]] = set()
    for color, center in zip(colors, centers):
        mode = random.choice(SCATTER_MODES)
        forbidden = all_crosses - cross_by_center[center]
        points = _sample_offsets(mode, forbidden, center, used_input)
        if points is None:
            return None

        for r, c in points:
            inp[r, c] = color
            used_input.add((r, c))

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid median-cross input")
