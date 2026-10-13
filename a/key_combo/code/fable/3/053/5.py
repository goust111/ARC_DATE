from __future__ import annotations

from itertools import combinations

import numpy as np


BACKGROUND = 0
PALETTE = np.arange(1, 10, dtype=int)
MAX_GENERATION_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _rectangle_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    coords = np.argwhere(grid != bg)
    point_set = {(int(r), int(c)) for r, c in coords}
    row_to_cols: dict[int, set[int]] = {}
    for r, c in point_set:
        row_to_cols.setdefault(r, set()).add(c)

    candidates: set[tuple[int, int, int, int]] = set()
    for r0, r1 in combinations(sorted(row_to_cols), 2):
        common_cols = sorted(row_to_cols[r0] & row_to_cols[r1])
        if len(common_cols) < 2:
            continue
        for c0, c1 in combinations(common_cols, 2):
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            if inner_h <= 0 or inner_w <= 0:
                continue
            if inner_h % 2 != 0 or inner_w % 2 != 0:
                continue
            if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
                continue
            if not np.all(grid[r0, c0 + 1 : c1] == bg):
                continue
            if not np.all(grid[r1, c0 + 1 : c1] == bg):
                continue
            if not np.all(grid[r0 + 1 : r1, c0] == bg):
                continue
            if not np.all(grid[r0 + 1 : r1, c1] == bg):
                continue
            candidates.add((r0, c0, r1, c1))
    return sorted(candidates)


def _paint_solution(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    r_mid = r0 + 1 + inner_h // 2
    c_mid = c0 + 1 + inner_w // 2

    tl = int(grid[r0, c0])
    tr = int(grid[r0, c1])
    bl = int(grid[r1, c0])
    br = int(grid[r1, c1])

    out = np.full_like(grid, bg)
    out[r0, c0] = tl
    out[r0, c1] = tr
    out[r1, c0] = bl
    out[r1, c1] = br

    out[r0 + 1 : r_mid, c0 + 1 : c_mid] = tl
    out[r0 + 1 : r_mid, c_mid:c1] = tr
    out[r_mid:r1, c0 + 1 : c_mid] = bl
    out[r_mid:r1, c_mid:c1] = br
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique four-corner singleton rectangle and fill its interior quadrants with the corner colors.

    Input structure:
    - The grid has a dominant black background plus several colored singleton pixels.
    - Exactly four of those pixels occupy two rows and two columns as the corners of one axis-aligned rectangle.
    - The rectangle's strict interior is empty and has even height and even width; all other pixels are noise.

    Rule:
    - Identify the unique set of four singleton corners that completes a full rectangle.
    - Erase every other foreground pixel.
    - Split the rectangle's strict interior into four equal quadrants and fill each quadrant with its nearest corner's color, while keeping the original four corner pixels.

    Key insight:
    - The real markers are the only four pixels that jointly certify themselves by occupying exactly two shared rows and two shared columns; once found, those same corners dictate the quadrant colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    candidates = _rectangle_candidates(g, bg)
    assert len(candidates) == 1, "expected exactly one qualifying corner rectangle"
    return _paint_solution(g, candidates[0], bg)


def _sample_spaced_values(
    rng: np.random.Generator,
    candidates: list[int],
    count: int,
    forbidden: set[int],
) -> list[int] | None:
    if count == 0:
        return []
    usable = [v for v in candidates if int(v) not in forbidden]
    if len(usable) < count:
        return None
    for _ in range(80):
        picks = sorted(int(v) for v in rng.choice(usable, size=count, replace=False))
        if all(b - a > 1 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _all_points_separated(points: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _choose_side_counts(rng: np.random.Generator, total_noise: int) -> dict[str, int]:
    row_sides = ["top", "bottom"]
    col_sides = ["left", "right"]
    min_active = max(2, (total_noise + 1) // 2)
    max_active = min(4, total_noise)

    for _ in range(80):
        active_count = int(rng.integers(min_active, max_active + 1))
        active = {
            row_sides[int(rng.integers(0, len(row_sides)))],
            col_sides[int(rng.integers(0, len(col_sides)))],
        }
        remaining = [side for side in ("top", "bottom", "left", "right") if side not in active]
        rng.shuffle(remaining)
        active.update(remaining[: active_count - len(active)])

        counts = {side: 0 for side in ("top", "bottom", "left", "right")}
        for side in active:
            counts[side] = 1

        extras = total_noise - active_count
        while extras > 0:
            expandable = [side for side in active if counts[side] < 2]
            if not expandable:
                break
            side = expandable[int(rng.integers(0, len(expandable)))]
            counts[side] += 1
            extras -= 1
        if extras == 0:
            return counts

    raise RuntimeError("failed to allocate side counts")


def _place_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    counts: dict[str, int],
) -> list[tuple[int, int]] | None:
    height, width = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)

    side_candidates = {
        "top": list(range(0, c0 - 1)) + list(range(c1 + 2, width)),
        "bottom": list(range(0, c0 - 1)) + list(range(c1 + 2, width)),
        "left": list(range(0, r0 - 1)) + list(range(r1 + 2, height)),
        "right": list(range(0, r0 - 1)) + list(range(r1 + 2, height)),
    }

    chosen: dict[str, list[int]] = {}
    for first, second in (("top", "bottom"), ("left", "right")):
        order = sorted((first, second), key=lambda side: counts[side], reverse=True)
        used: set[int] = set()
        for side in order:
            picks = _sample_spaced_values(rng, side_candidates[side], counts[side], used)
            if picks is None:
                return None
            chosen[side] = picks
            used.update(picks)

    noise = [(r0, c) for c in chosen["top"]]
    noise += [(r1, c) for c in chosen["bottom"]]
    noise += [(r, c0) for r in chosen["left"]]
    noise += [(r, c1) for r in chosen["right"]]
    return noise


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(12, 25))
    width = int(rng.integers(12, 25))

    inner_h_choices = list(range(2, min(12, height - 6) + 1, 2))
    inner_w_choices = list(range(2, min(12, width - 6) + 1, 2))
    if not inner_h_choices or not inner_w_choices:
        return None

    inner_h = int(rng.choice(inner_h_choices))
    inner_w = int(rng.choice(inner_w_choices))
    r0 = int(rng.integers(2, height - inner_h - 3))
    c0 = int(rng.integers(2, width - inner_w - 3))
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    total_noise = int(rng.integers(3, 6))
    counts = _choose_side_counts(rng, total_noise)
    noise = _place_noise(rng, (height, width), (r0, c0, r1, c1), counts)
    if noise is None:
        return None

    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    all_points = corners + noise
    if not _all_points_separated(all_points):
        return None

    color_count = len(all_points)
    colors = [int(c) for c in rng.choice(PALETTE, size=color_count, replace=False)]

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for (r, c), color in zip(all_points, colors):
        grid[r, c] = color

    candidates = _rectangle_candidates(grid, BACKGROUND)
    if candidates != [(r0, c0, r1, c1)]:
        return None

    expected = _paint_solution(grid, (r0, c0, r1, c1), BACKGROUND)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
