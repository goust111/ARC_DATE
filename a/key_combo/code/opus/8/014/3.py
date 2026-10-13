import itertools
from typing import Iterable

import numpy as np


BG = 0
MISSING = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
QUADRANT_ORDER = ("tl", "tr", "bl", "br")


def _axis_period(arr: np.ndarray, axis: int) -> int:
    length = arr.shape[axis]
    if axis == 0:
        for p in range(1, length + 1):
            if all(np.array_equal(arr[r], arr[r % p]) for r in range(length)):
                return p
    else:
        for p in range(1, length + 1):
            if all(np.array_equal(arr[:, c], arr[:, c % p]) for c in range(length)):
                return p
    raise AssertionError("period not found")


def _tile_from_periods(pattern: np.ndarray, row_period: int, col_period: int) -> np.ndarray:
    rows = np.arange(pattern.shape[0])[:, None] % row_period
    cols = np.arange(pattern.shape[1])[None, :] % col_period
    return pattern[:row_period, :col_period][rows, cols]


def _split_quadrants(grid: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "grid must be split by a central cross"
    mid_r = h // 2
    mid_c = w // 2
    assert np.all(g[mid_r, :] == BG)
    assert np.all(g[:, mid_c] == BG)
    quads = [
        g[:mid_r, :mid_c].copy(),
        g[:mid_r, mid_c + 1 :].copy(),
        g[mid_r + 1 :, :mid_c].copy(),
        g[mid_r + 1 :, mid_c + 1 :].copy(),
    ]
    shapes = {q.shape for q in quads}
    assert len(shapes) == 1
    return mid_r, mid_c, quads


def _paste_quadrant(grid: np.ndarray, quadrant: np.ndarray, index: int, mid_r: int, mid_c: int) -> None:
    if index == 0:
        grid[:mid_r, :mid_c] = quadrant
    elif index == 1:
        grid[:mid_r, mid_c + 1 :] = quadrant
    elif index == 2:
        grid[mid_r + 1 :, :mid_c] = quadrant
    elif index == 3:
        grid[mid_r + 1 :, mid_c + 1 :] = quadrant
    else:
        raise ValueError("invalid quadrant index")


def _compatible(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all((a == MISSING) | (b == MISSING) | (a == b)))


def _find_majority_group(quadrants: list[np.ndarray]) -> tuple[tuple[int, int, int], int]:
    matches = []
    for combo in itertools.combinations(range(4), 3):
        if not all(_compatible(quadrants[i], quadrants[j]) for i, j in itertools.combinations(combo, 2)):
            continue
        outlier = next(idx for idx in range(4) if idx not in combo)
        if all(not _compatible(quadrants[outlier], quadrants[i]) for i in combo):
            matches.append((combo, outlier))
    assert len(matches) == 1, "expected one compatible triple and one outlier"
    return matches[0]


def _restore_pattern(quadrants: Iterable[np.ndarray]) -> np.ndarray:
    stack = np.stack(list(quadrants), axis=0)
    h, w = stack.shape[1:]
    merged = np.full((h, w), -1, dtype=int)
    for r in range(h):
        for c in range(w):
            values = stack[:, r, c]
            real = values[values != MISSING]
            assert real.size > 0, "every majority position must be visible in at least one quadrant"
            assert np.all(real == real[0]), "majority quadrants disagree at a visible position"
            merged[r, c] = int(real[0])
    row_period = _axis_period(merged, axis=0)
    col_period = _axis_period(merged, axis=1)
    repaired = _tile_from_periods(merged, row_period, col_period)
    assert np.array_equal(repaired, merged)
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the wildcard-compatible quadrant majority and replace the outlier with the restored periodic pattern.

    Input structure:
    - The grid has odd height and width, with a black separator cross on the central row and column.
    - The four resulting quadrants have equal shape.
    - Three quadrants share one hidden two-color periodic pattern but contain magenta missing cells.
    - The remaining quadrant is a fully visible anomalous pattern that is not wildcard-compatible with that majority.

    Rule:
    - Split the grid into the four quadrants, excluding the separator cross.
    - Compare quadrants pairwise while treating magenta as a wildcard that matches any color.
    - Find the unique triple of mutually compatible quadrants and treat the fourth quadrant as the outlier.
    - Reconstruct the complete majority pattern by taking the non-magenta value at each position, infer its row and column periods, and retile the quadrant from that periodic tile.
    - Replace the outlier quadrant with the restored periodic pattern, leaving the separator and the three majority quadrants unchanged.

    Key insight:
    - The same compatibility test simultaneously identifies both the target quadrant to overwrite and the three redundant partial observations needed to recover the missing periodic tile.
    """
    g = np.asarray(grid, dtype=int)
    mid_r, mid_c, quads = _split_quadrants(g)
    group, outlier = _find_majority_group(quads)
    repaired = _restore_pattern(quads[i] for i in group)
    out = g.copy()
    _paste_quadrant(out, repaired, outlier, mid_r, mid_c)
    return out


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (int(rng.integers(h)), int(rng.integers(w)))
    mask[seed] = True
    frontier = {seed}
    while mask.sum() < target_area and frontier:
        cell = list(frontier)[int(rng.integers(len(frontier)))]
        r, c = cell
        candidates = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                candidates.append((nr, nc))
        if not candidates:
            frontier.remove(cell)
            continue
        pick = candidates[int(rng.integers(len(candidates)))]
        mask[pick] = True
        frontier.add(pick)
        frontier.add(cell)
    return mask


def _make_periodic_pattern(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, tuple[int, int]]:
    colors = tuple(int(x) for x in rng.choice(PALETTE, size=2, replace=False))
    mode = rng.choice(["rows", "cols", "grid"])
    for _ in range(200):
        if mode == "rows":
            row_period = int(rng.integers(2, min(h - 1, 4) + 1))
            col_period = 1
        elif mode == "cols":
            row_period = 1
            col_period = int(rng.integers(2, min(w - 1, 4) + 1))
        else:
            row_period = int(rng.integers(2, min(h - 1, 4) + 1))
            col_period = int(rng.integers(2, min(w - 1, 4) + 1))

        tile01 = rng.integers(0, 2, size=(row_period, col_period))
        if tile01.min() == tile01.max():
            continue
        frac = tile01.mean()
        if not (0.25 <= frac <= 0.75):
            continue
        tile = np.where(tile01 == 0, colors[0], colors[1])
        pattern = tile[np.arange(h)[:, None] % row_period, np.arange(w)[None, :] % col_period]
        actual_r = _axis_period(pattern, axis=0)
        actual_c = _axis_period(pattern, axis=1)
        if actual_r == 1 and actual_c == 1:
            continue
        if mode == "rows" and actual_r == 1:
            continue
        if mode == "cols" and actual_c == 1:
            continue
        if mode == "grid" and (actual_r == 1 or actual_c == 1):
            continue
        return pattern, colors
    raise RuntimeError("failed to build periodic pattern")


def _make_visibility_masks(
    rng: np.random.Generator, h: int, w: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    size = h * w
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    support = cells[:3]
    remainder = cells[3:]
    visible = np.ones((3, h, w), dtype=bool)

    for idx, (r, c) in enumerate(support):
        visible[:, r, c] = False
        visible[idx, r, c] = True

    patterns = np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    )
    weights = np.array([14, 7, 7, 7, 2, 2, 2], dtype=float)
    weights /= weights.sum()

    for r, c in remainder:
        visible[:, r, c] = patterns[int(rng.choice(len(patterns), p=weights))]

    missing_counts = (~visible).sum(axis=(1, 2))
    if np.any(missing_counts < max(2, size // 10)):
        raise RuntimeError("too few missing cells")
    if np.any(missing_counts > max(6, (size * 2) // 3)):
        raise RuntimeError("too many missing cells")
    return visible, support


def _paint_majority_quadrants(pattern: np.ndarray, visible: np.ndarray) -> list[np.ndarray]:
    quadrants = []
    for idx in range(3):
        quad = pattern.copy()
        quad[~visible[idx]] = MISSING
        quadrants.append(quad)
    return quadrants


def _other_majority_color(colors: tuple[int, int], value: int) -> int:
    return int(colors[0] if value == colors[1] else colors[1])


def _make_anomaly(
    rng: np.random.Generator,
    majority: np.ndarray,
    majority_quads: list[np.ndarray],
    support: list[tuple[int, int]],
    majority_colors: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    h, w = majority.shape
    size = h * w
    same_palette = bool(rng.integers(2))
    if same_palette:
        colors = majority_colors
    else:
        remaining = [c for c in PALETTE if c not in majority_colors]
        colors = tuple(int(x) for x in rng.choice(remaining, size=2, replace=False))

    for _ in range(200):
        blob_area = int(rng.integers(max(3, size // 4), max(4, (size * 3) // 5) + 1))
        blob = _random_connected_mask(rng, h, w, blob_area)
        anomaly = np.full((h, w), colors[0], dtype=int)
        anomaly[blob] = colors[1]

        if same_palette:
            for idx, (r, c) in enumerate(support):
                anomaly[r, c] = _other_majority_color(majority_colors, int(majority[r, c]))
                if idx == 0 and h > 2:
                    anomaly[(r + 1) % h, c] = _other_majority_color(majority_colors, int(majority[(r + 1) % h, c]))
        else:
            for r, c in support:
                anomaly[r, c] = colors[1]

        if len(np.unique(anomaly)) < 2:
            continue
        if same_palette and int(np.sum(anomaly != majority)) < max(4, size // 3):
            continue
        if all(not _compatible(anomaly, quad) for quad in majority_quads):
            return anomaly, same_palette
    raise RuntimeError("failed to build anomaly")


def _build_grid(quadrants: list[np.ndarray], qh: int, qw: int) -> np.ndarray:
    grid = np.full((2 * qh + 1, 2 * qw + 1), BG, dtype=int)
    grid[:qh, :qw] = quadrants[0]
    grid[:qh, qw + 1 :] = quadrants[1]
    grid[qh + 1 :, :qw] = quadrants[2]
    grid[qh + 1 :, qw + 1 :] = quadrants[3]
    return grid


def _generic_semantic_checks(grid: np.ndarray) -> bool:
    try:
        mid_r, mid_c, quads = _split_quadrants(grid)
        group, outlier = _find_majority_group(quads)
        stack = np.stack([quads[i] for i in group], axis=0)
        visible_counts = np.sum(stack != MISSING, axis=0)
        if np.any(visible_counts < 1):
            return False
        for local_idx in range(3):
            only_this = np.ones(stack.shape[1:], dtype=bool)
            for j in range(3):
                if j == local_idx:
                    only_this &= stack[j] != MISSING
                else:
                    only_this &= stack[j] == MISSING
            if not np.any(only_this):
                return False
            if not np.any(stack[local_idx] == MISSING):
                return False
        repaired = _restore_pattern(quads[i] for i in group)
        out = main(grid)
        expected = grid.copy()
        _paste_quadrant(expected, repaired, outlier, mid_r, mid_c)
        return np.array_equal(out, expected)
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        qh = int(rng.integers(4, 13))
        qw = int(rng.integers(4, 13))
        if 2 * qh + 1 > 30 or 2 * qw + 1 > 30:
            continue
        try:
            majority_pattern, majority_colors = _make_periodic_pattern(rng, qh, qw)
            visible, support = _make_visibility_masks(rng, qh, qw)
            majority_quads = _paint_majority_quadrants(majority_pattern, visible)
            anomaly, _same_palette = _make_anomaly(
                rng, majority_pattern, majority_quads, support, majority_colors
            )
        except RuntimeError:
            continue

        anomaly_index = int(rng.integers(4))
        quadrants = [None] * 4
        maj_iter = iter(majority_quads)
        for idx in range(4):
            quadrants[idx] = anomaly if idx == anomaly_index else next(maj_iter)
        grid = _build_grid(quadrants, qh, qw)

        if not _generic_semantic_checks(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
