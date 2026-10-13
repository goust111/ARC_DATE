from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7)
ROW_SWAP = np.array([1, 0, 2, 4, 3], dtype=int)

GRID_SHAPE = (5, 13)
PILLAR_COLS = (1, 4, 7, 10)
SEGMENTS = ((0, 1), (3, 4))

GENERATION_RETRIES = 300
MODE_OPTIONS = (
    "sparse_changed",
    "mixed_counterexample",
    "dense_mixed",
)


def _find_pillars(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == 5, "expected a 5-row grid"

    cols: list[int] = []
    for c in range(g.shape[1]):
        col = g[:, c]
        if np.all(col == BG):
            continue
        assert int(col[2]) == GREY, "each pillar must have a grey center"
        colors = [int(col[r]) for r in (0, 1, 3, 4)]
        assert all(color not in (BG, GREY) for color in colors), "pillar colors must be non-background and non-grey"
        assert len(set(colors)) == 4, "each pillar must use four distinct non-grey colors"
        cols.append(int(c))

    assert len(cols) >= 2, "expected at least two colored pillars"
    for left, right in zip(cols, cols[1:]):
        assert right > left + 1, "adjacent pillars must be separated by background columns"
        assert np.all(g[:, left + 1 : right] == BG), "only background may appear between pillars"
    return cols


def _flip_pillar(column: np.ndarray) -> np.ndarray:
    flipped = np.asarray(column, dtype=int).copy()
    flipped[[0, 1]] = flipped[[1, 0]]
    flipped[[3, 4]] = flipped[[4, 3]]
    return flipped


def _flip_grid(grid: np.ndarray, cols: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for c in cols:
        out[:, c] = _flip_pillar(out[:, c])
    return out


def _connection_signature(grid: np.ndarray, cols: list[int]) -> set[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    sig: set[tuple[int, int, int]] = set()
    for pair_idx, (left, right) in enumerate(zip(cols, cols[1:])):
        for r in range(g.shape[0]):
            color = int(g[r, left])
            if color in (BG, GREY):
                continue
            if int(g[r, right]) == color:
                sig.add((pair_idx, r, color))
    return sig


def _segment_match_counts(grid: np.ndarray, cols: list[int]) -> list[int]:
    g = np.asarray(grid, dtype=int)
    counts: list[int] = []
    for left, right in zip(cols, cols[1:]):
        for rows in SEGMENTS:
            counts.append(sum(int(g[r, left] == g[r, right]) for r in rows))
    return counts


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cols = _find_pillars(g)
    out = _flip_grid(g, cols)

    for left, right in zip(cols, cols[1:]):
        for r in range(out.shape[0]):
            color = int(out[r, left])
            if color in (BG, GREY):
                continue
            if int(out[r, right]) == color:
                out[r, left + 1 : right] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse each colored two-pixel segment around the fixed grey center, then connect only the post-flip same-color horizontal neighbors.

    Input structure:
    - The grid has five rows with black background elsewhere.
    - Several 5x1 colored pillars appear in separate columns, with only background between neighboring pillars.
    - Each pillar has a fixed grey center pixel on row 3, plus two distinct non-grey pixels above and two distinct non-grey pixels below.

    Rule:
    - For every pillar, keep the grey center fixed, swap the two pixels above it, and swap the two pixels below it.
    - After all pillars are flipped, inspect each neighboring pillar pair row by row.
    - Whenever the two pillar pixels on the same row now share the same non-grey color, fill every background cell between those pillars on that row with that color.
    - Leave all other cells unchanged.

    Key insight:
    - The local pillar reversals change which rows line up across neighboring pillars, so the horizontal bridges must be computed from the flipped configuration rather than from the original one.
    """
    return _oracle_output(np.asarray(grid, dtype=int))


def _sample_pillar(rng: np.random.Generator) -> np.ndarray:
    colors = rng.permutation(np.array(PALETTE, dtype=int))[:4]
    return np.array([colors[0], colors[1], GREY, colors[2], colors[3]], dtype=int)


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    for c in PILLAR_COLS:
        grid[:, c] = _sample_pillar(rng)
    return grid


def _matches_mode(grid: np.ndarray, cols: list[int], mode: str) -> bool:
    pre_sig = _connection_signature(grid, cols)
    post_grid = _flip_grid(grid, cols)
    post_sig = _connection_signature(post_grid, cols)
    counts = _segment_match_counts(grid, cols)

    n0 = sum(count == 0 for count in counts)
    n1 = sum(count == 1 for count in counts)
    n2 = sum(count == 2 for count in counts)

    if pre_sig == post_sig:
        return False
    if not post_sig:
        return False
    if not (post_sig - pre_sig):
        return False
    if not (pre_sig - post_sig):
        return False
    if len(post_sig) > 6:
        return False

    if mode == "sparse_changed":
        return n1 >= 1 and n2 == 0 and n0 >= 1
    if mode == "mixed_counterexample":
        return n1 >= 1 and n2 >= 1 and n0 >= 1
    if mode == "dense_mixed":
        return n1 >= 2 and n2 >= 1
    raise AssertionError(f"unknown mode: {mode}")


def _validate_generated_grid(grid: np.ndarray, mode: str) -> bool:
    if not isinstance(grid, np.ndarray) or grid.shape != GRID_SHAPE:
        return False

    try:
        cols = _find_pillars(grid)
    except AssertionError:
        return False

    if cols != list(PILLAR_COLS):
        return False
    if any(np.any(grid[:, c] != BG) for c in range(grid.shape[1]) if c not in PILLAR_COLS):
        return False

    pre_sig = _connection_signature(grid, cols)
    flipped = _flip_grid(grid, cols)
    post_sig = _connection_signature(flipped, cols)
    counts = _segment_match_counts(grid, cols)

    if not _matches_mode(grid, cols, mode):
        return False
    if len(pre_sig | post_sig) < 2:
        return False
    if len(set(int(v) for v in grid[:, PILLAR_COLS].flatten() if int(v) != GREY)) < 3:
        return False
    if not any(count == 0 for count in counts):
        return False
    if not any(count == 1 for count in counts):
        return False
    if not np.array_equal(flipped[:, PILLAR_COLS], grid[ROW_SWAP][:, PILLAR_COLS]):
        return False

    expected = _oracle_output(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    for left, right in zip(cols, cols[1:]):
        gap = actual[:, left + 1 : right]
        for r in range(actual.shape[0]):
            target = int(actual[r, left])
            if (int(actual[r, right]) == target) and target not in (BG, GREY):
                if gap.shape[1] == 0 or not np.all(gap[r] == target):
                    return False
            else:
                if gap.shape[1] != 0 and np.any(gap[r] != BG):
                    return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        mode = MODE_OPTIONS[int(rng.integers(0, len(MODE_OPTIONS)))]
        candidate = _build_candidate(rng)
        if _validate_generated_grid(candidate, mode):
            return candidate
    raise RuntimeError("failed to generate a valid local-reverse adjacency input")
