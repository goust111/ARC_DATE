import random
from collections import Counter

import numpy as np


BLACK = 0
MAROON = 9
STRIPE_COLORS = tuple(range(1, 9))
SIDES = ("left", "right", "top", "bottom")
GRID_SIZE_RANGE = (12, 20)
STRIPE_COUNT_RANGE = (2, 4)
MIN_GAP = 2
MAX_ATTEMPTS = 300


def _marker_side(grid: np.ndarray) -> str:
    h, w = grid.shape
    side_counts = {
        "left": int(np.sum(grid[:, 0] == MAROON)),
        "right": int(np.sum(grid[:, w - 1] == MAROON)),
        "top": int(np.sum(grid[0, :] == MAROON)),
        "bottom": int(np.sum(grid[h - 1, :] == MAROON)),
    }
    side = max(side_counts, key=side_counts.get)
    assert side_counts[side] > 0, "expected maroon markers on one border"
    return side


def _repeated_color(values: np.ndarray) -> int | None:
    counts = Counter(int(v) for v in values if int(v) not in (BLACK, MAROON))
    repeated = [(color, count) for color, count in counts.items() if count >= 2]
    if not repeated:
        return None
    repeated.sort(key=lambda item: (-item[1], item[0]))
    if len(repeated) > 1 and repeated[0][1] == repeated[1][1]:
        raise AssertionError(f"ambiguous stripe colors: {repeated}")
    return repeated[0][0]


def _row_stripes(grid: np.ndarray) -> dict[int, int]:
    stripes: dict[int, int] = {}
    for r in range(grid.shape[0]):
        color = _repeated_color(grid[r, :])
        if color is not None:
            stripes[r] = color
    return stripes


def _col_stripes(grid: np.ndarray) -> dict[int, int]:
    stripes: dict[int, int] = {}
    for c in range(grid.shape[1]):
        color = _repeated_color(grid[:, c])
        if color is not None:
            stripes[c] = color
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the maroon-marked stripes and reveal the orthogonal stripes hidden beneath them.

    Input structure:
    - The background is Black.
    - Several one-pixel-wide horizontal and vertical colored stripes cross the grid.
    - Each stripe has its own color, and an overlap cell shows only the color of the stripe on top.
    - Maroon marker pixels lie on exactly one border; a marked row means a horizontal stripe
      should be removed, while a marked column means a vertical stripe should be removed.

    Rule:
    - Detect the marker side to decide whether rows or columns are being deleted.
    - Recover every horizontal/vertical stripe by finding the unique non-Black, non-Maroon color
      that repeats along that row or column despite crossings.
    - Remove each marked stripe completely.
    - At every cell of a removed stripe, restore the orthogonal stripe color if such a stripe
      passes there; otherwise leave Black.
    - Remove the Maroon markers themselves.

    Key insight:
    - A crossing hides exactly one of two stripes, so after deleting the marked stripe the only
      possible color to restore is the color of the orthogonal stripe running through that cell.
    """
    g = np.asarray(grid, dtype=int)
    side = _marker_side(g)
    row_colors = _row_stripes(g)
    col_colors = _col_stripes(g)
    out = g.copy()
    out[out == MAROON] = BLACK

    if side in ("left", "right"):
        marker_col = 0 if side == "left" else g.shape[1] - 1
        removed_rows = [r for r in range(g.shape[0]) if int(g[r, marker_col]) == MAROON]
        for r in removed_rows:
            for c in range(g.shape[1]):
                out[r, c] = col_colors.get(c, BLACK)
    else:
        marker_row = 0 if side == "top" else g.shape[0] - 1
        removed_cols = [c for c in range(g.shape[1]) if int(g[marker_row, c]) == MAROON]
        for c in removed_cols:
            for r in range(g.shape[0]):
                out[r, c] = row_colors.get(r, BLACK)

    return out


def _sample_spaced_positions(
    rng: random.Random, start: int, stop: int, count: int, gap: int = MIN_GAP
) -> list[int] | None:
    candidates = list(range(start, stop))
    for _ in range(100):
        picks = sorted(rng.sample(candidates, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _paint_markers(grid: np.ndarray, side: str, positions: list[int]) -> None:
    if side == "left":
        for r in positions:
            grid[r, 0] = MAROON
    elif side == "right":
        for r in positions:
            grid[r, -1] = MAROON
    elif side == "top":
        for c in positions:
            grid[0, c] = MAROON
    else:
        for c in positions:
            grid[-1, c] = MAROON


def _render_scene(
    shape: tuple[int, int],
    h_rows: list[int],
    h_colors: list[int],
    v_cols: list[int],
    v_colors: list[int],
    top_is_horizontal: np.ndarray,
    active_h: list[bool],
    active_v: list[bool],
    marker_side: str | None = None,
    marker_positions: list[int] | None = None,
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)

    for row, color, active in zip(h_rows, h_colors, active_h):
        if active:
            grid[row, 1 : w - 1] = color

    for col, color, active in zip(v_cols, v_colors, active_v):
        if active:
            grid[1 : h - 1, col] = color

    for hi, row in enumerate(h_rows):
        if not active_h[hi]:
            continue
        for vi, col in enumerate(v_cols):
            if not active_v[vi]:
                continue
            grid[row, col] = h_colors[hi] if bool(top_is_horizontal[hi, vi]) else v_colors[vi]

    if marker_side is not None and marker_positions is not None:
        _paint_markers(grid, marker_side, marker_positions)

    return grid


def _changed_restoration_exists(
    marker_side: str,
    marked: set[int],
    top_is_horizontal: np.ndarray,
) -> bool:
    if marker_side in ("left", "right"):
        for hi in marked:
            if np.any(top_is_horizontal[hi, :]):
                return True
        return False

    for vi in marked:
        if np.any(~top_is_horizontal[:, vi]):
            return True
    return False


def _mixed_crossings_for_marked(
    marker_side: str,
    marked: set[int],
    top_is_horizontal: np.ndarray,
) -> bool:
    if marker_side in ("left", "right"):
        values = [bool(top_is_horizontal[hi, vi]) for hi in marked for vi in range(top_is_horizontal.shape[1])]
    else:
        values = [bool(top_is_horizontal[hi, vi]) for hi in range(top_is_horizontal.shape[0]) for vi in marked]
    return any(values) and not all(values)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    row_slots = h - 2
    col_slots = w - 2
    max_h = min(STRIPE_COUNT_RANGE[1], (row_slots + 1) // 2)
    max_v = min(STRIPE_COUNT_RANGE[1], (col_slots + 1) // 2)
    if max_h < STRIPE_COUNT_RANGE[0] or max_v < STRIPE_COUNT_RANGE[0]:
        return None

    n_h = rng.randint(STRIPE_COUNT_RANGE[0], max_h)
    n_v = rng.randint(STRIPE_COUNT_RANGE[0], max_v)
    if n_h + n_v > len(STRIPE_COLORS):
        return None

    h_rows = _sample_spaced_positions(rng, 1, h - 1, n_h)
    v_cols = _sample_spaced_positions(rng, 1, w - 1, n_v)
    if h_rows is None or v_cols is None:
        return None

    colors = rng.sample(STRIPE_COLORS, n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]
    top_is_horizontal = np.array(
        [[rng.choice((False, True)) for _ in range(n_v)] for _ in range(n_h)],
        dtype=bool,
    )

    marker_side = rng.choice(SIDES)
    removable_count = n_h if marker_side in ("left", "right") else n_v
    if removable_count < 2:
        return None
    marked_count = rng.randint(1, removable_count - 1)
    marked = set(rng.sample(range(removable_count), marked_count))

    if not _changed_restoration_exists(marker_side, marked, top_is_horizontal):
        return None
    if not _mixed_crossings_for_marked(marker_side, marked, top_is_horizontal):
        return None

    active_h_in = [True] * n_h
    active_v_in = [True] * n_v
    marker_positions = [h_rows[i] for i in sorted(marked)] if marker_side in ("left", "right") else [v_cols[i] for i in sorted(marked)]
    inp = _render_scene(
        (h, w),
        h_rows,
        h_colors,
        v_cols,
        v_colors,
        top_is_horizontal,
        active_h_in,
        active_v_in,
        marker_side,
        marker_positions,
    )

    active_h_out = [i not in marked for i in range(n_h)] if marker_side in ("left", "right") else [True] * n_h
    active_v_out = [True] * n_v if marker_side in ("left", "right") else [i not in marked for i in range(n_v)]
    expected = _render_scene(
        (h, w),
        h_rows,
        h_colors,
        v_cols,
        v_colors,
        top_is_horizontal,
        active_h_out,
        active_v_out,
    )

    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
