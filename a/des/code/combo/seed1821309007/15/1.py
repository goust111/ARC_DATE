import numpy as np
from numpy.random import default_rng


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int, blocked: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or blocked[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if not blocked[nr, nc] and grid[nr, nc] == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _axis_row_and_col(grid: np.ndarray, bg: int) -> tuple[int, int]:
    row_candidates = [r for r in range(grid.shape[0]) if np.all(grid[r] != bg)]
    col_candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] != bg)]
    assert len(row_candidates) == 1, "expected exactly one full non-background row"
    assert len(col_candidates) == 1, "expected exactly one full non-background column"
    return int(row_candidates[0]), int(col_candidates[0])


def _majority_color(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _axis_markers(grid: np.ndarray, axis_r: int, axis_c: int) -> tuple[list[int], list[int]]:
    row_values = np.delete(grid[axis_r], axis_c)
    col_values = np.delete(grid[:, axis_c], axis_r)
    x_base = _majority_color(row_values)
    y_base = _majority_color(col_values)

    x_markers = [c for c in range(grid.shape[1]) if c != axis_c and int(grid[axis_r, c]) != x_base]
    y_markers = [r for r in range(grid.shape[0]) if r != axis_r and int(grid[r, axis_c]) != y_base]
    assert x_markers, "expected at least one X-axis marker"
    assert y_markers, "expected at least one Y-axis marker"
    return x_markers, y_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the full colored row and full colored column as axes, then map each foreground
    object into a compressed matrix by its aligned Y-axis row marker and X-axis column marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    axis_r, axis_c = _axis_row_and_col(g, bg)
    x_markers, y_markers = _axis_markers(g, axis_r, axis_c)
    x_index = {c: i for i, c in enumerate(x_markers)}
    y_index = {r: i for i, r in enumerate(y_markers)}

    blocked = np.zeros_like(g, dtype=bool)
    blocked[axis_r, :] = True
    blocked[:, axis_c] = True
    comps = _find_components(g, bg=bg, blocked=blocked)
    assert comps, "expected at least one foreground object"

    out = np.full((len(y_markers), len(x_markers)), bg, dtype=int)
    for comp in comps:
        rows = {r for r, _ in comp["coords"]}
        cols = {c for _, c in comp["coords"]}
        hit_rows = [r for r in rows if r in y_index]
        hit_cols = [c for c in cols if c in x_index]
        assert len(hit_rows) == 1, "each object must align to exactly one Y-axis marker row"
        assert len(hit_cols) == 1, "each object must align to exactly one X-axis marker column"
        rr = y_index[hit_rows[0]]
        cc = x_index[hit_cols[0]]
        assert out[rr, cc] == bg, "multiple objects mapped to the same output cell"
        out[rr, cc] = comp["color"]
    return out


# --------------- generator ---------------

BG = 0
AXIS_MIN_SPACING = 3
GRID_H_RANGE = (12, 21)
GRID_W_RANGE = (12, 21)
MARKER_COUNT_RANGE = (2, 5)
OBJECT_COUNT_MAX = 7
MAX_ATTEMPTS = 400
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, -1)),
    ((0, 0), (1, 0), (-1, 0)),
    ((0, 0), (0, 1), (1, 1)),
)


def _spaced_positions(rng: np.random.Generator, values: list[int], count: int) -> list[int] | None:
    order = list(rng.permutation(values))
    chosen: list[int] = []
    for value in order:
        if all(abs(value - prev) >= AXIS_MIN_SPACING for prev in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _shape_fits(
    grid: np.ndarray,
    occupied: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    shape: tuple[tuple[int, int], ...],
    axis_r: int,
    axis_c: int,
    y_rows: list[int],
    x_cols: list[int],
) -> bool:
    h, w = grid.shape
    y_set = set(y_rows)
    x_set = set(x_cols)
    for dr, dc in shape:
        r = anchor_r + dr
        c = anchor_c + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r == axis_r or c == axis_c or occupied[r, c]:
            return False
        if dr != 0 and r in y_set:
            return False
        if dc != 0 and c in x_set:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    if h < 2 * AXIS_MIN_SPACING + 3 or w < 2 * AXIS_MIN_SPACING + 3:
        return None

    axis_r = int(rng.integers(2, h - 2))
    axis_c = int(rng.integers(2, w - 2))

    max_x = min(MARKER_COUNT_RANGE[1], len(range(w)) // AXIS_MIN_SPACING + 1)
    max_y = min(MARKER_COUNT_RANGE[1], len(range(h)) // AXIS_MIN_SPACING + 1)
    x_count = int(rng.integers(MARKER_COUNT_RANGE[0], max_x + 1))
    y_count = int(rng.integers(MARKER_COUNT_RANGE[0], max_y + 1))

    x_candidates = [c for c in range(1, w - 1) if c != axis_c]
    y_candidates = [r for r in range(1, h - 1) if r != axis_r]
    x_cols = _spaced_positions(rng, x_candidates, x_count)
    y_rows = _spaced_positions(rng, y_candidates, y_count)
    if x_cols is None or y_rows is None:
        return None

    palette = list(rng.permutation(PALETTE))
    axis_colors = palette[:2] if rng.random() < 0.5 else [palette[0], palette[0]]
    x_base, y_base = int(axis_colors[0]), int(axis_colors[1])
    object_colors = [color for color in PALETTE if color not in {x_base, y_base}]
    rng.shuffle(object_colors)
    marker_pool = [color for color in PALETTE if color not in {x_base, y_base}]
    marker_colors = [int(rng.choice(marker_pool)) for _ in range(x_count + y_count)]
    if len(object_colors) < 1:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[axis_r, :] = x_base
    grid[:, axis_c] = y_base
    grid[axis_r, axis_c] = int(rng.choice([x_base, y_base]))

    for i, c in enumerate(x_cols):
        grid[axis_r, c] = marker_colors[i]
    for i, r in enumerate(y_rows):
        grid[r, axis_c] = marker_colors[x_count + i]

    all_pairs = [(r, c) for r in y_rows for c in x_cols]
    rng.shuffle(all_pairs)
    object_count = int(rng.integers(1, min(len(all_pairs), OBJECT_COUNT_MAX) + 1))
    chosen_pairs = all_pairs[:object_count]
    if len(object_colors) < object_count:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    occupied[axis_r, :] = True
    occupied[:, axis_c] = True
    for idx, (r, c) in enumerate(chosen_pairs):
        placed = False
        for shape in rng.permutation(len(SHAPES)):
            offsets = SHAPES[int(shape)]
            if _shape_fits(grid, occupied, r, c, offsets, axis_r, axis_c, y_rows, x_cols):
                for dr, dc in offsets:
                    grid[r + dr, c + dc] = object_colors[idx]
                    occupied[r + dr, c + dc] = True
                placed = True
                break
        if not placed:
            return None

    # Reject accidental extra full rows/cols.
    if sum(np.all(grid[r] != BG) for r in range(h)) != 1:
        return None
    if sum(np.all(grid[:, c] != BG) for c in range(w)) != 1:
        return None

    # Reject cases where main's structural assumptions do not hold.
    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
