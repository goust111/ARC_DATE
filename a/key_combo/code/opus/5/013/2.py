import hashlib
import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

PANEL_COLORS = (BLUE, RED, YELLOW, MAGENTA)
PANEL_SIZES = (2, 3)
MAX_ATTEMPTS = 600

_SEEN_OUTPUT_SIGS: set[str] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bbox")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _next_growth_cols(cols: list[int]) -> list[int]:
    return [(left + right) // 2 for left, right in zip(cols, cols[1:]) if right - left == 2]


def _growth_rows(seed_cols: list[int], upper_h: int) -> list[tuple[int, list[int], int]]:
    rows = [(0, list(seed_cols), ORANGE)]
    cur_cols = list(seed_cols)
    cur_r = 0
    next_color = MAGENTA
    while True:
        next_r = cur_r + 2
        if next_r >= upper_h:
            break
        child_cols = _next_growth_cols(cur_cols)
        if not child_cols:
            break
        rows.append((int(next_r), child_cols, int(next_color)))
        cur_cols = child_cols
        cur_r = next_r
        next_color = ORANGE if next_color == MAGENTA else MAGENTA
    return rows


def _render_growth(seed_cols: list[int], upper_h: int) -> tuple[np.ndarray, list[tuple[int, list[int], int]]]:
    width = max(seed_cols) + 1
    upper = np.full((upper_h, width), GREEN, dtype=int)
    rows = _growth_rows(seed_cols, upper_h)
    for r, cols, color in rows:
        upper[r, cols] = color
    return upper, rows


def _cropped_mask(seed_cols: list[int], upper_h: int) -> np.ndarray:
    upper, _rows = _render_growth(seed_cols, upper_h)
    mask = np.isin(upper, (ORANGE, MAGENTA))
    return _crop_bbox(mask.astype(int), _bbox(mask))


def _extract_panel(lower: np.ndarray) -> np.ndarray:
    panel_mask = lower != BLACK
    box = _bbox(panel_mask)
    panel = _crop_bbox(lower, box)
    if panel.shape[0] != panel.shape[1] or panel.shape[0] not in PANEL_SIZES:
        raise ValueError("expected a square 2x2 or 3x3 panel")
    if np.any(panel == BLACK):
        raise ValueError("panel must be fully colored")
    if not set(np.unique(panel)).issubset(set(PANEL_COLORS)):
        raise ValueError("panel uses unexpected colors")
    return panel


def _separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    if len(rows) != 1:
        raise ValueError("expected exactly one full grey separator row")
    return int(rows[0])


def _seed_row(upper: np.ndarray) -> tuple[int, list[int]]:
    candidates = [r for r in range(upper.shape[0]) if np.any(upper[r] == ORANGE)]
    if not candidates:
        raise ValueError("missing orange seed row")
    r = int(candidates[0])
    cols = [int(c) for c in np.flatnonzero(upper[r] == ORANGE)]
    if not (3 <= len(cols) <= 5):
        raise ValueError("seed row must contain 3 to 5 pixels")
    return r, cols


def _spec_output(grid: np.ndarray) -> np.ndarray:
    sep = _separator_row(grid)
    upper = grid[:sep]
    lower = grid[sep + 1 :]
    seed_r, seed_cols = _seed_row(upper)
    local_upper_h = int(upper.shape[0] - seed_r)
    local_seed_cols = [int(c) for c in seed_cols]
    mask = _cropped_mask(local_seed_cols, local_upper_h)
    panel = _extract_panel(lower)
    return np.kron(mask.astype(int), panel)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a midpoint mask from the top seed row, then Kronecker-expand a color panel through that mask.

    Input structure:
    - A full grey (5) horizontal separator splits the grid into an upper growth zone and a lower panel zone.
    - The upper zone has green (3) background and one top seed row containing 3 to 5 orange (7) pixels.
    - The lower zone has black (0) background and exactly one fully colored 2x2 or 3x3 panel using colors from {1,2,4,6}.

    Rule:
    - Starting from the seed row, generate new active rows every other line: for each adjacent active pair separated by exactly one background cell, place a child at the midpoint two rows below.
    - Alternate generated-row colors magenta (6), orange (7), magenta (6), ... and stop when no child can be generated or the next row would leave the upper zone.
    - Convert the seed row plus all generated rows into a binary mask, crop it to the tight foreground bbox, then use that mask as the outer factor of a Kronecker product with the lower color panel; foreground cells copy the panel and background cells become black blocks.

    Key insight:
    - The seed-row spacing controls the recursive growth shape, and that cropped growth mask determines exactly where the small color panel is tiled in the output.
    """
    g = np.asarray(grid, dtype=int)
    return _spec_output(g)


def _all_seed_specs() -> list[dict]:
    specs: list[dict] = []
    for count in range(3, 6):
        for diffs in itertools.product((2, 3), repeat=count - 1):
            cols = [0]
            for diff in diffs:
                cols.append(cols[-1] + diff)
            if cols[-1] > 9:
                continue
            layers = []
            cur = cols
            while True:
                nxt = _next_growth_cols(cur)
                if not nxt:
                    break
                layers.append(nxt)
                cur = nxt
            if len(layers) < 2:
                continue
            specs.append(
                {
                    "cols": cols,
                    "diffs": diffs,
                    "full_depth": len(layers),
                    "has_gap3": any(diff == 3 for diff in diffs),
                }
            )
    return specs


SEED_SPECS = _all_seed_specs()


def _sample_seed_spec() -> dict:
    want_gap3 = random.random() < 0.55
    candidates = [spec for spec in SEED_SPECS if spec["has_gap3"] == want_gap3]
    if not candidates:
        candidates = SEED_SPECS
    return random.choice(candidates)


def _sample_visible_depth(spec: dict) -> int:
    full_depth = int(spec["full_depth"])
    if full_depth >= 3 and random.random() < 0.45:
        return random.randint(2, full_depth - 1)
    return full_depth


def _make_panel() -> np.ndarray | None:
    size = random.choice(PANEL_SIZES)
    n_colors = random.randint(2, min(4, size * size))
    chosen = random.sample(PANEL_COLORS, n_colors)
    flat = chosen[:]
    while len(flat) < size * size:
        flat.append(random.choice(chosen))
    random.shuffle(flat)
    panel = np.array(flat, dtype=int).reshape(size, size)
    if len(np.unique(panel)) != n_colors:
        return None
    if len({tuple(row) for row in panel}) == 1:
        return None
    if len({tuple(col) for col in panel.T}) == 1:
        return None
    return panel


def _build_input(seed_cols: list[int], upper_h: int, panel: np.ndarray) -> np.ndarray:
    shifted_seed_cols = [c + 1 for c in seed_cols]
    seed_w = shifted_seed_cols[-1] + 2
    panel_size = int(panel.shape[0])
    grid_w = max(seed_w, panel_size + 2)
    lower_h = panel_size + 2
    grid_h = upper_h + 1 + lower_h

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[:upper_h] = GREEN
    grid[upper_h] = GREY
    grid[upper_h + 1 :] = BLACK
    grid[0, shifted_seed_cols] = ORANGE
    grid[upper_h + 2 : upper_h + 2 + panel_size, 1 : 1 + panel_size] = panel
    return grid


def _dense_baseline_output(seed_count: int, upper_h: int, panel: np.ndarray) -> np.ndarray:
    seed_cols = [1 + 2 * i for i in range(seed_count)]
    mask = _cropped_mask(seed_cols, upper_h)
    return np.kron(mask.astype(int), panel)


def _validate_case(grid: np.ndarray, spec: dict, visible_depth: int, panel: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    sep = _separator_row(grid)
    upper = grid[:sep]
    lower = grid[sep + 1 :]
    if not np.all(grid[sep] == GREY):
        return False
    if np.any(upper[1:] != GREEN):
        return False

    seed_r, seed_cols = _seed_row(upper)
    if seed_r != 0:
        return False
    diffs = [b - a for a, b in zip(seed_cols, seed_cols[1:])]
    if any(diff not in (2, 3) for diff in diffs):
        return False

    extracted_panel = _extract_panel(lower)
    if not np.array_equal(extracted_panel, panel):
        return False

    grown_rows = _growth_rows(seed_cols, upper.shape[0])
    if len(grown_rows) - 1 != visible_depth:
        return False
    if visible_depth < 2:
        return False

    full_depth = int(spec["full_depth"])
    if visible_depth < full_depth:
        last_r, last_cols, _ = grown_rows[-1]
        if last_r + 2 != upper.shape[0]:
            return False
        if not _next_growth_cols(last_cols):
            return False
    else:
        if visible_depth != full_depth:
            return False
        if _next_growth_cols(grown_rows[-1][1]):
            return False

    mask = _cropped_mask(seed_cols, upper.shape[0])
    out = np.kron(mask.astype(int), panel)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if np.array_equal(out, np.kron(mask.astype(int), np.ones_like(panel))):
        return False

    seed_only = np.kron(np.ones((1, mask.shape[1]), dtype=int), panel)
    if np.array_equal(out, seed_only):
        return False

    if spec["has_gap3"]:
        baseline = _dense_baseline_output(len(seed_cols), upper.shape[0], panel)
        if np.array_equal(out, baseline):
            return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        spec = _sample_seed_spec()
        visible_depth = _sample_visible_depth(spec)
        upper_h = 2 * visible_depth + 2

        panel = _make_panel()
        if panel is None:
            continue

        grid = _build_input(list(spec["cols"]), upper_h, panel)
        if not _validate_case(grid, spec, visible_depth, panel):
            continue

        out = main(grid)
        sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        if sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_OUTPUT_SIGS.add(sig)
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
