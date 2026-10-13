import random

import numpy as np


BLACK = 0
RED = 2
NON_RED_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]


def _marker_rows(grid: np.ndarray) -> np.ndarray:
    return np.any(grid == RED, axis=1)


def _compute_offsets(marker_rows: np.ndarray) -> np.ndarray:
    marker_rows = np.asarray(marker_rows, dtype=bool)
    h = int(marker_rows.size)
    offsets = np.zeros(h, dtype=int)
    direction = 1
    for r in range(1, h):
        if marker_rows[r - 1]:
            direction *= -1
        offsets[r] = offsets[r - 1] + direction
    return offsets


def _render_with_offsets(grid: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    offsets = np.asarray(offsets, dtype=int)
    h, w = g.shape
    min_offset = int(offsets.min())
    max_offset = int(offsets.max())
    out = np.zeros((h, w + max_offset - min_offset), dtype=int)
    for r in range(h):
        start = int(offsets[r] - min_offset)
        out[r, start : start + w] = g[r]
    return out


def _has_consecutive(rows: list[int]) -> bool:
    return any(b == a + 1 for a, b in zip(rows, rows[1:]))


def _every_trigger_matters(marker_rows: np.ndarray) -> bool:
    base = _compute_offsets(marker_rows)
    for r in np.flatnonzero(marker_rows):
        alt = marker_rows.copy()
        alt[int(r)] = False
        if np.array_equal(_compute_offsets(alt), base):
            return False
    return True


def _wrong_offsets_current_row(marker_rows: np.ndarray) -> np.ndarray:
    marker_rows = np.asarray(marker_rows, dtype=bool)
    h = int(marker_rows.size)
    offsets = np.zeros(h, dtype=int)
    direction = 1
    for r in range(1, h):
        if marker_rows[r]:
            direction *= -1
        offsets[r] = offsets[r - 1] + direction
    return offsets


def _wrong_offsets_all_right(marker_rows: np.ndarray) -> np.ndarray:
    return np.arange(marker_rows.size, dtype=int)


def _wrong_offsets_all_left(marker_rows: np.ndarray) -> np.ndarray:
    return -np.arange(marker_rows.size, dtype=int)


def _wrong_offsets_first_flip_only(marker_rows: np.ndarray) -> np.ndarray:
    marker_rows = np.asarray(marker_rows, dtype=bool)
    h = int(marker_rows.size)
    offsets = np.zeros(h, dtype=int)
    direction = 1
    flipped = False
    for r in range(1, h):
        if marker_rows[r - 1] and not flipped:
            direction *= -1
            flipped = True
        offsets[r] = offsets[r - 1] + direction
    return offsets


def main(grid: np.ndarray) -> np.ndarray:
    """Shear each row onto a black canvas using a zigzag offset that reverses after marker rows.

    Input structure:
    - The input is a fully populated rectangular color grid whose row contents stay intact.
    - Some rows contain at least one red marker pixel, and red is used only for this marker role.
    - The transformation depends only on which rows contain red, not on the marker columns.

    Rule:
    - Row 0 keeps offset 0, and the initial offset-change direction is to the right.
    - For every later row, add the current direction (+1 or -1) to the previous row's offset; if the previous row contains any red pixel, reverse the direction before placing the next row.
    - Copy every row unchanged onto a black canvas after shifting all offsets so the minimum offset becomes column 0.

    Key insight:
    - The row positions come from one cumulative state machine: each red row changes the direction for all later rows, producing a content-driven zigzag shear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "grid must be non-empty"

    marker_rows = _marker_rows(g)
    offsets = _compute_offsets(marker_rows)
    return _render_with_offsets(g, offsets)


# --------------- generator ---------------

MIN_HEIGHT, MAX_HEIGHT = 5, 11
MIN_WIDTH, MAX_WIDTH = 4, 10
MIN_COLOR_COUNT, MAX_COLOR_COUNT = 4, 6
MAX_OUTPUT_SIDE = 30
GEN_RETRIES = 400

FILL_MODES = ("noise", "row_runs", "column_bands", "diagonal_bands")
TRIGGER_MODES = ("single", "multi", "consecutive")


def _weighted_choice(seq, weights):
    return random.choices(seq, weights=weights, k=1)[0]


def _candidate_trigger_rows(h: int) -> dict[str, list[list[int]]]:
    cached = getattr(_candidate_trigger_rows, "_cache", {})
    if h in cached:
        return cached[h]

    classes: dict[str, list[list[int]]] = {mode: [] for mode in TRIGGER_MODES}
    for mask in range(1, 1 << (h - 1)):
        rows = [r for r in range(h - 1) if (mask >> r) & 1]
        marker_rows = np.zeros(h, dtype=bool)
        marker_rows[rows] = True
        offsets = _compute_offsets(marker_rows)
        diffs = set(map(int, np.diff(offsets)))
        if not (int(offsets.min()) < 0 < int(offsets.max())):
            continue
        if diffs != {-1, 1}:
            continue
        if not _every_trigger_matters(marker_rows):
            continue

        if len(rows) == 1:
            classes["single"].append(rows)
        elif _has_consecutive(rows):
            if len(rows) >= 3:
                classes["consecutive"].append(rows)
        else:
            classes["multi"].append(rows)

    cached[h] = classes
    _candidate_trigger_rows._cache = cached
    return classes


def _sample_trigger_rows(h: int) -> list[int] | None:
    classes = _candidate_trigger_rows(h)
    available = [mode for mode in TRIGGER_MODES if classes[mode]]
    if not available:
        return None
    mode_weights = {"single": 3, "multi": 3, "consecutive": 4}
    mode = _weighted_choice(available, [mode_weights[m] for m in available])
    candidates = classes[mode]
    candidate_weights = []
    for rows in candidates:
        weight = 1
        if 0 in rows:
            weight += 2
        if 1 in rows:
            weight += 1
        if len(rows) >= 3:
            weight += 1
        candidate_weights.append(weight)
    return list(_weighted_choice(candidates, candidate_weights))


def _ensure_palette_coverage(grid: np.ndarray, palette: list[int]) -> None:
    positions = random.sample(range(grid.size), len(palette))
    rows, cols = np.divmod(positions, grid.shape[1])
    grid[rows, cols] = palette


def _fill_base_grid(h: int, w: int, palette: list[int]) -> np.ndarray:
    arr = np.zeros((h, w), dtype=int)
    mode = random.choice(FILL_MODES)

    if mode == "noise":
        arr[:] = np.random.choice(palette, size=(h, w))
    elif mode == "row_runs":
        for r in range(h):
            cuts = []
            if w > 1:
                max_cuts = min(3, w - 1)
                n_cuts = random.randint(0, max_cuts)
                if n_cuts > 0:
                    cuts = sorted(random.sample(range(1, w), n_cuts))
            starts = [0] + cuts
            ends = cuts + [w]
            last = None
            for c0, c1 in zip(starts, ends):
                choices = [color for color in palette if color != last] or palette
                last = random.choice(choices)
                arr[r, c0:c1] = last
    elif mode == "column_bands":
        for c in range(w):
            arr[:, c] = random.choice(palette)
        for _ in range(max(1, h // 2)):
            rr = random.randrange(h)
            cc = random.randrange(w)
            arr[rr, cc] = random.choice(palette)
    else:
        band = random.randint(1, min(3, max(1, w - 1)))
        phase = random.randrange(len(palette))
        rows, cols = np.indices((h, w))
        arr[:] = np.asarray(palette)[((rows + cols) // band + phase) % len(palette)]
        if random.random() < 0.5:
            arr = np.fliplr(arr)

    _ensure_palette_coverage(arr, palette)

    # Give every row a visible signature so row identity stays readable after the shear.
    for r in range(h):
        c0 = (2 * r + 1) % w
        c1 = (3 * r + 2) % w
        arr[r, c0] = palette[r % len(palette)]
        arr[r, c1] = palette[(r + 2) % len(palette)]
    return arr


def _place_markers(base: np.ndarray, trigger_rows: list[int], palette: list[int]) -> np.ndarray:
    grid = base.copy()
    h, w = grid.shape
    for r in trigger_rows:
        max_markers = 1 if w <= 4 else 2
        marker_count = random.randint(1, max_markers)
        cols = random.sample(range(w), marker_count)
        for c in cols:
            grid[r, c] = RED
        # Keep at least one non-red anchor so the row is not visually reduced to "just markers".
        safe_choices = [c for c in range(w) if c not in cols]
        if safe_choices:
            c = random.choice(safe_choices)
            grid[r, c] = random.choice(palette)
    assert np.all(grid != BLACK)
    assert np.all(np.isin(grid, [BLACK, RED] + NON_RED_COLORS))
    return grid


def _rows_are_distinct(grid: np.ndarray) -> bool:
    seen = set()
    for row in grid:
        key = tuple(map(int, row))
        if key in seen:
            return False
        seen.add(key)
    return True


def _semantic_checks(inp: np.ndarray) -> bool:
    if inp.ndim != 2:
        return False
    h, w = map(int, inp.shape)
    if h < MIN_HEIGHT or h > MAX_HEIGHT or w < MIN_WIDTH or w > MAX_WIDTH:
        return False
    if np.any(inp == BLACK):
        return False

    marker_rows = _marker_rows(inp)
    marked = list(map(int, np.flatnonzero(marker_rows)))
    if not marked or marked[-1] == h - 1:
        return False
    if len(marked) > max(3, h // 2):
        return False
    if not _rows_are_distinct(inp):
        return False

    offsets = _compute_offsets(marker_rows)
    if not (int(offsets.min()) < 0 < int(offsets.max())):
        return False
    if set(map(int, np.diff(offsets))) != {-1, 1}:
        return False
    if not _every_trigger_matters(marker_rows):
        return False

    out = main(inp)
    if out.shape[0] != h or max(out.shape) > MAX_OUTPUT_SIDE:
        return False
    if out.shape[1] != w + int(offsets.max()) - int(offsets.min()):
        return False

    wrong_renderers = [
        _wrong_offsets_all_right,
        _wrong_offsets_all_left,
        _wrong_offsets_current_row,
    ]
    if int(marker_rows.sum()) > 1:
        wrong_renderers.append(_wrong_offsets_first_flip_only)
    for wrong in wrong_renderers:
        if np.array_equal(out, _render_with_offsets(inp, wrong(marker_rows))):
            return False

    # Marker columns are irrelevant; moving a marker within the same row should change
    # only row content, not the offset trace.
    for r in marked:
        red_cols = list(map(int, np.flatnonzero(inp[r] == RED)))
        non_red_cols = list(map(int, np.flatnonzero(inp[r] != RED)))
        if red_cols and non_red_cols:
            moved = inp.copy()
            src = red_cols[0]
            dst = non_red_cols[0]
            moved[r, src] = moved[r, dst]
            moved[r, dst] = RED
            if not np.array_equal(_compute_offsets(_marker_rows(moved)), offsets):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    max_w = min(MAX_WIDTH, MAX_OUTPUT_SIDE - h + 1)
    if max_w < MIN_WIDTH:
        return None
    w = random.randint(MIN_WIDTH, max_w)

    trigger_rows = _sample_trigger_rows(h)
    if trigger_rows is None:
        return None

    palette_size = random.randint(MIN_COLOR_COUNT, MAX_COLOR_COUNT)
    palette = random.sample(NON_RED_COLORS, palette_size)
    base = _fill_base_grid(h, w, palette)
    inp = _place_markers(base, trigger_rows, palette)

    if not _semantic_checks(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid zigzag-shear input")
