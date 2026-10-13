import random
import numpy as np


BG = 0
PIN = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (14, 22)
GRID_WIDTH_RANGE = (5, 8)
PALETTE_SIZE_RANGE = (4, 6)
GEN_RETRY_LIMIT = 400

CHAIN_TEMPLATES = [
    [0, 1, 1, 0],
    [0, 1, 1, 0, 2],
    [0, 1, 0, 0, 1],
    [0, 1, 2, 2, 1, 0],
    [0, 1, 2, 2, 1, 0, 3],
    [0, 1, 0, 0, 1, 0],
]

SINGLE_TEMPLATES = [
    [0, 0, 1],
    [0, 1, 1, 2],
    [0, 0, 1, 2],
    [0, 1, 1, 2, 3],
    [0, 1, 2, 2, 0],
]

STABLE_TEMPLATES = [
    [0, 1],
    [0, 1, 2],
    [0, 1, 0],
    [0, 1, 2, 0],
    [0, 1, 2, 1],
]

PIN_TEMPLATES = [
    [[0], [0]],
    [[0, 1], [1]],
    [[0, 1], [1, 0]],
    [[0], [1], [1]],
    [[0, 1], [1], [0]],
    [[0], [1, 0], [1]],
]


def _settle_segment(segment: np.ndarray) -> np.ndarray:
    segment = np.asarray(segment, dtype=int)
    colors = segment[segment != BG]
    out = np.zeros_like(segment)
    out[len(segment) - len(colors) :] = colors
    return out


def _erase_runs(segment: np.ndarray) -> tuple[np.ndarray, bool]:
    out = np.array(segment, dtype=int, copy=True)
    changed = False
    i = 0
    while i < len(out):
        val = int(out[i])
        if val == BG:
            i += 1
            continue
        j = i + 1
        while j < len(out) and int(out[j]) == val:
            j += 1
        if j - i >= 2:
            out[i:j] = BG
            changed = True
        i = j
    return out, changed


def _solve_segment(segment: np.ndarray) -> tuple[np.ndarray, int]:
    cur = np.array(segment, dtype=int, copy=True)
    rounds = 0
    while True:
        settled = _settle_segment(cur)
        reduced, changed = _erase_runs(settled)
        if not changed:
            return settled, rounds
        cur = reduced
        rounds += 1


def _solve_segment_one_round(segment: np.ndarray) -> np.ndarray:
    settled = _settle_segment(segment)
    reduced, changed = _erase_runs(settled)
    if not changed:
        return settled
    return _settle_segment(reduced)


def _solve_column(column: np.ndarray) -> tuple[np.ndarray, dict]:
    col = np.asarray(column, dtype=int)
    height = len(col)
    out = np.zeros(height, dtype=int)
    pin_rows = np.flatnonzero(col == PIN).tolist()
    out[pin_rows] = PIN

    start = 0
    max_rounds = 0
    had_elimination = False
    for stop in pin_rows + [height]:
        segment = col[start:stop]
        solved, rounds = _solve_segment(segment)
        out[start:stop] = solved
        max_rounds = max(max_rounds, rounds)
        had_elimination |= rounds > 0
        start = stop + 1

    info = {
        "pins": len(pin_rows),
        "active": bool(np.any((col != BG) & (col != PIN))),
        "max_rounds": max_rounds,
        "had_elimination": had_elimination,
    }
    return out, info


def _solve_column_one_round(column: np.ndarray) -> np.ndarray:
    col = np.asarray(column, dtype=int)
    height = len(col)
    out = np.zeros(height, dtype=int)
    pin_rows = np.flatnonzero(col == PIN).tolist()
    out[pin_rows] = PIN

    start = 0
    for stop in pin_rows + [height]:
        out[start:stop] = _solve_segment_one_round(col[start:stop])
        start = stop + 1
    return out


def _solve_column_ignore_pins(column: np.ndarray) -> np.ndarray:
    col = np.asarray(column, dtype=int)
    work = np.array(col, copy=True)
    pin_rows = np.flatnonzero(work == PIN)
    work[pin_rows] = BG
    solved, _ = _solve_segment(work)
    solved[pin_rows] = PIN
    return solved


def _solve_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = np.zeros((height, width), dtype=int)
    infos = []
    for c in range(width):
        out[:, c], info = _solve_column(g[:, c])
        infos.append(info)
    return out, infos


def _solve_grid_one_round(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = np.zeros((height, width), dtype=int)
    for c in range(width):
        out[:, c] = _solve_column_one_round(g[:, c])
    return out


def _solve_grid_ignore_pins(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = np.zeros((height, width), dtype=int)
    for c in range(width):
        out[:, c] = _solve_column_ignore_pins(g[:, c])
    return out


def _settle_grid_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = np.zeros((height, width), dtype=int)
    for c in range(width):
        col = g[:, c]
        pin_rows = np.flatnonzero(col == PIN).tolist()
        out[pin_rows, c] = PIN
        start = 0
        for stop in pin_rows + [height]:
            out[start:stop, c] = _settle_segment(col[start:stop])
            start = stop + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each column by repeated falling and same-color segment cancellation around fixed gray pins.

    Input structure:
    - The grid has a black background with floating colored single cells in multiple columns.
    - Some cells are gray pins that stay fixed in place and split a column into independent vertical compartments.
    - Colors may repeat within a column, with gaps between blocks and with pins sometimes separating matching colors.

    Rule:
    - Process every column independently. In each pin-delimited compartment, all colored cells fall straight down until resting on the bottom, a pin, or another settled colored cell.
    - After settling, remove every maximal vertical run of adjacent equal-colored cells whose length is at least two, turning those cells black.
    - Then let the remaining cells in that compartment fall again and repeat the settle-remove cycle until no more same-color runs remain. Keep all gray pins unchanged.

    Key insight:
    - This is a chain-reaction gravity process: eliminations create new gaps, those gaps cause further falls, and later meetings of same-color cells can trigger additional rounds that a one-pass solve would miss.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    out, _ = _solve_grid(g)
    return out


def _symbol_count(template) -> int:
    if template and isinstance(template[0], list):
        return len({symbol for seq in template for symbol in seq})
    return len(set(template))


def _pick_template(templates, max_colors: int):
    valid = [template for template in templates if _symbol_count(template) <= max_colors]
    return random.choice(valid)


def _colorize_sequence(template: list[int], palette: list[int]) -> list[int]:
    symbols = sorted(set(template))
    chosen = random.sample(palette, len(symbols))
    mapping = {symbol: color for symbol, color in zip(symbols, chosen)}
    return [mapping[symbol] for symbol in template]


def _colorize_segments(template: list[list[int]], palette: list[int]) -> list[list[int]]:
    symbols = sorted({symbol for seq in template for symbol in seq})
    chosen = random.sample(palette, len(symbols))
    mapping = {symbol: color for symbol, color in zip(symbols, chosen)}
    return [[mapping[symbol] for symbol in seq] for seq in template]


def _role_sequences(role: str, palette: list[int]) -> tuple[list[list[int]], list[bool]]:
    if role == "chain":
        chain = _colorize_sequence(_pick_template(CHAIN_TEMPLATES, len(palette)), palette)
        if random.random() < 0.5:
            return [chain], [True]
        stable = _colorize_sequence(_pick_template(STABLE_TEMPLATES, len(palette)), palette)
        if random.random() < 0.5:
            return [chain, stable], [True, True]
        return [stable, chain], [True, True]

    if role == "single":
        single = _colorize_sequence(_pick_template(SINGLE_TEMPLATES, len(palette)), palette)
        if random.random() < 0.45:
            stable = _colorize_sequence(_pick_template(STABLE_TEMPLATES, len(palette)), palette)
            if random.random() < 0.5:
                return [single, stable], [True, True]
            return [stable, single], [True, True]
        return [single], [True]

    if role == "stable":
        stable = _colorize_sequence(_pick_template(STABLE_TEMPLATES, len(palette)), palette)
        if random.random() < 0.35:
            extra = _colorize_sequence(_pick_template(STABLE_TEMPLATES, len(palette)), palette)
            if random.random() < 0.5:
                return [stable, extra], [True, True]
            return [extra, stable], [True, True]
        return [stable], [True]

    if role == "pin":
        segments = _colorize_segments(_pick_template(PIN_TEMPLATES, len(palette)), palette)
        return segments, [True] * len(segments)

    raise ValueError(f"unknown role: {role}")


def _segment_min_length(sequence: list[int], force_float: bool) -> int:
    extra = 1 if force_float and sequence else 0
    return max(len(sequence) + extra, 2 if sequence else 1)


def _distribute(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _sample_rows(start: int, end: int, count: int, force_float: bool) -> list[int]:
    if count == 0:
        return []
    choices = list(range(start, end + 1))
    assert len(choices) >= count
    bottom_pack = list(range(end - count + 1, end + 1))
    for _ in range(40):
        rows = sorted(random.sample(choices, count))
        if not force_float or rows != bottom_pack:
            return rows
    if force_float:
        rows = bottom_pack[:]
        rows[0] -= 1
        rows.sort()
        return rows
    return bottom_pack


def _render_column(height: int, sequences: list[list[int]], force_flags: list[bool]) -> np.ndarray | None:
    mins = [_segment_min_length(seq, force) for seq, force in zip(sequences, force_flags)]
    needed = sum(mins) + len(sequences) - 1
    if needed > height:
        return None

    lengths = mins[:]
    for idx, extra in enumerate(_distribute(height - needed, len(sequences))):
        lengths[idx] += extra

    col = np.zeros(height, dtype=int)
    cursor = 0
    for idx, (sequence, force_float, seg_len) in enumerate(zip(sequences, force_flags, lengths)):
        start = cursor
        end = cursor + seg_len - 1
        rows = _sample_rows(start, end, len(sequence), force_float)
        for row, color in zip(rows, sequence):
            col[row] = int(color)
        cursor = end + 1
        if idx < len(sequences) - 1:
            col[cursor] = PIN
            cursor += 1
    return col


def _build_column(height: int, role: str, palette: list[int]) -> np.ndarray | None:
    for _ in range(60):
        sequences, force_flags = _role_sequences(role, palette)
        col = _render_column(height, sequences, force_flags)
        if col is not None:
            return col
    return None


def _active_color_stats(grid: np.ndarray) -> tuple[int, int]:
    mask = (grid != BG) & (grid != PIN)
    if not np.any(mask):
        return 0, 0
    return int(mask.sum()), int(len(np.unique(grid[mask])))


def _valid_attempt(grid: np.ndarray, roles: list[str]) -> bool:
    out, infos = _solve_grid(grid)
    one_round = _solve_grid_one_round(grid)
    ignore_pins = _solve_grid_ignore_pins(grid)
    settled = _settle_grid_only(grid)

    if np.array_equal(grid, settled):
        return False
    if np.array_equal(grid, out):
        return False
    if np.array_equal(out, one_round):
        return False
    if np.array_equal(out, ignore_pins):
        return False

    max_rounds = max(info["max_rounds"] for info in infos)
    if max_rounds < 2:
        return False
    if not any(info["max_rounds"] == 1 for info in infos):
        return False
    if not any(info["max_rounds"] == 0 and info["active"] for info in infos):
        return False
    if not any(info["pins"] > 0 for info in infos):
        return False
    if not any(info["pins"] == 0 and info["active"] for info in infos):
        return False

    active_cols = sum(info["active"] for info in infos)
    if active_cols < 4:
        return False

    inp_cells, inp_colors = _active_color_stats(grid)
    out_cells, out_colors = _active_color_stats(out)
    if inp_cells < 12 or inp_colors < 3:
        return False
    if out_cells < 3 or out_colors < 2:
        return False

    pin_sensitive = False
    chain_sensitive = False
    for c in range(grid.shape[1]):
        actual_col, info = _solve_column(grid[:, c])
        if not np.array_equal(actual_col, _solve_column_ignore_pins(grid[:, c])):
            pin_sensitive = True
        if info["max_rounds"] >= 2 and not np.array_equal(actual_col, _solve_column_one_round(grid[:, c])):
            chain_sensitive = True
    if not pin_sensitive or not chain_sensitive:
        return False

    # Ensure the intended role mix really appears, not just by label.
    role_set = set(roles)
    if not {"chain", "single", "stable", "pin"}.issubset(role_set):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(PALETTE, palette_size)

    roles = ["chain", "single", "stable", "pin"]
    extra_roles = ["stable", "single", "pin", "chain"]
    extra_weights = [0.35, 0.25, 0.25, 0.15]
    for _ in range(width - len(roles)):
        roles.append(random.choices(extra_roles, weights=extra_weights, k=1)[0])
    random.shuffle(roles)

    grid = np.zeros((height, width), dtype=int)
    for c, role in enumerate(roles):
        col = _build_column(height, role, palette)
        if col is None:
            return None
        grid[:, c] = col

    if not _valid_attempt(grid, roles):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-reaction gravity puzzle input")
