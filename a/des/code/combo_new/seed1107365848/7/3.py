import random

import numpy as np


BG = 5
ARROW = 8
COLOR_POOL = (0, 1, 2, 3, 4, 6, 7, 9)

GRID_H_RANGE = (12, 25)
GRID_W_RANGE = (14, 29)
N_MAP_RANGE = (2, 4)
N_RECT_RANGE = (4, 8)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 7)
ACTIVE_ROW_CAP = 5

GEN_RETRIES = 400
PLACEMENT_RETRIES = 250


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _parse_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    row = grid[0, 1:]
    mapping: dict[int, int] = {}
    for start, end in _runs_1d(row != bg):
        assert end == start + 1, "each legend group must contain exactly two adjacent cells"
        src = int(row[start])
        dst = int(row[end])
        assert src != bg and dst != bg, "legend colors must be foreground"
        assert src != dst, "legend source and target colors must differ"
        assert src not in mapping, "legend source colors must be unique"
        mapping[src] = dst
    assert mapping, "expected at least one legend pair on the top row"
    return mapping


def _active_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(1, grid.shape[0]) if int(grid[r, 0]) == ARROW]


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    mapping = _parse_legend(g, bg)
    out = g.copy()
    h, w = out.shape

    for r in _active_rows(g):
        c = 1
        while c < w:
            color = int(out[r, c])
            if color == bg:
                c += 1
                continue
            end = c + 1
            while end < w and int(out[r, end]) == color:
                end += 1
            if color in mapping:
                out[r, c:end] = mapping[color]
            c = end

    out[0, :] = bg
    out[:, 0] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor horizontal bands on arrow-marked rows using the source→target legend on the top edge.

    Input structure:
    - The background is gray.
    - The top row contains a color legend: each two-cell foreground pair encodes one source color followed by its target color, and pairs are separated by gray cells.
    - The leftmost column contains single azure marker pixels whose rows are the rows to process.
    - Away from the top row and left column, the scene contains several solid colored rectangles on the gray background.

    Rule:
    - Read the top-row legend as a source→target mapping.
    - For each row marked by an azure pixel in the left column, scan that row from left to right.
    - Whenever the scan reaches a non-background run whose color is a legend source color, recolor that whole contiguous run to the mapped target color.
    - Leave unmarked rows unchanged, and leave non-source runs unchanged even on marked rows.
    - In the output, clear the legend row and the marker column back to the gray background.

    Key insight:
    The top edge and left edge play different control roles: the top row tells which colors can change, while the azure edge markers tell which scanlines are activated.
    """
    return _apply_rule(grid)


def _legend_width(n_pairs: int) -> int:
    return 3 * n_pairs - 1


def _max_n_pairs(width: int) -> int:
    return min(N_MAP_RANGE[1], width // 3)


def _rect_padding_ok(candidate: dict[str, int], rects: list[dict[str, int]]) -> bool:
    r0, c0, r1, c1 = candidate["r0"], candidate["c0"], candidate["r1"], candidate["c1"]
    for rect in rects:
        if not (
            r1 + 1 < rect["r0"]
            or rect["r1"] + 1 < r0
            or c1 + 1 < rect["c0"]
            or rect["c1"] + 1 < c0
        ):
            return False
    return True


def _place_rectangles(
    height: int, width: int, colors: list[int], rng: random.Random
) -> list[dict[str, int]] | None:
    max_area = int((height - 2) * (width - 2) * 0.38)
    area_used = 0
    rects: list[dict[str, int]] = []

    max_h = min(RECT_H_RANGE[1], height - 4)
    max_w = min(RECT_W_RANGE[1], width - 4)
    if max_h < RECT_H_RANGE[0] or max_w < RECT_W_RANGE[0]:
        return None

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rh = rng.randint(RECT_H_RANGE[0], max_h)
            rw = rng.randint(RECT_W_RANGE[0], max_w)
            if area_used + rh * rw > max_area:
                continue

            r0 = rng.randint(2, height - rh - 1)
            c0 = rng.randint(2, width - rw - 1)
            candidate = {
                "r0": r0,
                "c0": c0,
                "r1": r0 + rh - 1,
                "c1": c0 + rw - 1,
                "color": int(color),
            }
            if not _rect_padding_ok(candidate, rects):
                continue

            rects.append(candidate)
            area_used += rh * rw
            placed = True
            break
        if not placed:
            return None

    return rects


def _rows_hit_by(rects: list[dict[str, int]], colors: set[int]) -> set[int]:
    rows: set[int] = set()
    for rect in rects:
        if rect["color"] not in colors:
            continue
        rows.update(range(rect["r0"], rect["r1"] + 1))
    return rows


def _choose_active_rows(
    height: int,
    rects: list[dict[str, int]],
    source_colors: set[int],
    rng: random.Random,
) -> list[int] | None:
    eligible_rows = list(range(2, height - 1))
    source_rows = sorted(_rows_hit_by(rects, source_colors))
    if not source_rows:
        return None

    quiet_rows = [r for r in eligible_rows if r not in source_rows]
    n_active = rng.randint(1, min(ACTIVE_ROW_CAP, len(eligible_rows)))

    include_quiet = bool(quiet_rows) and rng.choice([False, True])
    rows = {rng.choice(source_rows)}
    if include_quiet and len(rows) < n_active:
        rows.add(rng.choice(quiet_rows))

    remaining_pool = [r for r in eligible_rows if r not in rows]
    while len(rows) < n_active and remaining_pool:
        pick = rng.choice(remaining_pool)
        rows.add(pick)
        remaining_pool.remove(pick)

    active_rows = sorted(rows)
    changed_rect_exists = False
    partial_band_exists = False
    for rect in rects:
        if rect["color"] not in source_colors:
            continue
        covered = [r for r in active_rows if rect["r0"] <= r <= rect["r1"]]
        if covered:
            changed_rect_exists = True
        if covered and len(covered) < (rect["r1"] - rect["r0"] + 1):
            partial_band_exists = True

    if not changed_rect_exists or not partial_band_exists:
        return None
    return active_rows


def _build_input(
    height: int,
    width: int,
    mapping: dict[int, int],
    rects: list[dict[str, int]],
    active_rows: list[int],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)

    pairs = list(mapping.items())
    rng.shuffle(pairs)
    start = rng.randint(1, width - _legend_width(len(pairs)))
    c = start
    for idx, (src, dst) in enumerate(pairs):
        grid[0, c] = int(src)
        grid[0, c + 1] = int(dst)
        c += 2
        if idx != len(pairs) - 1:
            c += 1

    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]

    for row in active_rows:
        grid[row, 0] = ARROW

    return grid


def _expected_output(
    grid: np.ndarray,
    mapping: dict[int, int],
    rects: list[dict[str, int]],
    active_rows: list[int],
) -> np.ndarray:
    out = grid.copy()
    active = set(active_rows)
    for rect in rects:
        color = rect["color"]
        if color not in mapping:
            continue
        for row in range(rect["r0"], rect["r1"] + 1):
            if row in active:
                out[row, rect["c0"] : rect["c1"] + 1] = mapping[color]

    out[0, :] = BG
    out[:, 0] = BG
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)

    max_pairs = _max_n_pairs(width)
    if max_pairs < N_MAP_RANGE[0]:
        return None
    n_pairs = rng.randint(N_MAP_RANGE[0], max_pairs)

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    source_colors = colors[:n_pairs]
    target_colors = colors[n_pairs : 2 * n_pairs]
    mapping = {int(src): int(dst) for src, dst in zip(source_colors, target_colors)}
    stable_colors = target_colors + colors[2 * n_pairs :]
    if not stable_colors:
        stable_colors = target_colors[:]

    n_rectangles = rng.randint(max(N_RECT_RANGE[0], n_pairs + 1), N_RECT_RANGE[1])
    rect_colors = source_colors[:] + [rng.choice(stable_colors)]
    while len(rect_colors) < n_rectangles:
        rect_colors.append(rng.choice(source_colors + stable_colors))
    rng.shuffle(rect_colors)

    rects = _place_rectangles(height, width, rect_colors, rng)
    if rects is None:
        return None

    active_rows = _choose_active_rows(height, rects, set(source_colors), rng)
    if active_rows is None:
        return None

    grid = _build_input(height, width, mapping, rects, active_rows, rng)
    if _mode_color(grid) != BG:
        return None
    if _parse_legend(grid, BG) != mapping:
        return None
    if _active_rows(grid) != active_rows:
        return None

    expected = _expected_output(grid, mapping, rects, active_rows)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    changed_core = actual[1:, 1:] != grid[1:, 1:]
    if not np.any(changed_core):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
