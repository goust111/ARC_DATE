from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location
import random


def _line_runs(pattern: np.ndarray, axis: int) -> list[tuple[int, int, tuple[int, ...]]]:
    lines = pattern if axis == 0 else pattern.T
    runs: list[tuple[int, int, tuple[int, ...]]] = []
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or not np.array_equal(lines[i], lines[start]):
            runs.append((start, i, tuple(map(int, lines[start]))))
            start = i
    return runs


def _stretch_run_index(runs: list[tuple[int, int, tuple[int, ...]]]) -> int:
    color_counts = [len(set(line)) for _start, _end, line in runs]
    max_count = max(color_counts)
    candidates = [i for i, count in enumerate(color_counts) if count == max_count]
    if len(candidates) == 1:
        return candidates[0]

    lengths = [end - start for start, end, _line in runs]
    max_length = max(lengths[i] for i in candidates)
    candidates = [i for i in candidates if lengths[i] == max_length]
    assert len(candidates) == 1, "expected one structurally stretchable line run"
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_length: int) -> list[int]:
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    fixed_length = sum(end - start for i, (start, end, _line) in enumerate(runs) if i != stretch_i)
    stretch_length = int(target_length) - fixed_length
    assert stretch_length > 0, "target is too small for the fixed exemplar bands"

    indices: list[int] = []
    for i, (start, end, _line) in enumerate(runs):
        if i == stretch_i:
            indices.extend([start] * stretch_length)
        else:
            indices.extend(range(start, end))
    assert len(indices) == int(target_length)
    return indices


def _stretch_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row_indices = _axis_indices(pattern, 0, shape[0])
    col_indices = _axis_indices(pattern, 1, shape[1])
    return pattern[np.ix_(row_indices, col_indices)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each solid rectangle with a size-adjusted copy of the multicolor exemplar rectangle.

    Input structure
    - The background separates several solid rectangular foreground objects.
    - Exactly one rectangle is a multicolor exemplar; its four corners share the outer color.
    - Every other foreground rectangle is solid and uses that same outer color.

    Rule
    - Run-length encode the exemplar's identical rows and identical columns.
    - In each axis, keep all boundary runs fixed and resize the single interior run:
      choose the run whose representative line has the most colors, using unique
      run width/height only for the degenerate equal-complexity stripe case.
    - Replace each solid outer-color rectangle by the resulting stretched exemplar pattern.

    Key insight
    - The exemplar is not scaled geometrically; only its interior row and column runs stretch while the visible bands stay fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) >= 2, "expected one exemplar and at least one target rectangle"

    regions: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for component in components:
        b = component["bbox"]
        patch = crop_bbox(g, b)
        assert int(component["area"]) == bbox_h(b) * bbox_w(b), "components must be solid rectangles"
        assert np.all(patch != bg), "component bbox must not contain background holes"
        regions.append((b, patch))

    exemplar_regions = [(b, patch) for b, patch in regions if len(np.unique(patch)) > 1]
    assert len(exemplar_regions) == 1, "expected exactly one multicolor exemplar"
    exemplar_bbox, exemplar = exemplar_regions[0]

    outer = int(exemplar[0, 0])
    corners = [exemplar[0, 0], exemplar[0, -1], exemplar[-1, 0], exemplar[-1, -1]]
    assert all(int(corner) == outer for corner in corners), "exemplar corners must share outer color"

    out = g.copy()
    for b, patch in regions:
        if b == exemplar_bbox:
            continue
        colors = np.unique(patch)
        assert len(colors) == 1 and int(colors[0]) == outer, "targets must be solid outer-color rectangles"
        r0, c0, r1, c1 = b
        out[r0 : r1 + 1, c0 : c1 + 1] = _stretch_pattern(exemplar, patch.shape)

    return out

# --------------- generator (o5x) ---------------

# Canvas and placement.
GRID_HEIGHT_BOUNDS = (27, 30)
GRID_WIDTH = 30
BACKGROUND = 0
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1

# Palette and band counts.
COLOR_RANGE = range(1, 10)
PALETTE_SIZE_BOUNDS = (3, 4)
ROW_RUN_COUNT_BOUNDS = (2, 5)
COL_RUN_COUNT_BOUNDS = (3, 7)

# Exemplar and run-size bounds.
EXEMPLAR_HEIGHT_BOUNDS = (9, 11)
EXEMPLAR_WIDTH_BOUNDS = (11, 16)
RUN_WIDTH_BOUNDS = (1, 5)
STRETCH_WIDTH_MIN = 2

# Target rectangles and retries.
TARGET_HEIGHT_BOUNDS = (7, 15)
TARGET_WIDTH_BOUNDS = (9, 16)
SINGLE_TARGET_PROB = 0.35
MIN_STRETCH_LINE_COLORS = 2
MAX_GENERATION_ATTEMPTS = 300


def _gen_sequence(
    n: int,
    colors: list[int],
    *,
    fixed: dict[int, int],
    min_distinct: int,
    avoid_adjacent: bool = True,
) -> list[int] | None:
    fixed = {int(k): int(v) for k, v in fixed.items()}
    if any(k < 0 or k >= n for k in fixed):
        return None

    seq: list[int | None] = [None] * n

    def rec(i: int) -> bool:
        if i == n:
            return len(set(int(v) for v in seq if v is not None)) >= int(min_distinct)

        choices = [fixed[i]] if i in fixed else list(colors)
        random.shuffle(choices)
        for val in choices:
            val = int(val)
            if avoid_adjacent and i > 0 and seq[i - 1] == val:
                continue
            if avoid_adjacent and i + 1 in fixed and fixed[i + 1] == val:
                continue
            seq[i] = val
            if rec(i + 1):
                return True
            seq[i] = None
        return False

    if rec(0):
        return [int(v) for v in seq if v is not None]
    return None


def _gen_distribute(total: int, count: int, lo: int, hi: int) -> list[int] | None:
    vals: list[int] = []
    remaining = int(total)
    for i in range(int(count)):
        slots = int(count) - i - 1
        a = max(int(lo), remaining - slots * int(hi))
        b = min(int(hi), remaining - slots * int(lo))
        if a > b:
            return None
        val = random.randint(a, b)
        vals.append(val)
        remaining -= val
    return vals if remaining == 0 else None


def _gen_run_widths(count: int, total_lo: int, total_hi: int, stretch_i: int) -> list[int] | None:
    run_lo, run_hi = RUN_WIDTH_BOUNDS
    stretch_options: list[tuple[int, int, int]] = []
    for stretch_w in range(STRETCH_WIDTH_MIN, run_hi + 1):
        lo = stretch_w + (count - 1) * run_lo
        hi = stretch_w + (count - 1) * (stretch_w - 1)
        a = max(int(total_lo), lo)
        b = min(int(total_hi), hi)
        if a <= b:
            stretch_options.append((stretch_w, a, b))
    if not stretch_options:
        return None

    stretch_w, a, b = random.choice(stretch_options)
    total = random.randint(a, b)
    rest = _gen_distribute(total - stretch_w, count - 1, run_lo, stretch_w - 1)
    if rest is None:
        return None

    widths: list[int] = []
    rest_i = 0
    for i in range(count):
        if i == stretch_i:
            widths.append(stretch_w)
        else:
            widths.append(rest[rest_i])
            rest_i += 1
    return widths


def _fill_seed_stretch_lines(
    seed: np.ndarray,
    colors: list[int],
    outer: int,
    non_outer: list[int],
    row_stretch: int,
    col_stretch: int,
) -> bool:
    row_count, col_count = seed.shape

    row_fixed: dict[int, int] = {}
    if row_stretch in (0, row_count - 1):
        row_fixed[0] = int(outer)
        row_fixed[col_count - 1] = int(outer)
    if not (row_stretch in (0, row_count - 1) and col_stretch in (0, col_count - 1)):
        row_fixed[col_stretch] = int(random.choice(non_outer))

    row_seq = _gen_sequence(
        col_count,
        colors,
        fixed=row_fixed,
        min_distinct=MIN_STRETCH_LINE_COLORS,
    )
    if row_seq is None:
        return False
    seed[row_stretch, :] = row_seq

    col_fixed: dict[int, int] = {row_stretch: int(row_seq[col_stretch])}
    if col_stretch in (0, col_count - 1):
        col_fixed[0] = int(outer)
        col_fixed[row_count - 1] = int(outer)

    col_seq = _gen_sequence(
        row_count,
        colors,
        fixed=col_fixed,
        min_distinct=MIN_STRETCH_LINE_COLORS,
    )
    if col_seq is None:
        col_seq = _gen_sequence(
            row_count,
            colors,
            fixed=col_fixed,
            min_distinct=MIN_STRETCH_LINE_COLORS,
            avoid_adjacent=False,
        )
    if col_seq is None:
        return False
    seed[:, col_stretch] = col_seq
    return True


def _is_valid_banded_exemplar(
    exemplar: np.ndarray,
    colors: list[int],
    row_count: int,
    col_count: int,
    row_stretch: int,
    col_stretch: int,
) -> bool:
    ex_h_lo, ex_h_hi = EXEMPLAR_HEIGHT_BOUNDS
    ex_w_lo, ex_w_hi = EXEMPLAR_WIDTH_BOUNDS
    if not (ex_h_lo <= exemplar.shape[0] <= ex_h_hi):
        return False
    if not (ex_w_lo <= exemplar.shape[1] <= ex_w_hi):
        return False
    if len(set(map(int, exemplar.ravel()))) != len(colors):
        return False

    row_runs = _line_runs(exemplar, 0)
    col_runs = _line_runs(exemplar, 1)
    if len(row_runs) != row_count or len(col_runs) != col_count:
        return False
    if _stretch_run_index(row_runs) != row_stretch:
        return False
    if _stretch_run_index(col_runs) != col_stretch:
        return False
    return True


def _gen_banded_exemplar() -> tuple[np.ndarray, int, int, int] | None:
    colors = random.sample(COLOR_RANGE, random.randint(*PALETTE_SIZE_BOUNDS))
    outer = int(random.choice(colors))
    non_outer = [c for c in colors if c != outer]

    run_hi = RUN_WIDTH_BOUNDS[1]
    row_count_options = [
        n
        for n in range(ROW_RUN_COUNT_BOUNDS[0], ROW_RUN_COUNT_BOUNDS[1] + 1)
        if n * run_hi >= EXEMPLAR_HEIGHT_BOUNDS[0] and n <= EXEMPLAR_HEIGHT_BOUNDS[1]
    ]
    row_count = random.choice(row_count_options)
    col_count = random.randint(*COL_RUN_COUNT_BOUNDS)

    row_stretch = random.randrange(row_count)
    if row_count == ROW_RUN_COUNT_BOUNDS[0]:
        col_stretch = random.randrange(1, col_count - 1)
    else:
        col_stretch = random.randrange(col_count)

    row_widths = _gen_run_widths(row_count, *EXEMPLAR_HEIGHT_BOUNDS, row_stretch)
    col_widths = _gen_run_widths(col_count, *EXEMPLAR_WIDTH_BOUNDS, col_stretch)
    if row_widths is None or col_widths is None:
        return None

    seed = np.full((row_count, col_count), int(outer), dtype=int)
    if not _fill_seed_stretch_lines(seed, colors, outer, non_outer, row_stretch, col_stretch):
        return None
    for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        seed[r, c] = int(outer)

    exemplar = np.repeat(np.repeat(seed, row_widths, axis=0), col_widths, axis=1)
    if not _is_valid_banded_exemplar(exemplar, colors, row_count, col_count, row_stretch, col_stretch):
        return None

    fixed_rows = sum(w for i, w in enumerate(row_widths) if i != row_stretch)
    fixed_cols = sum(w for i, w in enumerate(col_widths) if i != col_stretch)
    return exemplar, int(outer), int(fixed_rows), int(fixed_cols)


def _gen_target_shapes(fixed_rows: int, fixed_cols: int) -> list[tuple[int, int]] | None:
    min_target_h = max(TARGET_HEIGHT_BOUNDS[0], fixed_rows + 1)
    min_target_w = max(TARGET_WIDTH_BOUNDS[0], fixed_cols + 1)
    if min_target_h > TARGET_HEIGHT_BOUNDS[1] or min_target_w > TARGET_WIDTH_BOUNDS[1]:
        return None

    target_count = 1 if random.random() < SINGLE_TARGET_PROB else 2
    return [
        (random.randint(min_target_h, TARGET_HEIGHT_BOUNDS[1]), random.randint(min_target_w, TARGET_WIDTH_BOUNDS[1]))
        for _ in range(target_count)
    ]


def _place_objects(
    objects: list[tuple[np.ndarray, np.ndarray]],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    out = np.zeros((height, width), dtype=int)
    inp = np.zeros((height, width), dtype=int)

    random.shuffle(objects)
    objects.sort(key=lambda item: item[0].size, reverse=True)
    try:
        for out_patch, in_patch in objects:
            r, c = random_free_location(
                out,
                out_patch,
                transparent=BACKGROUND,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
            )
            paste(out, out_patch, r, c)
            paste(inp, in_patch, r, c)
    except ValueError:
        return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    built = _gen_banded_exemplar()
    if built is None:
        return None
    exemplar, outer, fixed_rows, fixed_cols = built

    target_shapes = _gen_target_shapes(fixed_rows, fixed_cols)
    if target_shapes is None:
        return None

    height = random.randint(*GRID_HEIGHT_BOUNDS)
    objects: list[tuple[np.ndarray, np.ndarray]] = [(exemplar, exemplar)]
    for shape in target_shapes:
        stretched = _stretch_pattern(exemplar, shape)
        solid = np.full(shape, int(outer), dtype=int)
        objects.append((stretched, solid))

    placed = _place_objects(objects, height, GRID_WIDTH)
    if placed is None:
        return None

    inp, out = placed
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 40f6cd08 input")
