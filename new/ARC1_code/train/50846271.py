import itertools

import numpy as np

from arc_utils import Color
import math
import random


def _red_spans_in_non_black_runs(line: np.ndarray) -> list[int]:
    spans: list[int] = []
    i = 0
    n = len(line)
    while i < n:
        if int(line[i]) == Color.BLACK:
            i += 1
            continue

        reds: list[int] = []
        j = i
        while j < n and int(line[j]) != Color.BLACK:
            if int(line[j]) == Color.RED:
                reds.append(j)
            j += 1

        if len(reds) >= 2:
            spans.append(max(reds) - min(reds) + 1)
        i = j

    return spans


def _infer_radius(g: np.ndarray) -> int:
    spans: list[int] = []
    for r in range(g.shape[0]):
        spans.extend(_red_spans_in_non_black_runs(g[r, :]))
    for c in range(g.shape[1]):
        spans.extend(_red_spans_in_non_black_runs(g[:, c]))

    odd_spans = [span for span in spans if span >= 3 and span % 2 == 1]
    assert odd_spans
    radius = (max(odd_spans) - 1) // 2
    assert radius >= 1
    return radius


def _plus_cells(shape: tuple[int, int], r: int, c: int, radius: int) -> tuple[tuple[int, int], ...]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for dc in range(-radius, radius + 1):
        cc = c + dc
        if 0 <= cc < w:
            cells.append((r, cc))
    for dr in range(-radius, radius + 1):
        rr = r + dr
        if 0 <= rr < h and rr != r:
            cells.append((rr, c))
    return tuple(cells)


def _candidate_pluses(g: np.ndarray, radius: int) -> list[tuple[int, int, frozenset[tuple[int, int]], tuple[tuple[int, int], ...]]]:
    h, w = g.shape
    candidates: list[tuple[int, int, frozenset[tuple[int, int]], tuple[tuple[int, int], ...]]] = []
    for r in range(h):
        for c in range(w):
            cells = _plus_cells(g.shape, r, c, radius)
            if any(int(g[rr, cc]) == Color.BLACK for rr, cc in cells):
                continue

            red_cells = frozenset((rr, cc) for rr, cc in cells if int(g[rr, cc]) == Color.RED)
            if len(red_cells) < 3:
                continue

            row_red = sum(
                int(g[r, cc]) == Color.RED
                for cc in range(max(0, c - radius), min(w, c + radius + 1))
            )
            col_red = sum(
                int(g[rr, c]) == Color.RED
                for rr in range(max(0, r - radius), min(h, r + radius + 1))
            )
            if row_red >= 1 and col_red >= 1:
                candidates.append((r, c, red_cells, cells))

    assert candidates
    return candidates


def _select_pluses(
    candidates: list[tuple[int, int, frozenset[tuple[int, int]], tuple[tuple[int, int], ...]]],
    all_red: set[tuple[int, int]],
) -> list[tuple[int, int, frozenset[tuple[int, int]], tuple[tuple[int, int], ...]]]:
    covered: set[tuple[int, int]] = set()
    selected: list[tuple[int, int, frozenset[tuple[int, int]], tuple[tuple[int, int], ...]]] = []

    while covered != all_red:
        useful = [candidate for candidate in candidates if candidate[2] - covered]
        assert useful

        best_new_count = max(len(candidate[2] - covered) for candidate in useful)
        best = [candidate for candidate in useful if len(candidate[2] - covered) == best_new_count]
        for a, b in itertools.combinations(best, 2):
            assert not ((a[2] - covered) & (b[2] - covered))

        for candidate in sorted(best, key=lambda item: (item[0], item[1])):
            new_red = candidate[2] - covered
            if new_red:
                selected.append(candidate)
                covered.update(candidate[2])

    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey cells that complete each partially red equal-arm plus sign.

    Input structure
    The grid contains black background, grey distractor/substrate cells, and red marker pixels.
    The red pixels lie on one or more incomplete axis-aligned plus signs with one shared radius.

    Rule
    Infer the shared radius from the largest odd red span inside a non-black horizontal or vertical run.
    Find radius-sized plus masks whose in-bounds cells are all non-black and whose row/column both contain red evidence.
    Select the masks that cover all red pixels, strongest new red evidence first, and recolor only their grey cells teal.

    Key insight
    Black cells break candidate spans; within the noisy grey field, the valid pluses are the crosses that explain the red markers.
    """
    g = np.asarray(grid, dtype=int)
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.GREY, Color.RED}

    all_red = set(map(tuple, np.argwhere(g == Color.RED)))
    assert all_red

    radius = _infer_radius(g)
    selected = _select_pluses(_candidate_pluses(g, radius), all_red)

    out = g.copy()
    for _r, _c, _red_cells, cells in selected:
        for rr, cc in cells:
            if out[rr, cc] == Color.GREY:
                out[rr, cc] = Color.TEAL

    return out

# --------------- generator (o5x) ---------------

# Canvas and plus geometry bounds.
GRID_HEIGHT_RANGE = (11, 20)
GRID_WIDTH_RANGE = (12, 22)
PLUS_RADIUS_RANGE = (2, 3)
PLUS_COUNT_RANGE = (1, 4)

# Red marker constraints within each true plus.
MIN_RED_CELLS = 3
RED_FRACTION_RANGE = (0.33, 0.78)
SIGNAL_AXES = ("row", "col")

# Background color proportions after pluses and red markers are fixed.
BLACK_FRACTION_RANGE = (0.40, 0.57)
GREY_FRACTION_RANGE = (0.40, 0.55)

# Retry budgets for placement and whole-grid generation.
CENTER_PLACEMENT_RETRIES = 100
GENERATION_RETRIES = 300


def _gen_plus_cells(shape: tuple[int, int], r: int, c: int, radius: int) -> tuple[tuple[int, int], ...]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for dc in range(-radius, radius + 1):
        cc = c + dc
        if 0 <= cc < w:
            cells.append((r, cc))
    for dr in range(-radius, radius + 1):
        rr = r + dr
        if 0 <= rr < h and rr != r:
            cells.append((rr, c))
    return tuple(cells)


def _gen_place_centers(h: int, w: int, radius: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(radius, h - radius) for c in range(radius, w - radius)]
    if not candidates:
        return None

    for _ in range(CENTER_PLACEMENT_RETRIES):
        random.shuffle(candidates)
        centers: list[tuple[int, int]] = []
        occupied: set[tuple[int, int]] = set()

        for r, c in candidates:
            cells = set(_gen_plus_cells((h, w), r, c, radius))
            if any(abs(r - rr) + abs(c - cc) <= 2 * radius for rr, cc in centers):
                continue
            if cells & occupied:
                continue
            if any(
                (rr, cc - 1) in occupied
                or (rr, cc + 1) in occupied
                or (rr - 1, cc) in occupied
                or (rr + 1, cc) in occupied
                for rr, cc in cells
            ):
                continue

            centers.append((r, c))
            occupied.update(cells)
            if len(centers) == count:
                return centers

    return None


def _gen_red_cells(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    *,
    signal_axis: str | None,
) -> set[tuple[int, int]]:
    h, w = shape
    r, c = center
    row_cells = [(r, cc) for cc in range(c - radius, c + radius + 1) if 0 <= cc < w]
    col_cells = [(rr, c) for rr in range(r - radius, r + radius + 1) if 0 <= rr < h]
    cells = set(row_cells) | set(col_cells)
    min_fraction, max_fraction = RED_FRACTION_RANGE
    min_count = max(MIN_RED_CELLS, math.ceil(min_fraction * len(cells)))
    max_count = min(len(cells), math.floor(max_fraction * len(cells)))

    if signal_axis == "row":
        chosen = set(row_cells)
        chosen.add(random.choice([cell for cell in col_cells if cell != (r, c)]))
        target = random.randint(len(chosen), max_count)
    elif signal_axis == "col":
        chosen = set(col_cells)
        chosen.add(random.choice([cell for cell in row_cells if cell != (r, c)]))
        target = random.randint(len(chosen), max_count)
    else:
        chosen = {
            random.choice([cell for cell in row_cells if cell != (r, c)]),
            random.choice([cell for cell in col_cells if cell != (r, c)]),
        }
        target = random.randint(min_count, max_count)

    remaining = list(cells - chosen)
    random.shuffle(remaining)
    chosen.update(remaining[: target - len(chosen)])
    return chosen


def _gen_same_true_arm(
    a: tuple[int, int],
    b: tuple[int, int],
    centers: list[tuple[int, int]],
    owner: dict[tuple[int, int], int],
    *,
    axis: str,
) -> bool:
    ia = owner.get(a)
    ib = owner.get(b)
    if ia is None or ia != ib:
        return False
    cr, cc = centers[ia]
    if axis == "row":
        return a[0] == b[0] == cr
    return a[1] == b[1] == cc


def _gen_break_line_red_runs(
    out: np.ndarray,
    centers: list[tuple[int, int]],
    owner: dict[tuple[int, int], int],
    axis: str,
) -> bool:
    line_count = out.shape[0] if axis == "row" else out.shape[1]
    for index in range(line_count):
        if axis == "row":
            reds = [c for c in range(out.shape[1]) if int(out[index, c]) == Color.RED]
        else:
            reds = [r for r in range(out.shape[0]) if int(out[r, index]) == Color.RED]

        for a, b in zip(reds, reds[1:]):
            if axis == "row":
                first = (index, a)
                second = (index, b)
                has_black = any(int(out[index, c]) == Color.BLACK for c in range(a + 1, b))
                choices = [(index, c) for c in range(a + 1, b) if (index, c) not in owner]
            else:
                first = (a, index)
                second = (b, index)
                has_black = any(int(out[r, index]) == Color.BLACK for r in range(a + 1, b))
                choices = [(r, index) for r in range(a + 1, b) if (r, index) not in owner]
            if _gen_same_true_arm(first, second, centers, owner, axis=axis):
                continue
            if has_black:
                continue
            if not choices:
                return False
            rr, cc = random.choice(choices)
            out[rr, cc] = Color.BLACK
    return True


def _gen_break_accidental_red_runs(
    out: np.ndarray,
    centers: list[tuple[int, int]],
    owner: dict[tuple[int, int], int],
) -> bool:
    return _gen_break_line_red_runs(out, centers, owner, "row") and _gen_break_line_red_runs(
        out, centers, owner, "col"
    )


def _gen_candidate_has_plus_evidence(out: np.ndarray, r: int, c: int, radius: int) -> bool:
    h, w = out.shape
    row_red = sum(
        int(out[r, cc]) == Color.RED
        for cc in range(max(0, c - radius), min(w, c + radius + 1))
    )
    col_red = sum(
        int(out[rr, c]) == Color.RED
        for rr in range(max(0, r - radius), min(h, r + radius + 1))
    )
    return row_red >= 1 and col_red >= 1


def _gen_block_false_candidates(
    out: np.ndarray,
    centers: list[tuple[int, int]],
    radius: int,
    true_plus_cells: set[tuple[int, int]],
) -> bool:
    h, w = out.shape
    true_centers = set(centers)
    candidate_centers = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidate_centers)

    for r, c in candidate_centers:
        if (r, c) in true_centers:
            continue
        cells = _gen_plus_cells((h, w), r, c, radius)
        if any(int(out[rr, cc]) == Color.BLACK for rr, cc in cells):
            continue

        red_count = sum(int(out[rr, cc]) == Color.RED for rr, cc in cells)
        if red_count < MIN_RED_CELLS:
            continue

        if not _gen_candidate_has_plus_evidence(out, r, c, radius):
            continue

        blockers = [(rr, cc) for rr, cc in cells if (rr, cc) not in true_plus_cells]
        if not blockers:
            return False
        rr, cc = random.choice(blockers)
        out[rr, cc] = Color.BLACK

    return True


def _gen_add_background_black(
    out: np.ndarray,
    true_plus_cells: set[tuple[int, int]],
    red_count: int,
) -> bool:
    h, w = out.shape
    total = h * w
    current_black = int(np.sum(out == Color.BLACK))

    black_min, black_max = BLACK_FRACTION_RANGE
    grey_min, grey_max = GREY_FRACTION_RANGE
    black_low = math.ceil(black_min * total)
    black_high = math.floor(black_max * total)
    grey_low = math.ceil(grey_min * total)
    grey_high = math.floor(grey_max * total)

    target_low = max(black_low, total - red_count - grey_high, current_black)
    target_high = min(black_high, total - red_count - grey_low)

    free = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in true_plus_cells and int(out[r, c]) != Color.BLACK
    ]
    target_high = min(target_high, current_black + len(free))
    if target_low > target_high:
        return False

    target = random.randint(target_low, target_high)
    random.shuffle(free)
    for r, c in free[: target - current_black]:
        out[r, c] = Color.BLACK
    return True


def _gen_layout_params() -> tuple[int, int, int, int]:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    radius = random.randint(*PLUS_RADIUS_RANGE)
    count = random.randint(*PLUS_COUNT_RANGE)
    return h, w, radius, count


def _gen_render_pluses(
    h: int,
    w: int,
    radius: int,
    centers: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[tuple[int, int], int], set[tuple[int, int]], int]:
    out = np.full((h, w), Color.GREY, dtype=int)
    owner: dict[tuple[int, int], int] = {}
    true_plus_cells: set[tuple[int, int]] = set()
    red_count = 0

    signal_index = random.randrange(len(centers))
    signal_axis = random.choice(SIGNAL_AXES)
    for i, center in enumerate(centers):
        cells = _gen_plus_cells((h, w), center[0], center[1], radius)
        for cell in cells:
            owner[cell] = i
            true_plus_cells.add(cell)
            out[cell] = Color.TEAL

        axis = signal_axis if i == signal_index else None
        reds = _gen_red_cells((h, w), center, radius, signal_axis=axis)
        red_count += len(reds)
        for rr, cc in reds:
            out[rr, cc] = Color.RED

    return out, owner, true_plus_cells, red_count


def _gen_matches_solver(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    h, w, radius, count = _gen_layout_params()

    centers = _gen_place_centers(h, w, radius, count)
    if centers is None:
        return None

    out, owner, true_plus_cells, red_count = _gen_render_pluses(h, w, radius, centers)

    if not _gen_break_accidental_red_runs(out, centers, owner):
        return None
    if not _gen_block_false_candidates(out, centers, radius, true_plus_cells):
        return None

    if not _gen_add_background_black(out, true_plus_cells, red_count):
        return None

    inp = out.copy()
    inp[inp == Color.TEAL] = Color.GREY

    if not _gen_matches_solver(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 50846271 input")
