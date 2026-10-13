from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

CAT_NONE = 0
CAT_LEFT = 1
CAT_RIGHT = 2
CAT_BOTH = 3

GEN_RETRIES = 400
FILL_RETRIES = 300


def _find_bottom_bar_row(grid: np.ndarray) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(np.isin(row, (GREY, RED))) and np.count_nonzero(row == RED) <= 1:
            candidates.append(r)
    assert candidates and candidates[-1] == grid.shape[0] - 1, "expected a bottom grey bar row"
    return int(candidates[-1])


def _find_separator_col(grid: np.ndarray, bar_row: int) -> int:
    candidates = []
    for c in range(grid.shape[1]):
        if not np.all(grid[:bar_row, c] == RED):
            continue
        if grid[bar_row, c] not in (RED, GREY):
            continue
        left = grid[:bar_row, :c]
        right = grid[:bar_row, c + 1 :]
        if left.size == 0 or left.shape != right.shape:
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one red separator column"
    return int(candidates[0])


def _alternate_moves(blue_starts: list[tuple[int, int]], red_starts: list[tuple[int, int]]) -> list[tuple[int, tuple[int, int]]]:
    moves: list[tuple[int, tuple[int, int]]] = []
    bi = 0
    ri = 0
    blue_turn = True
    while bi < len(blue_starts) or ri < len(red_starts):
        if blue_turn:
            if bi < len(blue_starts):
                moves.append((BLUE, blue_starts[bi]))
                bi += 1
            elif ri < len(red_starts):
                moves.append((RED, red_starts[ri]))
                ri += 1
        else:
            if ri < len(red_starts):
                moves.append((RED, red_starts[ri]))
                ri += 1
            elif bi < len(blue_starts):
                moves.append((BLUE, blue_starts[bi]))
                bi += 1
        blue_turn = not blue_turn
    return moves


def _simulate_categories(
    categories: np.ndarray,
    *,
    block_unmoved: bool = False,
    green_blocks: bool = True,
    paint_green: bool = True,
    all_down: bool = False,
    all_up: bool = False,
    return_trace: bool = False,
):
    active_h, width = categories.shape
    out = np.full((active_h + 1, width), BLACK, dtype=int)
    out[-1, :] = GREY

    settled = np.zeros((active_h, width), dtype=bool)
    trace = {"moves": [], "start_conflict": False}

    green_coords = [tuple(map(int, rc)) for rc in np.argwhere(categories == CAT_BOTH)]
    if paint_green:
        for r, c in green_coords:
            out[r, c] = GREEN
    if green_blocks:
        for r, c in green_coords:
            settled[r, c] = True

    if all_down or all_up:
        active = [tuple(map(int, rc)) for rc in np.argwhere(categories != CAT_NONE)]
        active.sort(key=lambda rc: (-rc[0], rc[1]) if all_down else (rc[0], rc[1]))
        blue_starts = active if all_down else []
        red_starts = active if all_up else []
    else:
        blue_starts = [tuple(map(int, rc)) for rc in np.argwhere(categories == CAT_LEFT)]
        red_starts = [tuple(map(int, rc)) for rc in np.argwhere(categories == CAT_RIGHT)]
        blue_starts.sort(key=lambda rc: (-rc[0], rc[1]))
        red_starts.sort(key=lambda rc: (rc[0], rc[1]))

    moves = _alternate_moves(blue_starts, red_starts)

    pending = np.zeros((active_h, width), dtype=bool)
    if block_unmoved:
        for color, (r, c) in moves:
            pending[r, c] = True

    for color, (r0, c0) in moves:
        if block_unmoved:
            pending[r0, c0] = False

        if settled[r0, c0]:
            trace["start_conflict"] = True

        occupied = settled | pending
        r = r0
        if color == BLUE:
            while r + 1 < active_h and not occupied[r + 1, c0]:
                r += 1
            if r + 1 == active_h:
                stop_reason = "grey"
            elif categories[r + 1, c0] == CAT_BOTH and green_blocks:
                stop_reason = "green"
            else:
                stop_reason = "mover"
        else:
            while r - 1 >= 0 and not occupied[r - 1, c0]:
                r -= 1
            if r == 0:
                stop_reason = "top"
            elif categories[r - 1, c0] == CAT_BOTH and green_blocks:
                stop_reason = "green"
            else:
                stop_reason = "mover"

        settled[r, c0] = True
        out[r, c0] = color
        trace["moves"].append(
            {
                "color": color,
                "start": (r0, c0),
                "end": (r, c0),
                "distance": abs(r - r0),
                "stop": stop_reason,
            }
        )

    if return_trace:
        return out, trace
    return out


def _solve_from_masks(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    assert left_mask.shape == right_mask.shape and left_mask.ndim == 2
    categories = np.full(left_mask.shape, CAT_NONE, dtype=int)
    categories[left_mask & ~right_mask] = CAT_LEFT
    categories[~left_mask & right_mask] = CAT_RIGHT
    categories[left_mask & right_mask] = CAT_BOTH
    return _simulate_categories(categories)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge the aligned orange masks by provenance, then slide left-only and right-only pixels in opposite directions on a reduced canvas.

    Input structure:
    - The grid has two equal black/orange mask halves separated by a central red divider.
    - A grey bottom bar spans the scene; the divider occupies the rows above that bar.
    - The two halves are aligned cell-by-cell and may overlap, differ, or both be empty at any given coordinate.

    Rule:
    - Find the bottom grey bar and the red separator, then split the rows above the bar into left and right masks.
    - Classify each aligned active coordinate as left-only, right-only, or both-sides.
    - Output a single half-width canvas with the same height: keep the grey bottom bar, paint both-sides cells green and fixed, slide left-only cells downward as blue, and slide right-only cells upward as red.
    - Blue starts are processed from bottom to top, red starts from top to bottom, alternating blue then red; movers are blocked only by the grey bar, fixed green cells, and already settled movers.

    Key insight:
    - The OR-merged active set is not enough: each pixel's motion comes from which input half supplied it, so the solver must recover provenance before simulating collisions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D grid"

    bar_row = _find_bottom_bar_row(g)
    sep = _find_separator_col(g, bar_row)

    left = g[:bar_row, :sep]
    right = g[:bar_row, sep + 1 :]
    assert left.shape == right.shape and left.size > 0
    assert set(map(int, np.unique(left))) <= {BLACK, ORANGE}
    assert set(map(int, np.unique(right))) <= {BLACK, ORANGE}

    return _solve_from_masks(left == ORANGE, right == ORANGE)


def _column_safe(col: np.ndarray) -> bool:
    green_rows = np.flatnonzero(col == CAT_BOTH)
    bounds = [-1] + [int(v) for v in green_rows] + [len(col)]
    for lo, hi in zip(bounds, bounds[1:]):
        seg = col[lo + 1 : hi]
        if seg.size == 0:
            continue
        movers = [int(v) for v in seg if int(v) in (CAT_LEFT, CAT_RIGHT)]
        transitions = sum(a != b for a, b in zip(movers, movers[1:]))
        if transitions > 1:
            return False
    return True


def _try_place(categories: np.ndarray, rng: np.random.Generator, cat: int, max_per_col: int) -> bool:
    active_h, width = categories.shape
    coords = [(r, c) for r in range(active_h) for c in range(width) if categories[r, c] == CAT_NONE]
    rng.shuffle(coords)
    for r, c in coords:
        if cat == CAT_BOTH and not (1 <= r < active_h - 1):
            continue
        if cat == CAT_LEFT and r == active_h - 1:
            continue
        if cat == CAT_RIGHT and r == 0:
            continue
        if np.count_nonzero(categories[:, c] != CAT_NONE) >= max_per_col:
            continue
        categories[r, c] = cat
        if _column_safe(categories[:, c]):
            return True
        categories[r, c] = CAT_NONE
    return False


def _make_input_from_categories(categories: np.ndarray) -> np.ndarray:
    active_h, width = categories.shape
    left_mask = (categories == CAT_LEFT) | (categories == CAT_BOTH)
    right_mask = (categories == CAT_RIGHT) | (categories == CAT_BOTH)

    grid = np.full((active_h + 1, 2 * width + 1), BLACK, dtype=int)
    grid[-1, :] = GREY
    grid[:active_h, width] = RED
    grid[:active_h, :width][left_mask] = ORANGE
    grid[:active_h, width + 1 :][right_mask] = ORANGE
    return grid


def _wrong_no_green_block(categories: np.ndarray) -> np.ndarray:
    return _simulate_categories(categories, green_blocks=False, paint_green=True)


def _wrong_static_blockers(categories: np.ndarray) -> np.ndarray:
    return _simulate_categories(categories, block_unmoved=True)


def _wrong_all_down(categories: np.ndarray) -> np.ndarray:
    return _simulate_categories(categories, all_down=True, green_blocks=False, paint_green=False)


def _wrong_all_up(categories: np.ndarray) -> np.ndarray:
    return _simulate_categories(categories, all_up=True, green_blocks=False, paint_green=False)


def _validate_categories(categories: np.ndarray) -> bool:
    active_h, width = categories.shape
    counts = {
        CAT_LEFT: int(np.count_nonzero(categories == CAT_LEFT)),
        CAT_RIGHT: int(np.count_nonzero(categories == CAT_RIGHT)),
        CAT_BOTH: int(np.count_nonzero(categories == CAT_BOTH)),
    }
    if min(counts.values()) < 2:
        return False

    left_total = counts[CAT_LEFT] + counts[CAT_BOTH]
    right_total = counts[CAT_RIGHT] + counts[CAT_BOTH]
    if not (8 <= left_total <= 12 and 8 <= right_total <= 12):
        return False

    if not np.any(categories[1 : active_h - 1] == CAT_BOTH):
        return False

    per_col = np.count_nonzero(categories != CAT_NONE, axis=0)
    if np.max(per_col) > 5:
        return False

    if sum(np.count_nonzero(categories[:, c] == CAT_LEFT) >= 2 for c in range(width)) < 1:
        return False
    if sum(np.count_nonzero(categories[:, c] == CAT_RIGHT) >= 2 for c in range(width)) < 1:
        return False

    out, trace = _simulate_categories(categories, return_trace=True)
    if trace["start_conflict"]:
        return False

    blue_moves = [m for m in trace["moves"] if m["color"] == BLUE]
    red_moves = [m for m in trace["moves"] if m["color"] == RED]
    if not blue_moves or not red_moves:
        return False
    if not any(m["distance"] > 0 for m in blue_moves):
        return False
    if not any(m["distance"] > 0 for m in red_moves):
        return False
    if not any(m["stop"] in {"green", "mover"} for m in blue_moves):
        return False
    if not any(m["stop"] in {"green", "mover"} for m in red_moves):
        return False

    if np.array_equal(out, _wrong_static_blockers(categories)):
        return False
    if np.array_equal(out, _wrong_no_green_block(categories)):
        return False
    if np.array_equal(out, _wrong_all_down(categories)):
        return False
    if np.array_equal(out, _wrong_all_up(categories)):
        return False

    inp = _make_input_from_categories(categories)
    if not np.array_equal(main(inp), out):
        return False
    return True


def _place_base_patterns(categories: np.ndarray, rng: np.random.Generator) -> bool:
    active_h, width = categories.shape
    cols = list(map(int, rng.choice(width, size=5, replace=False)))
    split_a, split_b, blue_col, red_col, mixed_col = cols

    for col in (split_a, split_b):
        g = int(rng.integers(2, active_h - 2))
        b = int(rng.integers(0, g - 1))
        r = int(rng.integers(g + 2, active_h))
        if categories[b, col] != CAT_NONE or categories[g, col] != CAT_NONE or categories[r, col] != CAT_NONE:
            return False
        categories[b, col] = CAT_LEFT
        categories[g, col] = CAT_BOTH
        categories[r, col] = CAT_RIGHT

    blue_rows = sorted(map(int, rng.choice(np.arange(0, active_h - 1), size=2, replace=False)))
    red_rows = sorted(map(int, rng.choice(np.arange(1, active_h), size=2, replace=False)))
    for r in blue_rows:
        if categories[r, blue_col] != CAT_NONE:
            return False
        categories[r, blue_col] = CAT_LEFT
    for r in red_rows:
        if categories[r, red_col] != CAT_NONE:
            return False
        categories[r, red_col] = CAT_RIGHT

    mixed_blue = int(rng.integers(0, active_h - 4))
    mixed_red = int(rng.integers(mixed_blue + 3, active_h - 1))
    if categories[mixed_blue, mixed_col] != CAT_NONE or categories[mixed_red, mixed_col] != CAT_NONE:
        return False
    categories[mixed_blue, mixed_col] = CAT_LEFT
    categories[mixed_red, mixed_col] = CAT_RIGHT

    return all(_column_safe(categories[:, c]) for c in cols)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(10, 14))
    width = int(rng.integers(6, 9))
    active_h = height - 1
    categories = np.full((active_h, width), CAT_NONE, dtype=int)

    if not _place_base_patterns(categories, rng):
        return None

    green_target = int(rng.integers(2, 4))
    left_total = int(rng.integers(8, 13))
    right_total = int(rng.integers(8, 13))
    blue_target = left_total - green_target
    red_target = right_total - green_target
    if blue_target < 5 or red_target < 5:
        return None

    max_per_col = 5
    for _ in range(green_target - 2):
        if not _try_place(categories, rng, CAT_BOTH, max_per_col):
            return None

    extras = [CAT_LEFT] * (blue_target - 5) + [CAT_RIGHT] * (red_target - 5)
    rng.shuffle(extras)
    for cat in extras:
        if not _try_place(categories, rng, cat, max_per_col):
            return None

    if not _validate_categories(categories):
        return None
    return _make_input_from_categories(categories)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
