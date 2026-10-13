from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present
import itertools
import random


def _maximal_marker_windows(g: np.ndarray) -> list[tuple[int, int, frozenset[tuple[int, int]]]]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, frozenset[tuple[int, int]]]] = []

    for r in range(height - 2):
        for c in range(width - 2):
            window = g[r : r + 3, c : c + 3]
            if not np.all(window != Color.BLACK):
                continue
            if window[1, 1] != Color.GREY or not np.any(window == Color.RED):
                continue

            red_cells = frozenset(
                (int(r + rr), int(c + cc))
                for rr, cc in np.argwhere(window == Color.RED)
            )
            candidates.append((r, c, red_cells))

    selected = [
        candidate
        for candidate in candidates
        if not any(candidate[2] < other[2] for other in candidates)
    ]
    assert selected, "expected at least one marked 3x3 foreground region"

    seen_red: set[tuple[int, int]] = set()
    for _r, _c, red_cells in selected:
        assert seen_red.isdisjoint(red_cells), "maximal red marker regions must be disjoint"
        seen_red.update(red_cells)

    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each maximal solid 3x3 foreground region marked by red pixels, making its center yellow and its other grey cells orange.

    Input structure
    - The grid contains black empty cells plus grey foreground material with red
      marker pixels embedded in some solid 3x3 foreground windows.
    - A candidate window is a 3x3 region with no black cells, at least one red
      pixel, and a grey center.
    - Overlapping candidate windows may exist when one sees only part of the red
      marker; the intended marker window has the maximal set of red pixels.

    Rule
    - Select the candidate 3x3 windows whose red pixel set is not a strict subset
      of another candidate's red pixel set.
    - In each selected window, recolor the grey center pixel to yellow.
    - Recolor every other grey pixel in the selected window to orange, leaving
      red pixels and the rest of the grid unchanged.

    Key insight
    - The red pixels identify local 3x3 foreground blocks; maximal red-set
      containment removes the overlapping partial views of the same marked block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(colors_present(g)) <= {Color.BLACK, Color.RED, Color.GREY}

    out = g.copy()
    painted: set[tuple[int, int]] = set()

    for r, c, _red_cells in _maximal_marker_windows(g):
        for rr in range(r, r + 3):
            for cc in range(c, c + 3):
                assert g[rr, cc] != Color.BLACK
                if g[rr, cc] == Color.GREY:
                    assert (rr, cc) not in painted, "selected windows must not overlap on grey cells"
                    painted.add((rr, cc))
                    out[rr, cc] = (
                        Color.YELLOW if (rr, cc) == (r + 1, c + 1) else Color.ORANGE
                    )

    return out

# --------------- generator (o5x) ---------------

# Canvas and color values.
GEN_BLACK, GEN_RED, GEN_YELLOW, GEN_GREY, GEN_ORANGE = 0, 2, 4, 5, 7
UNFILLED, CANVAS_SIZE, WINDOW_SIZE, CENTER_OFFSET = -1, 19, 3, 1
WINDOW_TOP_MAX = CANVAS_SIZE - WINDOW_SIZE
TOTAL_CELLS = CANVAS_SIZE * CANVAS_SIZE

# Marker layout and red-count distribution.
SELECTED_WINDOW_COUNT, MIN_WINDOW_SEPARATION = 4, 4
MIN_RED_PER_WINDOW, MAX_RED_PER_WINDOW = 1, 6
MIN_TOTAL_RED, MAX_TOTAL_RED = 10, 16
MAX_SUBSET_CANDIDATES = 3

# Background color-count bounds matched to the eval examples.
MIN_BLACK_CELLS, MAX_BLACK_CELLS = 126, 169
MIN_INPUT_GREY_CELLS, MAX_INPUT_GREY_CELLS = 177, 224

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300

ALL_WINDOW_POSITIONS = [
    (r, c) for r in range(WINDOW_TOP_MAX + 1) for c in range(WINDOW_TOP_MAX + 1)
]
RED_OFFSETS = [
    (dr, dc) for dr in range(WINDOW_SIZE) for dc in range(WINDOW_SIZE)
    if (dr, dc) != (CENTER_OFFSET, CENTER_OFFSET)
]
RED_COUNT_RANGE = range(MIN_RED_PER_WINDOW, MAX_RED_PER_WINDOW + 1)
RED_COUNT_TUPLES = {
    total: [
        vals
        for vals in itertools.product(RED_COUNT_RANGE, repeat=SELECTED_WINDOW_COUNT)
        if sum(vals) == total
    ]
    for total in range(MIN_TOTAL_RED, MAX_TOTAL_RED + 1)
}


def _window_cells(r: int, c: int) -> set[tuple[int, int]]:
    return {
        (rr, cc) for rr in range(r, r + WINDOW_SIZE) for cc in range(c, c + WINDOW_SIZE)
    }


def _positions_compatible(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    no_overlap = dr >= WINDOW_SIZE or dc >= WINDOW_SIZE
    observed_separation = max(dr, dc) >= MIN_WINDOW_SEPARATION
    return no_overlap and observed_separation


def _choose_positions() -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []

    for _ in range(SELECTED_WINDOW_COUNT):
        valid = [
            pos for pos in ALL_WINDOW_POSITIONS
            if all(_positions_compatible(pos, old) for old in positions)
        ]
        if not valid:
            return None
        positions.append(random.choice(valid))

    return positions


def _choose_red_counts() -> tuple[int, ...]:
    total = random.randint(MIN_TOTAL_RED, MAX_TOTAL_RED)
    return random.choice(RED_COUNT_TUPLES[total])


def _candidate_reds(red_cells):
    candidates: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for r, c in ALL_WINDOW_POSITIONS:
        seen = frozenset(
            (rr, cc)
            for rr, cc in red_cells
            if r <= rr < r + WINDOW_SIZE and c <= cc < c + WINDOW_SIZE
        )
        if seen:
            candidates[(r, c)] = seen
    return candidates


def _choose_marker_reds(positions, red_counts):
    selected_reds: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    red_cells: set[tuple[int, int]] = set()
    selected_cells: set[tuple[int, int]] = set()

    for (r, c), count in zip(positions, red_counts):
        offsets = random.sample(RED_OFFSETS, count)
        reds = frozenset((r + dr, c + dc) for dr, dc in offsets)
        selected_reds[(r, c)] = reds
        red_cells.update(reds)
        selected_cells.update(_window_cells(r, c))

    return selected_reds, red_cells, selected_cells


def _choose_allowed_subset_candidates(candidates, selected_reds, red_cells):
    intended = set(selected_reds)
    selected_red_sets = list(selected_reds.values())

    for pos, reds in selected_reds.items():
        if candidates.get(pos) != reds:
            return None

    eligible_extra: list[tuple[int, int]] = []
    candidate_positions: list[tuple[int, int]] = []
    for pos, reds in candidates.items():
        if pos in intended:
            continue

        candidate_positions.append(pos)
        center = (pos[0] + 1, pos[1] + 1)
        strict_owners = [chosen for chosen in selected_red_sets if reds < chosen]
        if len(strict_owners) == 1 and center not in red_cells:
            eligible_extra.append(pos)

    desired_extra = random.randint(0, MAX_SUBSET_CANDIDATES)
    if len(eligible_extra) < desired_extra:
        return None

    random.shuffle(eligible_extra)
    allowed_extra = set(eligible_extra[:desired_extra])
    unwanted = [pos for pos in candidate_positions if pos not in allowed_extra]

    return allowed_extra, unwanted


def _choose_blockers(unwanted, protected):
    blockers: set[tuple[int, int]] = set()
    random.shuffle(unwanted)
    for pos in unwanted:
        cells = _window_cells(*pos)
        if cells & blockers:
            continue
        choices = [cell for cell in cells if cell not in protected]
        if not choices:
            return None
        blockers.add(random.choice(choices))
    return blockers


def _paint_selected_windows(out, positions, selected_reds) -> None:
    for r, c in positions:
        window = out[r : r + WINDOW_SIZE, c : c + WINDOW_SIZE]
        window[:, :] = GEN_ORANGE
        window[CENTER_OFFSET, CENTER_OFFSET] = GEN_YELLOW
        reds = selected_reds[(r, c)]
        for rr, cc in reds:
            out[rr, cc] = GEN_RED


def _apply_allowed_extra_windows(out, allowed_extra) -> bool:
    for r, c in allowed_extra:
        window = out[r : r + WINDOW_SIZE, c : c + WINDOW_SIZE]
        if np.any(window == GEN_BLACK):
            return False
        window[window == UNFILLED] = GEN_GREY
    return True


def _render_fixed_output(positions, selected_reds, blockers, allowed_extra):
    out = np.full((CANVAS_SIZE, CANVAS_SIZE), UNFILLED, dtype=int)

    if blockers:
        rows, cols = zip(*blockers)
        out[rows, cols] = GEN_BLACK
    _paint_selected_windows(out, positions, selected_reds)

    if not _apply_allowed_extra_windows(out, allowed_extra):
        return None
    return out


def _fill_random_background(out: np.ndarray, total_red: int) -> bool:
    fixed_black = int(np.count_nonzero(out == GEN_BLACK))
    fixed_input_grey = int(
        np.count_nonzero((out == GEN_GREY) | (out == GEN_YELLOW) | (out == GEN_ORANGE))
    )
    unknown = list(zip(*np.where(out == UNFILLED)))

    black_low = max(
        MIN_BLACK_CELLS,
        TOTAL_CELLS - total_red - MAX_INPUT_GREY_CELLS,
        fixed_black,
    )
    black_high = min(
        MAX_BLACK_CELLS,
        TOTAL_CELLS - total_red - MIN_INPUT_GREY_CELLS,
        fixed_black + len(unknown),
        TOTAL_CELLS - total_red - fixed_input_grey,
    )
    if black_low > black_high:
        return False

    black_target = random.randint(black_low, black_high)
    need_black = black_target - fixed_black
    random.shuffle(unknown)
    for i, (r, c) in enumerate(unknown):
        out[r, c] = GEN_BLACK if i < need_black else GEN_GREY
    return True


def _make_input_from_output(out, positions):
    inp = out.copy()
    for r, c in positions:
        window = inp[r : r + WINDOW_SIZE, c : c + WINDOW_SIZE]
        window[(window == GEN_YELLOW) | (window == GEN_ORANGE)] = GEN_GREY
    return inp


def _gen_attempt() -> np.ndarray | None:
    positions = _choose_positions()
    if positions is None:
        return None

    red_counts = _choose_red_counts()
    selected_reds, red_cells, selected_cells = _choose_marker_reds(positions, red_counts)
    candidates = _candidate_reds(red_cells)
    candidate_plan = _choose_allowed_subset_candidates(
        candidates, selected_reds, red_cells
    )
    if candidate_plan is None:
        return None

    allowed_extra, unwanted = candidate_plan
    protected = set(selected_cells)
    for pos in allowed_extra:
        protected.update(_window_cells(*pos))
    blockers = _choose_blockers(unwanted, protected)
    if blockers is None:
        return None

    out = _render_fixed_output(positions, selected_reds, blockers, allowed_extra)
    if out is None or not _fill_random_background(out, int(sum(red_counts))):
        return None

    inp = _make_input_from_output(out, positions)
    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
