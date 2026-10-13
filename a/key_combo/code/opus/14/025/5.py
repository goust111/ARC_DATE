from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np

try:
    from arc_utils import colors_present
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_025_5",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    colors_present = _ARC_UTILS.colors_present


BACKGROUND = 0
MARKER_COLORS = (1, 2, 3)
CORNER_COLOR = 4

GRID_MIN = 14
GRID_MAX = 19
RECT_COUNT_CHOICES = (2, 3)
RECT_SPAN_MIN = 3
RECT_SPAN_MAX = 6
OUTER_MARGIN = 1
MIN_AXIS_GAP = 1
GENERATION_ATTEMPTS = 500


Rect = tuple[int, int, int, int]
Marker = tuple[int, int, int]


def _bbox_overlaps(a: Rect, b: Rect) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _rectangles_from_corners(grid: np.ndarray) -> list[Rect]:
    yellow = {tuple(map(int, rc)) for rc in np.argwhere(grid == CORNER_COLOR)}
    assert yellow and len(yellow) % 4 == 0, "yellow corner count must be a positive multiple of four"

    rows = sorted({r for r, _c in yellow})
    cols = sorted({c for _r, c in yellow})
    candidates: list[tuple[Rect, frozenset[tuple[int, int]]]] = []
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    corners = frozenset(((r0, c0), (r0, c1), (r1, c0), (r1, c1)))
                    if corners.issubset(yellow):
                        assert r1 - r0 >= 2 and c1 - c0 >= 2, "rectangle must have non-empty interior"
                        candidates.append(((r0, c0, r1, c1), corners))
    assert candidates, "no corner-defined rectangles found"

    by_corner: dict[tuple[int, int], list[int]] = {corner: [] for corner in yellow}
    for idx, (_rect, corners) in enumerate(candidates):
        for corner in corners:
            by_corner[corner].append(idx)

    solutions: list[list[Rect]] = []

    def search(remaining: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append([candidates[idx][0] for idx in chosen])
            return

        anchor = min(remaining)
        for idx in by_corner[anchor]:
            rect, corners = candidates[idx]
            if not corners.issubset(remaining):
                continue
            if any(_bbox_overlaps(rect, candidates[other][0]) for other in chosen):
                continue
            search(remaining - set(corners), chosen + [idx])

    search(set(yellow), [])
    assert len(solutions) == 1, "yellow corners must define a unique rectangle partition"
    return sorted(solutions[0])


def _marker_pixels(grid: np.ndarray) -> list[Marker]:
    markers: list[Marker] = []
    for color in colors_present(grid, exclude={BACKGROUND, CORNER_COLOR}):
        assert int(color) in MARKER_COLORS, "unexpected marker color"
        coords = np.argwhere(grid == int(color))
        for r, c in coords:
            markers.append((int(r), int(c), int(color)))
    return markers


def _aligned_markers(rect: Rect, markers: list[Marker]) -> list[dict[str, object]]:
    r0, c0, r1, c1 = rect
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    candidates: list[dict[str, object]] = []
    for r, c, color in markers:
        row_aligned = r0 <= r <= r1 and (c < c0 or c > c1)
        col_aligned = c0 <= c <= c1 and (r < r0 or r > r1)
        if row_aligned == col_aligned:
            continue
        candidates.append(
            {
                "cell": (r, c),
                "color": int(color),
                "kind": "row" if row_aligned else "col",
                "distance": abs(r - cr) + abs(c - cc),
            }
        )
    return candidates


def _winner(candidates: list[dict[str, object]]) -> dict[str, object]:
    assert candidates, "each rectangle needs at least one aligned marker"
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(item["distance"]),
            0 if item["kind"] == "row" else 1,
            tuple(item["cell"]),
        ),
    )
    best = ranked[0]
    best_key = (
        int(best["distance"]),
        0 if best["kind"] == "row" else 1,
    )
    assert (
        len([item for item in ranked if (int(item["distance"]), 0 if item["kind"] == "row" else 1) == best_key]) == 1
    ), "distance rule must yield a unique winner after row-priority tie-break"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each yellow-corner rectangle interior with the color of the nearest aligned singleton marker.

    Input structure:
    - The background is black.
    - Yellow single pixels appear in groups of four, each group occupying two rows and two columns and defining one hidden rectangle.
    - Red, blue, or green singleton markers lie outside the rectangles but are row-aligned or column-aligned with at least one rectangle.

    Rule:
    - Detect every rectangle whose four corners are marked by yellow pixels.
    - For each rectangle, collect all colored singleton markers whose row falls inside the rectangle's row span while their column stays outside the rectangle, or whose column falls inside the rectangle's column span while their row stays outside.
    - Compute each candidate marker's Manhattan distance to the rectangle center at `((r_top+r_bottom)//2, (c_left+c_right)//2)`.
    - Choose the nearest candidate; if a row-aligned and a column-aligned marker tie on distance, prefer the row-aligned marker.
    - Fill the rectangle interior with the chosen marker color, keep the yellow corner pixels, and erase all singleton markers.

    Key insight:
    - The yellow corners encode geometry while the colored singletons encode appearance, and both channels must be combined because aligned markers compete by center distance rather than by one-to-one positional matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rectangles = _rectangles_from_corners(g)
    markers = _marker_pixels(g)
    assert markers, "expected colored singleton markers"

    out = np.full_like(g, BACKGROUND)
    out[g == CORNER_COLOR] = CORNER_COLOR

    used_winners: set[tuple[int, int]] = set()
    for rect in rectangles:
        candidates = _aligned_markers(rect, markers)
        winner = _winner(candidates)
        winner_cell = tuple(winner["cell"])
        assert winner_cell not in used_winners, "nearest winners must be distinct across rectangles"
        used_winners.add(winner_cell)

        r0, c0, r1, c1 = rect
        out[r0 + 1 : r1, c0 + 1 : c1] = int(winner["color"])

    return out


def _gen_axis_intervals(k: int) -> tuple[int, list[tuple[int, int]]]:
    for _ in range(100):
        spans = [random.randint(RECT_SPAN_MIN, RECT_SPAN_MAX) for _ in range(k)]
        lengths = [span + 1 for span in spans]
        minimum = 2 * OUTER_MARGIN + sum(lengths) + MIN_AXIS_GAP * (k - 1)
        if minimum > GRID_MAX:
            continue
        size = random.randint(max(GRID_MIN, minimum), GRID_MAX)
        slack = size - minimum
        extras = [0] * (k + 1)
        for _ in range(slack):
            extras[random.randrange(k + 1)] += 1

        intervals: list[tuple[int, int]] = []
        start = OUTER_MARGIN + extras[0]
        for i, length in enumerate(lengths):
            end = start + length - 1
            intervals.append((start, end))
            if i < k - 1:
                start = end + 1 + MIN_AXIS_GAP + extras[i + 1]
        return size, intervals
    raise RuntimeError("failed to build non-overlapping axis intervals")


def _outside_index_set(size: int, intervals: list[tuple[int, int]]) -> list[int]:
    return [
        idx
        for idx in range(size)
        if not any(lo <= idx <= hi for lo, hi in intervals)
    ]


def _candidate_pool(
    rect: Rect,
    shape: tuple[int, int],
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    h, w = shape
    r0, c0, r1, c1 = rect
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2

    outside_rows = _outside_index_set(h, row_intervals)
    outside_cols = _outside_index_set(w, col_intervals)

    row_candidates: list[dict[str, object]] = []
    for r in range(r0, r1 + 1):
        for c in outside_cols:
            if (r, c) in occupied:
                continue
            row_candidates.append(
                {"cell": (r, c), "kind": "row", "distance": abs(r - cr) + abs(c - cc)}
            )

    col_candidates: list[dict[str, object]] = []
    for r in outside_rows:
        for c in range(c0, c1 + 1):
            if (r, c) in occupied:
                continue
            col_candidates.append(
                {"cell": (r, c), "kind": "col", "distance": abs(r - cr) + abs(c - cc)}
            )
    return row_candidates, col_candidates


def _pick_marker_plan(
    rect: Rect,
    scenario: str,
    winner_color: int,
    shape: tuple[int, int],
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> dict[str, object] | None:
    row_candidates, col_candidates = _candidate_pool(rect, shape, row_intervals, col_intervals, occupied)
    if not row_candidates or not col_candidates:
        return None

    chosen: list[dict[str, object]] = []
    winner_cell: tuple[int, int]
    winner_kind: str
    if scenario == "row":
        valid_rows = [cand for cand in row_candidates if any(int(other["distance"]) > int(cand["distance"]) for other in col_candidates)]
        if not valid_rows:
            return None
        row_winner = random.choice(valid_rows)
        col_losers = [cand for cand in col_candidates if int(cand["distance"]) > int(row_winner["distance"])]
        if not col_losers:
            return None
        chosen = [row_winner, random.choice(col_losers)]
        winner_cell = tuple(row_winner["cell"])
        winner_kind = "row"
    elif scenario == "col":
        valid_cols = [cand for cand in col_candidates if any(int(other["distance"]) > int(cand["distance"]) for other in row_candidates)]
        if not valid_cols:
            return None
        col_winner = random.choice(valid_cols)
        row_losers = [cand for cand in row_candidates if int(cand["distance"]) > int(col_winner["distance"])]
        if not row_losers:
            return None
        chosen = [col_winner, random.choice(row_losers)]
        winner_cell = tuple(col_winner["cell"])
        winner_kind = "col"
    else:
        equal_pairs = [
            (row_cand, col_cand)
            for row_cand in row_candidates
            for col_cand in col_candidates
            if int(row_cand["distance"]) == int(col_cand["distance"])
        ]
        if not equal_pairs:
            return None
        row_winner, col_tie = random.choice(equal_pairs)
        chosen = [row_winner, col_tie]
        winner_cell = tuple(row_winner["cell"])
        winner_kind = "row"

    winner_key = (
        min(chosen, key=lambda item: (int(item["distance"]), 0 if item["kind"] == "row" else 1))["distance"],
        0 if winner_kind == "row" else 1,
    )
    extras = [
        cand
        for cand in row_candidates + col_candidates
        if tuple(cand["cell"]) not in {tuple(item["cell"]) for item in chosen}
        and (int(cand["distance"]), 0 if cand["kind"] == "row" else 1) > winner_key
    ]
    if extras and random.random() < 0.65:
        chosen.append(random.choice(extras))

    colors = [winner_color]
    alt_colors = [color for color in MARKER_COLORS if color != winner_color]
    colors.append(random.choice(alt_colors))
    while len(colors) < len(chosen):
        colors.append(random.choice(MARKER_COLORS))

    placements = []
    for item, color in zip(chosen, colors):
        placements.append(
            {
                "cell": tuple(item["cell"]),
                "kind": str(item["kind"]),
                "distance": int(item["distance"]),
                "color": int(color),
            }
        )

    return {
        "scenario": scenario,
        "winner_cell": winner_cell,
        "winner_kind": winner_kind,
        "winner_color": int(winner_color),
        "placements": placements,
    }


def _scenario_list(k: int) -> list[str]:
    if k == 3:
        scenarios = ["row", "col", "tie"]
        random.shuffle(scenarios)
        return scenarios
    if random.random() < 0.7:
        scenarios = ["tie", random.choice(["row", "col"])]
        random.shuffle(scenarios)
        return scenarios
    return random.sample(["row", "col", "tie"], k)


def _render_expected(shape: tuple[int, int], rectangles: list[Rect], plans: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        out[r0, c0] = CORNER_COLOR
        out[r0, c1] = CORNER_COLOR
        out[r1, c0] = CORNER_COLOR
        out[r1, c1] = CORNER_COLOR
    for rect, plan in zip(rectangles, plans):
        r0, c0, r1, c1 = rect
        out[r0 + 1 : r1, c0 + 1 : c1] = int(plan["winner_color"])
    return out


def _validate_generation(
    grid: np.ndarray,
    rectangles: list[Rect],
    plans: list[dict[str, object]],
) -> bool:
    try:
        solved = main(grid)
    except Exception:
        return False

    if not np.array_equal(solved, _render_expected(grid.shape, rectangles, plans)):
        return False

    markers = _marker_pixels(grid)
    if len(markers) < 2 * len(rectangles):
        return False

    winner_cells: set[tuple[int, int]] = set()
    for marker in markers:
        aligned_count = sum(
            1
            for rect in rectangles
            if any(tuple(candidate["cell"]) == marker[:2] for candidate in _aligned_markers(rect, markers))
        )
        if aligned_count != 1:
            return False

    for rect, plan in zip(rectangles, plans):
        candidates = _aligned_markers(rect, markers)
        if len(candidates) < 2:
            return False
        if {candidate["kind"] for candidate in candidates} != {"row", "col"}:
            return False

        winner = _winner(candidates)
        if tuple(winner["cell"]) != tuple(plan["winner_cell"]):
            return False
        if int(winner["color"]) != int(plan["winner_color"]):
            return False
        if tuple(winner["cell"]) in winner_cells:
            return False
        winner_cells.add(tuple(winner["cell"]))

        scenario = str(plan["scenario"])
        if scenario == "row":
            if winner["kind"] != "row":
                return False
            if not any(candidate["kind"] == "col" and int(candidate["distance"]) > int(winner["distance"]) for candidate in candidates):
                return False
        elif scenario == "col":
            if winner["kind"] != "col":
                return False
            if not any(candidate["kind"] == "row" and int(candidate["distance"]) > int(winner["distance"]) for candidate in candidates):
                return False
        else:
            if winner["kind"] != "row":
                return False
            if not any(candidate["kind"] == "col" and int(candidate["distance"]) == int(winner["distance"]) for candidate in candidates):
                return False
    return True


def _attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    h, row_intervals = _gen_axis_intervals(rect_count)
    w, col_intervals = _gen_axis_intervals(rect_count)
    shuffled_cols = col_intervals[:]
    random.shuffle(shuffled_cols)
    rectangles = [
        (row_lo, col_lo, row_hi, col_hi)
        for (row_lo, row_hi), (col_lo, col_hi) in zip(row_intervals, shuffled_cols)
    ]
    rectangles = sorted(rectangles)

    assert all(not _bbox_overlaps(a, b) for i, a in enumerate(rectangles) for b in rectangles[i + 1 :])

    winner_colors = random.sample(list(MARKER_COLORS), rect_count)
    scenarios = _scenario_list(rect_count)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for r0, c0, r1, c1 in rectangles:
        for cell in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            grid[cell] = CORNER_COLOR
            occupied.add(cell)

    plans: list[dict[str, object]] = []
    for rect, scenario, winner_color in zip(rectangles, scenarios, winner_colors):
        plan = _pick_marker_plan(rect, scenario, winner_color, grid.shape, row_intervals, shuffled_cols, occupied)
        if plan is None:
            return None
        for placement in plan["placements"]:
            occupied.add(tuple(placement["cell"]))
        plans.append(plan)

    for plan in plans:
        for placement in plan["placements"]:
            r, c = placement["cell"]
            grid[r, c] = int(placement["color"])

    if len({plan["winner_color"] for plan in plans}) != rect_count:
        return None
    if not _validate_generation(grid, rectangles, plans):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_ATTEMPTS} attempts")
