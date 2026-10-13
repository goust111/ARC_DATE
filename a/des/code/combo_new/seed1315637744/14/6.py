import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
RECT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200
LAYOUTS = (
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
    (2, 2),
)


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components(points: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(points)
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(comp)
    return comps


def _rect_points(rect: dict) -> set[tuple[int, int]]:
    r0 = int(rect["r0"])
    c0 = int(rect["c0"])
    r1 = int(rect["r1"])
    c1 = int(rect["c1"])
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _boxes_have_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    min_gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return max(row_gap, col_gap) >= min_gap


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, RED)):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "every foreground color must appear"
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape[0] * patch.shape[1] == len(coords), "object must be a solid rectangle"
        assert np.all(patch == color), "object must be monochrome"
        rectangles.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    assert rectangles, "expected at least one rectangle"
    return rectangles


def _detect_cuts(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    red_points = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert red_points and len(red_points) % 2 == 0, "red markers must form pairs"

    rows: defaultdict[int, list[int]] = defaultdict(list)
    cols: defaultdict[int, list[int]] = defaultdict(list)
    for r, c in red_points:
        rows[r].append(c)
        cols[c].append(r)
    for values in rows.values():
        values.sort()
    for values in cols.values():
        values.sort()

    horizontal: list[dict] = []
    vertical: list[dict] = []
    for r, c in red_points:
        row_count = len(rows[r])
        col_count = len(cols[c])
        if row_count == 2 and col_count == 1:
            if c == rows[r][0]:
                horizontal.append({"axis": "h", "line": r, "lo": rows[r][0], "hi": rows[r][1]})
        elif row_count == 1 and col_count == 2:
            if r == cols[c][0]:
                vertical.append({"axis": "v", "line": c, "lo": cols[c][0], "hi": cols[c][1]})
        else:
            raise AssertionError("each red pixel must have exactly one row or column partner")

    assert 2 * (len(horizontal) + len(vertical)) == len(red_points), "every red pixel must be paired once"
    horizontal.sort(key=lambda cut: (int(cut["line"]), int(cut["lo"]), int(cut["hi"])))
    vertical.sort(key=lambda cut: (int(cut["line"]), int(cut["lo"]), int(cut["hi"])))
    return horizontal, vertical


def _hits_horizontal(comp: list[tuple[int, int]], cut: dict) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    line = int(cut["line"])
    lo = int(cut["lo"])
    hi = int(cut["hi"])
    return r0 <= line < r1 and c0 < hi and c1 > lo


def _hits_vertical(comp: list[tuple[int, int]], cut: dict) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    line = int(cut["line"])
    lo = int(cut["lo"])
    hi = int(cut["hi"])
    return c0 <= line < c1 and r0 < hi and r1 > lo


def _apply_horizontal(points: set[tuple[int, int]], cut: dict) -> tuple[set[tuple[int, int]], int]:
    line = int(cut["line"])
    out: set[tuple[int, int]] = set()
    affected = 0
    for comp in _components(points):
        if not _hits_horizontal(comp, cut):
            out.update(comp)
            continue
        affected += 1
        for r, c in comp:
            out.add((r - 1, c) if r <= line else (r + 1, c))
    assert len(out) == len(points), "horizontal cut must preserve pixel count"
    return out, affected


def _apply_vertical(points: set[tuple[int, int]], cut: dict) -> tuple[set[tuple[int, int]], int]:
    line = int(cut["line"])
    out: set[tuple[int, int]] = set()
    affected = 0
    for comp in _components(points):
        if not _hits_vertical(comp, cut):
            out.update(comp)
            continue
        affected += 1
        for r, c in comp:
            out.add((r, c - 1) if c <= line else (r, c + 1))
    assert len(out) == len(points), "vertical cut must preserve pixel count"
    return out, affected


def _simulate(
    shape: tuple[int, int],
    rectangles: list[dict],
    horizontal_cuts: list[dict],
    vertical_cuts: list[dict],
) -> tuple[np.ndarray, list[int], list[int]]:
    h, w = shape
    out = np.full(shape, BG, dtype=int)
    h_effects = [0] * len(horizontal_cuts)
    v_effects = [0] * len(vertical_cuts)

    for rect in rectangles:
        points = _rect_points(rect)
        for i, cut in enumerate(horizontal_cuts):
            points, affected = _apply_horizontal(points, cut)
            h_effects[i] += affected
        for i, cut in enumerate(vertical_cuts):
            points, affected = _apply_vertical(points, cut)
            v_effects[i] += affected
        for r, c in points:
            assert 0 <= r < h and 0 <= c < w, "shifted rectangle must stay in bounds"
            assert out[r, c] == BG, "shifted rectangles must not overlap"
            out[r, c] = int(rect["color"])

    return out, h_effects, v_effects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split every rectangle along the invisible lines between paired red markers and separate the pieces.

    Input structure:
    - The background is black.
    - Several non-red colors each form one solid axis-aligned rectangle.
    - Additional red pixels appear only as isolated same-row or same-column pairs; each pair defines one hidden cut line segment.

    Rule:
    - Remove the red pixels from the output.
    - For every horizontal red pair, treat the space between that row and the next row as a horizontal knife segment spanning between the two red endpoints.
    - Any rectangle fragment that crosses that knife segment is split into upper and lower parts; the upper part moves up by one cell and the lower part moves down by one cell.
    - After applying all horizontal cuts from top to bottom, do the same for vertical red pairs from left to right, moving left/right halves one cell outward.

    Key insight:
    - The red pixels are not paths to draw; they are paired markers for invisible knives, and the cuts act on whole rectangle fragments in sequence rather than on colors independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    rectangles = _extract_rectangles(g)
    horizontal_cuts, vertical_cuts = _detect_cuts(g)
    out, _, _ = _simulate(g.shape, rectangles, horizontal_cuts, vertical_cuts)
    return out


def _build_base_grid(shape: tuple[int, int], rectangles: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rectangles:
        grid[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = int(rect["color"])
    return grid


def _horizontal_candidates(grid: np.ndarray, rectangles: list[dict]) -> dict[int, list[dict]]:
    h, w = grid.shape
    by_row: dict[int, list[dict]] = defaultdict(list)
    for line in range(0, h - 1):
        crossers = [
            (int(rect["c0"]), int(rect["c1"]), idx)
            for idx, rect in enumerate(rectangles)
            if int(rect["r0"]) <= line < int(rect["r1"])
        ]
        if not crossers:
            continue
        crossers.sort()
        for i in range(len(crossers)):
            for j in range(i, len(crossers)):
                left_bound = 0 if i == 0 else crossers[i - 1][1] + 1
                right_bound = w - 1 if j == len(crossers) - 1 else crossers[j + 1][0] - 1
                left_options = [c for c in range(left_bound, crossers[i][0]) if grid[line, c] == BG]
                right_options = [c for c in range(crossers[j][1] + 1, right_bound + 1) if grid[line, c] == BG]
                if not left_options or not right_options:
                    continue
                by_row[line].append(
                    {
                        "axis": "h",
                        "line": line,
                        "rect_ids": tuple(item[2] for item in crossers[i : j + 1]),
                        "left_options": left_options,
                        "right_options": right_options,
                    }
                )
    return by_row


def _vertical_candidates(grid: np.ndarray, rectangles: list[dict]) -> dict[int, list[dict]]:
    h, w = grid.shape
    by_col: dict[int, list[dict]] = defaultdict(list)
    for line in range(0, w - 1):
        crossers = [
            (int(rect["r0"]), int(rect["r1"]), idx)
            for idx, rect in enumerate(rectangles)
            if int(rect["c0"]) <= line < int(rect["c1"])
        ]
        if not crossers:
            continue
        crossers.sort()
        for i in range(len(crossers)):
            for j in range(i, len(crossers)):
                top_bound = 0 if i == 0 else crossers[i - 1][1] + 1
                bottom_bound = h - 1 if j == len(crossers) - 1 else crossers[j + 1][0] - 1
                top_options = [r for r in range(top_bound, crossers[i][0]) if grid[r, line] == BG]
                bottom_options = [r for r in range(crossers[j][1] + 1, bottom_bound + 1) if grid[r, line] == BG]
                if not top_options or not bottom_options:
                    continue
                by_col[line].append(
                    {
                        "axis": "v",
                        "line": line,
                        "rect_ids": tuple(item[2] for item in crossers[i : j + 1]),
                        "top_options": top_options,
                        "bottom_options": bottom_options,
                    }
                )
    return by_col


def _assign_horizontal_endpoints(
    candidates: list[dict],
    forbidden_cols: set[int],
) -> list[dict] | None:
    chosen: list[dict] = []
    used = set(forbidden_cols)

    def backtrack(i: int) -> bool:
        if i == len(candidates):
            return True
        cand = candidates[i]
        lefts = list(cand["left_options"])
        rights = list(cand["right_options"])
        random.shuffle(lefts)
        random.shuffle(rights)
        for left in lefts:
            if left in used:
                continue
            used.add(left)
            for right in rights:
                if right in used or left >= right:
                    continue
                used.add(right)
                chosen.append({"axis": "h", "line": cand["line"], "lo": left, "hi": right})
                if backtrack(i + 1):
                    return True
                chosen.pop()
                used.remove(right)
            used.remove(left)
        return False

    return chosen if backtrack(0) else None


def _assign_vertical_endpoints(
    candidates: list[dict],
    forbidden_rows: set[int],
) -> list[dict] | None:
    chosen: list[dict] = []
    used = set(forbidden_rows)

    def backtrack(i: int) -> bool:
        if i == len(candidates):
            return True
        cand = candidates[i]
        tops = list(cand["top_options"])
        bottoms = list(cand["bottom_options"])
        random.shuffle(tops)
        random.shuffle(bottoms)
        for top in tops:
            if top in used:
                continue
            used.add(top)
            for bottom in bottoms:
                if bottom in used or top >= bottom:
                    continue
                used.add(bottom)
                chosen.append({"axis": "v", "line": cand["line"], "lo": top, "hi": bottom})
                if backtrack(i + 1):
                    return True
                chosen.pop()
                used.remove(bottom)
            used.remove(top)
        return False

    return chosen if backtrack(0) else None


def _sample_rectangles(shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    count = random.randint(2, 5)
    colors = random.sample(RECT_COLORS, count)
    rectangles: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rect_h = random.randint(2, min(7, h - 4))
            rect_w = random.randint(2, min(7, w - 4))
            r0 = random.randint(1, h - rect_h - 2)
            c0 = random.randint(1, w - rect_w - 2)
            candidate = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if all(_boxes_have_gap(candidate, (int(r["r0"]), int(r["c0"]), int(r["r1"]), int(r["c1"]))) for r in rectangles):
                rectangles.append(
                    {
                        "color": int(color),
                        "r0": r0,
                        "c0": c0,
                        "r1": r0 + rect_h - 1,
                        "c1": c0 + rect_w - 1,
                    }
                )
                placed = True
                break
        if not placed:
            return None
    return rectangles


def _choose_cut_plan(grid: np.ndarray, rectangles: list[dict]) -> tuple[list[dict], list[dict]] | None:
    horizontal_options = _horizontal_candidates(grid, rectangles)
    vertical_options = _vertical_candidates(grid, rectangles)
    row_keys = list(horizontal_options)
    col_keys = list(vertical_options)
    layouts = [layout for layout in LAYOUTS if layout[0] <= len(row_keys) and layout[1] <= len(col_keys)]
    if not layouts:
        return None

    for _ in range(80):
        nh, nv = random.choice(layouts)
        selected_rows = random.sample(row_keys, nh)
        selected_cols = random.sample(col_keys, nv)
        h_candidates = [random.choice(horizontal_options[row]) for row in selected_rows]
        v_candidates = [random.choice(vertical_options[col]) for col in selected_cols]
        vertical_lines = {int(cand["line"]) for cand in v_candidates}
        horizontal_lines = {int(cand["line"]) for cand in h_candidates}
        horizontal_cuts = _assign_horizontal_endpoints(h_candidates, vertical_lines)
        if horizontal_cuts is None:
            continue
        vertical_cuts = _assign_vertical_endpoints(v_candidates, horizontal_lines)
        if vertical_cuts is None:
            continue
        horizontal_cuts.sort(key=lambda cut: int(cut["line"]))
        vertical_cuts.sort(key=lambda cut: int(cut["line"]))
        return horizontal_cuts, vertical_cuts
    return None


def _build_input(
    shape: tuple[int, int],
    rectangles: list[dict],
    horizontal_cuts: list[dict],
    vertical_cuts: list[dict],
) -> np.ndarray:
    grid = _build_base_grid(shape, rectangles)
    for cut in horizontal_cuts:
        grid[int(cut["line"]), int(cut["lo"])] = RED
        grid[int(cut["line"]), int(cut["hi"])] = RED
    for cut in vertical_cuts:
        grid[int(cut["lo"]), int(cut["line"])] = RED
        grid[int(cut["hi"]), int(cut["line"])] = RED
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(15, 26), random.randint(15, 26))
    rectangles = _sample_rectangles(shape)
    if rectangles is None:
        return None

    base = _build_base_grid(shape, rectangles)
    cut_plan = _choose_cut_plan(base, rectangles)
    if cut_plan is None:
        return None
    horizontal_cuts, vertical_cuts = cut_plan

    try:
        expected, h_effects, v_effects = _simulate(shape, rectangles, horizontal_cuts, vertical_cuts)
    except AssertionError:
        return None
    if not horizontal_cuts and not vertical_cuts:
        return None
    if horizontal_cuts and min(h_effects) <= 0:
        return None
    if vertical_cuts and min(v_effects) <= 0:
        return None

    inp = _build_input(shape, rectangles, horizontal_cuts, vertical_cuts)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
