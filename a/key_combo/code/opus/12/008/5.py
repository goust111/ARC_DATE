from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
DEPTH_COLORS = {1: 8, 2: 4, 3: 3}
HALF_OPTIONS = (2, 3, 4, 5, 6)  # odd side lengths 5..13
GEN_RETRIES = 1200


Rect = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rect_from_center(cr: int, cc: int, half_h: int, half_w: int) -> Rect:
    return (cr - half_h, cc - half_w, cr + half_h, cc + half_w)


def _rect_center(rect: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _rect_halves(rect: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return ((r1 - r0) // 2, (c1 - c0) // 2)


def _border_cells(rect: Rect) -> frozenset[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return frozenset(cells)


def _side_cells(rect: Rect, side: str) -> frozenset[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return frozenset((r0, c) for c in range(c0, c1 + 1))
    if side == "bottom":
        return frozenset((r1, c) for c in range(c0, c1 + 1))
    if side == "left":
        return frozenset((r, c0) for r in range(r0, r1 + 1))
    if side == "right":
        return frozenset((r, c1) for r in range(r0, r1 + 1))
    raise ValueError(side)


def _interior_overlap_exists(rects: list[Rect]) -> bool:
    for i, a in enumerate(rects):
        ar0, ac0, ar1, ac1 = a
        for b in rects[i + 1 :]:
            br0, bc0, br1, bc1 = b
            rr0 = max(ar0 + 1, br0 + 1)
            cc0 = max(ac0 + 1, bc0 + 1)
            rr1 = min(ar1 - 1, br1 - 1)
            cc1 = min(ac1 - 1, bc1 - 1)
            if rr0 <= rr1 and cc0 <= cc1:
                return True
    return False


def _find_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    markers = []
    for r, c in np.argwhere(g == RED):
        rr = int(r)
        cc = int(c)
        if all(
            not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == RED)
            for nr, nc in _neighbors4(rr, cc)
        ):
            markers.append((rr, cc))
    return sorted(markers)


def _candidate_rectangles(
    grid: np.ndarray, markers: list[tuple[int, int]]
) -> list[list[dict[str, object]]]:
    red_cells = {tuple(map(int, p)) for p in np.argwhere(grid == RED)}
    marker_set = set(markers)
    border_red = red_cells - marker_set
    height, width = map(int, grid.shape)

    all_candidates: list[list[dict[str, object]]] = []
    for cr, cc in markers:
        candidates = []
        for half_h in HALF_OPTIONS:
            if cr - half_h < 0 or cr + half_h >= height:
                continue
            for half_w in HALF_OPTIONS:
                if cc - half_w < 0 or cc + half_w >= width:
                    continue
                rect = _rect_from_center(cr, cc, half_h, half_w)
                border = _border_cells(rect)
                if border & marker_set:
                    continue
                if border <= border_red:
                    candidates.append({"rect": rect, "border": border})
        all_candidates.append(candidates)
    return all_candidates


def _search_rectangle_solutions(
    candidate_lists: list[list[dict[str, object]]],
    red_border_cells: set[tuple[int, int]],
    limit: int = 2,
) -> list[list[dict[str, object]]]:
    order = sorted(range(len(candidate_lists)), key=lambda i: len(candidate_lists[i]))
    suffix_cover: list[set[tuple[int, int]]] = [set() for _ in range(len(order) + 1)]
    for pos in range(len(order) - 1, -1, -1):
        idx = order[pos]
        cover = set(suffix_cover[pos + 1])
        for cand in candidate_lists[idx]:
            cover.update(cand["border"])
        suffix_cover[pos] = cover

    chosen: list[dict[str, object] | None] = [None] * len(candidate_lists)
    solutions: list[list[dict[str, object]]] = []

    def backtrack(pos: int, covered: set[tuple[int, int]]) -> None:
        if len(solutions) >= limit:
            return
        uncovered = red_border_cells - covered
        if not uncovered <= suffix_cover[pos]:
            return
        if pos == len(order):
            if not uncovered:
                solutions.append([cand for cand in chosen if cand is not None])
            return

        idx = order[pos]
        for cand in candidate_lists[idx]:
            chosen[idx] = cand
            backtrack(pos + 1, covered | set(cand["border"]))
            chosen[idx] = None

    backtrack(0, set())
    return solutions


def _solve_rectangles(grid: np.ndarray) -> list[Rect]:
    markers = _find_markers(grid)
    assert markers, "expected at least one isolated red center marker"

    red_border_cells = {tuple(map(int, p)) for p in np.argwhere(grid == RED)} - set(markers)
    candidate_lists = _candidate_rectangles(grid, markers)
    assert all(cands for cands in candidate_lists), "every marker must admit a rectangle candidate"

    solutions = _search_rectangle_solutions(candidate_lists, red_border_cells, limit=2)
    assert len(solutions) == 1, "rectangle ranges must be uniquely determined"
    return [cand["rect"] for cand in solutions[0]]


def _render_output(shape: tuple[int, int], rects: list[Rect], red_mask: np.ndarray) -> np.ndarray:
    depth = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        depth[r0 + 1 : r1, c0 + 1 : c1] += 1

    out = np.zeros(shape, dtype=int)
    out[depth == 1] = DEPTH_COLORS[1]
    out[depth == 2] = DEPTH_COLORS[2]
    out[depth == 3] = DEPTH_COLORS[3]
    out[depth >= 4] = 6
    out[red_mask] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover overlapping hollow rectangles from centered red markers and recolor pixels by interior overlap depth.

    Input structure
    - The grid has a black background and red foreground only.
    - Foreground consists of 3-4 hollow axis-aligned red rectangles plus one isolated red center marker for each rectangle.
    - Rectangle borders may overlap each other, and border fragments may pass through other rectangles' interiors.

    Rule
    - Identify the isolated red marker pixels as rectangle centers.
    - Reconstruct one odd-by-odd rectangle around each center so that every chosen border is red and, together, the chosen borders explain all non-marker red pixels.
    - For each non-red cell, count how many reconstructed rectangle interiors strictly contain it; keep depth 0 black, map depth 1 to teal, 2 to yellow, 3 to green, and 4+ to magenta.
    - Preserve every original red border pixel and every red center marker pixel.

    Key insight
    - A single center can admit multiple symmetric rectangle candidates when other borders intrude on its row or column, so the true rectangles are recovered by a global exact-cover style fit of all red border fragments at once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BG, RED}), "expected only black and red in the input"

    rects = _solve_rectangles(g)
    return _render_output(g.shape, rects, g == RED)


def _draw_input(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        for r, c in _border_cells(rect):
            grid[r, c] = RED
        cr, cc = _rect_center(rect)
        grid[cr, cc] = RED
    return grid


def _centers_match_markers(grid: np.ndarray, rects: list[Rect]) -> bool:
    expected = sorted(_rect_center(rect) for rect in rects)
    return expected == _find_markers(grid)


def _clean_side_count(rects: list[Rect], idx: int) -> int:
    other_borders = set()
    for j, rect in enumerate(rects):
        if j != idx:
            other_borders.update(_border_cells(rect))
    return sum(
        _side_cells(rects[idx], side).isdisjoint(other_borders)
        for side in ("top", "bottom", "left", "right")
    )


def _all_have_two_clean_sides(rects: list[Rect]) -> bool:
    return all(_clean_side_count(rects, idx) >= 2 for idx in range(len(rects)))


def _blocked_direction_count(rects: list[Rect]) -> int:
    all_borders = [set(_border_cells(rect)) for rect in rects]
    total = 0
    for idx, rect in enumerate(rects):
        cr, cc = _rect_center(rect)
        half_h, half_w = _rect_halves(rect)
        other = set().union(*(all_borders[j] for j in range(len(rects)) if j != idx))
        for dr, dc, limit in (
            (-1, 0, half_h),
            (1, 0, half_h),
            (0, -1, half_w),
            (0, 1, half_w),
        ):
            if any((cr + dr * step, cc + dc * step) in other for step in range(2, limit)):
                total += 1
    return total


def _contains_point_interior(rect: Rect, point: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    pr, pc = point
    return r0 < pr < r1 and c0 < pc < c1


def _center_values_for_overlap(limit: int, half: int, target: int, strict: bool = True) -> list[int]:
    pad = half - 1 if strict else half
    lo = max(half, target - pad)
    hi = min(limit - 1 - half, target + pad)
    return list(range(lo, hi + 1))


def _marker_zone(center: tuple[int, int]) -> set[tuple[int, int]]:
    zone = {center}
    zone.update(_neighbors4(*center))
    return zone


def _rects_marker_isolated(rects: list[Rect]) -> bool:
    centers = [_rect_center(rect) for rect in rects]
    borders = [_border_cells(rect) for rect in rects]
    for i, center in enumerate(centers):
        zone = _marker_zone(center)
        if any(zone & borders[j] for j in range(len(rects)) if j != i):
            return False
        if any(abs(center[0] - other[0]) + abs(center[1] - other[1]) == 1 for j, other in enumerate(centers) if j != i):
            return False
    return True


def _interior_intersection(a: Rect, b: Rect) -> tuple[int, int, int, int] | None:
    rr0 = max(a[0] + 1, b[0] + 1)
    cc0 = max(a[1] + 1, b[1] + 1)
    rr1 = min(a[2] - 1, b[2] - 1)
    cc1 = min(a[3] - 1, b[3] - 1)
    if rr0 > rr1 or cc0 > cc1:
        return None
    return (rr0, cc0, rr1, cc1)


def _build_anchor_and_blocker(shape: tuple[int, int]) -> tuple[Rect, Rect, tuple[int, int]] | None:
    height, width = shape
    direction = random.choice(("up", "down", "left", "right"))
    for _ in range(240):
        if direction in ("up", "down"):
            ahh = random.choice((4, 5, 6))
            ahw = random.choice(HALF_OPTIONS)
            bhh = random.choice(HALF_OPTIONS)
            bhw = random.choice(HALF_OPTIONS)
            dist = random.randint(2, ahh - 2)

            acr_lo = ahh if direction == "down" else dist + 2 * bhh
            acr_hi = height - 1 - ahh if direction == "up" else height - 1 - dist - 2 * bhh
            if acr_lo > acr_hi:
                continue
            acr = random.randint(acr_lo, acr_hi)
            acc = random.randint(ahw, width - 1 - ahw)
            bcr = acr - dist - bhh if direction == "up" else acr + dist + bhh

            col_lo = max(bhw, acc - (bhw - 1))
            col_hi = min(width - 1 - bhw, acc + (bhw - 1))
            if col_lo > col_hi:
                continue
            bcc = random.randint(col_lo, col_hi)
        else:
            ahh = random.choice(HALF_OPTIONS)
            ahw = random.choice((4, 5, 6))
            bhh = random.choice(HALF_OPTIONS)
            bhw = random.choice(HALF_OPTIONS)
            dist = random.randint(2, ahw - 2)

            acc_lo = ahw if direction == "right" else dist + 2 * bhw
            acc_hi = width - 1 - ahw if direction == "left" else width - 1 - dist - 2 * bhw
            if acc_lo > acc_hi:
                continue
            acc = random.randint(acc_lo, acc_hi)
            acr = random.randint(ahh, height - 1 - ahh)
            bcc = acc - dist - bhw if direction == "left" else acc + dist + bhw

            row_lo = max(bhh, acr - (bhh - 1))
            row_hi = min(height - 1 - bhh, acr + (bhh - 1))
            if row_lo > row_hi:
                continue
            bcr = random.randint(row_lo, row_hi)

        anchor = _rect_from_center(acr, acc, ahh, ahw)
        blocker = _rect_from_center(bcr, bcc, bhh, bhw)
        if not _rects_marker_isolated([anchor, blocker]):
            continue
        if not _all_have_two_clean_sides([anchor, blocker]):
            continue
        overlap = _interior_intersection(anchor, blocker)
        if overlap is None:
            continue
        rr0, cc0, rr1, cc1 = overlap
        point = (random.randint(rr0, rr1), random.randint(cc0, cc1))
        return anchor, blocker, point
    return None


def _random_overlap_rect(
    shape: tuple[int, int],
    overlap_point: tuple[int, int],
    used_centers: set[tuple[int, int]],
    existing_rects: list[Rect],
) -> Rect | None:
    height, width = shape
    candidates: list[Rect] = []
    for half_h in HALF_OPTIONS:
        rows = _center_values_for_overlap(height, half_h, overlap_point[0], strict=True)
        if not rows:
            continue
        for half_w in HALF_OPTIONS:
            cols = _center_values_for_overlap(width, half_w, overlap_point[1], strict=True)
            if not cols:
                continue
            for cr in rows:
                for cc in cols:
                    center = (cr, cc)
                    if center in used_centers:
                        continue
                    rect = _rect_from_center(cr, cc, half_h, half_w)
                    trial = existing_rects + [rect]
                    if _rects_marker_isolated(trial) and _all_have_two_clean_sides(trial):
                        candidates.append(rect)
    if not candidates:
        return None
    return random.choice(candidates)


def _semantic_scene_ok(grid: np.ndarray, rects: list[Rect]) -> bool:
    if not _centers_match_markers(grid, rects):
        return False

    if not _interior_overlap_exists(rects):
        return False

    if not _all_have_two_clean_sides(rects):
        return False

    if _blocked_direction_count(rects) < 1:
        return False

    markers = _find_markers(grid)
    candidate_lists = _candidate_rectangles(grid, markers)
    if not all(candidate_lists):
        return False

    red_border_cells = {tuple(map(int, p)) for p in np.argwhere(grid == RED)} - set(markers)
    solutions = _search_rectangle_solutions(candidate_lists, red_border_cells, limit=3)
    if len(solutions) != 1:
        return False

    solved_rects = sorted(tuple(cand["rect"]) for cand in solutions[0])
    if solved_rects != sorted(rects):
        return False

    expected = _render_output(grid.shape, rects, grid == RED)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    rect_count = 3 if random.random() < 0.75 else 4

    pair = _build_anchor_and_blocker((height, width))
    if pair is None:
        return None
    anchor, blocker, overlap_point = pair
    rects = [anchor, blocker]
    used_centers = {_rect_center(anchor), _rect_center(blocker)}

    if not _rects_marker_isolated(rects):
        return None

    while len(rects) < rect_count:
        rect = _random_overlap_rect((height, width), overlap_point, used_centers, rects)
        if rect is None:
            return None
        rects.append(rect)
        used_centers.add(_rect_center(rect))

    if not _rects_marker_isolated(rects):
        return None

    random.shuffle(rects)
    grid = _draw_input((height, width), rects)
    if not _semantic_scene_ok(grid, rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangle scene")
