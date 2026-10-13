from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 2000


def _legend_colors(g: np.ndarray) -> list[int]:
    colors = [int(v) for v in g[:, 0] if int(v) != BLACK]
    assert 1 <= len(colors) <= 6, "left legend must contain one to six colors"
    assert GREY not in colors, "grey is reserved for corner markers"
    assert len(set(colors)) == len(colors), "legend colors must be unique"
    return colors


def _corner_set(r0: int, c0: int, r1: int, c1: int) -> frozenset[tuple[int, int]]:
    return frozenset(((r0, c0), (r0, c1), (r1, c0), (r1, c1)))


def _rectangle_candidates(g: np.ndarray, legend: set[int]) -> list[dict]:
    marker_pts = {
        (int(r), int(c))
        for r, c in np.argwhere(g == GREY)
        if int(c) > 0
    }
    assert marker_pts and len(marker_pts) % 4 == 0, "grey markers must form rectangle corners"

    rows = sorted({r for r, _ in marker_pts})
    cols = sorted({c for _, c in marker_pts})
    candidates: list[dict] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            if r1 - r0 < 2:
                continue
            common_cols = [c for c in cols if (r0, c) in marker_pts and (r1, c) in marker_pts]
            for j, c0 in enumerate(common_cols):
                for c1 in common_cols[j + 1 :]:
                    if c1 - c0 < 2:
                        continue
                    corners = _corner_set(r0, c0, r1, c1)
                    if corners in seen:
                        continue
                    window = g[r0 : r1 + 1, c0 : c1 + 1]
                    if not any(int(v) in legend for v in np.unique(window)):
                        continue
                    candidates.append({"bbox": (r0, c0, r1, c1), "corners": corners})
                    seen.add(corners)
    assert candidates, "expected at least one marker-defined rectangle"
    return candidates


def _exact_cover_rectangles(candidates: list[dict], marker_pts: set[tuple[int, int]]) -> list[dict]:
    by_marker: dict[tuple[int, int], list[int]] = {pt: [] for pt in marker_pts}
    for i, cand in enumerate(candidates):
        for pt in cand["corners"]:
            if pt in by_marker:
                by_marker[pt].append(i)

    solutions: list[list[int]] = []

    def search(unused: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not unused:
            solutions.append(chosen[:])
            return
        pivot = min(unused, key=lambda pt: len(by_marker[pt]))
        for idx in by_marker[pivot]:
            corners = candidates[idx]["corners"]
            if corners <= unused:
                search(unused - corners, chosen + [idx])

    search(set(marker_pts), [])
    assert len(solutions) == 1, "corner markers must define rectangles unambiguously"
    return [candidates[i] for i in solutions[0]]


def _recover_rectangles(g: np.ndarray, order: list[int]) -> list[dict]:
    priority = {color: rank for rank, color in enumerate(order)}
    marker_pts = {
        (int(r), int(c))
        for r, c in np.argwhere(g == GREY)
        if int(c) > 0
    }
    candidates = _rectangle_candidates(g, set(order))
    rects = _exact_cover_rectangles(candidates, marker_pts)

    cover = np.zeros(g.shape, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1

    recovered: list[dict] = []
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        evidence: Counter[int] = Counter()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                v = int(g[r, c])
                if v in priority and cover[r, c] == 1:
                    evidence[v] += 1
        if not evidence:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    v = int(g[r, c])
                    if v in priority:
                        evidence[v] += 1
        assert evidence, "each rectangle must have visible color evidence"
        best_count = max(evidence.values())
        best = [color for color, count in evidence.items() if count == best_count]
        assert len(best) == 1, "rectangle color evidence must not tie"
        recovered.append({"bbox": rect["bbox"], "color": best[0]})
    return recovered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-render marked overlapping rectangles using the priority order encoded by the left color bar.

    Input structure:
    - The background is black, and the first column is a vertical legend of distinct non-grey colors.
    - The remaining columns contain solid axis-aligned rectangles drawn with those legend colors.
    - Each rectangle has four visible grey corner pixels marking its exact bounding box, even when rectangles overlap.

    Rule:
    - Read the first column from top to bottom as increasing layer priority.
    - Recover every rectangle from a unique set of four grey corner markers and determine its color from visible non-grey cells.
    - For every position covered by one or more rectangles, keep the color whose legend entry has the highest priority.
    - Return the scene without the legend column, replacing all grey marker pixels by the color that wins at that position.

    Key insight:
    - The visible overlap colors in the input are not reliable; the left legend supplies the true layer order, while grey corner markers reveal the full hidden rectangle extents.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w >= 4, "grid must include legend and rectangle area"

    order = _legend_colors(g)
    priority = {color: rank for rank, color in enumerate(order)}
    rects = _recover_rectangles(g, order)

    out = np.full((h, w - 1), BLACK, dtype=int)
    for rect in sorted(rects, key=lambda item: priority[item["color"]]):
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 - 1 : c1] = rect["color"]
    return out


def _area_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _corners_area(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _render_area(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int, int]],
    draw_order: list[int],
    add_markers: bool,
) -> np.ndarray:
    area = np.full((height, width), BLACK, dtype=int)
    for idx in draw_order:
        r0, c0, r1, c1, color = rects[idx]
        area[r0 : r1 + 1, c0 : c1 + 1] = color
    if add_markers:
        for r0, c0, r1, c1, _ in rects:
            for r, c in _corners_area((r0, c0, r1, c1)):
                area[r, c] = GREY
    return area


def _expected_output(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int, int]],
    legend_order: list[int],
) -> np.ndarray:
    rank = {color: i for i, color in enumerate(legend_order)}
    draw_order = sorted(range(len(rects)), key=lambda idx: rank[rects[idx][4]])
    return _render_area(height, width, rects, draw_order, add_markers=False)


def _has_clear_evidence(rects: list[tuple[int, int, int, int, int]]) -> bool:
    cover: Counter[tuple[int, int]] = Counter()
    all_corners: set[tuple[int, int]] = set()
    cells_by_rect: list[set[tuple[int, int]]] = []
    for r0, c0, r1, c1, _ in rects:
        cells = _area_cells((r0, c0, r1, c1))
        cells_by_rect.append(cells)
        cover.update(cells)
        all_corners.update(_corners_area((r0, c0, r1, c1)))
    for cells in cells_by_rect:
        if not any(cover[p] == 1 and p not in all_corners for p in cells):
            return False
    return True


def _has_visible_overlap(rects: list[tuple[int, int, int, int, int]]) -> bool:
    cover: Counter[tuple[int, int]] = Counter()
    all_corners: set[tuple[int, int]] = set()
    for r0, c0, r1, c1, _ in rects:
        cover.update(_area_cells((r0, c0, r1, c1)))
        all_corners.update(_corners_area((r0, c0, r1, c1)))
    return any(count > 1 and p not in all_corners for p, count in cover.items())


def _unique_corners(rects: list[tuple[int, int, int, int, int]]) -> bool:
    corners: list[tuple[int, int]] = []
    for r0, c0, r1, c1, _ in rects:
        corners.extend(_corners_area((r0, c0, r1, c1)))
    return len(corners) == len(set(corners))


def _random_bbox(rng: random.Random, height: int, width: int) -> tuple[int, int, int, int]:
    rh = rng.randint(3, min(8, height))
    rw = rng.randint(3, min(8, width))
    r0 = rng.randint(0, height - rh)
    c0 = rng.randint(0, width - rw)
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _choose_display_order(
    rng: random.Random,
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int, int]],
    legend_order: list[int],
    want_wrong_overlap: bool,
) -> list[int] | None:
    rank = {color: i for i, color in enumerate(legend_order)}
    priority_order = sorted(range(len(rects)), key=lambda idx: rank[rects[idx][4]])
    expected = _render_area(height, width, rects, priority_order, add_markers=False)
    all_corners: set[tuple[int, int]] = set()
    for r0, c0, r1, c1, _ in rects:
        all_corners.update(_corners_area((r0, c0, r1, c1)))

    if not want_wrong_overlap:
        return priority_order

    choices = list(range(len(rects)))
    for _ in range(80):
        rng.shuffle(choices)
        if choices == priority_order:
            continue
        shown = _render_area(height, width, rects, choices, add_markers=False)
        for r in range(height):
            for c in range(width):
                if (r, c) not in all_corners and int(shown[r, c]) != int(expected[r, c]):
                    return choices[:]
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(2, 6)
    legend_order = rng.sample(list(PALETTE), color_count)
    height = rng.randint(max(8, color_count + 3), 22)
    width = rng.randint(8, 23)

    rects: list[tuple[int, int, int, int, int]] = []
    for color in legend_order:
        r0, c0, r1, c1 = _random_bbox(rng, height, width)
        rects.append((r0, c0, r1, c1, color))

    if not _unique_corners(rects):
        return None
    if not _has_visible_overlap(rects):
        return None
    if not _has_clear_evidence(rects):
        return None

    expected = _expected_output(height, width, rects, legend_order)
    want_wrong_overlap = rng.random() < 0.55
    display_order = _choose_display_order(rng, height, width, rects, legend_order, want_wrong_overlap)
    if display_order is None:
        return None

    area = _render_area(height, width, rects, display_order, add_markers=True)
    grid = np.full((height, width + 1), BLACK, dtype=int)
    for r, color in enumerate(legend_order):
        grid[r, 0] = color
    grid[:, 1:] = area

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if int((grid == BLACK).sum()) < grid.size // 4:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(solved, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid layered-rectangle input")
