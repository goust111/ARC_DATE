from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _on_bbox_boundary(r: int, c: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return r == r0 or r == r1 or c == c0 or c == c1


def _find_marker_rectangles(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, BBox]]]:
    marker_rects: list[tuple[int, BBox]] = []
    other_colors: list[int] = []

    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        coords = np.argwhere(grid == color)
        if len(coords) == 3:
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            if r0 < r1 and c0 < c1 and all(_on_bbox_boundary(int(r), int(c), box) for r, c in coords):
                marker_rects.append((int(color), box))
                continue
        other_colors.append(int(color))

    assert marker_rects, "expected at least one 3-point marker group"
    assert len(other_colors) == 1, "expected exactly one sand color distinct from marker colors"
    return other_colors[0], marker_rects


def _settle_sand(grid: np.ndarray, sand_color: int, rects: list[BBox]) -> np.ndarray:
    h, w = map(int, grid.shape)
    out = grid.copy()
    out[out == sand_color] = _get_bg(grid)

    for c in range(w):
        blocked = np.zeros(h, dtype=bool)
        for r0, c0, r1, c1 in rects:
            if c0 <= c <= c1:
                blocked[r0 : r1 + 1] = True

        sand_rows = sorted((int(r) for r in np.flatnonzero(grid[:, c] == sand_color)), reverse=True)
        final_rows: list[int] = []
        for start_r in sand_rows:
            assert not blocked[start_r], "sand should not start inside an invisible tray"
            target_r = start_r
            while target_r + 1 < h and not blocked[target_r + 1]:
                target_r += 1
            blocked[target_r] = True
            final_rows.append(target_r)

        for r in final_rows:
            assert out[r, c] == _get_bg(grid), "settled sand may not overwrite marker pixels"
            out[r, c] = sand_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let scattered sand pixels fall vertically until they rest on invisible tray rectangles implied by three same-color markers.

    Input structure:
    - The background is black.
    - Several marker colors each appear exactly three times, and each same-color triple lies on the boundary of its shared bounding box.
    - One extra sand color appears as isolated pixels scattered in the upper part of the grid.

    Rule:
    - For every three-pixel marker group, infer its full bounding box and treat that whole rectangle as a solid invisible obstacle.
    - Move each sand pixel straight downward within its own column until the next step would enter an invisible rectangle or a previously settled sand pixel; if nothing blocks it, it lands on the bottom row.
    - Keep every marker pixel fixed and visible, and do not draw the inferred rectangles themselves.

    Key insight:
    - The three sparse markers are not a partial shape to complete visually; they only define hidden collision rectangles that can be recovered from the final sand heights.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = _get_bg(g)
    sand_color, marker_rects = _find_marker_rectangles(g, bg)
    rects = [box for _color, box in marker_rects]
    return _settle_sand(g, sand_color, rects)


GEN_BG = 0
GEN_ATTEMPTS = 400
GEN_COLORS = tuple(range(1, 10))
GEN_GRID_H = (15, 25)
GEN_GRID_W = (16, 27)
GEN_BOX_COUNT = (2, 4)
GEN_BOX_H = (2, 4)
GEN_BOX_W = (4, 7)
GEN_BETWEEN_GAP = 2
GEN_SIDE_MARGIN = 1
GEN_TOP_ROW_MIN = 6
GEN_BOTTOM_MARGIN = 2
GEN_SAND_TOTAL = (6, 12)
GEN_STACK_SIZE = (2, 3)


def _randint_inclusive(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _random_composition(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(x) for x in rng.choice(np.arange(total + parts - 1), size=parts - 1, replace=False))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(int(cut - prev - 1))
        prev = cut
    return values


def _choose_boxes(rng: np.random.Generator, h: int, w: int) -> list[BBox] | None:
    count = _randint_inclusive(rng, GEN_BOX_COUNT[0], GEN_BOX_COUNT[1])
    widths = [_randint_inclusive(rng, GEN_BOX_W[0], GEN_BOX_W[1]) for _ in range(count)]
    heights = [_randint_inclusive(rng, GEN_BOX_H[0], GEN_BOX_H[1]) for _ in range(count)]

    min_width = (
        2 * GEN_SIDE_MARGIN
        + sum(widths)
        + GEN_BETWEEN_GAP * (count - 1)
    )
    if min_width > w:
        return None

    extra = int(w - min_width)
    gap_bonus = _random_composition(rng, extra, count + 1)

    c = GEN_SIDE_MARGIN + gap_bonus[0]
    boxes: list[BBox] = []
    for i, (box_h, box_w) in enumerate(zip(heights, widths)):
        max_r0 = h - box_h - GEN_BOTTOM_MARGIN
        if max_r0 < GEN_TOP_ROW_MIN:
            return None
        r0 = _randint_inclusive(rng, GEN_TOP_ROW_MIN, max_r0)
        boxes.append((r0, c, r0 + box_h - 1, c + box_w - 1))
        c += box_w
        if i < count - 1:
            c += GEN_BETWEEN_GAP + gap_bonus[i + 1]
    return boxes


def _boundary_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    cells.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    cells.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return cells


def _pick_markers(rng: np.random.Generator, box: BBox) -> list[Coord]:
    combos = []
    for cells in itertools.combinations(_boundary_cells(box), 3):
        coords = np.array(cells, dtype=int)
        if _bbox(coords) == box:
            combos.append(list(cells))
    assert combos, "box boundary should admit at least one defining 3-point subset"
    return combos[_randint_inclusive(rng, 0, len(combos) - 1)]


def _sand_target_rows(grid: np.ndarray, sand_color: int) -> dict[int, list[int]]:
    rows_by_col: dict[int, list[int]] = {}
    for c in range(grid.shape[1]):
        rows = [int(r) for r in np.flatnonzero(grid[:, c] == sand_color)]
        if rows:
            rows_by_col[c] = rows
    return rows_by_col


def _sample_sand_columns(
    rng: np.random.Generator, boxes: list[BBox], width: int
) -> dict[int, int] | None:
    covered_cols = sorted({c for _r0, c0, _r1, c1 in boxes for c in range(c0, c1 + 1)})
    free_cols = [c for c in range(width) if c not in covered_cols]
    if not free_cols:
        return None

    inactive_count = _randint_inclusive(rng, 1, max(1, len(boxes) - 1))
    inactive = set(int(i) for i in rng.choice(len(boxes), size=inactive_count, replace=False))
    active_cols = sorted(
        {
            c
            for i, (_r0, c0, _r1, c1) in enumerate(boxes)
            if i not in inactive
            for c in range(c0, c1 + 1)
        }
    )
    if not active_cols:
        return None

    total = _randint_inclusive(rng, GEN_SAND_TOTAL[0], GEN_SAND_TOTAL[1])
    stack_col = int(rng.choice(active_cols if rng.random() < 0.7 else free_cols))
    counts: dict[int, int] = {stack_col: _randint_inclusive(rng, GEN_STACK_SIZE[0], GEN_STACK_SIZE[1])}

    must_have = [int(rng.choice(active_cols)), int(rng.choice(free_cols))]
    for col in must_have:
        counts[col] = counts.get(col, 0) + 1

    candidate_cols = sorted(set(active_cols + free_cols))
    while sum(counts.values()) < total:
        col = int(rng.choice(candidate_cols))
        counts[col] = counts.get(col, 0) + 1

    for i, box in enumerate(boxes):
        if i in inactive:
            _r0, c0, _r1, c1 = box
            assert not any(c0 <= col <= c1 for col in counts), "inactive trays must receive no sand above"

    return counts


def _place_sands(
    rng: np.random.Generator,
    grid: np.ndarray,
    counts: dict[int, int],
    boxes: list[BBox],
    sand_color: int,
) -> bool:
    top_limit = min(r0 for r0, _c0, _r1, _c1 in boxes) - 2
    if top_limit < 2:
        return False

    placed: list[Coord] = []
    for col, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rows = list(range(top_limit + 1))
        rng.shuffle(rows)
        chosen: list[int] = []
        for row in rows:
            if any(abs(row - rr) <= 1 and abs(col - cc) <= 1 for rr, cc in placed):
                continue
            chosen.append(int(row))
            placed.append((int(row), int(col)))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            return False

    for r, c in placed:
        grid[r, c] = sand_color
    return True


def _build_grid(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint_inclusive(rng, GEN_GRID_H[0], GEN_GRID_H[1])
    w = _randint_inclusive(rng, GEN_GRID_W[0], GEN_GRID_W[1])
    boxes = _choose_boxes(rng, h, w)
    if boxes is None:
        return None

    needed_colors = len(boxes) + 1
    colors = rng.choice(np.array(GEN_COLORS, dtype=int), size=needed_colors, replace=False).tolist()
    sand_color = int(colors[0])
    marker_colors = [int(c) for c in colors[1:]]

    grid = np.full((h, w), GEN_BG, dtype=int)
    for color, box in zip(marker_colors, boxes):
        for r, c in _pick_markers(rng, box):
            grid[r, c] = color

    sand_counts = _sample_sand_columns(rng, boxes, w)
    if sand_counts is None:
        return None
    if not _place_sands(rng, grid, sand_counts, boxes, sand_color):
        return None

    before = grid.copy()
    after = main(before)
    if np.array_equal(before, after):
        return None

    rows_by_col = _sand_target_rows(after, sand_color)
    if not any(len(rows) >= 2 for rows in rows_by_col.values()):
        return None
    covered_cols = {c for _r0, c0, _r1, c1 in boxes for c in range(c0, c1 + 1)}
    if not any(col not in covered_cols for col in rows_by_col):
        return None

    return before


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for invisible-tray sand puzzle")
