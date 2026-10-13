from __future__ import annotations

from collections import deque
import random

import numpy as np


TEAL = 8
PALETTE = tuple(range(10))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator settings.
GRID_HEIGHT_RANGE = (8, 13)
GRID_WIDTH_RANGE = (8, 13)
RECT_COUNT_RANGE = (2, 5)
RECT_SIDE_RANGE = range(2, 6)
OCCUPANCY_PCT_RANGE = (22, 38)
INPUT_GAP = 1
OUTPUT_GAP = 1
PLACEMENT_NODE_BUDGET = 12000
GENERATION_ATTEMPTS = 400
SHIFT_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIMS = [(h, w, h * w) for h in RECT_SIDE_RANGE for w in RECT_SIDE_RANGE]

BBox = tuple[int, int, int, int]


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).reshape(-1)
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = exclude or set()
    colors = [int(color) for color in np.unique(np.asarray(grid, dtype=int)) if int(color) not in excluded]
    colors.sort()
    return colors


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    mask = g == int(color)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comp = np.array(coords, dtype=int)
            components.append(comp)

    components.sort(key=lambda comp: (_bbox_from_coords(comp)[0], _bbox_from_coords(comp)[1]))
    return components


def _rectangle_moves(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    marker_coords = {tuple(map(int, coord)) for coord in np.argwhere(g == TEAL)}
    assert marker_coords, "expected at least one teal control pixel"

    moves: list[tuple[int, int, int, int, int, int, int]] = []
    used_markers: set[tuple[int, int]] = set()
    colors = _colors_present(g, exclude={int(bg), TEAL})

    for color in colors:
        for coords in _find_color_components(g, color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            rect_h, rect_w = region.shape

            marks = np.argwhere(region == TEAL)
            assert marks.shape == (1, 2), "each rectangle must contain exactly one teal corner"
            marker_r, marker_c = map(int, marks[0])
            assert marker_r in (0, rect_h - 1) and marker_c in (0, rect_w - 1), "teal marker must sit on a corner"

            expected = np.full((rect_h, rect_w), int(color), dtype=int)
            expected[marker_r, marker_c] = TEAL
            assert np.array_equal(region, expected), "foreground object must be a filled rectangle with one missing corner"

            dr = -1 if marker_r == 0 else 1
            dc = -1 if marker_c == 0 else 1
            tr0 = r0 + dr
            tc0 = c0 + dc
            tr1 = r1 + dr
            tc1 = c1 + dc
            h, w = g.shape
            assert 0 <= tr0 <= tr1 < h and 0 <= tc0 <= tc1 < w, "shifted rectangle must stay in bounds"

            used_markers.add((r0 + marker_r, c0 + marker_c))
            moves.append((r0, c0, r1, c1, int(color), dr, dc))

    assert used_markers == marker_coords, "every teal marker must belong to one rectangle"
    moves.sort(key=lambda item: (item[0], item[1], item[4]))
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each teal-cornered filled rectangle one cell diagonally toward its teal corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    moves = _rectangle_moves(g, bg)

    out = g.copy()
    for r0, c0, r1, c1, _color, _dr, _dc in moves:
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    occupied = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1, color, dr, dc in moves:
        tr0 = r0 + dr
        tc0 = c0 + dc
        tr1 = r1 + dr
        tc1 = c1 + dc
        assert not occupied[tr0 : tr1 + 1, tc0 : tc1 + 1].any(), "shifted rectangles must not overlap"
        assert np.all(out[tr0 : tr1 + 1, tc0 : tc1 + 1] == int(bg)), "target region must be empty after erasing sources"
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = int(color)
        occupied[tr0 : tr1 + 1, tc0 : tc1 + 1] = True

    return out


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(box: BBox, grid_h: int, grid_w: int, gap: int) -> BBox:
    return (
        max(0, box[0] - gap),
        max(0, box[1] - gap),
        min(grid_h - 1, box[2] + gap),
        min(grid_w - 1, box[3] + gap),
    )


def _choose_sizes(count: int, area: int) -> list[tuple[int, int]] | None:
    min_pct, max_pct = OCCUPANCY_PCT_RANGE
    low = (min_pct * area + 99) // 100
    high = (max_pct * area) // 100

    sums: list[set[int]] = [{0}]
    for _ in range(count):
        next_sums: set[int] = set()
        for prev in sums[-1]:
            for _h, _w, cells in DIMS:
                next_sums.add(prev + cells)
        sums.append(next_sums)

    targets = [total for total in sums[count] if low <= total <= high]
    if not targets:
        return None

    target = random.choice(targets)
    sizes: list[tuple[int, int]] = []
    for remaining in range(count, 0, -1):
        choices = [(h, w, cells) for h, w, cells in DIMS if target - cells in sums[remaining - 1]]
        h, w, cells = random.choice(choices)
        sizes.append((h, w))
        target -= cells

    random.shuffle(sizes)
    return sizes


def _candidate_boxes(grid_h: int, grid_w: int, rect_h: int, rect_w: int, dr: int, dc: int) -> list[tuple[BBox, BBox]]:
    r0_min = 0 if dr == -1 else 1
    r0_max = grid_h - rect_h - 1 if dr == -1 else grid_h - rect_h
    c0_min = 0 if dc == -1 else 1
    c0_max = grid_w - rect_w - 1 if dc == -1 else grid_w - rect_w
    if r0_min > r0_max or c0_min > c0_max:
        return []

    boxes: list[tuple[BBox, BBox]] = []
    for in_r0 in range(r0_min, r0_max + 1):
        for in_c0 in range(c0_min, c0_max + 1):
            in_bbox = (in_r0, in_c0, in_r0 + rect_h - 1, in_c0 + rect_w - 1)
            out_bbox = (in_r0 + dr, in_c0 + dc, in_r0 + dr + rect_h - 1, in_c0 + dc + rect_w - 1)
            boxes.append((in_bbox, out_bbox))

    random.shuffle(boxes)
    return boxes


def _conflicts(
    in_bbox: BBox,
    out_bbox: BBox,
    placed: list[dict[str, object] | None],
    grid_h: int,
    grid_w: int,
) -> bool:
    expanded_in = _expand_bbox(in_bbox, grid_h, grid_w, INPUT_GAP)
    expanded_out = _expand_bbox(out_bbox, grid_h, grid_w, OUTPUT_GAP)

    for other in placed:
        if other is None:
            continue
        if _bbox_intersects(expanded_in, other["expanded_in"]):
            return True
        if _bbox_intersects(expanded_out, other["expanded_out"]):
            return True
    return False


def _choose_directions(count: int) -> list[tuple[int, int]]:
    dirs = random.sample(SHIFT_DIRS, k=min(count, len(SHIFT_DIRS)))
    while len(dirs) < count:
        dirs.append(random.choice(SHIFT_DIRS))
    random.shuffle(dirs)
    return dirs


def _place_rectangles(rects: list[dict[str, int]], grid_h: int, grid_w: int) -> list[dict[str, object]] | None:
    placed: list[dict[str, object] | None] = [None] * len(rects)
    order = sorted(range(len(rects)), key=lambda idx: rects[idx]["h"] * rects[idx]["w"], reverse=True)
    node_budget = PLACEMENT_NODE_BUDGET

    def rec(pos: int) -> bool:
        nonlocal node_budget
        if pos == len(order):
            return True
        if node_budget <= 0:
            return False
        node_budget -= 1

        idx = order[pos]
        rect = rects[idx]
        for in_bbox, out_bbox in _candidate_boxes(grid_h, grid_w, rect["h"], rect["w"], rect["dr"], rect["dc"]):
            if _conflicts(in_bbox, out_bbox, placed, grid_h, grid_w):
                continue

            placed[idx] = {
                **rect,
                "in_bbox": in_bbox,
                "out_bbox": out_bbox,
                "expanded_in": _expand_bbox(in_bbox, grid_h, grid_w, INPUT_GAP),
                "expanded_out": _expand_bbox(out_bbox, grid_h, grid_w, OUTPUT_GAP),
            }
            if rec(pos + 1):
                return True
            placed[idx] = None
        return False

    if not rec(0):
        return None
    return [item for item in placed if item is not None]


def _render_pair(placed: list[dict[str, object]], grid_h: int, grid_w: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    out = np.full((grid_h, grid_w), int(bg), dtype=int)

    for rect in placed:
        rect_h = int(rect["h"])
        rect_w = int(rect["w"])
        color = int(rect["color"])
        dr = int(rect["dr"])
        dc = int(rect["dc"])

        in_r0, in_c0, _in_r1, _in_c1 = rect["in_bbox"]
        out_r0, out_c0, _out_r1, _out_c1 = rect["out_bbox"]

        out[out_r0 : out_r0 + rect_h, out_c0 : out_c0 + rect_w] = int(color)

        marked = np.full((rect_h, rect_w), int(color), dtype=int)
        marker_r = 0 if dr == -1 else rect_h - 1
        marker_c = 0 if dc == -1 else rect_w - 1
        marked[marker_r, marker_c] = TEAL
        inp[in_r0 : in_r0 + rect_h, in_c0 : in_c0 + rect_w] = marked

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    area = grid_h * grid_w

    bg = random.choice([color for color in PALETTE if color != TEAL])
    max_rects = min(RECT_COUNT_RANGE[1], len(PALETTE) - 2)
    rect_count = random.randint(RECT_COUNT_RANGE[0], max_rects)
    sizes = _choose_sizes(rect_count, area)
    if sizes is None:
        return None

    colors = random.sample([color for color in PALETTE if color not in {bg, TEAL}], rect_count)
    directions = _choose_directions(rect_count)
    rects: list[dict[str, int]] = []
    for (rect_h, rect_w), color, (dr, dc) in zip(sizes, colors, directions):
        rects.append({"h": rect_h, "w": rect_w, "color": int(color), "dr": dr, "dc": dc})

    placed = _place_rectangles(rects, grid_h, grid_w)
    if placed is None:
        return None

    inp, expected = _render_pair(placed, grid_h, grid_w, bg)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
