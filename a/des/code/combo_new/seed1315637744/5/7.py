from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = tuple(range(1, 10))


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = map(int, mask.shape)
    labels = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []
    next_id = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
            next_id += 1

    return labels, components


def _component_info(grid: np.ndarray, idx: int, coords: np.ndarray) -> dict[str, object]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    colors = np.unique(grid[rows, cols])
    area = int(coords.shape[0])
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    is_line = (height == 1 or width == 1) and area == max(height, width)
    orientation = "h" if height == 1 and is_line else "v" if width == 1 and is_line else None
    return {
        "idx": idx,
        "coords": coords,
        "area": area,
        "colors": colors,
        "monochrome": len(colors) == 1,
        "color": int(colors[0]) if len(colors) == 1 else None,
        "bbox": (r0, c0, r1, c1),
        "is_line": is_line,
        "orientation": orientation,
    }


def _scan_first_component(
    labels: np.ndarray, start_r: int, start_c: int, dr: int, dc: int
) -> tuple[int | None, int]:
    h, w = map(int, labels.shape)
    gap = 0
    r, c = start_r, start_c
    while 0 <= r < h and 0 <= c < w:
        comp_id = int(labels[r, c])
        if comp_id != -1:
            return comp_id, gap
        gap += 1
        r += dr
        c += dc
    return None, gap


def _arrow_endpoints(comp: dict[str, object]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    coords = comp["coords"]
    assert isinstance(coords, np.ndarray)
    orientation = comp["orientation"]
    if orientation == "h":
        row = int(coords[0, 0])
        cols = np.sort(coords[:, 1])
        return [((row, int(cols[0])), (0, -1)), ((row, int(cols[-1])), (0, 1))]
    assert orientation == "v"
    col = int(coords[0, 1])
    rows = np.sort(coords[:, 0])
    return [((int(rows[0]), col), (-1, 0)), ((int(rows[-1]), col), (1, 0))]


def _ordered_arrow_cells(comp: dict[str, object], tip: tuple[int, int]) -> list[tuple[int, int]]:
    coords = comp["coords"]
    assert isinstance(coords, np.ndarray)
    orientation = comp["orientation"]
    if orientation == "h":
        row = int(coords[0, 0])
        cols = np.sort(coords[:, 1])
        if int(tip[1]) == int(cols[0]):
            return [(row, int(c)) for c in cols]
        return [(row, int(c)) for c in cols[::-1]]

    col = int(coords[0, 1])
    rows = np.sort(coords[:, 0])
    if int(tip[0]) == int(rows[0]):
        return [(int(r), col) for r in rows]
    return [(int(r), col) for r in rows[::-1]]


def _count_color_runs(grid: np.ndarray, cells: list[tuple[int, int]]) -> int:
    colors = [int(grid[r, c]) for r, c in cells]
    runs = 1
    for prev, cur in zip(colors, colors[1:]):
        if cur != prev:
            runs += 1
    return runs


def _dilate_cells(
    cells: set[tuple[int, int]], blocked: np.ndarray, steps: int
) -> set[tuple[int, int]]:
    grown = set(cells)
    h, w = map(int, blocked.shape)
    for _ in range(steps):
        frontier: set[tuple[int, int]] = set()
        for r, c in grown:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in grown or blocked[nr, nc]:
                    continue
                frontier.add((nr, nc))
        grown |= frontier
    return grown


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each straight arrow indicator and dilate the pointed monochrome object by the arrow's color-layer count.

    Input structure:
    - The background is black.
    - Each non-background connected component is either a small monochrome target object or a straight horizontal/vertical arrow indicator.
    - Target objects are 4-connected monochrome shapes, while each arrow is a 2-to-4-pixel line whose nearer endpoint points toward exactly one target object across at least one black cell.
    - Arrow cells may use one to three consecutive color segments from tip to tail.

    Rule:
    - Identify every arrow by finding the unique line component whose one outward endpoint sees a target object first along its row or column.
    - Pair that arrow with the seen object, and count the number of color runs along the arrow from tip to tail; this count is K.
    - Erase all arrow cells to black.
    - For each paired target object, perform K rounds of 4-neighbor dilation in its own color, but never grow onto any other originally non-black cell.

    Key insight:
    - The arrow encodes two things at once: geometry tells which object is selected, and visible color segmentation along the arrow tells how many dilation layers to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    labels, raw_components = _component_labels(g != BG)
    assert raw_components, "expected at least one non-background component"

    components = [_component_info(g, idx, coords) for idx, coords in enumerate(raw_components)]
    object_ids = {
        int(comp["idx"])
        for comp in components
        if bool(comp["monochrome"]) and 3 <= int(comp["area"]) <= 6
    }
    arrow_candidates = [
        comp for comp in components if bool(comp["is_line"]) and 2 <= int(comp["area"]) <= 4
    ]

    pairs: list[dict[str, object]] = []
    for comp in arrow_candidates:
        hits: list[tuple[int, tuple[int, int]]] = []
        for tip, (dr, dc) in _arrow_endpoints(comp):
            obj_id, gap = _scan_first_component(labels, tip[0] + dr, tip[1] + dc, dr, dc)
            if obj_id is not None and gap >= 1 and obj_id in object_ids and obj_id != int(comp["idx"]):
                hits.append((obj_id, tip))
        if len(hits) != 1:
            continue
        obj_id, tip = hits[0]
        ordered = _ordered_arrow_cells(comp, tip)
        pairs.append(
            {
                "arrow_id": int(comp["idx"]),
                "object_id": int(obj_id),
                "k": _count_color_runs(g, ordered),
            }
        )

    arrow_ids = {int(pair["arrow_id"]) for pair in pairs}
    target_ids = {int(pair["object_id"]) for pair in pairs}
    all_ids = {int(comp["idx"]) for comp in components}
    assert pairs, "expected at least one arrow-object pair"
    assert len(arrow_ids) == len(pairs), "each arrow must pair once"
    assert len(target_ids) == len(pairs), "each object must be targeted once"
    assert not (arrow_ids & target_ids), "a component cannot be both arrow and target"
    assert all_ids == arrow_ids | target_ids, "expected only arrows and target objects"

    out = g.copy()
    for arrow_id in arrow_ids:
        coords = components[arrow_id]["coords"]
        assert isinstance(coords, np.ndarray)
        out[coords[:, 0], coords[:, 1]] = BG

    occupied = g != BG
    paint: dict[tuple[int, int], int] = {}
    for pair in pairs:
        comp = components[int(pair["object_id"])]
        color = comp["color"]
        coords = comp["coords"]
        assert isinstance(coords, np.ndarray)
        assert color is not None
        seed = {(int(r), int(c)) for r, c in coords}
        blocked = occupied.copy()
        blocked[coords[:, 0], coords[:, 1]] = False
        grown = _dilate_cells(seed, blocked, int(pair["k"]))
        for cell in grown:
            prev = paint.get(cell)
            assert prev is None or prev == int(color), "expanded objects should not conflict"
            paint[cell] = int(color)

    for (r, c), color in paint.items():
        out[r, c] = color
    return out


SHAPE_LIBRARY = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
]
PAIR_COUNT_OPTIONS = (2, 3, 4)
REGION_MIN = 10
REGION_MAX = 12
REGION_GAP = 2
OUTER_MARGIN = 1
PAIR_ATTEMPTS = 200
SCENE_ATTEMPTS = 400


def _normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _random_shape() -> tuple[tuple[int, int], ...]:
    shape = random.choice(SHAPE_LIBRARY)
    coords = [(int(r), int(c)) for r, c in shape]
    for _ in range(random.randrange(4)):
        coords = [(c, -r) for r, c in coords]
    if random.choice([False, True]):
        coords = [(r, -c) for r, c in coords]
    return _normalize_shape(coords)


def _facing_boundary_cells(
    shape: tuple[tuple[int, int], ...], direction: str
) -> list[tuple[int, int]]:
    if direction == "right":
        by_row: dict[int, int] = {}
        for r, c in shape:
            by_row[r] = c if r not in by_row else min(by_row[r], c)
        return [(r, c) for r, c in by_row.items()]
    if direction == "left":
        by_row = {}
        for r, c in shape:
            by_row[r] = c if r not in by_row else max(by_row[r], c)
        return [(r, c) for r, c in by_row.items()]
    if direction == "down":
        by_col: dict[int, int] = {}
        for r, c in shape:
            by_col[c] = r if c not in by_col else min(by_col[c], r)
        return [(r, c) for c, r in by_col.items()]
    by_col = {}
    for r, c in shape:
        by_col[c] = r if c not in by_col else max(by_col[c], r)
    return [(r, c) for c, r in by_col.items()]


def _arrow_from_anchor(
    anchor: tuple[int, int], direction: str, gap: int, length: int
) -> list[tuple[int, int]]:
    ar, ac = anchor
    if direction == "right":
        tip = (ar, ac - gap - 1)
        return [(tip[0], tip[1] - i) for i in range(length)]
    if direction == "left":
        tip = (ar, ac + gap + 1)
        return [(tip[0], tip[1] + i) for i in range(length)]
    if direction == "down":
        tip = (ar - gap - 1, ac)
        return [(tip[0] - i, tip[1]) for i in range(length)]
    tip = (ar + gap + 1, ac)
    return [(tip[0] + i, tip[1]) for i in range(length)]


def _manhattan_footprint(
    shape: tuple[tuple[int, int], ...], radius: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in shape:
        for dr in range(-radius, radius + 1):
            rem = radius - abs(dr)
            for dc in range(-rem, rem + 1):
                cells.add((r + dr, c + dc))
    return cells


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    out: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _layout_regions() -> tuple[int, int, list[tuple[int, int, int, int]]]:
    n_pairs = random.choice(PAIR_COUNT_OPTIONS)
    if n_pairs == 2:
        rows, cols = random.choice([(1, 2), (2, 1)])
    else:
        rows, cols = 2, 2

    row_heights = [random.randint(REGION_MIN, REGION_MAX) for _ in range(rows)]
    col_widths = [random.randint(REGION_MIN, REGION_MAX) for _ in range(cols)]
    h = 2 * OUTER_MARGIN + sum(row_heights) + REGION_GAP * (rows - 1)
    w = 2 * OUTER_MARGIN + sum(col_widths) + REGION_GAP * (cols - 1)
    assert h <= 30 and w <= 30

    row_starts: list[int] = []
    cur = OUTER_MARGIN
    for height in row_heights:
        row_starts.append(cur)
        cur += height + REGION_GAP

    col_starts: list[int] = []
    cur = OUTER_MARGIN
    for width in col_widths:
        col_starts.append(cur)
        cur += width + REGION_GAP

    slots = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    random.shuffle(slots)
    regions = [
        (row_starts[rr], col_starts[cc], row_heights[rr], col_widths[cc])
        for rr, cc in slots[:n_pairs]
    ]
    return h, w, regions


def _place_pair_in_region(
    region: tuple[int, int, int, int], object_color: int, arrow_palette: list[int]
) -> dict[str, object] | None:
    r0, c0, h, w = region
    for _ in range(PAIR_ATTEMPTS):
        shape = _random_shape()
        k = random.choice([1, 2, 3])
        length = random.randint(max(2, k), 4)
        gap = random.randint(1, 3)
        direction = random.choice(["left", "right", "up", "down"])
        boundary = _facing_boundary_cells(shape, direction)
        if not boundary:
            continue
        anchor = random.choice(boundary)
        arrow_cells_local = _arrow_from_anchor(anchor, direction, gap, length)
        local_union = set(shape) | set(arrow_cells_local) | _manhattan_footprint(shape, k)
        min_r = min(r for r, _ in local_union)
        max_r = max(r for r, _ in local_union)
        min_c = min(c for _, c in local_union)
        max_c = max(c for _, c in local_union)

        off_r_lo = r0 - min_r
        off_r_hi = r0 + h - 1 - max_r
        off_c_lo = c0 - min_c
        off_c_hi = c0 + w - 1 - max_c
        if off_r_lo > off_r_hi or off_c_lo > off_c_hi:
            continue

        off_r = random.randint(off_r_lo, off_r_hi)
        off_c = random.randint(off_c_lo, off_c_hi)
        object_cells = {(off_r + r, off_c + c) for r, c in shape}
        arrow_cells = [(off_r + r, off_c + c) for r, c in arrow_cells_local]

        segment_lengths = _random_composition(length, k)
        segment_colors = random.sample(arrow_palette, k)
        arrow_colors: list[int] = []
        for seg_len, seg_color in zip(segment_lengths, segment_colors):
            arrow_colors.extend([int(seg_color)] * int(seg_len))

        return {
            "object_cells": object_cells,
            "object_color": int(object_color),
            "arrow_cells": arrow_cells,
            "arrow_colors": arrow_colors,
            "k": int(k),
        }
    return None


def _build_scene(
    h: int, w: int, records: list[dict[str, object]]
) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for record in records:
        object_cells = record["object_cells"]
        arrow_cells = record["arrow_cells"]
        object_color = int(record["object_color"])
        arrow_colors = record["arrow_colors"]
        assert isinstance(object_cells, set)
        assert isinstance(arrow_cells, list)
        assert isinstance(arrow_colors, list)

        if any(cell in occupied for cell in object_cells):
            return None
        if any(cell in occupied for cell in arrow_cells):
            return None

        for r, c in object_cells:
            grid[r, c] = object_color
            occupied.add((r, c))
        for (r, c), color in zip(arrow_cells, arrow_colors):
            grid[r, c] = int(color)
            occupied.add((r, c))

    out = grid.copy()
    occupied_mask = grid != BG
    paint: dict[tuple[int, int], int] = {}
    for record in records:
        object_cells = record["object_cells"]
        arrow_cells = record["arrow_cells"]
        object_color = int(record["object_color"])
        k = int(record["k"])
        assert isinstance(object_cells, set)
        assert isinstance(arrow_cells, list)

        for r, c in arrow_cells:
            out[r, c] = BG

        blocked = occupied_mask.copy()
        for r, c in object_cells:
            blocked[r, c] = False
        grown = _dilate_cells(set(object_cells), blocked, k)
        for cell in grown:
            prev = paint.get(cell)
            if prev is not None and prev != object_color:
                return None
            paint[cell] = object_color

    for (r, c), color in paint.items():
        out[r, c] = color
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        h, w, regions = _layout_regions()
        object_colors = random.sample(ALL_COLORS, len(regions))
        arrow_palette = [color for color in ALL_COLORS if color not in object_colors]

        records: list[dict[str, object]] = []
        ok = True
        for region, object_color in zip(regions, object_colors):
            record = _place_pair_in_region(region, int(object_color), arrow_palette)
            if record is None:
                ok = False
                break
            records.append(record)
        if not ok:
            continue

        built = _build_scene(h, w, records)
        if built is None:
            continue
        inp, expected = built

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("failed to generate a valid input")
