import numpy as np


BG = 0
YELLOW = 4
RECT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_SHAPES = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)}),
    "right": frozenset({(1, 0), (1, 1), (0, 2), (1, 2), (2, 2)}),
}
MAX_GENERATION_ATTEMPTS = 500
MAX_PLACEMENT_ATTEMPTS = 250

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _normalize(cells: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _classify_arrow(cells: list[tuple[int, int]]) -> str | None:
    if len(cells) != 5:
        return None
    shape = _normalize(cells)
    for direction, expected in ARROW_SHAPES.items():
        if shape == expected:
            return direction
    return None


def _front_edge(cells: list[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    if direction == "up":
        edge = min(r for r, _c in cells)
        return [(r, c) for r, c in cells if r == edge]
    if direction == "down":
        edge = max(r for r, _c in cells)
        return [(r, c) for r, c in cells if r == edge]
    if direction == "left":
        edge = min(c for _r, c in cells)
        return [(r, c) for r, c in cells if c == edge]
    edge = max(c for _r, c in cells)
    return [(r, c) for r, c in cells if c == edge]


def _component_from_cell(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    color = int(grid[start])
    stack = [start]
    seen = {start}
    comp = []
    while stack:
        r, c = stack.pop()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return comp


def _target_rectangle(
    grid: np.ndarray, arrow_cells: list[tuple[int, int]], direction: str
) -> tuple[int, list[tuple[int, int]], tuple[int, int, int, int]] | None:
    h, w = grid.shape
    dr, dc = DIRS[direction]
    hits = []
    for r, c in _front_edge(arrow_cells, direction):
        rr = r + dr
        cc = c + dc
        dist = 1
        while 0 <= rr < h and 0 <= cc < w:
            value = int(grid[rr, cc])
            if value not in (BG, YELLOW):
                cells = _component_from_cell(grid, (rr, cc))
                hits.append((dist, rr, cc, value, cells))
                break
            rr += dr
            cc += dc
            dist += 1
    if not hits:
        return None
    hits.sort(key=lambda item: (item[0], item[1], item[2]))
    _dist, _rr, _cc, color, cells = hits[0]
    return color, cells, _bbox(cells)


def _stamp_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _rect_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _shift_bbox(
    bbox: tuple[int, int, int, int], direction: str, amount: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    dr, dc = DIRS[direction]
    return r0 - dr * amount, c0 - dc * amount, r1 - dr * amount, c1 - dc * amount


def _arrow_cells(direction: str, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in ARROW_SHAPES[direction]]


def _in_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_occupied(occ: np.ndarray, cells: list[tuple[int, int]], margin: int = 1) -> bool:
    h, w = occ.shape
    for r, c in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and occ[rr, cc]:
                    return True
    return False


def _reserve(occ: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occ[r, c] = True


def _build_expected(h: int, w: int, objects: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for obj in objects:
        _stamp_rect(out, obj["out_bbox"], obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each selected rectangle opposite its yellow arrow by the rectangle's own axis length.

    Input structure:
    - The background is black.
    - Several solid monochrome rectangles are separated from one another.
    - Near each rectangle sits a yellow 5-pixel arrow marker whose 3x3 shape points along a horizontal or vertical ray toward that rectangle.

    Rule:
    - Detect every yellow arrow's direction from its 3x3 arrow shape.
    - From the arrow's front edge, ray-cast in that direction and take the first non-black, non-yellow object hit as the target rectangle.
    - Measure that rectangle along the arrow axis: width for left/right arrows and height for up/down arrows.
    - Clear all yellow arrows, erase each targeted rectangle from its old position, and redraw it shifted in the opposite direction by that measured length.

    Key insight:
    - The arrow tells you which rectangle to select, not the direction of motion; the rectangle moves the other way, and the travel distance comes from the rectangle's own size.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    moves = []
    seen_targets: set[frozenset[tuple[int, int]]] = set()
    arrow_components = _components(g == YELLOW)
    arrow_components.sort(key=lambda comp: _bbox(comp))

    for comp in arrow_components:
        direction = _classify_arrow(comp)
        if direction is None:
            continue
        target = _target_rectangle(g, comp, direction)
        if target is None:
            continue
        color, cells, bbox = target
        key = frozenset(cells)
        if key in seen_targets:
            continue
        seen_targets.add(key)
        height, width = _bbox_hw(bbox)
        amount = width if direction in ("left", "right") else height
        moves.append((color, cells, _shift_bbox(bbox, direction, amount)))

    out[out == YELLOW] = BG
    for _color, cells, _dst_bbox in moves:
        for r, c in cells:
            out[r, c] = BG
    for color, _cells, dst_bbox in moves:
        _stamp_rect(out, dst_bbox, color)

    return out


def _sample_candidate(
    rng: np.random.Generator, h: int, w: int, direction: str, rect_h: int, rect_w: int, gap: int
) -> dict | None:
    if direction == "right":
        c_lo = max(rect_w, gap + 3)
        c_hi = w - rect_w
        if c_lo > c_hi:
            return None
        r0 = int(rng.integers(0, h - rect_h + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        row_lo = max(r0, 1)
        row_hi = min(r0 + rect_h - 1, h - 2)
        if row_lo > row_hi:
            return None
        center_row = int(rng.integers(row_lo, row_hi + 1))
        arrow_top = center_row - 1
        arrow_left = c0 - gap - 3
    elif direction == "left":
        c_lo = 0
        c_hi = min(w - 2 * rect_w, w - rect_w - gap - 3)
        if c_lo > c_hi:
            return None
        r0 = int(rng.integers(0, h - rect_h + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        row_lo = max(r0, 1)
        row_hi = min(r0 + rect_h - 1, h - 2)
        if row_lo > row_hi:
            return None
        center_row = int(rng.integers(row_lo, row_hi + 1))
        arrow_top = center_row - 1
        arrow_left = c0 + rect_w + gap
    elif direction == "down":
        r_lo = max(rect_h, gap + 3)
        r_hi = h - rect_h
        if r_lo > r_hi:
            return None
        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(0, w - rect_w + 1))
        col_lo = max(c0, 1)
        col_hi = min(c0 + rect_w - 1, w - 2)
        if col_lo > col_hi:
            return None
        center_col = int(rng.integers(col_lo, col_hi + 1))
        arrow_top = r0 - gap - 3
        arrow_left = center_col - 1
    else:
        r_lo = 0
        r_hi = min(h - 2 * rect_h, h - rect_h - gap - 3)
        if r_lo > r_hi:
            return None
        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(0, w - rect_w + 1))
        col_lo = max(c0, 1)
        col_hi = min(c0 + rect_w - 1, w - 2)
        if col_lo > col_hi:
            return None
        center_col = int(rng.integers(col_lo, col_hi + 1))
        arrow_top = r0 + rect_h + gap
        arrow_left = center_col - 1

    rect_bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    length = rect_w if direction in ("left", "right") else rect_h
    out_bbox = _shift_bbox(rect_bbox, direction, length)
    arrow_cells = _arrow_cells(direction, arrow_top, arrow_left)
    if not _in_bounds(arrow_cells, h, w):
        return None
    return {
        "direction": direction,
        "rect_bbox": rect_bbox,
        "out_bbox": out_bbox,
        "arrow_cells": arrow_cells,
    }


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    h = int(rng.integers(20, 26))
    w = int(rng.integers(20, 26))
    count = int(rng.integers(2, 5))

    while True:
        directions = rng.choice(tuple(DIRS), size=count, replace=True).tolist()
        has_horizontal = any(d in ("left", "right") for d in directions)
        has_vertical = any(d in ("up", "down") for d in directions)
        if has_horizontal and has_vertical:
            break

    colors = rng.choice(RECT_COLORS, size=count, replace=False).tolist()
    specs = []
    for color, direction in zip(colors, directions):
        specs.append(
            {
                "color": int(color),
                "direction": direction,
                "rect_h": int(rng.integers(2, 6)),
                "rect_w": int(rng.integers(2, 6)),
                "gap": int(rng.integers(1, 5)),
            }
        )
    specs.sort(key=lambda spec: spec["rect_h"] * spec["rect_w"], reverse=True)

    input_occ = np.zeros((h, w), dtype=bool)
    output_occ = np.zeros((h, w), dtype=bool)
    objects = []

    for spec in specs:
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            candidate = _sample_candidate(
                rng,
                h,
                w,
                spec["direction"],
                spec["rect_h"],
                spec["rect_w"],
                spec["gap"],
            )
            if candidate is None:
                continue

            rect_cells = _rect_cells(candidate["rect_bbox"])
            out_cells = _rect_cells(candidate["out_bbox"])
            all_input_cells = rect_cells + candidate["arrow_cells"]

            if _touches_occupied(input_occ, all_input_cells, margin=1):
                continue
            if _touches_occupied(output_occ, out_cells, margin=1):
                continue

            obj = {
                "color": spec["color"],
                "direction": spec["direction"],
                "rect_bbox": candidate["rect_bbox"],
                "out_bbox": candidate["out_bbox"],
                "arrow_cells": candidate["arrow_cells"],
            }
            objects.append(obj)
            _reserve(input_occ, all_input_cells)
            _reserve(output_occ, out_cells)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        _stamp_rect(grid, obj["rect_bbox"], obj["color"])
        for r, c in obj["arrow_cells"]:
            grid[r, c] = YELLOW

    expected = _build_expected(h, w, objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
