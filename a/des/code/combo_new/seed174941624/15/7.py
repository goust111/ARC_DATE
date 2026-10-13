from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
MARKER = 9
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SHAPE_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

GEN_RETRIES = 400
PLACE_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_bg(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]))
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _holes_from_component(coords: list[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    holes: list[set[tuple[int, int]]] = []
    for comp in _components(~mask):
        if any(
            rr == 0 or cc == 0 or rr == mask.shape[0] - 1 or cc == mask.shape[1] - 1
            for rr, cc in comp
        ):
            continue
        holes.append({(r0 + rr, c0 + cc) for rr, cc in comp})
    return holes


def _translate(cells: set[tuple[int, int]] | list[tuple[int, int]], dr: int, dc: int, k: int = 1) -> set[tuple[int, int]]:
    return {(r + dr * k, c + dc * k) for r, c in cells}


def _first_nonoverlap_translation(
    hole: set[tuple[int, int]],
    solid: set[tuple[int, int]],
    direction: str,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    dr, dc = DIR_VECTORS[direction]
    for k in range(1, h + w + 1):
        moved = _translate(hole, dr, dc, k)
        if any(r < 0 or c < 0 or r >= h or c >= w for r, c in moved):
            continue
        if moved & solid:
            continue
        return moved
    raise AssertionError("hole extrusion should fit inside the grid")


def _bar_relation(
    shape_bbox: tuple[int, int, int, int], bar: list[tuple[int, int]]
) -> tuple[tuple[int, int], str] | None:
    r0, c0, r1, c1 = shape_bbox
    bar_rows = sorted({r for r, _ in bar})
    bar_cols = sorted({c for _, c in bar})

    if len(bar_rows) == 1:
        row = bar_rows[0]
        bc0, bc1 = min(bar_cols), max(bar_cols)
        if bc1 - bc0 + 1 != len(bar):
            return None
        overlap = max(0, min(c1, bc1) - max(c0, bc0) + 1)
        if overlap <= 0:
            return None
        center_penalty = abs((bc0 + bc1) - (c0 + c1))
        if row == r0 - 2:
            return ((overlap, -center_penalty), "up")
        if row == r1 + 2:
            return ((overlap, -center_penalty), "down")
        return None

    if len(bar_cols) == 1:
        col = bar_cols[0]
        br0, br1 = min(bar_rows), max(bar_rows)
        if br1 - br0 + 1 != len(bar):
            return None
        overlap = max(0, min(r1, br1) - max(r0, br0) + 1)
        if overlap <= 0:
            return None
        center_penalty = abs((br0 + br1) - (r0 + r1))
        if col == c0 - 2:
            return ((overlap, -center_penalty), "left")
        if col == c1 + 2:
            return ((overlap, -center_penalty), "right")
        return None

    return None


def _match_shapes_to_bars(
    shapes: list[dict[str, object]], bars: list[list[tuple[int, int]]]
) -> dict[int, str]:
    candidates: list[tuple[tuple[int, int], int, int, str]] = []
    for si, shape in enumerate(shapes):
        for bi, bar in enumerate(bars):
            rel = _bar_relation(shape["bbox"], bar)
            if rel is None:
                continue
            score, direction = rel
            candidates.append((score, si, bi, direction))

    candidates.sort(key=lambda item: item[0], reverse=True)
    shape_dirs: dict[int, str] = {}
    used_bars: set[int] = set()
    used_shapes: set[int] = set()
    for _score, si, bi, direction in candidates:
        if si in used_shapes or bi in used_bars:
            continue
        used_shapes.add(si)
        used_bars.add(bi)
        shape_dirs[si] = direction
    return shape_dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrude each shape's interior holes outside the shape in the maroon-indicated direction.

    Input structure:
    - The background is black.
    - The grid contains several 4-connected single-color shapes, each with one or more enclosed black holes.
    - A short maroon horizontal or vertical bar sits one cell away from each shape on exactly one side.

    Rule:
    - Detect every monochrome shape and the enclosed background regions inside its bounding box.
    - Match each shape to its nearby maroon bar; the bar's side relative to the shape gives the extrusion direction.
    - Fill each original hole with the shape color, then translate that hole rigidly along the extrusion direction until it first sits completely outside the solid shape while remaining directly adjacent to the outer boundary.
    - Draw the translated hole with the shape color and erase every maroon bar back to background.

    Key insight:
    - The maroon bar encodes direction only; the real transformation is that interior background cavities disappear and reappear outside as solid copies in the shape's own color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _border_bg(g)

    out = g.copy()
    out[g == MARKER] = bg

    shapes: list[dict[str, object]] = []
    for color in [int(v) for v in np.unique(g) if int(v) not in {bg, MARKER}]:
        for comp in _components(g == color):
            holes = _holes_from_component(comp)
            shapes.append(
                {
                    "color": int(color),
                    "coords": set(comp),
                    "bbox": _bbox(comp),
                    "holes": holes,
                    "solid": set(comp) | set().union(*holes) if holes else set(comp),
                }
            )

    bars = [comp for comp in _components(g == MARKER) if len(comp) in (2, 3)]
    shape_dirs = _match_shapes_to_bars(shapes, bars)

    for si, direction in shape_dirs.items():
        shape = shapes[si]
        color = int(shape["color"])
        solid = set(shape["solid"])
        holes = shape["holes"]
        for hole in holes:
            for r, c in hole:
                out[r, c] = color
        for hole in holes:
            moved = _first_nonoverlap_translation(hole, solid, direction, (h, w))
            for r, c in moved:
                out[r, c] = color

    return out


def _random_polyomino(height: int, width: int) -> set[tuple[int, int]]:
    area_max = height * width
    area = random.randint(1, area_max)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < area and frontier:
        choices = list(frontier)
        weights = []
        for r, c in choices:
            support = sum((r + dr, c + dc) in cells for dr, dc in DIR4)
            weights.append((support + 1) ** 2)
        cell = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)
    return cells


def _carve_holes(mask: np.ndarray, rect: tuple[int, int, int, int], target: int) -> bool:
    r0, c0, r1, c1 = rect
    reserved = np.zeros_like(mask, dtype=bool)
    for _hole_idx in range(target):
        placed = False
        for _ in range(80):
            max_h = min(3, r1 - r0 - 1)
            max_w = min(3, c1 - c0 - 1)
            if max_h < 1 or max_w < 1:
                return False
            hh = random.randint(1, max_h)
            hw = random.randint(1, max_w)
            rr0 = random.randint(r0 + 1, r1 - hh)
            cc0 = random.randint(c0 + 1, c1 - hw)
            rr1 = rr0 + hh - 1
            cc1 = cc0 + hw - 1
            if reserved[rr0 - 1 : rr1 + 2, cc0 - 1 : cc1 + 2].any():
                continue
            hole = {(rr0 + r, cc0 + c) for r, c in _random_polyomino(hh, hw)}
            if any(not mask[r, c] for r, c in hole):
                continue
            for r, c in hole:
                mask[r, c] = False
            reserved[rr0 - 1 : rr1 + 2, cc0 - 1 : cc1 + 2] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _add_protrusion(mask: np.ndarray, rect: tuple[int, int, int, int], side: str) -> None:
    r0, c0, r1, c1 = rect
    if side in {"left", "right"}:
        height = random.randint(2, min(4, r1 - r0 + 1))
        thickness = random.randint(1, 2)
        top = random.randint(r0, r1 - height + 1)
        if side == "left":
            mask[top : top + height, c0 - thickness : c0] = True
        else:
            mask[top : top + height, c1 + 1 : c1 + 1 + thickness] = True
    else:
        width = random.randint(2, min(4, c1 - c0 + 1))
        thickness = random.randint(1, 2)
        left = random.randint(c0, c1 - width + 1)
        if side == "up":
            mask[r0 - thickness : r0, left : left + width] = True
        else:
            mask[r1 + 1 : r1 + 1 + thickness, left : left + width] = True


def _normalize_cells(*groups: set[tuple[int, int]]) -> tuple[list[set[tuple[int, int]]], tuple[int, int]]:
    all_cells = set().union(*groups)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    shifted = [{(r - min_r, c - min_c) for r, c in group} for group in groups]
    return shifted, (-min_r, -min_c)


def _choose_bar_cells(
    shape_cells: set[tuple[int, int]], extruded: set[tuple[int, int]], direction: str
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = _bbox(shape_cells)
    lengths = [2, 3]
    random.shuffle(lengths)

    if direction in {"up", "down"}:
        bar_row = r0 - 2 if direction == "up" else r1 + 2
        for length in lengths:
            if c1 - c0 + 1 < length:
                continue
            starts = list(range(c0, c1 - length + 2))
            random.shuffle(starts)
            for start in starts:
                cells = {(bar_row, c) for c in range(start, start + length)}
                if cells & extruded:
                    continue
                return cells
        return None

    bar_col = c0 - 2 if direction == "left" else c1 + 2
    for length in lengths:
        if r1 - r0 + 1 < length:
            continue
        starts = list(range(r0, r1 - length + 2))
        random.shuffle(starts)
        for start in starts:
            cells = {(r, bar_col) for r in range(start, start + length)}
            if cells & extruded:
                continue
            return cells
    return None


def _make_object(color: int) -> tuple[np.ndarray, np.ndarray] | None:
    direction = random.choice(tuple(DIR_VECTORS))
    base_h = random.randint(7, 11)
    base_w = random.randint(7, 11)
    canvas = np.zeros((base_h + 8, base_w + 8), dtype=bool)
    rect = (3, 3, 3 + base_h - 1, 3 + base_w - 1)
    r0, c0, r1, c1 = rect
    canvas[r0 : r1 + 1, c0 : c1 + 1] = True

    protrusion_sides = [side for side in DIR_VECTORS if side != direction]
    random.shuffle(protrusion_sides)
    for side in protrusion_sides[: random.randint(0, 2)]:
        _add_protrusion(canvas, rect, side)

    hole_target = random.choice([1, 2, 3])
    if not _carve_holes(canvas, rect, hole_target):
        return None

    comps = _components(canvas)
    if len(comps) != 1:
        return None
    shape_cells = set(comps[0])
    holes = _holes_from_component(comps[0])
    if len(holes) != hole_target:
        return None

    solid = set(shape_cells) | set().union(*holes)
    extruded_groups = []
    for hole in holes:
        dr, dc = DIR_VECTORS[direction]
        for k in range(1, 60):
            moved = _translate(hole, dr, dc, k)
            if moved & solid:
                continue
            extruded_groups.append(moved)
            break
        else:
            return None

    extruded = set().union(*extruded_groups)
    if extruded & solid:
        return None
    bar_cells = _choose_bar_cells(shape_cells, extruded, direction)
    if bar_cells is None:
        return None

    groups, _shift = _normalize_cells(shape_cells, solid, extruded, bar_cells)
    shape_shifted, solid_shifted, extruded_shifted, bar_shifted = groups

    all_cells = solid_shifted | extruded_shifted | bar_shifted
    h = max(r for r, _ in all_cells) + 1
    w = max(c for _, c in all_cells) + 1
    if h > 18 or w > 18:
        return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for r, c in shape_shifted:
        inp[r, c] = color
    for r, c in bar_shifted:
        inp[r, c] = MARKER
    for r, c in solid_shifted | extruded_shifted:
        out[r, c] = color

    if np.any((inp == color) & (inp == MARKER)):
        return None
    if np.array_equal(inp, out):
        return None
    return inp, out


def _try_place(locals_in: list[np.ndarray], locals_out: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    min_h = max(arr.shape[0] for arr in locals_in) + 2
    min_w = max(arr.shape[1] for arr in locals_in) + 2
    reserve = sum((arr.shape[0] + 2) * (arr.shape[1] + 2) for arr in locals_in)

    for _ in range(60):
        h = random.randint(max(14, min_h), 30)
        w = random.randint(max(14, min_w), 30)
        if h * w < reserve + 20:
            continue

        occupied = np.zeros((h, w), dtype=bool)
        placements: list[tuple[int, int]] = [(-1, -1)] * len(locals_in)
        order = sorted(range(len(locals_in)), key=lambda i: locals_in[i].size, reverse=True)
        ok = True
        for idx in order:
            arr = locals_in[idx]
            ah, aw = arr.shape
            positions = [
                (r, c)
                for r in range(1, h - ah)
                for c in range(1, w - aw)
                if not occupied[r - 1 : r + ah + 1, c - 1 : c + aw + 1].any()
            ]
            if not positions:
                ok = False
                break
            r, c = random.choice(positions)
            placements[idx] = (r, c)
            occupied[r - 1 : r + ah + 1, c - 1 : c + aw + 1] = True
        if not ok:
            continue

        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        for idx, (r0, c0) in enumerate(placements):
            arr_in = locals_in[idx]
            arr_out = locals_out[idx]
            for rr, cc in np.argwhere(arr_in != BG):
                inp[r0 + rr, c0 + cc] = int(arr_in[rr, cc])
            for rr, cc in np.argwhere(arr_out != BG):
                out[r0 + rr, c0 + cc] = int(arr_out[rr, cc])
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        obj_count = random.choice([1, 2, 3])
        colors = random.sample(SHAPE_COLORS, obj_count)
        locals_in: list[np.ndarray] = []
        locals_out: list[np.ndarray] = []
        ok = True
        for color in colors:
            obj = _make_object(color)
            if obj is None:
                ok = False
                break
            arr_in, arr_out = obj
            locals_in.append(arr_in)
            locals_out.append(arr_out)
        if not ok:
            continue

        placed = _try_place(locals_in, locals_out)
        if placed is None:
            continue
        inp, expected = placed

        bars = _components(inp == MARKER)
        if len(bars) != obj_count:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp

    raise AssertionError("failed to generate a valid puzzle instance")
