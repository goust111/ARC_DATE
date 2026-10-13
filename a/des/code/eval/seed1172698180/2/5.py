from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]


BG = 0
OBJECT_COLORS = (1, 2, 3, 4, 5, 6)
PORTAL_COLORS = (7, 8, 9)
ORIENTATIONS = ("horizontal", "vertical")

GRID_MIN = 14
GRID_MAX = 18
OBJECT_SIZE_RANGE = (3, 8)
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120
PORTAL_OPTION_LIMIT = 48


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    ]


def _connected_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _portal_pairs(grid: np.ndarray, bg: int) -> dict[int, tuple[Coord, Coord]]:
    h, w = grid.shape
    out: dict[int, tuple[Coord, Coord]] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(coords) != 2:
            continue
        (r0, c0), (r1, c1) = coords
        if r0 != r1 and c0 != c1:
            continue
        isolated = True
        for r, c in coords:
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords and int(grid[nr, nc]) != bg:
                    isolated = False
                    break
            if not isolated:
                break
        if isolated:
            out[color] = tuple(sorted(coords))
    assert out, "expected at least one portal pair"
    return out


def _object_components(grid: np.ndarray, bg: int, portals: dict[int, tuple[Coord, Coord]]) -> list[dict]:
    portal_cells = {pt for pair in portals.values() for pt in pair}
    mask = (grid != int(bg)).copy()
    for r, c in portal_cells:
        mask[r, c] = False
    comps = _connected_components(mask)
    objects = []
    for comp in comps:
        coords = sorted(comp)
        objects.append(
            {
                "coords": coords,
                "coord_set": set(coords),
                "size": len(coords),
            }
        )
    return objects


def _teleport_copy(
    coords: list[Coord],
    source_anchor: Coord,
    target_anchor: Coord,
    orientation: str,
) -> list[Coord]:
    sr, sc = source_anchor
    tr, tc = target_anchor
    out: list[Coord] = []
    if orientation == "horizontal":
        for r, c in coords:
            out.append((tr - (r - sr), tc + (c - sc)))
    else:
        assert orientation == "vertical"
        for r, c in coords:
            out.append((tr + (r - sr), tc - (c - sc)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat isolated same-colored pixel pairs as portals and mirror-copy any object
    crossed by the portal segment to the far side of the opposite endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)
    portals = _portal_pairs(g, bg)
    objects = _object_components(g, bg, portals)

    out = g.copy()
    for portal_color, endpoints in portals.items():
        (r0, c0), (r1, c1) = endpoints
        if r0 == r1:
            orientation = "horizontal"
            line_row = r0
            lo, hi = sorted((c0, c1))
            for obj in objects:
                on_line = [(r, c) for r, c in obj["coords"] if r == line_row and lo <= c <= hi]
                if not on_line:
                    continue
                assert len(on_line) == 1, "each associated object must hit a portal line once"
                anchor = on_line[0]
                d0 = abs(anchor[1] - c0)
                d1 = abs(anchor[1] - c1)
                assert d0 != d1, "portal anchor must have a unique nearer endpoint"
                if d0 < d1:
                    near, far, d = (r0, c0), (r1, c1), d0
                else:
                    near, far, d = (r1, c1), (r0, c0), d1
                step = 1 if far[1] > near[1] else -1
                target_anchor = (anchor[0], far[1] + step * d)
                for nr, nc in _teleport_copy(obj["coords"], anchor, target_anchor, orientation):
                    assert 0 <= nr < h and 0 <= nc < w
                    out[nr, nc] = portal_color
            continue

        assert c0 == c1, "portal endpoints must align horizontally or vertically"
        orientation = "vertical"
        line_col = c0
        lo, hi = sorted((r0, r1))
        for obj in objects:
            on_line = [(r, c) for r, c in obj["coords"] if c == line_col and lo <= r <= hi]
            if not on_line:
                continue
            assert len(on_line) == 1, "each associated object must hit a portal line once"
            anchor = on_line[0]
            d0 = abs(anchor[0] - r0)
            d1 = abs(anchor[0] - r1)
            assert d0 != d1, "portal anchor must have a unique nearer endpoint"
            if d0 < d1:
                near, far, d = (r0, c0), (r1, c1), d0
            else:
                near, far, d = (r1, c1), (r0, c0), d1
            step = 1 if far[0] > near[0] else -1
            target_anchor = (far[0] + step * d, anchor[1])
            for nr, nc in _teleport_copy(obj["coords"], anchor, target_anchor, orientation):
                assert 0 <= nr < h and 0 <= nc < w
                out[nr, nc] = portal_color

    return out


def _shape_cells() -> list[Coord]:
    for _ in range(SHAPE_ATTEMPTS):
        target = random.randint(*OBJECT_SIZE_RANGE)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier: set[Coord] = set()
            for r, c in cells:
                for nr, nc in _neighbors8(r, c):
                    if abs(nr) > 3 or abs(nc) > 3:
                        continue
                    if nr == 0 or nc == 0:
                        continue
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            options = list(frontier)
            options.sort(key=lambda pt: (abs(pt[0]) + abs(pt[1]), abs(pt[0]), abs(pt[1]), pt))
            cells.add(random.choice(options[: min(len(options), 10)]))
        if len(cells) != target:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) - min(rows) > 5 or max(cols) - min(cols) > 5:
            continue
        return sorted(cells)
    raise RuntimeError("failed to sample a shape")


def _colorize_shape(coords: list[Coord]) -> list[tuple[int, int, int]]:
    n_colors = min(len(coords), random.choice((2, 2, 3)))
    palette = random.sample(OBJECT_COLORS, n_colors)
    shuffled = coords[:]
    random.shuffle(shuffled)
    assigned: list[tuple[int, int, int]] = []
    for idx, (r, c) in enumerate(shuffled):
        color = palette[idx] if idx < n_colors else random.choice(palette)
        assigned.append((r, c, color))
    assigned.sort(key=lambda x: (x[0], x[1]))
    return assigned


def _fits_with_gap(cells: list[Coord], occupied: set[Coord], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - 2, r + 3):
            for cc in range(c - 2, c + 3):
                if (rr, cc) in occupied:
                    return False
    return True


def _place_objects(shapes: list[list[tuple[int, int, int]]], h: int, w: int) -> list[dict] | None:
    order = sorted(range(len(shapes)), key=lambda idx: len(shapes[idx]), reverse=True)
    occupied: set[Coord] = set()
    placed: list[dict | None] = [None] * len(shapes)
    for idx in order:
        shape = shapes[idx]
        rows = [r for r, _c, _color in shape]
        cols = [c for _r, c, _color in shape]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        row_lo = max(3, -min_r)
        row_hi = min(h - 4, h - 1 - max_r)
        col_lo = max(3, -min_c)
        col_hi = min(w - 4, w - 1 - max_c)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates = [
            (ar, ac)
            for ar in range(row_lo, row_hi + 1)
            for ac in range(col_lo, col_hi + 1)
            if _fits_with_gap([(ar + r, ac + c) for r, c, _color in shape], occupied, h, w)
        ]
        if not candidates:
            return None
        anchor_r, anchor_c = random.choice(candidates)
        abs_cells = [(anchor_r + r, anchor_c + c, color) for r, c, color in shape]
        placed[idx] = {
            "anchor": (anchor_r, anchor_c),
            "cells": abs_cells,
            "coords": [(r, c) for r, c, _color in abs_cells],
        }
        occupied.update((r, c) for r, c, _color in abs_cells)

    assert all(obj is not None for obj in placed)
    return [obj for obj in placed if obj is not None]


def _line_hits(cell: Coord, start: Coord, end: Coord) -> bool:
    r, c = cell
    if start[0] == end[0] == r:
        lo, hi = sorted((start[1], end[1]))
        return lo <= c <= hi
    if start[1] == end[1] == c:
        lo, hi = sorted((start[0], end[0]))
        return lo <= r <= hi
    return False


def _endpoint_isolated(pt: Coord, object_cells: set[Coord], portal_cells: set[Coord], h: int, w: int) -> bool:
    r, c = pt
    if not (0 <= r < h and 0 <= c < w):
        return False
    if pt in object_cells or pt in portal_cells:
        return False
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < h and 0 <= nc < w and ((nr, nc) in object_cells or (nr, nc) in portal_cells):
            return False
    return True


def _portal_options(
    h: int,
    w: int,
    objects: list[dict],
    assoc: dict,
    object_cells: set[Coord],
    portal_cells: set[Coord],
    reserved_output: set[Coord],
) -> list[dict]:
    obj = objects[assoc["object_index"]]
    anchor = obj["anchor"]
    orientation = assoc["orientation"]
    coords = obj["coords"]
    coord_set = set(coords)
    options: list[dict] = []

    for near_side in (-1, 1):
        for near_dist in range(2, 5):
            for far_dist in range(near_dist + 1, 7):
                if orientation == "horizontal":
                    left = anchor[1] - far_dist if near_side == 1 else anchor[1] - near_dist
                    right = anchor[1] + near_dist if near_side == 1 else anchor[1] + far_dist
                    if not (0 <= left < right < w):
                        continue
                    if near_side == 1:
                        near_pt = (anchor[0], right)
                        far_pt = (anchor[0], left)
                    else:
                        near_pt = (anchor[0], left)
                        far_pt = (anchor[0], right)
                    target_anchor = (anchor[0], far_pt[1] - near_side * near_dist)
                else:
                    top = anchor[0] - far_dist if near_side == 1 else anchor[0] - near_dist
                    bottom = anchor[0] + near_dist if near_side == 1 else anchor[0] + far_dist
                    if not (0 <= top < bottom < h):
                        continue
                    if near_side == 1:
                        near_pt = (bottom, anchor[1])
                        far_pt = (top, anchor[1])
                    else:
                        near_pt = (top, anchor[1])
                        far_pt = (bottom, anchor[1])
                    target_anchor = (far_pt[0] - near_side * near_dist, anchor[1])

                if not _endpoint_isolated(near_pt, object_cells, portal_cells, h, w):
                    continue
                if not _endpoint_isolated(far_pt, object_cells, portal_cells, h, w):
                    continue

                blocked = False
                for cell in object_cells:
                    if _line_hits(cell, near_pt, far_pt):
                        if cell not in coord_set or cell != anchor:
                            blocked = True
                            break
                if blocked:
                    continue
                if any(_line_hits(cell, near_pt, far_pt) for cell in portal_cells):
                    continue

                copy_cells = _teleport_copy(coords, anchor, target_anchor, orientation)
                copy_set = set(copy_cells)
                if len(copy_set) != len(coords):
                    continue
                if any(not (0 <= r < h and 0 <= c < w) for r, c in copy_set):
                    continue
                if copy_set & object_cells:
                    continue
                if copy_set & portal_cells:
                    continue
                if copy_set & reserved_output:
                    continue

                options.append(
                    {
                        "endpoints": tuple(sorted((near_pt, far_pt))),
                        "copy_cells": copy_set,
                    }
                )
    random.shuffle(options)
    return options[:PORTAL_OPTION_LIMIT]


def _assign_portals(
    h: int,
    w: int,
    objects: list[dict],
    associations: list[dict],
) -> list[dict] | None:
    object_cells = {cell for obj in objects for cell in obj["coords"]}

    def rec(
        idx: int,
        portal_cells: set[Coord],
        reserved_output: set[Coord],
        chosen: list[dict],
    ) -> list[dict] | None:
        if idx == len(associations):
            return chosen

        assoc = associations[idx]
        options = _portal_options(h, w, objects, assoc, object_cells, portal_cells, reserved_output)
        if not options:
            return None
        for option in options:
            result = rec(
                idx + 1,
                portal_cells | set(option["endpoints"]),
                reserved_output | option["copy_cells"],
                chosen + [{**assoc, **option}],
            )
            if result is not None:
                return result
        return None

    return rec(0, set(), set(), [])


def _build_plan() -> tuple[list[list[tuple[int, int, int]]], list[dict]]:
    use_dual = random.random() < 0.5
    if use_dual:
        n_objects = 2
        extra_orientation = random.choice(ORIENTATIONS)
        associations = [
            {"object_index": 0, "orientation": "horizontal", "color": PORTAL_COLORS[0]},
            {"object_index": 0, "orientation": "vertical", "color": PORTAL_COLORS[1]},
            {"object_index": 1, "orientation": extra_orientation, "color": PORTAL_COLORS[2]},
        ]
    else:
        n_objects = random.choice((2, 3))
        orientations = ["horizontal", "vertical"]
        if n_objects == 3:
            orientations.append(random.choice(ORIENTATIONS))
        random.shuffle(orientations)
        associations = [
            {"object_index": idx, "orientation": orientations[idx], "color": PORTAL_COLORS[idx]}
            for idx in range(n_objects)
        ]

    shapes = [_colorize_shape(_shape_cells()) for _ in range(n_objects)]
    random.shuffle(associations)
    return shapes, associations


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shapes, associations = _build_plan()
    objects = _place_objects(shapes, h, w)
    if objects is None:
        return None

    assignments = _assign_portals(h, w, objects, associations)
    if assignments is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c, color in obj["cells"]:
            grid[r, c] = color
    for assignment in assignments:
        color = assignment["color"]
        for r, c in assignment["endpoints"]:
            grid[r, c] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[list(map(int, values)).index(BG)]) if BG in values else 0
    if bg_count <= (h * w) // 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid portal-copy puzzle")
