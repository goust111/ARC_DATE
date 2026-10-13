from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG_COLOR = 0
SOURCE_COLOR = 5
PORTAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("left", "right", "up", "down")

MAX_GENERATION_ATTEMPTS = 300
TRIPLET_ATTEMPTS = 200
SHAPE_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_source_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != SOURCE_COLOR:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != SOURCE_COLOR:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _portal_pairs(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere((grid != BG_COLOR) & (grid != SOURCE_COLOR)):
        by_color[int(grid[r, c])].append((int(r), int(c)))

    assert by_color, "expected at least one portal pair"
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color, coords in by_color.items():
        assert len(coords) == 2, "each portal color must mark exactly two pixels"
        a, b = sorted(coords)
        pairs[int(color)] = (a, b)
    return pairs


def _visible_object_index(
    grid: np.ndarray,
    marker: tuple[int, int],
    coord_to_object: dict[tuple[int, int], int],
) -> int | None:
    h, w = map(int, grid.shape)
    seen: set[int] = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = marker
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            value = int(grid[r, c])
            if value == BG_COLOR:
                continue
            if value == SOURCE_COLOR:
                seen.add(coord_to_object[(r, c)])
            break

    assert len(seen) <= 1, "a portal marker may face at most one source object"
    return next(iter(seen), None) if seen else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each grey source object through its aligned portal pair.

    Input structure
    - Background is black.
    - Source objects are 4-connected grey components.
    - Each portal pair is two singleton pixels of the same non-black, non-grey color.
    - Every source object has exactly one portal marker with unobstructed row/column sight.

    Rule
    - Find the aligned marker A for each source object.
    - Find the paired same-color marker B.
    - Translate the entire source object by the vector from A to B.
    - Recolor the translated copy with the portal color.
    - Erase the original grey object and keep the portal markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = map(int, g.shape)

    objects = _find_source_components(g)
    assert objects, "expected at least one grey source object"
    pairs = _portal_pairs(g)

    coord_to_object: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(objects):
        for cell in comp:
            coord_to_object[cell] = idx

    object_to_marker: dict[int, tuple[int, tuple[int, int]]] = {}
    for color, markers in pairs.items():
        for marker in markers:
            obj_idx = _visible_object_index(g, marker, coord_to_object)
            if obj_idx is None:
                continue
            assert obj_idx not in object_to_marker, "each source object must face exactly one portal marker"
            object_to_marker[obj_idx] = (int(color), marker)

    assert len(object_to_marker) == len(objects), "every source object must match one portal marker"

    out = g.copy()
    out[g == SOURCE_COLOR] = BG_COLOR

    for obj_idx, comp in enumerate(objects):
        color, marker_a = object_to_marker[obj_idx]
        marker1, marker2 = pairs[color]
        marker_b = marker2 if marker_a == marker1 else marker1
        dr = int(marker_b[0] - marker_a[0])
        dc = int(marker_b[1] - marker_a[1])

        for r, c in comp:
            nr = int(r + dr)
            nc = int(c + dc)
            assert 0 <= nr < h and 0 <= nc < w, "teleported object must stay in bounds"
            assert int(out[nr, nc]) == BG_COLOR, "teleported objects may not overlap portals or each other"
            out[nr, nc] = int(color)

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(SHAPE_ATTEMPTS):
        target_area = random.randint(1, 6)
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in sorted(cells):
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            random.shuffle(frontier)
            placed = False
            for nr, nc in frontier:
                candidate = set(cells)
                candidate.add((nr, nc))
                normalized = _normalize_shape(candidate)
                height = max(r for r, _ in normalized) + 1
                width = max(c for _, c in normalized) + 1
                if height <= 4 and width <= 4:
                    cells = set(normalized)
                    placed = True
                    break
            if not placed:
                break
        if len(cells) == target_area:
            return _normalize_shape(cells)
    raise RuntimeError("failed to build a source shape")


def _shape_cells(shape: tuple[tuple[int, int], ...], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _boundary_candidates(shape: tuple[tuple[int, int], ...]) -> dict[str, list[tuple[int, int]]]:
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in shape:
        by_row[int(r)].append(int(c))
        by_col[int(c)].append(int(r))

    left = sorted((r, min(cols)) for r, cols in by_row.items())
    right = sorted((r, max(cols)) for r, cols in by_row.items())
    up = sorted((min(rows), c) for c, rows in by_col.items())
    down = sorted((max(rows), c) for c, rows in by_col.items())
    return {"left": left, "right": right, "up": up, "down": down}


def _marker_position(
    top: int,
    left: int,
    anchor: tuple[int, int],
    side: str,
    gap: int,
) -> tuple[int, int]:
    ar, ac = anchor
    if side == "left":
        return top + ar, left + ac - gap
    if side == "right":
        return top + ar, left + ac + gap
    if side == "up":
        return top + ar - gap, left + ac
    assert side == "down"
    return top + ar + gap, left + ac


def _sample_top_left(
    height: int,
    width: int,
    shape: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    side: str,
    gap: int,
) -> tuple[int, int] | None:
    shape_h, shape_w = _shape_dims(shape)
    ar, ac = anchor

    top_min = 0
    top_max = height - shape_h
    left_min = 0
    left_max = width - shape_w

    if side == "left":
        left_min = max(left_min, gap - ac)
    elif side == "right":
        left_max = min(left_max, width - 1 - ac - gap)
    elif side == "up":
        top_min = max(top_min, gap - ar)
    else:
        top_max = min(top_max, height - 1 - ar - gap)

    if top_min > top_max or left_min > left_max:
        return None
    return random.randint(top_min, top_max), random.randint(left_min, left_max)


def _corridor_cells(marker: tuple[int, int], anchor_cell: tuple[int, int]) -> set[tuple[int, int]]:
    mr, mc = marker
    ar, ac = anchor_cell
    corridor: set[tuple[int, int]] = set()
    if mr == ar:
        step = 1 if ac > mc else -1
        for c in range(mc + step, ac, step):
            corridor.add((mr, c))
    else:
        step = 1 if ar > mr else -1
        for r in range(mr + step, ar, step):
            corridor.add((r, mc))
    return corridor


def _render_scene(
    height: int,
    width: int,
    triplets: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG_COLOR, dtype=int)
    out = np.full((height, width), BG_COLOR, dtype=int)

    for triplet in triplets:
        color = int(triplet["color"])
        for r, c in triplet["source_cells"]:
            inp[r, c] = SOURCE_COLOR
        for marker in (triplet["marker_a"], triplet["marker_b"]):
            mr, mc = marker
            inp[mr, mc] = color
            out[mr, mc] = color
        for r, c in triplet["output_cells"]:
            out[r, c] = color

    return inp, out


def _occupied_cells(triplets: list[dict[str, object]]) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for triplet in triplets:
        occupied.update(triplet["source_cells"])
        occupied.update(triplet["output_cells"])
        occupied.add(triplet["marker_a"])
        occupied.add(triplet["marker_b"])
        occupied.update(triplet["corridor"])
    return occupied


def _try_add_triplet(
    height: int,
    width: int,
    triplets: list[dict[str, object]],
    color: int,
) -> dict[str, object] | None:
    reserved = _occupied_cells(triplets)

    for _ in range(TRIPLET_ATTEMPTS):
        shape = _random_shape()
        side = random.choice(SIDES)
        anchor = random.choice(_boundary_candidates(shape)[side])
        gap = random.randint(1, 4)

        src_top_left = _sample_top_left(height, width, shape, anchor, side, gap)
        dst_top_left = _sample_top_left(height, width, shape, anchor, side, gap)
        if src_top_left is None or dst_top_left is None:
            continue
        if src_top_left == dst_top_left:
            continue

        src_top, src_left = src_top_left
        dst_top, dst_left = dst_top_left
        source_cells = _shape_cells(shape, src_top, src_left)
        output_cells = _shape_cells(shape, dst_top, dst_left)
        marker_a = _marker_position(src_top, src_left, anchor, side, gap)
        marker_b = _marker_position(dst_top, dst_left, anchor, side, gap)
        anchor_cell = (src_top + anchor[0], src_left + anchor[1])
        corridor = _corridor_cells(marker_a, anchor_cell)

        new_reserved = set(source_cells) | set(output_cells) | {marker_a, marker_b} | corridor
        if len(new_reserved) != len(source_cells) + len(output_cells) + 2 + len(corridor):
            continue
        if new_reserved & reserved:
            continue

        triplet = {
            "color": int(color),
            "shape": shape,
            "side": side,
            "gap": int(gap),
            "anchor": anchor,
            "source_cells": source_cells,
            "output_cells": output_cells,
            "marker_a": marker_a,
            "marker_b": marker_b,
            "corridor": corridor,
        }
        trial = triplets + [triplet]
        inp, expected = _render_scene(height, width, trial)
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return triplet

    return None


def _generate_attempt() -> np.ndarray | None:
    n_pairs = random.randint(2, 4)
    height = random.randint(10 + 2 * n_pairs, 16 + 2 * n_pairs)
    width = random.randint(10 + 2 * n_pairs, 16 + 2 * n_pairs)
    colors = random.sample(PORTAL_COLORS, n_pairs)

    triplets: list[dict[str, object]] = []
    for color in colors:
        triplet = _try_add_triplet(height, width, triplets, color)
        if triplet is None:
            return None
        triplets.append(triplet)

    inp, expected = _render_scene(height, width, triplets)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-transport input")
