import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
SEPARATOR = 5
DRAW_COLORS = (1, 4, 6, 7, 8, 9)
MAX_GENERATION_TRIES = 400
MAX_LAYOUT_TRIES = 200
MAX_PIECE_TRIES = 120
INNER_MIN_EXTENT = 4
INNER_MAX_COORD = 6


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    nbrs = _neighbors4 if connectivity == 4 else _neighbors8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in nbrs(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _enclosed_cells(walls: np.ndarray) -> np.ndarray:
    h, w = walls.shape
    seen = np.zeros_like(walls, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not walls[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not walls[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return (~walls) & (~seen)


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        value = int(grid[0, c])
        if value == bg:
            continue
        if not np.all(grid[:, c] == value):
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one separator column"
    return candidates[0]


def _find_container(
    grid: np.ndarray, sep_col: int, bg: int
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, tuple[int, int]]:
    h, w = grid.shape
    left_cols = np.arange(w)[None, :] < sep_col
    origin_pts = np.argwhere((grid == RED) & left_cols)
    assert len(origin_pts) == 1, "expected one origin marker on the left"
    origin_r, origin_c = map(int, origin_pts[0])

    outline_mask = (grid != bg) & (grid != RED) & left_cols
    candidates = []
    for comp in _components(outline_mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(comp)
        wall = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            wall[r - r0, c - c0] = True
        inside = _enclosed_cells(wall)
        local_origin = (origin_r - r0, origin_c - c0)
        if 0 <= local_origin[0] < inside.shape[0] and 0 <= local_origin[1] < inside.shape[1]:
            if inside[local_origin]:
                candidates.append(((r0, c0, r1, c1), wall, inside, local_origin))
    assert len(candidates) == 1, "expected a single container around the origin"
    return candidates[0]


def _extract_pieces(
    grid: np.ndarray, sep_col: int, bg: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    h, w = grid.shape
    right_cols = np.arange(w)[None, :] > sep_col
    piece_mask = (grid != bg) & right_cols
    pieces = []
    for comp in _components(piece_mask, connectivity=8):
        colors = {int(grid[r, c]) for r, c in comp}
        body_colors = colors - {RED, GREEN}
        assert len(body_colors) == 1, "piece must have exactly one body color"
        body = next(iter(body_colors))

        red_pts = [(r, c) for r, c in comp if int(grid[r, c]) == RED]
        green_pts = [(r, c) for r, c in comp if int(grid[r, c]) == GREEN]
        assert len(red_pts) == 1 and len(green_pts) == 1, "piece must have one red and one green marker"

        r0, c0, r1, c1 = _bbox(comp)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        occupied = np.zeros_like(patch, dtype=bool)
        for r, c in comp:
            rr, cc = r - r0, c - c0
            occupied[rr, cc] = True
            value = int(grid[r, c])
            patch[rr, cc] = body if value in (RED, GREEN) else value

        place_r = red_pts[0][0] - r0
        place_c = green_pts[0][1] - c0
        pieces.append((place_r, place_c, patch, occupied))
    assert pieces, "expected at least one fragment"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_col = _find_separator_col(g, bg)
    (r0, c0, r1, c1), wall, inside, origin = _find_container(g, sep_col, bg)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    left_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    out[wall] = left_patch[wall]

    filled = np.zeros_like(inside, dtype=bool)
    for place_r, place_c, patch, occupied in _extract_pieces(g, sep_col, bg):
        for rr, cc in np.argwhere(occupied):
            tr = origin[0] + place_r + int(rr)
            tc = origin[1] + place_c + int(cc)
            assert 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1], "piece placement escaped the container crop"
            assert inside[tr, tc], "piece landed outside the container interior"
            assert out[tr, tc] == bg, "pieces must not overlap"
            out[tr, tc] = int(patch[rr, cc])
            filled[tr, tc] = True

    assert np.array_equal(filled, inside), "pieces must fill the interior exactly"
    return out


def _cell_set_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _has_hole(cells: set[tuple[int, int]]) -> bool:
    max_r = max(r for r, _ in cells) + 3
    max_c = max(c for _, c in cells) + 3
    occ = np.zeros((max_r, max_c), dtype=bool)
    for r, c in cells:
        occ[r + 1, c + 1] = True
    outside = np.zeros_like(occ, dtype=bool)
    q: deque[tuple[int, int]] = deque([(0, 0)])
    outside[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < occ.shape[0] and 0 <= nc < occ.shape[1] and not occ[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any((~occ) & (~outside)))


def _marker_support_ok(cells: set[tuple[int, int]], place_r: int, place_c: int) -> bool:
    row_cells = [cell for cell in cells if cell[0] == place_r]
    col_cells = [cell for cell in cells if cell[1] == place_c]
    return len(set(row_cells) | set(col_cells)) >= 2


def _choose_marker_cells(
    cells: set[tuple[int, int]], place_r: int, place_c: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    row_cells = [cell for cell in cells if cell[0] == place_r]
    col_cells = [cell for cell in cells if cell[1] == place_c]
    random.shuffle(row_cells)
    random.shuffle(col_cells)
    for red_cell in row_cells:
        for green_cell in col_cells:
            if red_cell != green_cell:
                return red_cell, green_cell
    raise AssertionError("marker cells must be distinct")


def _generate_local_shape(h: int, w: int, place_r: int, place_c: int) -> set[tuple[int, int]] | None:
    min_area = h + w - 1
    max_area = h * w
    for _ in range(MAX_PIECE_TRIES):
        target_area = random.randint(min_area, max_area)
        cells = {(0, 0)}
        path = ["D"] * (h - 1) + ["R"] * (w - 1)
        random.shuffle(path)
        r = c = 0
        for step in path:
            if step == "D":
                r += 1
            else:
                c += 1
            cells.add((r, c))
        while len(cells) < target_area:
            frontier = []
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if _marker_support_ok(cells, place_r, place_c) and not _has_hole(cells):
            return cells
    return None


def _touches(existing: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in existing:
                return True
    return False


def _generate_layout():
    piece_count = random.choice([2, 3, 3, 4, 4, 5])
    pieces = []

    h0 = random.randint(2, 4)
    w0 = random.randint(2, 4)
    local0 = _generate_local_shape(h0, w0, 0, 0)
    if local0 is None:
        return None
    union = set(local0)
    pieces.append({"h": h0, "w": w0, "place_r": 0, "place_c": 0, "cells": local0})

    for _ in range(piece_count - 1):
        placed = False
        for _ in range(MAX_LAYOUT_TRIES):
            h = random.randint(2, 4)
            w = random.randint(2, 4)
            place_r = random.randint(0, h - 1)
            place_c = random.randint(0, w - 1)
            local = _generate_local_shape(h, w, place_r, place_c)
            if local is None:
                continue
            absolute = {(place_r + r, place_c + c) for r, c in local}
            if absolute & union:
                continue
            if max(r for r, _ in absolute | union) > INNER_MAX_COORD:
                continue
            if max(c for _, c in absolute | union) > INNER_MAX_COORD:
                continue
            if not _touches(union, absolute):
                continue
            new_union = union | absolute
            if not _cell_set_connected(new_union):
                continue
            if _has_hole(new_union):
                continue
            pieces.append({"h": h, "w": w, "place_r": place_r, "place_c": place_c, "cells": local})
            union = new_union
            placed = True
            break
        if not placed:
            return None

    inner_h = max(r for r, _ in union) + 1
    inner_w = max(c for _, c in union) + 1
    if inner_h < INNER_MIN_EXTENT or inner_w < INNER_MIN_EXTENT:
        return None
    return pieces, union, inner_h, inner_w


def _build_container(
    cell_colors: dict[tuple[int, int], int], inner_h: int, inner_w: int, outline_color: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inner = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
    for r, c in cell_colors:
        inner[r + 1, c + 1] = True

    shell = np.zeros_like(inner)
    for r, c in np.argwhere(inner):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < shell.shape[0] and 0 <= nc < shell.shape[1] and not inner[nr, nc]:
                shell[nr, nc] = True

    container = np.full(inner.shape, BG, dtype=int)
    container[shell] = outline_color

    expected = container.copy()
    for (r, c), color in cell_colors.items():
        expected[r + 1, c + 1] = color

    input_patch = container.copy()
    input_patch[1, 1] = RED
    return input_patch, expected, shell, inner


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _render_piece_patch(spec: dict, color: int) -> np.ndarray:
    patch = np.full((spec["h"], spec["w"]), BG, dtype=int)
    for r, c in spec["cells"]:
        patch[r, c] = color
    red_cell, green_cell = _choose_marker_cells(spec["cells"], spec["place_r"], spec["place_c"])
    patch[red_cell] = RED
    patch[green_cell] = GREEN
    return patch


def _compose_input(container: np.ndarray, piece_patches: list[np.ndarray]) -> np.ndarray | None:
    container_h, container_w = container.shape
    total_piece_w = sum(p.shape[1] for p in piece_patches)
    right_w_min = max(12, total_piece_w + len(piece_patches) + 2)
    right_w = random.randint(right_w_min, right_w_min + 8)
    sep_col = random.randint(container_w + 2, container_w + 5)
    grid_w = sep_col + 1 + right_w

    max_piece_h = max(p.shape[0] for p in piece_patches)
    min_h = max(container_h + 2, max_piece_h + 3, 10)
    grid_h = random.randint(min_h, min_h + 8)

    if sep_col - container_w - 1 < 1:
        return None
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, sep_col] = SEPARATOR

    cont_r = random.randint(1, grid_h - container_h - 1)
    cont_c = random.randint(1, sep_col - container_w - 1)
    grid[cont_r : cont_r + container_h, cont_c : cont_c + container_w] = container

    boxes: list[tuple[int, int, int, int]] = []
    right_start = sep_col + 2
    right_end = grid_w - 1
    for patch in sorted(piece_patches, key=lambda p: int(p.shape[0] * p.shape[1]), reverse=True):
        ph, pw = patch.shape
        if right_end - right_start < pw:
            return None
        placed = False
        for _ in range(MAX_LAYOUT_TRIES):
            r0 = random.randint(1, grid_h - ph - 1)
            c0 = random.randint(right_start, right_end - pw)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_boxes_separated(box, other, gap=1) for other in boxes):
                grid[r0 : r0 + ph, c0 : c0 + pw] = patch
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _generate_layout()
    if layout is None:
        return None
    pieces, union, inner_h, inner_w = layout

    color_pool = list(DRAW_COLORS)
    random.shuffle(color_pool)
    outline_color = color_pool[0]
    piece_colors = color_pool[1 : 1 + len(pieces)]
    if len(piece_colors) != len(pieces):
        return None

    cell_colors: dict[tuple[int, int], int] = {}
    rendered_pieces = []
    for spec, color in zip(pieces, piece_colors):
        absolute = {(spec["place_r"] + r, spec["place_c"] + c) for r, c in spec["cells"]}
        for cell in absolute:
            cell_colors[cell] = color
        rendered_pieces.append(_render_piece_patch(spec, color))

    input_patch, expected, shell, inner = _build_container(cell_colors, inner_h, inner_w, outline_color)
    if len(_components(shell, connectivity=8)) != 1:
        return None
    if not np.array_equal(_enclosed_cells(shell), inner):
        return None

    grid = _compose_input(input_patch, rendered_pieces)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
