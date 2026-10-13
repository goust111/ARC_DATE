from __future__ import annotations

from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVE_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = (0, 1, 2, 3, 4, 6, 7, 8, 9)
BG_COLOR = 5

MIN_GRID = 16
MAX_GRID = 23
MIN_OBJECTS = 2
MAX_OBJECTS = 4
MIN_POINTERS = 2
MAX_POINTERS = 4
MIN_POINTER_LEN = 3
MAX_POINTER_LEN = 6
MIN_OBJECT_AREA = 5
MAX_OBJECT_AREA = 10

SHAPE_RETRIES = 80
OBJECT_PLACEMENT_RETRIES = 120
POINTER_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _ordered_line(coords: np.ndarray) -> list[tuple[int, int]] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        r = int(rows[0])
        c0 = int(np.min(cols))
        c1 = int(np.max(cols))
        if c1 - c0 + 1 != len(coords):
            return None
        return [(r, c) for c in range(c0, c1 + 1)]
    if np.all(cols == cols[0]):
        c = int(cols[0])
        r0 = int(np.min(rows))
        r1 = int(np.max(rows))
        if r1 - r0 + 1 != len(coords):
            return None
        return [(r, c) for r in range(r0, r1 + 1)]
    return None


def _pointer_info(grid: np.ndarray, coords: np.ndarray) -> dict[str, object] | None:
    ordered = _ordered_line(coords)
    if ordered is None or len(ordered) < 3:
        return None

    colors = [int(grid[r, c]) for r, c in ordered]
    values, counts = np.unique(colors, return_counts=True)
    if len(values) != 2:
        return None

    count_map = {int(v): int(n) for v, n in zip(values, counts)}
    arrow_colors = [color for color, count in count_map.items() if count == 1]
    if len(arrow_colors) != 1:
        return None

    arrow_color = int(arrow_colors[0])
    body_color = int(next(color for color in count_map if color != arrow_color))
    if count_map[body_color] != len(ordered) - 1:
        return None

    first_arrow = colors[0] == arrow_color
    last_arrow = colors[-1] == arrow_color
    if first_arrow == last_arrow:
        return None

    if ordered[0][0] == ordered[-1][0]:
        direction = (0, -1) if first_arrow else (0, 1)
    else:
        direction = (-1, 0) if first_arrow else (1, 0)

    return {
        "coords": np.array(ordered, dtype=int),
        "length": int(len(ordered)),
        "body_color": body_color,
        "arrow_color": arrow_color,
        "direction": direction,
    }


def _detect_pointers(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mask = g != bg
    pointers: list[dict[str, object]] = []
    pointer_mask = np.zeros_like(mask, dtype=bool)
    for comp in _components(mask):
        info = _pointer_info(g, comp)
        if info is None:
            continue
        pointers.append(info)
        for r, c in info["coords"]:
            pointer_mask[int(r), int(c)] = True
    pointers.sort(
        key=lambda info: (
            int(np.min(info["coords"][:, 0])),
            int(np.min(info["coords"][:, 1])),
            int(info["body_color"]),
            int(info["arrow_color"]),
        )
    )
    return pointers, pointer_mask, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each pointer by its own length and stamp its body color onto overlapped foreground objects.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pointers, pointer_mask, bg = _detect_pointers(g)

    out = g.copy()
    out[pointer_mask] = bg
    object_mask = (g != bg) & (~pointer_mask)

    for info in pointers:
        dr, dc = info["direction"]
        length = int(info["length"])
        body_color = int(info["body_color"])
        moved = info["coords"] + np.array([dr * length, dc * length], dtype=int)
        for rr, cc in moved:
            r = int(rr)
            c = int(cc)
            if 0 <= r < h and 0 <= c < w and object_mask[r, c]:
                out[r, c] = body_color

    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_polyomino(rng: np.random.Generator) -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        area = int(rng.integers(MIN_OBJECT_AREA, MAX_OBJECT_AREA + 1))
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            cells.add((nr, nc))

        norm = _normalize(cells)
        rows = [r for r, _ in norm]
        cols = [c for _, c in norm]
        height = max(rows) + 1
        width = max(cols) + 1
        if height < 2 or width < 2:
            continue
        if height > 6 or width > 6:
            continue
        return norm
    return None


def _translate(cells: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(r + top, c + left) for r, c in cells]


def _can_place_component(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in cell_set:
                return False
    return True


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _iter_segments(h: int, w: int, length: int, direction: tuple[int, int]) -> list[list[tuple[int, int]]]:
    segments: list[list[tuple[int, int]]] = []
    if direction[0] == 0:
        for r in range(h):
            for c0 in range(w - length + 1):
                segments.append([(r, c0 + i) for i in range(length)])
    else:
        for r0 in range(h - length + 1):
            for c in range(w):
                segments.append([(r0 + i, c) for i in range(length)])
    return segments


def _source_from_target(
    target: list[tuple[int, int]], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(r - dr * length, c - dc * length) for r, c in target]


def _choose_pointer_target(
    rng: np.random.Generator,
    h: int,
    w: int,
    direction: tuple[int, int],
    length: int,
    objects: list[dict[str, object]],
    all_object_cells: set[tuple[int, int]],
    occupied_input: np.ndarray,
    pointer_source_cells: set[tuple[int, int]],
    pointer_target_cells: set[tuple[int, int]],
    hit: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, object] | None] | None:
    segments = _iter_segments(h, w, length, direction)
    rng.shuffle(segments)

    if hit:
        order = list(range(len(objects)))
        rng.shuffle(order)
        for obj_idx in order:
            obj = objects[obj_idx]
            obj_cells = obj["cells"]
            other_cells = all_object_cells - obj_cells
            candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
            for target in segments:
                target_set = set(target)
                if target_set & pointer_source_cells:
                    continue
                if target_set & pointer_target_cells:
                    continue
                overlap = target_set & obj_cells
                if not overlap:
                    continue
                if target_set & other_cells:
                    continue
                source = _source_from_target(target, direction, length)
                if not _can_place_component(source, occupied_input):
                    continue
                candidates.append((len(overlap), target, source))
            if not candidates:
                continue
            good = [item for item in candidates if item[0] >= 2]
            chosen_pool = good if good else candidates
            _, target, source = chosen_pool[int(rng.integers(len(chosen_pool)))]
            return target, source, obj
        return None

    candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    for target in segments:
        target_set = set(target)
        if target_set & all_object_cells:
            continue
        if target_set & pointer_source_cells:
            continue
        if target_set & pointer_target_cells:
            continue
        source = _source_from_target(target, direction, length)
        if not _can_place_component(source, occupied_input):
            continue
        candidates.append((target, source))

    if not candidates:
        return None
    target, source = candidates[int(rng.integers(len(candidates)))]
    return target, source, None


def _pointer_colors(
    rng: np.random.Generator, target_object: dict[str, object] | None
) -> tuple[int, int]:
    forbidden = {BG_COLOR}
    if target_object is not None:
        forbidden.add(int(target_object["color"]))
    body_options = [color for color in PALETTE if color not in forbidden]
    body = int(body_options[int(rng.integers(len(body_options)))])
    arrow_options = [color for color in PALETTE if color != body]
    arrow = int(arrow_options[int(rng.integers(len(arrow_options)))])
    return body, arrow


def _paint_pointer(
    grid: np.ndarray,
    source: list[tuple[int, int]],
    direction: tuple[int, int],
    body_color: int,
    arrow_color: int,
) -> None:
    if direction == (0, 1) or direction == (1, 0):
        arrow_cell = source[-1]
    else:
        arrow_cell = source[0]
    for r, c in source:
        grid[r, c] = arrow_color if (r, c) == arrow_cell else body_color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    w = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied_input = np.zeros((h, w), dtype=bool)

    n_objects = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    object_colors = list(rng.choice(np.array(PALETTE, dtype=int), size=n_objects, replace=False))
    objects: list[dict[str, object]] = []

    for color in object_colors:
        placed = False
        for _ in range(OBJECT_PLACEMENT_RETRIES):
            shape = _random_polyomino(rng)
            if shape is None:
                continue
            height = max(r for r, _ in shape) + 1
            width = max(c for _, c in shape) + 1
            top = int(rng.integers(0, h - height + 1))
            left = int(rng.integers(0, w - width + 1))
            cells = _translate(shape, top, left)
            if not _can_place_component(cells, occupied_input):
                continue
            _paint(grid, cells, int(color))
            for r, c in cells:
                occupied_input[r, c] = True
            cell_set = set(cells)
            objects.append({"color": int(color), "cells": cell_set})
            placed = True
            break
        if not placed:
            return None

    all_object_cells = set().union(*(obj["cells"] for obj in objects))
    n_pointers = int(rng.integers(MIN_POINTERS, MAX_POINTERS + 1))
    hit_flags = [True, False] + [bool(rng.integers(0, 2)) for _ in range(n_pointers - 2)]
    rng.shuffle(hit_flags)

    pointer_source_cells: set[tuple[int, int]] = set()
    pointer_target_cells: set[tuple[int, int]] = set()

    for hit in hit_flags:
        placed = False
        for _ in range(POINTER_PLACEMENT_RETRIES):
            direction = MOVE_DIRS[int(rng.integers(len(MOVE_DIRS)))]
            length = int(rng.integers(MIN_POINTER_LEN, MAX_POINTER_LEN + 1))
            result = _choose_pointer_target(
                rng,
                h,
                w,
                direction,
                length,
                objects,
                all_object_cells,
                occupied_input,
                pointer_source_cells,
                pointer_target_cells,
                hit,
            )
            if result is None:
                continue
            target, source, target_object = result
            body_color, arrow_color = _pointer_colors(rng, target_object)
            _paint_pointer(grid, source, direction, body_color, arrow_color)
            for r, c in source:
                occupied_input[r, c] = True
                pointer_source_cells.add((r, c))
            for cell in target:
                pointer_target_cells.add(cell)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GENERATE_RETRIES} attempts")
