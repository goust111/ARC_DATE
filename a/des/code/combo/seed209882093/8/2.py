from __future__ import annotations

from itertools import permutations, product
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def _scale(d: tuple[int, int], k: int) -> tuple[int, int]:
    return (d[0] * k, d[1] * k)


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return (-d[1], d[0])


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return (d[1], -d[0])


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bg_color(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(counts.argmax())])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "cells": tuple(sorted(cells))})
    return comps


def _segment_info(comp: dict) -> dict:
    cells = list(comp["cells"])
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    color = int(comp["color"])
    if len(cells) == 1:
        cell = cells[0]
        return {
            "color": color,
            "cells": tuple(cells),
            "endpoints": (cell, cell),
        }
    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert len(cells) == c1 - c0 + 1
        ordered = tuple((row, c) for c in range(c0, c1 + 1))
        return {
            "color": color,
            "cells": ordered,
            "endpoints": (ordered[0], ordered[-1]),
        }
    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        assert len(cells) == r1 - r0 + 1
        ordered = tuple((r, col) for r in range(r0, r1 + 1))
        return {
            "color": color,
            "cells": ordered,
            "endpoints": (ordered[0], ordered[-1]),
        }
    raise AssertionError("foreground components must be 1-pixel-wide axis-aligned segments")


def _segment_orientations(segment: dict) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    a, b = segment["endpoints"]
    if a == b:
        return [(a, b)]
    return [(a, b), (b, a)]


def _gap_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    if a[0] == b[0]:
        dist = b[1] - a[1]
        if abs(dist) not in (2, 3):
            return None
        step = 1 if dist > 0 else -1
        return [(a[0], c) for c in range(a[1] + step, b[1], step)]
    if a[1] == b[1]:
        dist = b[0] - a[0]
        if abs(dist) not in (2, 3):
            return None
        step = 1 if dist > 0 else -1
        return [(r, a[1]) for r in range(a[0] + step, b[0], step)]
    return None


def _path_degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {cell: sum((nbr in cells) for nbr in _neighbors4(cell)) for cell in cells}


def _is_simple_path(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    degrees = _path_degrees(cells)
    if len(cells) == 1:
        return next(iter(degrees.values())) == 0
    if any(deg == 0 or deg > 2 for deg in degrees.values()):
        return False
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return False
    start = endpoints[0]
    stack = [start]
    seen = {start}
    while stack:
        cell = stack.pop()
        for nbr in _neighbors4(cell):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cells)


def _reconstruct_full_path(grid: np.ndarray, bg: int, segments: list[dict]) -> set[tuple[int, int]]:
    base_cells = set()
    for segment in segments:
        base_cells.update(segment["cells"])

    candidates: dict[frozenset[tuple[int, int]], set[tuple[int, int]]] = {}
    idxs = list(range(len(segments)))
    for perm in permutations(idxs):
        orientation_sets = [_segment_orientations(segments[i]) for i in perm]
        for oriented in product(*orientation_sets):
            full = set(base_cells)
            ok = True
            for i in range(len(perm) - 1):
                end_a = oriented[i][1]
                start_b = oriented[i + 1][0]
                gap = _gap_cells(end_a, start_b)
                if gap is None:
                    ok = False
                    break
                for cell in gap:
                    if int(grid[cell]) != bg:
                        ok = False
                        break
                if not ok:
                    break
                full.update(gap)
            if not ok:
                continue
            if not _is_simple_path(full):
                continue
            candidates.setdefault(frozenset(full), full)

    assert candidates, "failed to reconstruct a valid path"
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    # Valid tasks should be unique. If multiple survive, use a stable geometric tie-break.
    ordered = sorted(
        candidates.values(),
        key=lambda cells: (len(cells), tuple(sorted(cells))),
    )
    return ordered[-1]


def _terminal_endpoint_and_direction(path_cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    degrees = _path_degrees(path_cells)
    endpoints = sorted(cell for cell, deg in degrees.items() if deg == 1)
    assert len(endpoints) == 2
    end = endpoints[-1]
    inside = next(nbr for nbr in _neighbors4(end) if nbr in path_cells)
    return end, (end[0] - inside[0], end[1] - inside[1])


def _growth_cells(end: tuple[int, int], direction: tuple[int, int], seg_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    extension = [_add(end, _scale(direction, step)) for step in range(1, seg_count + 1)]
    tail = extension[-1]
    if direction[0] != 0:
        wing_dirs = [(0, -1), (0, 1)]
    else:
        wing_dirs = [(-1, 0), (1, 0)]
    arrow = [_add(tail, direction)] + [_add(tail, wing) for wing in wing_dirs]
    return extension, arrow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect scattered same-color segments into their unique simple path, then extend an arrow.

    The full path itself is uniquely reconstructible from the legal 1-2-cell gaps. Reversing the
    same completed path does not change the path cells, so the arrow is anchored to the later path
    endpoint in reading order to make the end choice deterministic.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    by_color: dict[int, list[dict]] = {}
    for comp in _components(g, bg):
        seg = _segment_info(comp)
        by_color.setdefault(seg["color"], []).append(seg)

    out = g.copy()
    h, w = g.shape

    for color, segments in by_color.items():
        assert 2 <= len(segments) <= 5
        full_path = _reconstruct_full_path(g, bg, segments)
        for r, c in full_path:
            out[r, c] = color

        end, direction = _terminal_endpoint_and_direction(full_path)
        extension, arrow = _growth_cells(end, direction, len(segments))
        for r, c in extension + arrow:
            assert 0 <= r < h and 0 <= c < w
            assert int(out[r, c]) in (bg, color)
            out[r, c] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
COLOR_COUNT_RANGE = (2, 4)
SEG_COUNT_RANGE = (2, 5)
SEG_LENGTHS = (1, 2, 3, 4)
GAP_LENGTHS = (1, 2)
GEN_ATTEMPTS = 300
PATH_ATTEMPTS = 200
COLOR_PLACE_ATTEMPTS = 100


def _line_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [_add(start, _scale(direction, step)) for step in range(length)]


def _in_bounds(cells: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    around: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    around.add((rr, cc))
    return around


def _segments_are_separate(segments: list[list[tuple[int, int]]]) -> bool:
    indexed = [{(int(r), int(c)) for r, c in seg} for seg in segments]
    for i in range(len(indexed)):
        for j in range(i + 1, len(indexed)):
            for r, c in indexed[i]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (r + dr, c + dc) in indexed[j]:
                            return False
    return True


def _growth_is_clean(path_cells: set[tuple[int, int]], extension: list[tuple[int, int]], arrow: list[tuple[int, int]]) -> bool:
    new_cells = extension + arrow
    if len(set(new_cells)) != len(new_cells):
        return False
    if any(cell in path_cells for cell in new_cells):
        return False
    tail = extension[-1]
    allowed_touch: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for i, cell in enumerate(extension):
        allowed = set()
        if i == 0:
            path_neighbors = {nbr for nbr in _neighbors4(cell) if nbr in path_cells}
            if len(path_neighbors) != 1:
                return False
            allowed.update(path_neighbors)
            allowed.add(extension[1] if len(extension) > 1 else tail)
        if i > 0:
            allowed.add(extension[i - 1])
        if i + 1 < len(extension):
            allowed.add(extension[i + 1])
        else:
            allowed.update(arrow)
        allowed_touch[cell] = allowed
    for cell in arrow:
        allowed_touch[cell] = {tail}

    full = path_cells | set(new_cells)
    for cell in new_cells:
        for nbr in _neighbors4(cell):
            if nbr in full and nbr not in allowed_touch[cell]:
                return False
    return True


def _generate_color_shape(h: int, w: int) -> dict | None:
    seg_count = random.randint(*SEG_COUNT_RANGE)
    lengths = [random.choice(SEG_LENGTHS) for _ in range(seg_count)]
    gaps = [random.choice(GAP_LENGTHS) for _ in range(seg_count - 1)]
    require_turn = random.random() < 0.75

    for _ in range(PATH_ATTEMPTS):
        start = (random.randrange(h), random.randrange(w))
        direction = random.choice(DIRS4)
        first = _line_cells(start, direction, lengths[0])
        if not _in_bounds(first, h, w):
            continue

        segments = [first]
        full_path = set(first)
        current_end = first[-1]
        current_dir = direction
        turn_count = 0
        ok = True

        for idx in range(1, seg_count):
            gap_len = gaps[idx - 1]
            choices = []
            connector_dirs = [current_dir, _left(current_dir), _right(current_dir)]
            random.shuffle(connector_dirs)
            for connector_dir in connector_dirs:
                next_start = _add(current_end, _scale(connector_dir, gap_len + 1))
                connector = [_add(current_end, _scale(connector_dir, step)) for step in range(1, gap_len + 1)]
                next_dirs = [connector_dir, _left(connector_dir), _right(connector_dir)]
                random.shuffle(next_dirs)
                for next_dir in next_dirs:
                    next_seg = _line_cells(next_start, next_dir, lengths[idx])
                    candidate = full_path | set(connector) | set(next_seg)
                    if not _in_bounds(next_seg, h, w):
                        continue
                    if len(candidate) != len(full_path) + len(connector) + len(next_seg):
                        continue
                    if not _is_simple_path(candidate):
                        continue
                    bend = int(connector_dir != current_dir) + int(next_dir != connector_dir)
                    choices.append((candidate, connector, next_seg, next_dir, bend))

            if not choices:
                ok = False
                break

            candidate, connector, next_seg, next_dir, bend = random.choice(choices)
            full_path = candidate
            segments.append(next_seg)
            current_end = next_seg[-1]
            current_dir = next_dir
            turn_count += bend

        if not ok:
            continue
        if require_turn and turn_count == 0:
            continue
        if not _segments_are_separate(segments):
            continue

        end, out_dir = _terminal_endpoint_and_direction(full_path)
        extension, arrow = _growth_cells(end, out_dir, seg_count)
        if not _in_bounds(extension + arrow, h, w):
            continue
        if not _growth_is_clean(full_path, extension, arrow):
            continue

        output_cells = full_path | set(extension) | set(arrow)
        return {
            "segments": [tuple(seg) for seg in segments],
            "path": full_path,
            "extension": extension,
            "arrow": arrow,
            "output_cells": output_cells,
        }

    return None


def _paint(grid: np.ndarray, cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        color_count = random.randint(*COLOR_COUNT_RANGE)
        colors = random.sample(range(1, 10), color_count)

        inp = np.full((h, w), BG, dtype=int)
        expected = np.full((h, w), BG, dtype=int)
        occupied: set[tuple[int, int]] = set()
        occupied_halo: set[tuple[int, int]] = set()

        ok = True
        for color in colors:
            placed = False
            for _ in range(COLOR_PLACE_ATTEMPTS):
                shape = _generate_color_shape(h, w)
                if shape is None:
                    continue
                if shape["output_cells"] & occupied_halo:
                    continue

                for seg in shape["segments"]:
                    _paint(inp, seg, color)
                _paint(expected, shape["path"], color)
                _paint(expected, shape["extension"], color)
                _paint(expected, shape["arrow"], color)

                occupied |= shape["output_cells"]
                occupied_halo |= _halo(shape["output_cells"], h, w)
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a valid input grid")
