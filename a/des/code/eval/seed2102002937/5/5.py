import random
from collections import defaultdict
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _valid_table_triples(triples: list[tuple[int, int, int]]) -> dict[frozenset[int], int] | None:
    pairs: dict[frozenset[int], int] = {}
    pair_colors: list[int] = []
    for a, b, mix in triples:
        if a in (BLACK, GREY) or b in (BLACK, GREY) or mix in (BLACK, GREY):
            return None
        if a == b:
            return None
        pair = frozenset((a, b))
        if pair in pairs:
            return None
        pairs[pair] = mix
        pair_colors.extend((a, b))
    if len(set(pair_colors)) != 3:
        return None
    return pairs


def _find_mix_table(grid: np.ndarray) -> tuple[dict[frozenset[int], int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, dict[frozenset[int], int], set[tuple[int, int]]]] = []

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not (r0 == 0 or c0 == 0 or r0 + 2 == h - 1 or c0 + 2 == w - 1):
                continue

            triples = [tuple(int(x) for x in grid[r, c0 : c0 + 3]) for r in range(r0, r0 + 3)]
            pairs = _valid_table_triples(triples)
            if pairs is not None:
                cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
                for rr in range(max(0, r0 - 1), min(h, r0 + 4)):
                    for cc in range(max(0, c0 - 1), min(w, c0 + 4)):
                        if int(grid[rr, cc]) == MAROON:
                            cells.add((rr, cc))
                edge_bonus = sum(1 for r, c in cells if r in (0, h - 1) or c in (0, w - 1))
                candidates.append((edge_bonus, -(r0 + c0), pairs, cells))

            triples = [tuple(int(grid[r, c]) for r in range(r0, r0 + 3)) for c in range(c0, c0 + 3)]
            pairs = _valid_table_triples(triples)
            if pairs is not None:
                cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
                for rr in range(max(0, r0 - 1), min(h, r0 + 4)):
                    for cc in range(max(0, c0 - 1), min(w, c0 + 4)):
                        if int(grid[rr, cc]) == MAROON:
                            cells.add((rr, cc))
                edge_bonus = sum(1 for r, c in cells if r in (0, h - 1) or c in (0, w - 1))
                candidates.append((edge_bonus, -(r0 + c0), pairs, cells))

    if not candidates:
        return {}, set()
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2], candidates[0][3]


def _straight_segment(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict | None:
    h, w = grid.shape
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    cells = set(coords)

    if len(rows) == 1:
        r = rows[0]
        if len(cells) != cols[-1] - cols[0] + 1:
            return None
        left, right = cols[0], cols[-1]
        marker_options: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if left - 1 >= 0 and int(grid[r, left - 1]) == GREY:
            marker_options.append(((r, left - 1), (r, right)))
        if right + 1 < w and int(grid[r, right + 1]) == GREY:
            marker_options.append(((r, right + 1), (r, left)))
        if len(marker_options) != 1:
            return None
        marker, unmarked = marker_options[0]
        return {"color": color, "marker": marker, "unmarked": unmarked, "cells": cells}

    if len(cols) == 1:
        c = cols[0]
        if len(cells) != rows[-1] - rows[0] + 1:
            return None
        top, bottom = rows[0], rows[-1]
        marker_options = []
        if top - 1 >= 0 and int(grid[top - 1, c]) == GREY:
            marker_options.append(((top - 1, c), (bottom, c)))
        if bottom + 1 < h and int(grid[bottom + 1, c]) == GREY:
            marker_options.append(((bottom + 1, c), (top, c)))
        if len(marker_options) != 1:
            return None
        marker, unmarked = marker_options[0]
        return {"color": color, "marker": marker, "unmarked": unmarked, "cells": cells}

    return None


def _find_segments(grid: np.ndarray, table_cells: set[tuple[int, int]]) -> list[dict]:
    colors = sorted(
        {
            int(grid[r, c])
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if (r, c) not in table_cells and int(grid[r, c]) not in (BLACK, GREY, MAROON)
        }
    )
    segments: list[dict] = []
    blocked = np.zeros_like(grid, dtype=bool)
    for r, c in table_cells:
        blocked[r, c] = True

    for color in colors:
        mask = (grid == color) & ~blocked
        for coords in _components(mask):
            segment = _straight_segment(grid, color, coords)
            if segment is not None:
                segments.append(segment)
    return segments


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("points are not aligned")


def _l_cells(a: tuple[int, int], b: tuple[int, int], elbow: tuple[int, int]) -> list[tuple[int, int]]:
    first = _line_cells(a, elbow)
    second = _line_cells(elbow, b)
    return first + second[1:]


def _path_is_clear(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    color: int,
    table_cells: set[tuple[int, int]],
) -> bool:
    for r, c in path:
        if (r, c) in table_cells:
            return False
        value = int(grid[r, c])
        if value not in (BLACK, GREY, color):
            return False
    return True


def _draw_connection(
    out: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    color: int,
    table_cells: set[tuple[int, int]],
) -> None:
    ar, ac = a
    br, bc = b
    if ar == br or ac == bc:
        choices = [_line_cells(a, b)]
    else:
        choices = [_l_cells(a, b, (ar, bc)), _l_cells(a, b, (br, ac))]

    chosen = choices[0]
    for path in choices:
        if _path_is_clear(out, path, color, table_cells):
            chosen = path
            break
    for r, c in chosen:
        if (r, c) not in table_cells:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    mix_table, table_cells = _find_mix_table(g)
    segments = _find_segments(g, table_cells)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for segment in segments:
        by_color[int(segment["color"])].append(segment)

    for color, color_segments in by_color.items():
        if len(color_segments) != 2:
            continue
        a, b = color_segments
        _draw_connection(out, a["marker"], b["marker"], color, table_cells)

    endpoints: list[tuple[int, tuple[int, int]]] = [
        (int(segment["color"]), segment["unmarked"]) for segment in segments
    ]
    for (color_a, a), (color_b, b) in combinations(endpoints, 2):
        if color_a == color_b:
            continue
        ar, ac = a
        br, bc = b
        if ar == br and abs(ac - bc) == 2:
            mid = (ar, (ac + bc) // 2)
        elif ac == bc and abs(ar - br) == 2:
            mid = ((ar + br) // 2, ac)
        else:
            continue
        mix = mix_table.get(frozenset((color_a, color_b)))
        if mix is not None and int(g[mid]) == BLACK and mid not in table_cells:
            out[mid] = mix

    return out


def _put_from_marker(
    grid: np.ndarray,
    color: int,
    marker: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> tuple[int, int]:
    h, w = grid.shape
    mr, mc = marker
    dr, dc = direction
    cells = [(mr + dr * i, mc + dc * i) for i in range(1, length + 1)]
    all_cells = [marker] + cells
    if any(not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BLACK for r, c in all_cells):
        raise ValueError("segment placement collision")
    grid[mr, mc] = GREY
    for r, c in cells:
        grid[r, c] = color
    return cells[-1]


def _put_from_unmarked(
    grid: np.ndarray,
    color: int,
    unmarked: tuple[int, int],
    direction_to_marker: tuple[int, int],
    length: int,
) -> tuple[int, int]:
    h, w = grid.shape
    ur, uc = unmarked
    dr, dc = direction_to_marker
    cells = [(ur + dr * i, uc + dc * i) for i in range(length)]
    marker = (ur + dr * length, uc + dc * length)
    all_cells = cells + [marker]
    if any(not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BLACK for r, c in all_cells):
        raise ValueError("segment placement collision")
    for r, c in cells:
        grid[r, c] = color
    grid[marker] = GREY
    return marker


def generate_input() -> np.ndarray:
    for _ in range(200):
        h = random.randint(26, 30)
        w = random.randint(26, 30)
        grid = np.zeros((h, w), dtype=int)

        line_colors = random.sample([1, 2, 3, 4, 6, 7, 8], 3)
        mix_colors = random.sample([c for c in [1, 2, 3, 4, 6, 7, 8] if c not in line_colors], 3)
        color_a, color_b, color_c = line_colors
        grid[0, 0:3] = [color_a, color_b, mix_colors[0]]
        grid[1, 0:3] = [color_a, color_c, mix_colors[1]]
        grid[2, 0:3] = [color_b, color_c, mix_colors[2]]
        grid[0:4, 3] = MAROON
        grid[3, 0:4] = MAROON

        try:
            r0 = random.randint(5, 7)
            c0 = random.randint(3, 5)
            len_a0 = random.randint(2, 3)
            len_a1 = random.randint(2, 4)
            _put_from_marker(grid, color_a, (r0, c0), (0, 1), len_a0)
            _put_from_marker(grid, color_a, (r0 + 4, c0 + random.randint(0, 2)), (0, 1), len_a1)

            gap_right = c0 + len_a0 + 2
            len_b0 = random.randint(2, 3)
            marker_b0 = _put_from_unmarked(grid, color_b, (r0, gap_right), (0, 1), len_b0)
            col_b1 = random.randint(max(marker_b0[1] + 3, 15), min(w - 6, 20))
            _put_from_marker(grid, color_b, (r0 + 8, col_b1), (0, 1), random.randint(2, 4))

            r2a = h - random.randint(8, 10)
            r2b = h - random.randint(4, 5)
            if r2b <= r2a:
                r2b = r2a + 3
            c2a = random.randint(4, 7)
            c2b = random.randint(13, min(w - 6, 18))
            _put_from_marker(grid, color_c, (r2a, c2a), (0, 1), random.randint(2, 4))
            _put_from_marker(grid, color_c, (r2b, c2b), (0, 1), random.randint(2, 4))

            result = main(grid)
            if result.shape == grid.shape and np.any(result != grid):
                return grid
        except (AssertionError, ValueError, IndexError):
            continue

    raise RuntimeError("failed to generate a valid input")
