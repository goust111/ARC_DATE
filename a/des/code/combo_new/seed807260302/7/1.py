from __future__ import annotations

from collections import deque
from itertools import permutations
import random

import numpy as np


BG = 0
GREY = 5
DIRECTIONS = ("up", "down", "left", "right")
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

L_PATTERNS = {
    "up": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "right": ((0, 0), (0, 1), (0, 2), (1, 0)),
    "down": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "left": ((0, 2), (1, 0), (1, 1), (1, 2)),
}

ARROW_PATTERNS = {
    "up": ((0, 1), (1, 0), (1, 2)),
    "down": ((1, 0), (1, 2), (2, 1)),
    "left": ((0, 1), (1, 0), (2, 1)),
    "right": ((0, 1), (1, 2), (2, 1)),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


NORMALIZED_L = {direction: _normalize(set(cells)) for direction, cells in L_PATTERNS.items()}


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _connected(cells: set[tuple[int, int]]) -> bool:
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


def _l_orientation(cells: set[tuple[int, int]]) -> str | None:
    if len(cells) != 4:
        return None
    norm = _normalize(cells)
    for direction, pattern in NORMALIZED_L.items():
        if norm == pattern:
            return direction
    return None


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == BG:
            continue
        if np.all(grid[:, c] == color):
            has_left_fg = np.any(grid[:, :c] != BG)
            has_right_fg = np.any(grid[:, c + 1 :] != BG)
            if has_left_fg and has_right_fg:
                candidates.append((c, color))
    assert len(candidates) == 1, "expected one full-height divider column"
    return candidates[0]


def _parse_legend(grid: np.ndarray, sep_col: int, sep_color: int) -> dict[str, int]:
    h, w = grid.shape
    found: list[tuple[str, int, frozenset[tuple[int, int]]]] = []
    for r0 in range(0, h - 2):
        for c0 in range(sep_col + 1, w - 3):
            box = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
            for direction, pattern in ARROW_PATTERNS.items():
                cells = frozenset((r0 + dr, c0 + dc) for dr, dc in pattern)
                if not all(int(grid[r, c]) == sep_color for r, c in cells):
                    continue
                if any(int(grid[r, c]) == sep_color for r, c in box - cells):
                    continue
                swatch_r, swatch_c = r0 + 1, c0 + 3
                swatch = int(grid[swatch_r, swatch_c])
                if swatch in (BG, sep_color):
                    continue
                found.append((direction, swatch, cells))

    mapping: dict[str, int] = {}
    used_cells: set[frozenset[tuple[int, int]]] = set()
    for direction, color, cells in found:
        assert cells not in used_cells, "legend arrow matched more than once"
        used_cells.add(cells)
        assert direction not in mapping, "duplicate legend direction"
        mapping[direction] = color

    assert set(mapping) == set(DIRECTIONS), "expected one legend entry for each direction"
    assert len(set(mapping.values())) == 4, "legend colors must be distinct"
    return mapping


def _attached_marker_split(component: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], str, set[tuple[int, int]]] | None:
    min_r = min(r for r, _c in component)
    max_r = max(r for r, _c in component)
    min_c = min(c for _r, c in component)
    max_c = max(c for _r, c in component)
    candidates: list[tuple[set[tuple[int, int]], str, set[tuple[int, int]]]] = []

    for direction, pattern in L_PATTERNS.items():
        ph = max(r for r, _c in pattern) + 1
        pw = max(c for _r, c in pattern) + 1
        for r0 in range(min_r, max_r - ph + 2):
            for c0 in range(min_c, max_c - pw + 2):
                marker = {(r0 + dr, c0 + dc) for dr, dc in pattern}
                if not marker <= component:
                    continue
                box = {(r, c) for r in range(r0, r0 + ph) for c in range(c0, c0 + pw)}
                if (box - marker) & component:
                    continue
                rest = component - marker
                if len(rest) < 5 or not _connected(rest):
                    continue
                contacts = 0
                for r, c in marker:
                    for nb in _neighbors4(r, c):
                        if nb in rest:
                            contacts += 1
                if 1 <= contacts <= 2:
                    candidates.append((marker, direction, rest))

    if not candidates:
        return None
    assert len(candidates) == 1, "ambiguous attached L marker"
    return candidates[0]


def _min_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def _pair_objects_to_markers(
    objects: list[set[tuple[int, int]]], markers: list[tuple[set[tuple[int, int]], str]]
) -> list[tuple[set[tuple[int, int]], tuple[set[tuple[int, int]], str]]]:
    assert len(objects) == len(markers) and objects, "objects and L markers must pair one-to-one"
    if len(objects) > 7:
        raise AssertionError("too many same-color pairings to disambiguate")

    distances = [[_min_distance(obj, marker_cells) for marker_cells, _direction in markers] for obj in objects]
    best_score: int | None = None
    best_perm: tuple[int, ...] | None = None
    best_count = 0
    for perm in permutations(range(len(markers))):
        score = sum(distances[i][perm[i]] for i in range(len(objects)))
        if best_score is None or score < best_score:
            best_score = score
            best_perm = perm
            best_count = 1
        elif score == best_score:
            best_count += 1

    assert best_perm is not None and best_count == 1, "ambiguous object-to-marker pairing"
    return [(objects[i], markers[best_perm[i]]) for i in range(len(objects))]


def _left_color_groups(left: np.ndarray, color: int) -> tuple[list[tuple[set[tuple[int, int]], str]], list[set[tuple[int, int]]]]:
    comps = _components(left == color)
    markers: list[tuple[set[tuple[int, int]], str]] = []
    objects: list[set[tuple[int, int]]] = []

    for comp in comps:
        orientation = _l_orientation(comp)
        if orientation is not None:
            markers.append((comp, orientation))
        elif len(comp) >= 5:
            objects.append(comp)
        else:
            raise AssertionError("unexpected tiny foreground component")

    if markers:
        return markers, objects

    for comp in comps:
        split = _attached_marker_split(comp)
        assert split is not None, "missing L marker for foreground object"
        marker, orientation, obj = split
        markers.append((marker, orientation))
        objects.append(obj)
    return markers, objects


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each marked left-side object through the direction legend.

    Input structure:
    - The canvas has a black background and one full-height grey divider column.
    - The left side contains same-color pairs: a small L marker and a nearby 4-connected foreground object of area at least five.
    - Each L marker has one of four visible orientations: up, down, left, or right.
    - The right side contains four grey three-pixel arrow indicators, each with a single colored swatch immediately to its right.

    Rule:
    - Read the right-side legend as a mapping from arrow direction to swatch color.
    - For each left-side foreground object, find its same-color L marker and the marker orientation.
    - Recolor all pixels of the object to the legend color for that orientation.
    - Leave the L markers, divider column, background, and legend unchanged.

    Key insight:
    - The L marker orientation is the lookup key, and the right-side legend is the lookup table that supplies the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"

    sep_col, sep_color = _find_separator(g)
    legend = _parse_legend(g, sep_col, sep_color)
    left = g[:, :sep_col]

    out = g.copy()
    colors = [int(c) for c in np.unique(left) if int(c) not in (BG, sep_color)]
    assert colors, "expected at least one marked foreground object"

    for color in colors:
        markers, objects = _left_color_groups(left, color)
        for obj, (_marker_cells, orientation) in _pair_objects_to_markers(objects, markers):
            new_color = legend[orientation]
            for r, c in obj:
                out[r, c] = new_color

    return out


def _random_blob(size: int, max_h: int = 5, max_w: int = 6) -> tuple[tuple[int, int], ...] | None:
    cells = {(0, 0)}
    for _ in range(size * 30):
        if len(cells) >= size:
            break
        r, c = random.choice(tuple(cells))
        nr, nc = random.choice(_neighbors4(r, c))
        trial = cells | {(nr, nc)}
        norm = _normalize(trial)
        height = max(rr for rr, _cc in norm) + 1
        width = max(cc for _rr, cc in norm) + 1
        if height <= max_h and width <= max_w:
            cells = set(norm)
    if len(cells) < size:
        return None
    return _normalize(cells)


def _random_object_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(100):
        style = random.choice(("blob", "rectangle", "bar_with_bumps"))
        if style == "rectangle":
            height = random.randint(2, 4)
            width = random.randint(3, 5)
            cells = {(r, c) for r in range(height) for c in range(width)}
            return _normalize(cells)

        if style == "bar_with_bumps":
            length = random.randint(5, 7)
            horizontal = random.choice((True, False))
            cells = {(0, c) for c in range(length)} if horizontal else {(r, 0) for r in range(length)}
            for _i in range(random.randint(1, 3)):
                if horizontal:
                    c = random.randrange(length)
                    cells.add((random.choice((-1, 1)), c))
                else:
                    r = random.randrange(length)
                    cells.add((r, random.choice((-1, 1))))
            if _connected(cells):
                return _normalize(cells)

        blob = _random_blob(random.randint(5, 12))
        if blob is not None:
            return blob

    return ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2))


def _compose_pair(direction: str, obj_shape: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    marker = tuple(L_PATTERNS[direction])
    marker_h = max(r for r, _c in marker) + 1
    marker_w = max(c for _r, c in marker) + 1
    obj_h = max(r for r, _c in obj_shape) + 1
    obj_w = max(c for _r, c in obj_shape) + 1

    side = random.choice(DIRECTIONS)
    gap = random.choice((1, 1, 2))
    if side in ("left", "right"):
        obj_r = random.randint(-(obj_h - 1), marker_h - 1)
        obj_c = marker_w + gap if side == "right" else -obj_w - gap
    else:
        obj_r = marker_h + gap if side == "down" else -obj_h - gap
        obj_c = random.randint(-(obj_w - 1), marker_w - 1)

    raw_marker = set(marker)
    raw_object = {(r + obj_r, c + obj_c) for r, c in obj_shape}
    all_cells = raw_marker | raw_object
    min_r = min(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    norm_marker = tuple(sorted((r - min_r, c - min_c) for r, c in raw_marker))
    norm_object = tuple(sorted((r - min_r, c - min_c) for r, c in raw_object))
    return norm_marker, norm_object


def _can_place(grid: np.ndarray, cells: set[tuple[int, int]], sep_col: int) -> bool:
    h, _w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < sep_col):
            return False
        if int(grid[r, c]) != BG:
            return False

    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < sep_col and (rr, cc) not in cells and int(grid[rr, cc]) != BG:
                    return False
    return True


def _draw_arrow(grid: np.ndarray, r0: int, c0: int, direction: str, swatch: int) -> None:
    for dr, dc in ARROW_PATTERNS[direction]:
        grid[r0 + dr, c0 + dc] = GREY
    grid[r0 + 1, c0 + 3] = swatch


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    right_w = random.randint(5, 7)
    max_left_w = 30 - right_w - 1
    if max_left_w < 13:
        return None
    left_w = random.randint(13, max_left_w)
    width = left_w + 1 + right_w

    grid = np.full((height, width), BG, dtype=int)
    sep_col = left_w
    grid[:, sep_col] = GREY

    colors = PALETTE[:]
    random.shuffle(colors)
    legend_colors = colors[:4]
    object_count = random.randint(2, min(4, len(colors) - 4))
    object_colors = colors[4 : 4 + object_count]

    legend_dirs = list(DIRECTIONS)
    random.shuffle(legend_dirs)
    legend = {direction: color for direction, color in zip(legend_dirs, legend_colors)}

    legend_total_h = 4 * 3 + 3
    max_start = height - legend_total_h - 1
    if max_start < 1:
        return None
    start = random.randint(1, max_start)
    row_tops = [start + i * 4 for i in range(4)]
    row_dirs = list(DIRECTIONS)
    random.shuffle(row_dirs)
    arrow_c = sep_col + 2
    for row, direction in zip(row_tops, row_dirs):
        _draw_arrow(grid, row, arrow_c, direction, legend[direction])

    object_dirs = random.sample(DIRECTIONS, object_count)
    records: list[tuple[tuple[tuple[int, int], ...], str]] = []
    for color, direction in zip(object_colors, object_dirs):
        placed = False
        for _ in range(250):
            obj_shape = _random_object_shape()
            marker_cells, obj_cells = _compose_pair(direction, obj_shape)
            all_pair = set(marker_cells) | set(obj_cells)
            pair_h = max(r for r, _c in all_pair) + 1
            pair_w = max(c for _r, c in all_pair) + 1
            if pair_h > height - 2 or pair_w > left_w - 2:
                continue
            r0 = random.randint(1, height - pair_h - 1)
            c0 = random.randint(1, left_w - pair_w - 1)
            abs_marker = tuple((r + r0, c + c0) for r, c in marker_cells)
            abs_object = tuple((r + r0, c + c0) for r, c in obj_cells)
            abs_all = set(abs_marker) | set(abs_object)
            if not _can_place(grid, abs_all, sep_col):
                continue
            for r, c in abs_marker:
                grid[r, c] = color
            for r, c in abs_object:
                grid[r, c] = color
            records.append((abs_object, direction))
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for obj_cells, direction in records:
        for r, c in obj_cells:
            expected[r, c] = legend[direction]

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
