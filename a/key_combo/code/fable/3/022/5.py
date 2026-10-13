from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
ALL_COLORS = tuple(range(10))
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left

TEMPLATES: dict[str, dict[int, tuple[tuple[int, int], ...]]] = {
    "hollow3": {
        0: ((0, 0), (1, -1), (1, 1)),
        1: ((0, 0), (-1, -1), (1, -1)),
        2: ((0, 0), (-1, -1), (-1, 1)),
        3: ((0, 0), (-1, 1), (1, 1)),
    },
    "filled4": {
        0: ((0, 0), (1, -1), (1, 0), (1, 1)),
        1: ((0, 0), (-1, -1), (0, -1), (1, -1)),
        2: ((0, 0), (-1, -1), (-1, 0), (-1, 1)),
        3: ((0, 0), (-1, 1), (0, 1), (1, 1)),
    },
    "stem4": {
        0: ((0, 0), (1, 0), (2, -1), (2, 1)),
        1: ((0, 0), (0, -1), (-1, -2), (1, -2)),
        2: ((0, 0), (-1, 0), (-2, -1), (-2, 1)),
        3: ((0, 0), (0, 1), (-1, 2), (1, 2)),
    },
    "midfilled6": {
        0: ((0, 0), (1, -1), (1, 0), (1, 1), (2, -1), (2, 1)),
        1: ((0, 0), (-1, -1), (0, -1), (1, -1), (-1, -2), (1, -2)),
        2: ((0, 0), (-1, -1), (-1, 0), (-1, 1), (-2, -1), (-2, 1)),
        3: ((0, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2)),
    },
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    offsets = DIRS4 if connectivity == 4 else tuple(
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    )
    out: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in offsets:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _template_cells(head: tuple[int, int], template: str, direction: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[template][direction]}


def _match_sprite_component(
    cells: list[tuple[int, int]],
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    cell_set = set(cells)
    for head in cell_set:
        for template, by_dir in TEMPLATES.items():
            for direction, offsets in by_dir.items():
                if len(offsets) != len(cell_set):
                    continue
                placed = _template_cells(head, template, direction)
                if placed == cell_set:
                    return template, int(direction), (int(head[0]), int(head[1])), placed
    return None


def _find_grey_square(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    assert bg != GREY, "grey is reserved for the square cloth, not the background"
    comps = _components(g == GREY, connectivity=4)
    assert len(comps) == 1, "expected exactly one grey square"
    r0, c0, r1, c1 = _bbox(comps[0])
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side > 0, "grey cloth must have a square bbox"
    assert len(comps[0]) == side * side, "grey cloth must be solid"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey square cannot contain holes"
    return side, (r0, c0, r1, c1)


def _find_sprite(g: np.ndarray, bg: int) -> dict:
    candidates: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {bg, GREY}):
        for comp in _components(g == color, connectivity=8):
            if len(comp) <= 1:
                continue
            matched = _match_sprite_component(comp)
            if matched is None:
                continue
            template, direction, head, cells = matched
            candidates.append(
                {
                    "color": int(color),
                    "template": template,
                    "direction": int(direction),
                    "head": head,
                    "cells": cells,
                }
            )

    assert len(candidates) == 1, "expected one unambiguous triangular sprite"
    return candidates[0]


def _marker_pixels(g: np.ndarray, bg: int, sprite_cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    markers: dict[tuple[int, int], int] = {}
    for r0, c0 in np.argwhere((g != bg) & (g != GREY)):
        r, c = int(r0), int(c0)
        if (r, c) in sprite_cells:
            continue
        markers[(r, c)] = int(g[r, c])
    return markers


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    ar, ac = a
    br, bc = b
    if ac == bc and br < ar:
        return 0
    if ar == br and bc > ac:
        return 1
    if ac == bc and br > ar:
        return 2
    if ar == br and bc < ac:
        return 3
    return None


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _consume_colors(
    start_head: tuple[int, int],
    markers: dict[tuple[int, int], int],
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    remaining = dict(markers)
    head = start_head
    colors: list[int] = []
    directions: list[int] = []
    path: list[tuple[int, int]] = [head]

    while remaining:
        by_dir: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        for pt in remaining:
            direction = _direction_between(head, pt)
            if direction is None:
                continue
            by_dir.setdefault(direction, []).append((_distance(head, pt), pt))

        usable = [direction for direction, pts in by_dir.items() if pts]
        assert len(usable) == 1, "each step must have exactly one reachable marker direction"
        direction = usable[0]
        by_dir[direction].sort()
        _, target = by_dir[direction][0]

        colors.append(int(remaining.pop(target)))
        directions.append(int(direction))
        head = target
        path.append(head)

    return colors, directions, path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Consume colored markers with a triangular sprite and weave the temporal color order into the grey square.

    Input structure:
    - The grid has one dominant background color, one triangular arrow sprite, and a solid grey square.
    - Every non-background, non-grey pixel outside the sprite is a colored single-pixel marker.
    - Grey is used only by the square, whose side length equals the number of markers.

    Rule:
    - Start at the sprite head.
    - At each step, among the four row/column half-lines from the current head, exactly one direction contains remaining markers.
    - Move to the nearest marker on that half-line, consume it, record its color, and turn the sprite to that movement direction.
    - Return only the former grey square area, with row i filled by the i-th consumed marker color.

    Key insight:
    - The cloth rows are ordered by the sprite's consumption time, not by marker reading order, color id, or spatial sorting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    side, _square_bbox = _find_grey_square(g, bg)
    sprite = _find_sprite(g, bg)
    markers = _marker_pixels(g, bg, sprite["cells"])
    assert len(markers) == side, "grey square side must equal the number of markers"

    colors, _directions, _path = _consume_colors(sprite["head"], markers)
    assert len(colors) == side

    out = np.empty((side, side), dtype=int)
    for r, color in enumerate(colors):
        out[r, :] = int(color)
    return out


# --------------- generator ---------------

MAX_ATTEMPTS = 600
MARKER_COUNT_RANGE = (4, 7)
GRID_SIZE_RANGE = (14, 30)
STEP_DISTANCE_RANGE = (2, 6)
SQUARE_GAP = 1


def _sample_segment_lengths(total: int, parts: int) -> list[int]:
    lengths = [1] * parts
    for _ in range(total - parts):
        lengths[random.randrange(parts)] += 1
    random.shuffle(lengths)
    return lengths


def _sample_directions(n: int) -> list[int]:
    segment_count = random.randint(2, min(5, n - 1))
    lengths = _sample_segment_lengths(n, segment_count)
    segment_dirs = [random.randrange(4)]
    for _ in range(segment_count - 1):
        turn = random.choice((-1, 1))
        segment_dirs.append((segment_dirs[-1] + turn) % 4)

    dirs: list[int] = []
    for direction, length in zip(segment_dirs, lengths):
        dirs.extend([direction] * length)
    return dirs


def _path_is_valid(heads: list[tuple[int, int]]) -> bool:
    remaining = set(heads[1:])
    if len(remaining) != len(heads) - 1:
        return False

    for i, head in enumerate(heads[:-1]):
        by_dir: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        for pt in remaining:
            direction = _direction_between(head, pt)
            if direction is None:
                continue
            by_dir.setdefault(direction, []).append((_distance(head, pt), pt))

        usable = [direction for direction, pts in by_dir.items() if pts]
        if len(usable) != 1:
            return False
        direction = usable[0]
        by_dir[direction].sort()
        if by_dir[direction][0][1] != heads[i + 1]:
            return False
        remaining.remove(heads[i + 1])

    return not remaining


def _has_required_path_structure(heads: list[tuple[int, int]], dirs: list[int]) -> bool:
    if len(set(dirs)) < 2:
        return False
    if not any(dirs[i] == dirs[i - 1] for i in range(1, len(dirs))):
        return False
    markers = heads[1:]
    if markers == sorted(markers):
        return False
    if markers == sorted(markers, key=lambda rc: (rc[1], rc[0])):
        return False
    return True


def _make_relative_path(n: int) -> tuple[list[tuple[int, int]], list[int]] | None:
    dirs = _sample_directions(n)
    heads = [(0, 0)]
    for direction in dirs:
        dr, dc = DIRS4[direction]
        dist = random.randint(*STEP_DISTANCE_RANGE)
        r, c = heads[-1]
        heads.append((r + dr * dist, c + dc * dist))

    if not _path_is_valid(heads):
        return None
    if not _has_required_path_structure(heads, dirs):
        return None
    return heads, dirs


def _translate_path(
    heads: list[tuple[int, int]],
    sprite_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    all_cells = set(heads) | sprite_cells
    r0, c0, r1, c1 = _bbox(all_cells)
    tr_lo, tr_hi = 1 - r0, h - 2 - r1
    tc_lo, tc_hi = 1 - c0, w - 2 - c1
    if tr_lo > tr_hi or tc_lo > tc_hi:
        return None

    tr = random.randint(tr_lo, tr_hi)
    tc = random.randint(tc_lo, tc_hi)
    moved_heads = [(r + tr, c + tc) for r, c in heads]
    moved_sprite = {(r + tr, c + tc) for r, c in sprite_cells}
    marker_cells = set(moved_heads[1:])
    if marker_cells & moved_sprite:
        return None
    return moved_heads, moved_sprite


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = (br > ar) - (br < ar)
    dc = (bc > ac) - (bc < ac)
    steps = max(abs(br - ar), abs(bc - ac))
    return {(ar + i * dr, ac + i * dc) for i in range(steps + 1)}


def _padded(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _square_cells(r: int, c: int, side: int) -> set[tuple[int, int]]:
    return {(rr, cc) for rr in range(r, r + side) for cc in range(c, c + side)}


def _choose_square(
    shape: tuple[int, int],
    side: int,
    heads: list[tuple[int, int]],
    sprite_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    corridor: set[tuple[int, int]] = set(sprite_cells)
    for a, b in zip(heads, heads[1:]):
        corridor |= _line_cells(a, b)
    blocked = _padded(corridor, shape, SQUARE_GAP)

    options: list[tuple[int, int]] = []
    for r in range(h - side + 1):
        for c in range(w - side + 1):
            cells = _square_cells(r, c, side)
            if cells & blocked:
                continue
            options.append((r, c))
    if not options:
        return None
    return random.choice(options)


def _render_case(
    shape: tuple[int, int],
    bg: int,
    actor: int,
    marker_colors: list[int],
    template: str,
    first_dir: int,
    heads: list[tuple[int, int]],
    sprite_cells: set[tuple[int, int]],
    square_pos: tuple[int, int],
) -> np.ndarray:
    side = len(marker_colors)
    grid = np.full(shape, int(bg), dtype=int)
    sr, sc = square_pos
    grid[sr : sr + side, sc : sc + side] = GREY
    for r, c in sprite_cells:
        grid[r, c] = int(actor)
    for (r, c), color in zip(heads[1:], marker_colors):
        assert int(grid[r, c]) == bg
        grid[r, c] = int(color)

    found = _find_sprite(grid, bg)
    assert found["template"] == template and found["direction"] == first_dir
    return grid


def _expected_from_colors(colors: list[int]) -> np.ndarray:
    side = len(colors)
    out = np.empty((side, side), dtype=int)
    for r, color in enumerate(colors):
        out[r, :] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*MARKER_COUNT_RANGE)
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    bg = random.choice([color for color in ALL_COLORS if color != GREY])
    actor = random.choice([color for color in ALL_COLORS if color not in (bg, GREY)])
    marker_palette = [color for color in ALL_COLORS if color not in (bg, GREY, actor)]
    if len(marker_palette) < n:
        return None
    marker_colors = random.sample(marker_palette, n)

    rel = _make_relative_path(n)
    if rel is None:
        return None
    rel_heads, dirs = rel
    template = random.choice(tuple(TEMPLATES))
    first_dir = dirs[0]
    rel_sprite = _template_cells(rel_heads[0], template, first_dir)

    translated = _translate_path(rel_heads, rel_sprite, (h, w))
    if translated is None:
        return None
    heads, sprite_cells = translated
    if not _path_is_valid(heads):
        return None

    square_pos = _choose_square((h, w), n, heads, sprite_cells)
    if square_pos is None:
        return None

    try:
        grid = _render_case(
            (h, w),
            bg,
            actor,
            marker_colors,
            template,
            first_dir,
            heads,
            sprite_cells,
            square_pos,
        )
        expected = _expected_from_colors(marker_colors)
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    consumed, consumed_dirs, consumed_path = _consume_colors(heads[0], dict(zip(heads[1:], marker_colors)))
    if consumed != marker_colors or consumed_path != heads:
        return None
    if not _has_required_path_structure(consumed_path, consumed_dirs):
        return None

    row_major_colors = [dict(zip(heads[1:], marker_colors))[pt] for pt in sorted(heads[1:])]
    col_major_colors = [
        dict(zip(heads[1:], marker_colors))[pt]
        for pt in sorted(heads[1:], key=lambda rc: (rc[1], rc[0]))
    ]
    if marker_colors == row_major_colors or marker_colors == col_major_colors:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/3/022/5 input")
