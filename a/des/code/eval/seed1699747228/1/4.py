from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
GATE_PALETTE = (1, 2, 3, 4, 6, 7, 8)

LAYER_RANGE = (2, 5)
INNER_SIDE_RANGE = (3, 5)
GRID_MAX = 30
GRID_MIN = 18
MIN_EXTRA_H = 9
MIN_EXTRA_W = 6
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords
    coord_set = set(coords)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in coords:
        degrees[(r, c)] = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c, h, w))

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2
    border_endpoints = [
        (r, c)
        for r, c in endpoints
        if r in (0, h - 1) or c in (0, w - 1)
    ]
    assert len(border_endpoints) == 1

    start = border_endpoints[0]
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        next_cells = [
            nb
            for nb in _neighbors4(cur[0], cur[1], h, w)
            if nb in coord_set and nb != prev
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1
        nxt = next_cells[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    assert len(ordered) == len(coords)
    assert ordered[-1] in endpoints and ordered[-1] != start
    return ordered


def _gate_sides(
    prev: tuple[int, int],
    cur: tuple[int, int],
    nxt: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    r, c = cur
    if prev[0] == nxt[0]:
        if 0 < r < h - 1:
            return (r - 1, c), (r + 1, c)
        return None
    if prev[1] == nxt[1]:
        if 0 < c < w - 1:
            return (r, c - 1), (r, c + 1)
        return None
    return None


def _extract_gate_sequence(grid: np.ndarray, path: list[tuple[int, int]]) -> list[int]:
    sequence: list[int] = []
    seen: set[int] = set()
    for i in range(1, len(path) - 1):
        sides = _gate_sides(path[i - 1], path[i], path[i + 1], grid.shape)
        if sides is None:
            continue
        (r1, c1), (r2, c2) = sides
        color = int(grid[r1, c1])
        if color == int(grid[r2, c2]) and color not in (BG, GREY, MAROON):
            assert color not in seen
            seen.add(color)
            sequence.append(color)
    assert sequence
    return sequence


def _frame_pixels(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pixels = [(r0, c) for c in range(c0, c1 + 1)]
    pixels.extend((r1, c) for c in range(c0, c1 + 1) if r1 != r0)
    pixels.extend((r, c0) for r in range(r0 + 1, r1))
    pixels.extend((r, c1) for r in range(r0 + 1, r1) if c1 != c0)
    return pixels


def _find_nested_frames(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    comps = _components(grid == MAROON)
    frames: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in comps:
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        bbox = (r0, c0, r1, c1)
        expected = set(_frame_pixels(bbox))
        actual = {tuple(map(int, rc)) for rc in coords}
        assert expected == actual
        frames.append((bbox, coords))

    assert frames
    frames.sort(
        key=lambda item: (
            (item[0][2] - item[0][0] + 1) * (item[0][3] - item[0][1] + 1),
            -item[0][0],
            -item[0][1],
        ),
        reverse=True,
    )
    for i in range(len(frames) - 1):
        outer = frames[i][0]
        inner = frames[i + 1][0]
        assert inner[0] == outer[0] + 2
        assert inner[1] == outer[1] + 2
        assert inner[2] == outer[2] - 2
        assert inner[3] == outer[3] - 2
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect gate colors along the grey path from the border endpoint, then recolor
    the nested maroon rectangle frames from outermost to innermost with that
    ordered sequence.
    """
    g = np.asarray(grid, dtype=int)
    path = _trace_path(g == GREY)
    sequence = _extract_gate_sequence(g, path)
    frames = _find_nested_frames(g)
    assert len(sequence) == len(frames)

    out = g.copy()
    for color, (_bbox, coords) in zip(sequence, frames):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _random_composition(total: int, parts: int, min_each: int) -> list[int]:
    assert total >= parts * min_each
    pieces = [min_each] * parts
    extra = total - parts * min_each
    for _ in range(extra):
        pieces[random.randrange(parts)] += 1
    random.shuffle(pieces)
    return pieces


def _make_target(layer_count: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    inner_h = random.randint(*INNER_SIDE_RANGE)
    inner_w = random.randint(*INNER_SIDE_RANGE)
    outer_h = inner_h + 4 * (layer_count - 1)
    outer_w = inner_w + 4 * (layer_count - 1)
    patch = np.full((outer_h, outer_w), BG, dtype=int)
    bboxes: list[tuple[int, int, int, int]] = []
    for layer in range(layer_count):
        r0 = 2 * layer
        c0 = 2 * layer
        r1 = outer_h - 1 - 2 * layer
        c1 = outer_w - 1 - 2 * layer
        patch[r0, c0 : c1 + 1] = MAROON
        patch[r1, c0 : c1 + 1] = MAROON
        patch[r0 : r1 + 1, c0] = MAROON
        patch[r0 : r1 + 1, c1] = MAROON
        bboxes.append((r0, c0, r1, c1))
    return patch, bboxes


def _draw_segment(
    path: list[tuple[int, int]],
    end: tuple[int, int],
) -> None:
    r0, c0 = path[-1]
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            path.append((r0, c))
    else:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            path.append((r, c0))


def _build_top_path(
    endpoint_row: int,
    endpoint_col: int,
    width: int,
) -> list[tuple[int, int]] | None:
    max_turns = min(4, endpoint_row // 2 - 1)
    if max_turns < 2:
        return None
    horizontal_count = random.randint(2, max_turns)
    vertical_lengths = _random_composition(
        endpoint_row,
        horizontal_count + 1,
        min_each=2,
    )
    row_levels = []
    total = 0
    for length in vertical_lengths:
        total += length
        row_levels.append(total)

    usable_cols = list(range(2, width - 2))
    if endpoint_col not in usable_cols:
        return None

    for _ in range(80):
        cols = [random.choice(usable_cols)]
        ok = True
        for _step in range(horizontal_count - 1):
            choices = [c for c in usable_cols if abs(c - cols[-1]) >= 2]
            if not choices:
                ok = False
                break
            cols.append(random.choice(choices))
        if not ok or abs(cols[-1] - endpoint_col) < 2:
            continue
        cols.append(endpoint_col)

        path = [(0, cols[0])]
        for i in range(horizontal_count):
            _draw_segment(path, (row_levels[i], cols[i]))
            _draw_segment(path, (row_levels[i], cols[i + 1]))
        _draw_segment(path, (row_levels[-1], cols[-1]))
        if len(path) == len(set(path)):
            return path
    return None


def _candidate_gates(
    grid: np.ndarray,
    path: list[tuple[int, int]],
) -> list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]]:
    candidates: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = []
    for i in range(1, len(path) - 1):
        sides = _gate_sides(path[i - 1], path[i], path[i + 1], grid.shape)
        if sides is None:
            continue
        if all(int(grid[r, c]) == BG for r, c in sides):
            candidates.append((i, sides))
    return candidates


def _choose_gate_positions(
    candidates: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]],
    count: int,
) -> list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] | None:
    for _ in range(120):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        chosen: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = []
        occupied: set[tuple[int, int]] = set()
        for item in shuffled:
            side_cells = set(item[1])
            if all(abs(item[0] - other[0]) >= 2 for other in chosen) and occupied.isdisjoint(side_cells):
                chosen.append(item)
                occupied |= side_cells
                if len(chosen) == count:
                    chosen.sort(key=lambda part: part[0])
                    return chosen
    return None


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        assert int(grid[r, c]) == BG
        grid[r, c] = GREY


def _paint_target(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    assert target.shape == patch.shape
    assert np.all(target == BG)
    target[patch != BG] = patch[patch != BG]


def _oracle_expected(
    grid: np.ndarray,
    layer_bboxes: list[tuple[int, int, int, int]],
    gate_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for color, bbox in zip(gate_colors, layer_bboxes):
        for r, c in _frame_pixels(bbox):
            out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(*LAYER_RANGE)
    target_patch, local_bboxes = _make_target(layer_count)
    target_h, target_w = target_patch.shape

    if target_h + MIN_EXTRA_H > GRID_MAX or target_w + MIN_EXTRA_W > GRID_MAX:
        return None

    extra_h = random.randint(MIN_EXTRA_H, GRID_MAX - target_h)
    extra_w = random.randint(MIN_EXTRA_W, GRID_MAX - target_w)
    top_margin = random.randint(8, extra_h - 1)
    bottom_margin = extra_h - top_margin
    left_margin = random.randint(2, extra_w - 2)
    right_margin = extra_w - left_margin

    height = target_h + top_margin + bottom_margin
    width = target_w + left_margin + right_margin
    if height < GRID_MIN or width < GRID_MIN:
        return None

    target_top = top_margin
    target_left = left_margin
    endpoint_row = target_top - 1
    endpoint_col = random.randint(target_left, target_left + target_w - 1)
    path = _build_top_path(endpoint_row, endpoint_col, width)
    if path is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    _paint_target(grid, target_patch, target_top, target_left)
    _paint_path(grid, path)

    candidates = _candidate_gates(grid, path)
    if len(candidates) < layer_count:
        return None
    chosen = _choose_gate_positions(candidates, layer_count)
    if chosen is None:
        return None

    gate_colors = random.sample(GATE_PALETTE, layer_count)
    for color, (_idx, sides) in zip(gate_colors, chosen):
        for r, c in sides:
            assert int(grid[r, c]) == BG
            grid[r, c] = int(color)

    global_bboxes = [
        (r0 + target_top, c0 + target_left, r1 + target_top, c1 + target_left)
        for r0, c0, r1, c1 in local_bboxes
    ]
    expected = _oracle_expected(grid, global_bboxes, gate_colors)

    rotation = random.randrange(4)
    inp = np.rot90(grid, rotation).copy()
    out = np.rot90(expected, rotation).copy()
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
