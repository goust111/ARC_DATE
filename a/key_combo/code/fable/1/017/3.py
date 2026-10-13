from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.is_dir() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, components_mask, in_bounds
except Exception:
    class Color:
        BLACK = 0
        GREEN = 3
        ORANGE = 7
        TEAL = 8

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = map(int, shape)
        return 0 <= int(r) < h and 0 <= int(c) < w

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        if int(connectivity) == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]

        seen = np.zeros_like(mask, dtype=bool)
        out: list[np.ndarray] = []
        for r, c in np.argwhere(mask):
            r = int(r)
            c = int(c)
            if seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if not in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
        return out


BG = Color.BLACK
PATH = Color.GREEN
RIGHT_TURN = Color.ORANGE
LEFT_TURN = Color.TEAL

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_STEPS = DIRS
GEN_RETRY_LIMIT = 300
GEN_ACTIVE_LEN_CHOICES = (2, 3, 4)
GEN_DISTRACTOR_LEN_CHOICES = (2, 3, 4)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _flip(color: int) -> int:
    color = int(color)
    assert color in (LEFT_TURN, RIGHT_TURN), "mirror color must be teal or orange"
    return RIGHT_TURN if color == LEFT_TURN else LEFT_TURN


def _turn(dir_idx: int, color: int) -> int:
    color = int(color)
    if color == LEFT_TURN:
        return (int(dir_idx) - 1) % 4
    assert color == RIGHT_TURN, "mirror color must be teal or orange"
    return (int(dir_idx) + 1) % 4


def _boundary_direction(shape: tuple[int, int], emitter: tuple[int, int]) -> int:
    h, w = map(int, shape)
    r, c = map(int, emitter)
    sides = [r == 0, c == w - 1, r == h - 1, c == 0]
    assert sum(sides) == 1, "emitter must lie on exactly one border side"
    if r == 0:
        return 2
    if c == w - 1:
        return 3
    if r == h - 1:
        return 0
    return 1


def _parse_strips(grid: np.ndarray) -> list[dict[str, object]]:
    mirror_mask = np.isin(grid, [LEFT_TURN, RIGHT_TURN])
    comps = components_mask(mirror_mask, connectivity=4)
    strips: list[dict[str, object]] = []
    for coords in comps:
        rows = coords[:, 0]
        cols = coords[:, 1]
        colors = np.unique(grid[rows, cols])
        assert len(colors) == 1, "each mirror strip must be monochrome"
        color = int(colors[0])
        assert color in (LEFT_TURN, RIGHT_TURN), "only teal/orange strips are allowed"

        same_row = int(rows.min()) == int(rows.max())
        same_col = int(cols.min()) == int(cols.max())
        assert same_row or same_col, "each strip must be axis-aligned"
        if same_row:
            span = int(cols.max()) - int(cols.min()) + 1
            assert int(coords.shape[0]) == span, "horizontal strip must be contiguous"
            cells = sorted((int(r), int(c)) for r, c in coords)
        else:
            span = int(rows.max()) - int(rows.min()) + 1
            assert int(coords.shape[0]) == span, "vertical strip must be contiguous"
            cells = sorted((int(r), int(c)) for r, c in coords)
        strips.append({"cells": cells, "color": color})

    assert strips, "expected at least one mirror strip"
    return strips


def _color_mask(colors: list[int]) -> int:
    mask = 0
    for idx, color in enumerate(colors):
        if int(color) == RIGHT_TURN:
            mask |= 1 << idx
    return mask


def _simulate_scene(
    shape: tuple[int, int],
    emitter: tuple[int, int],
    strip_cells: list[list[tuple[int, int]]],
    strip_colors: list[int],
) -> tuple[set[tuple[int, int]], list[int], list[int]]:
    h, w = map(int, shape)
    assert len(strip_cells) == len(strip_colors)

    cell_to_strip: dict[tuple[int, int], int] = {}
    for idx, cells in enumerate(strip_cells):
        assert cells, "each strip must contain at least one cell"
        for r, c in cells:
            r = int(r)
            c = int(c)
            assert in_bounds((h, w), r, c), "strip must stay in bounds"
            key = (r, c)
            assert key not in cell_to_strip, "strips must not overlap"
            cell_to_strip[key] = idx

    colors = [int(color) for color in strip_colors]
    current = tuple(map(int, emitter))
    assert in_bounds((h, w), current[0], current[1]), "emitter must stay in bounds"
    assert current not in cell_to_strip, "emitter cannot overlap a strip"

    dir_idx = _boundary_direction((h, w), current)
    visited = {current}
    hits: list[int] = []
    color_mask = _color_mask(colors)
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (current[0], current[1], dir_idx, color_mask)
        assert state not in seen_states, "beam must eventually exit instead of looping"
        seen_states.add(state)

        dr, dc = DIRS[dir_idx]
        nr = current[0] + dr
        nc = current[1] + dc
        if not in_bounds((h, w), nr, nc):
            break

        next_cell = (nr, nc)
        strip_idx = cell_to_strip.get(next_cell)
        if strip_idx is not None:
            color = colors[strip_idx]
            dir_idx = _turn(dir_idx, color)
            colors[strip_idx] = _flip(color)
            color_mask ^= 1 << strip_idx
            hits.append(strip_idx)
            continue

        current = next_cell
        visited.add(current)

    return visited, hits, colors


def _render_input(
    shape: tuple[int, int],
    emitter: tuple[int, int],
    strip_cells: list[list[tuple[int, int]]],
    strip_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for cells, color in zip(strip_cells, strip_colors):
        for r, c in cells:
            out[int(r), int(c)] = int(color)
    out[int(emitter[0]), int(emitter[1])] = PATH
    return out


def _render_output(
    shape: tuple[int, int],
    path_cells: set[tuple[int, int]],
    strip_cells: list[list[tuple[int, int]]],
    final_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for cells, color in zip(strip_cells, final_colors):
        for r, c in cells:
            out[int(r), int(c)] = int(color)
    for r, c in path_cells:
        out[int(r), int(c)] = PATH
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the green beam through teal/orange mirror strips, flipping each struck strip after every turn.

    Input structure:
    - Black background contains several one-cell-wide horizontal or vertical
      mirror strips colored teal or orange.
    - Exactly one green emitter pixel sits on the grid border and launches the
      beam perpendicularly inward.

    Rule:
    - Advance the beam one cell at a time, recoloring every visited background
      cell green.
    - If the next cell is a mirror strip cell, do not enter it: teal turns the
      beam left, orange turns it right, and the entire struck strip flips to the
      opposite mirror color.
    - Continue until the beam exits the grid; output the green path over the
      final mirror-strip colors.

    Key insight:
    - Mirror color encodes turn chirality, but every hit also toggles that same
      strip, so revisiting one strip later can force the opposite turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 1 <= h <= 30 and 1 <= w <= 30, "grid must stay within ARC limits"

    allowed = {BG, PATH, LEFT_TURN, RIGHT_TURN}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected color in grid"

    emitters = np.argwhere(g == PATH)
    assert emitters.shape == (1, 2), "expected exactly one green emitter pixel"
    emitter = tuple(map(int, emitters[0]))

    strips = _parse_strips(g)
    strip_cells = [list(strip["cells"]) for strip in strips]
    strip_colors = [int(strip["color"]) for strip in strips]

    path_cells, _hits, final_colors = _simulate_scene((h, w), emitter, strip_cells, strip_colors)
    return _render_output((h, w), path_cells, strip_cells, final_colors)


def _can_place_segment(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    *,
    forbid_path_neighbors: bool,
    safe_from_path_start: int,
) -> bool:
    h, w = map(int, shape)
    cell_set = set(cells)
    if not all(in_bounds((h, w), r, c) for r, c in cells):
        return False
    if any(cell in path_cells for cell in cells[safe_from_path_start:]):
        return False

    if forbid_path_neighbors:
        for cell in cells[safe_from_path_start:]:
            if any(nei in path_cells for nei in _neighbors4(cell)):
                return False

    for cell in cells:
        if cell in occupied and cell not in cell_set:
            return False
        for nei in _neighbors4(cell):
            if nei in occupied and nei not in cell_set:
                return False
    return True


def _extend_active_strips(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    strips: list[list[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()

    for anchor in anchors:
        candidates: list[list[tuple[int, int]]] = []
        dirs = random.sample(GEN_STEPS, len(GEN_STEPS))
        lengths = random.sample(list(GEN_ACTIVE_LEN_CHOICES), len(GEN_ACTIVE_LEN_CHOICES))
        for dr, dc in dirs:
            for length in lengths:
                cells = [(anchor[0] + dr * step, anchor[1] + dc * step) for step in range(length)]
                if anchor != cells[0]:
                    continue
                if any(cell in path_cells for cell in cells[1:]):
                    continue
                if not _can_place_segment(
                    shape,
                    cells,
                    path_cells,
                    occupied,
                    forbid_path_neighbors=True,
                    safe_from_path_start=1,
                ):
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        strips.append(chosen)
        occupied.update(chosen)

    return strips


def _place_distractors(
    shape: tuple[int, int],
    path_cells: set[tuple[int, int]],
    strips: list[list[tuple[int, int]]],
    colors: list[int],
) -> None:
    occupied = {cell for strip in strips for cell in strip}
    target = random.randint(0, 3)
    for _ in range(target):
        for _ in range(160):
            r = random.randrange(shape[0])
            c = random.randrange(shape[1])
            dr, dc = random.choice(GEN_STEPS)
            length = random.choice(GEN_DISTRACTOR_LEN_CHOICES)
            cells = [(r + dr * step, c + dc * step) for step in range(length)]
            if not _can_place_segment(
                shape,
                cells,
                path_cells,
                occupied,
                forbid_path_neighbors=True,
                safe_from_path_start=0,
            ):
                continue
            strips.append(cells)
            colors.append(random.choice([LEFT_TURN, RIGHT_TURN]))
            occupied.update(cells)
            break


def _build_simple_blueprint() -> dict[str, object]:
    excursions = random.choices([1, 2, 3], weights=[1, 2, 2], k=1)[0]
    rows: list[int]
    while True:
        rows = [0]
        signs: list[int] = []
        for _ in range(excursions):
            sign = random.choice([-1, 1])
            signs.append(sign)
            rows.append(rows[-1] + sign * random.randint(2, 4))
        if excursions == 1 or len(set(signs)) > 1 or random.random() < 0.35:
            break

    top_pad = random.randint(2, 4) - min(rows)
    rows = [row + top_pad for row in rows]
    bottom_pad = random.randint(2, 4)
    runs = [random.randint(2, 4) for _ in range(excursions)]

    col = 0
    anchors: list[tuple[int, int]] = []
    colors: list[int] = []
    for idx in range(excursions):
        col += runs[idx]
        src_row = rows[idx]
        dst_row = rows[idx + 1]
        anchors.append((src_row, col + 1))
        if dst_row < src_row:
            colors.append(LEFT_TURN)
            anchors.append((dst_row - 1, col))
            colors.append(RIGHT_TURN)
        else:
            colors.append(RIGHT_TURN)
            anchors.append((dst_row + 1, col))
            colors.append(LEFT_TURN)

    height = max(rows) + bottom_pad + 1
    width = col + 1 + random.randint(3, 7)
    expected_hits = list(range(len(anchors)))
    return {
        "shape": (height, width),
        "emitter": (rows[0], 0),
        "anchors": anchors,
        "colors": colors,
        "expected_hits": expected_hits,
        "repeated": False,
    }


def _build_loop_blueprint() -> dict[str, object]:
    rise = random.randint(2, 5)
    lead = random.randint(2, 5)
    top_run = random.randint(3, 6)
    top_pad = random.randint(2, 4)
    bottom_pad = random.randint(2, 4)
    right_pad = random.randint(3, 6)

    emitter = (top_pad + rise + 1, 0)
    anchors = [
        (top_pad + rise + 1, lead + 1),
        (top_pad, lead),
        (top_pad + 1, lead + top_run + 1),
        (top_pad + rise + 2, lead + top_run),
    ]
    colors = [LEFT_TURN, RIGHT_TURN, RIGHT_TURN, RIGHT_TURN]
    return {
        "shape": (top_pad + rise + 3 + bottom_pad, lead + top_run + 2 + right_pad),
        "emitter": emitter,
        "anchors": anchors,
        "colors": colors,
        "expected_hits": [0, 1, 2, 3, 0],
        "repeated": True,
    }


def _rotate_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    return np.rot90(inp, k).copy(), np.rot90(out, k).copy()


def _blueprint_pair() -> tuple[np.ndarray, np.ndarray] | None:
    spec = _build_loop_blueprint() if random.random() < 0.6 else _build_simple_blueprint()
    shape = tuple(map(int, spec["shape"]))
    emitter = tuple(map(int, spec["emitter"]))
    anchors = [tuple(map(int, cell)) for cell in spec["anchors"]]
    colors = [int(color) for color in spec["colors"]]
    expected_hits = list(map(int, spec["expected_hits"]))
    repeated = bool(spec["repeated"])

    point_strips = [[anchor] for anchor in anchors]
    try:
        point_path, point_hits, _point_final = _simulate_scene(shape, emitter, point_strips, colors)
    except AssertionError:
        return None
    if point_hits != expected_hits:
        return None

    full_strips = _extend_active_strips(shape, anchors, point_path)
    if full_strips is None:
        return None

    full_colors = list(colors)
    _place_distractors(shape, point_path, full_strips, full_colors)

    try:
        path_cells, hits, final_colors = _simulate_scene(shape, emitter, full_strips, full_colors)
    except AssertionError:
        return None
    if hits != expected_hits:
        return None

    if repeated:
        if len(set(hits)) == len(hits):
            return None
        if final_colors[0] != full_colors[0]:
            return None
    else:
        if len(set(hits)) != len(hits):
            return None
        for idx in expected_hits:
            if final_colors[idx] != _flip(full_colors[idx]):
                return None

    inp = _render_input(shape, emitter, full_strips, full_colors)
    out = _render_output(shape, path_cells, full_strips, final_colors)
    rot_inp, rot_out = _rotate_pair(inp, out)
    try:
        actual = main(rot_inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, rot_out):
        return None
    return rot_inp, rot_out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        pair = _blueprint_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid mirror-beam scene")
