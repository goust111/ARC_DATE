import random
from typing import Iterable

import numpy as np


DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
BG = 0
PALETTE = tuple(range(1, 10))
TURN_CHOICES = (0, 1, 2, 3)
ODD_TURNS = (1, 3)
EVEN_TURNS = (0, 2)
RECT_SIZE_CHOICES = ((2, 2), (2, 4), (4, 2))
GRID_MIN = 18
GRID_MAX = 28
PAIR_COUNT_CHOICES = (2, 3, 4)
SEGMENT_MIN = 1
SEGMENT_MAX = 5
PAIR_BUILD_ATTEMPTS = 160
PLACEMENT_ATTEMPTS = 220
GEN_ATTEMPTS = 320


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _component_lists(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _cells_from_bbox(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _draw_square(
    out: np.ndarray, bbox: tuple[int, int, int, int], color: int, side: int
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    sr0 = int(np.floor(center_r - (side - 1) / 2.0))
    sc0 = int(np.floor(center_c - (side - 1) / 2.0))
    sr1 = sr0 + side - 1
    sc1 = sc0 + side - 1
    rr0 = max(0, sr0)
    cc0 = max(0, sc0)
    rr1 = min(h - 1, sr1)
    cc1 = min(w - 1, sc1)
    if rr0 <= rr1 and cc0 <= cc1:
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)


def _square_cells(bbox: tuple[int, int, int, int], side: int) -> set[tuple[int, int]]:
    canvas = np.full((80, 80), -1, dtype=int)
    shifted_bbox = (bbox[0] + 30, bbox[1] + 30, bbox[2] + 30, bbox[3] + 30)
    _draw_square(canvas, shifted_bbox, 1, side)
    return {
        (int(r - 30), int(c - 30))
        for r, c in np.argwhere(canvas == 1)
    }


def _extract_object_mask(color_mask: np.ndarray) -> np.ndarray:
    h, w = color_mask.shape
    object_mask = np.zeros_like(color_mask, dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if color_mask[r : r + 2, c : c + 2].all():
                object_mask[r : r + 2, c : c + 2] = True
    return object_mask


def _ordered_path(path_mask: np.ndarray) -> list[tuple[int, int]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(path_mask)}
    assert coords, "each color must contain a nonempty connector path"
    if len(coords) == 1:
        return [next(iter(coords))]

    degree = {}
    for r, c in coords:
        degree[(r, c)] = sum((nr, nc) in coords for nr, nc in _neighbors4(r, c, *path_mask.shape))
    assert all(d <= 2 for d in degree.values()), "connector path cannot branch"
    endpoints = [pt for pt, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "connector path must have two endpoints"

    ordered = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        next_options = []
        for nr, nc in _neighbors4(cur[0], cur[1], *path_mask.shape):
            nxt = (nr, nc)
            if nxt in coords and nxt != prev:
                next_options.append(nxt)
        if not next_options:
            break
        assert len(next_options) == 1, "connector path must be a simple chain"
        prev, cur = cur, next_options[0]
        ordered.append(cur)
    assert len(ordered) == len(coords), "connector path ordering must cover every path pixel"
    return ordered


def _adjacent_object_dirs(endpoint: tuple[int, int], object_mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = object_mask.shape
    dirs: set[tuple[int, int]] = set()
    for nr, nc in _neighbors4(endpoint[0], endpoint[1], h, w):
        if object_mask[nr, nc]:
            dirs.add((endpoint[0] - nr, endpoint[1] - nc))
    return dirs


def _count_turns(path: list[tuple[int, int]], objects: list[np.ndarray]) -> int:
    endpoints = [path[0], path[-1]] if len(path) > 1 else [path[0]]
    touches: list[list[tuple[int, int]]] = []
    for object_mask in objects:
        touches.append([pt for pt in endpoints if _adjacent_object_dirs(pt, object_mask)])
    assert len(touches[0]) == 1 and len(touches[1]) == 1, "each object must meet one path endpoint"

    start_endpoint = touches[0][0]
    end_endpoint = touches[1][0]
    ordered = path if path[0] == start_endpoint else list(reversed(path))

    start_dirs = _adjacent_object_dirs(ordered[0], objects[0])
    end_dirs = _adjacent_object_dirs(ordered[-1], objects[1])
    assert len(start_dirs) == 1 and len(end_dirs) == 1, "path-object contacts must be unambiguous"

    directions = [next(iter(start_dirs))]
    directions.extend(
        (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
        for i in range(len(ordered) - 1)
    )
    end_dir = next(iter(end_dirs))
    directions.append((-end_dir[0], -end_dir[1]))
    return sum(directions[i] != directions[i - 1] for i in range(1, len(directions)))


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only same-color object pairs whose connector path has an odd number of turns, then redraw the surviving objects as centered squares sized by that turn count.

    Input structure:
    - The background is the dominant color.
    - Each non-background color forms one 4-connected structure made of two solid rectangular objects plus a one-pixel-wide orthogonal path of the same color.
    - Different colors do not intersect, and the path for each color touches the two rectangles only at its endpoints.

    Rule:
    - For each foreground color, isolate the two rectangular objects and the connecting path, then count how many times the path changes direction by 90 degrees.
    - If that turn count is even, erase the entire color to the background.
    - If that turn count is odd, erase the path and redraw both objects as solid squares of side length turn_count + 1, centered on the original object centers and clipped to the grid if needed.

    Key insight:
    - The path encodes both a binary decision and a size parameter: its turn parity decides whether the color survives, and its exact turn count decides the size of the replacement squares.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    out = np.full(g.shape, bg, dtype=int)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        color_mask = g == color
        assert len(_component_lists(color_mask)) == 1, "each foreground color must be one connected structure"

        object_mask = _extract_object_mask(color_mask)
        object_components = _component_lists(object_mask)
        assert len(object_components) == 2, "each color must contain exactly two rectangular objects"

        object_bboxes: list[tuple[int, int, int, int]] = []
        object_masks: list[np.ndarray] = []
        for coords in object_components:
            bbox = _bbox(coords)
            r0, c0, r1, c1 = bbox
            expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(coords) == expected_area, "each object must fill its rectangle"
            comp_mask = np.zeros_like(color_mask, dtype=bool)
            for r, c in coords:
                comp_mask[r, c] = True
            object_bboxes.append(bbox)
            object_masks.append(comp_mask)

        path_mask = color_mask & ~object_mask
        assert path_mask.any(), "each color must contain a connector path"
        assert len(_component_lists(path_mask)) == 1, "connector pixels must form one path component"
        ordered_path = _ordered_path(path_mask)
        turns = _count_turns(ordered_path, object_masks)

        if turns % 2 == 1:
            for bbox in object_bboxes:
                _draw_square(out, bbox, color, turns + 1)

    return out


def _direction_sequence(turns: int) -> list[int]:
    sequence = [random.randrange(4)]
    for _ in range(turns):
        prev = sequence[-1]
        sequence.append(random.choice(((prev + 1) % 4, (prev - 1) % 4)))
    return sequence


def _build_path(direction_sequence: list[int], lengths: list[int]) -> list[tuple[int, int]] | None:
    path = [(0, 0)]
    seen = {(0, 0)}
    for d_idx, length in zip(direction_sequence, lengths):
        dr, dc = DIRS[d_idx]
        for _ in range(length):
            pr, pc = path[-1]
            nxt = (pr + dr, pc + dc)
            if nxt in seen:
                return None
            for rr, cc in seen:
                if abs(rr - nxt[0]) + abs(cc - nxt[1]) == 1 and (rr, cc) != path[-1]:
                    return None
            path.append(nxt)
            seen.add(nxt)
    return path


def _rect_from_attachment(
    attach: tuple[int, int], out_dir: tuple[int, int], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = size
    if out_dir in ((0, 1), (0, -1)):
        offset = random.randrange(h)
        r0 = attach[0] - offset
        r1 = r0 + h - 1
        if out_dir == (0, 1):
            c1 = attach[1]
            c0 = c1 - w + 1
        else:
            c0 = attach[1]
            c1 = c0 + w - 1
    else:
        offset = random.randrange(w)
        c0 = attach[1] - offset
        c1 = c0 + w - 1
        if out_dir == (1, 0):
            r1 = attach[0]
            r0 = r1 - h + 1
        else:
            r0 = attach[0]
            r1 = r0 + h - 1
    return r0, c0, r1, c1


def _adjacent_count(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    return sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in DIRS)


def _pair_spec(turns: int) -> dict | None:
    for _ in range(PAIR_BUILD_ATTEMPTS):
        direction_sequence = _direction_sequence(turns)
        lengths = [random.randint(SEGMENT_MIN, SEGMENT_MAX) for _ in direction_sequence]
        path = _build_path(direction_sequence, lengths)
        if path is None:
            continue

        start_dir = DIRS[direction_sequence[0]]
        end_dir = DIRS[direction_sequence[-1]]
        rect_a = _rect_from_attachment((path[0][0] - start_dir[0], path[0][1] - start_dir[1]), start_dir, random.choice(RECT_SIZE_CHOICES))
        rect_b = _rect_from_attachment((path[-1][0] + end_dir[0], path[-1][1] + end_dir[1]), (-end_dir[0], -end_dir[1]), random.choice(RECT_SIZE_CHOICES))
        rect_a_cells = _cells_from_bbox(rect_a)
        rect_b_cells = _cells_from_bbox(rect_b)
        path_cells = set(path)

        if path_cells & rect_a_cells or path_cells & rect_b_cells or rect_a_cells & rect_b_cells:
            continue
        if _adjacent_count(path[0], rect_a_cells) != 1 or _adjacent_count(path[-1], rect_b_cells) != 1:
            continue
        if any(_adjacent_count(cell, rect_a_cells) for cell in path[1:]):
            continue
        if any(_adjacent_count(cell, rect_b_cells) for cell in path[:-1]):
            continue
        if any(_adjacent_count(cell, rect_b_cells) for cell in rect_a_cells):
            continue

        actual = path_cells | rect_a_cells | rect_b_cells
        bbox_all = _bbox(actual)
        if bbox_all[2] - bbox_all[0] + 1 > 12 or bbox_all[3] - bbox_all[1] + 1 > 12:
            continue

        out_cells: set[tuple[int, int]] = set()
        if turns % 2 == 1:
            out_cells |= _square_cells(rect_a, turns + 1)
            out_cells |= _square_cells(rect_b, turns + 1)
            out_bbox = _bbox(out_cells)
            if out_bbox[2] - out_bbox[0] + 1 > 14 or out_bbox[3] - out_bbox[1] + 1 > 14:
                continue

        return {
            "turns": turns,
            "path": path,
            "cells": actual,
            "rect_bboxes": [rect_a, rect_b],
            "out_cells": out_cells,
        }
    return None


def _translate_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_spec(
    spec: dict,
    h: int,
    w: int,
    forbidden_input: set[tuple[int, int]],
    forbidden_output: set[tuple[int, int]],
) -> tuple[dict, set[tuple[int, int]], set[tuple[int, int]]] | None:
    fit_cells = spec["cells"] | spec["out_cells"]
    min_r, min_c, max_r, max_c = _bbox(fit_cells)
    row_shifts = list(range(-min_r, h - max_r))
    col_shifts = list(range(-min_c, w - max_c))
    if not row_shifts or not col_shifts:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        dr = random.choice(row_shifts)
        dc = random.choice(col_shifts)
        actual = _translate_cells(spec["cells"], dr, dc)
        if actual & forbidden_input:
            continue
        out_cells = _translate_cells(spec["out_cells"], dr, dc)
        if out_cells & forbidden_output:
            continue
        placed = {
            "turns": spec["turns"],
            "cells": actual,
            "rect_bboxes": [
                (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
                for r0, c0, r1, c1 in spec["rect_bboxes"]
            ],
            "path": [(r + dr, c + dc) for r, c in spec["path"]],
        }
        return placed, _halo(actual, h, w), _halo(out_cells, h, w)
    return None


def _choose_turns(n_pairs: int) -> list[int]:
    parities = [i % 2 for i in range(n_pairs)]
    random.shuffle(parities)
    if n_pairs >= 3 and random.random() < 0.5:
        parities[random.randrange(n_pairs)] = random.randrange(2)
    turns = []
    for parity in parities:
        turns.append(random.choice(ODD_TURNS if parity else EVEN_TURNS))
    if all(t % 2 == 0 for t in turns):
        turns[0] = random.choice(ODD_TURNS)
    if all(t % 2 == 1 for t in turns):
        turns[0] = random.choice(EVEN_TURNS)
    random.shuffle(turns)
    return turns


def _expected_output(shape: tuple[int, int], placed_pairs: list[dict], colors: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair, color in zip(placed_pairs, colors):
        if pair["turns"] % 2 == 0:
            continue
        for bbox in pair["rect_bboxes"]:
            _draw_square(out, bbox, color, pair["turns"] + 1)
    return out


def _render_input(shape: tuple[int, int], placed_pairs: list[dict], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for pair, color in zip(placed_pairs, colors):
        for r, c in pair["cells"]:
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(PALETTE, pair_count)
    turns_by_pair = _choose_turns(pair_count)

    placed_pairs: list[dict] = []
    forbidden_input: set[tuple[int, int]] = set()
    forbidden_output: set[tuple[int, int]] = set()

    for turns in turns_by_pair:
        placed = None
        for _ in range(PAIR_BUILD_ATTEMPTS):
            spec = _pair_spec(turns)
            if spec is None:
                continue
            result = _place_spec(spec, h, w, forbidden_input, forbidden_output)
            if result is None:
                continue
            placed, input_halo, output_halo = result
            forbidden_input |= input_halo
            forbidden_output |= output_halo
            placed_pairs.append(placed)
            break
        if placed is None:
            return None

    grid = _render_input((h, w), placed_pairs, colors)
    expected = _expected_output((h, w), placed_pairs, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for combo_new seed168776016/15/4")
