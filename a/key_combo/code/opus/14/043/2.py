from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
WALL = 5
ORANGE = 7
TEAL = 8

MARKER_PALETTE = (BLUE, RED, YELLOW, ORANGE)

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
MIN_ROOM_SIDE = 5
GEN_ATTEMPTS = 400


Coord = tuple[int, int]
Shape = tuple[Coord, ...]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Coord]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[Coord] = deque([(r, c)])
        seen[r, c] = True
        comp: list[Coord] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        components.append(comp)
    return components


def _components_from_cells(cells: list[Coord] | tuple[Coord, ...]) -> list[list[Coord]]:
    remaining = set((int(r), int(c)) for r, c in cells)
    components: list[list[Coord]] = []
    while remaining:
        start = remaining.pop()
        q: deque[Coord] = deque([start])
        comp = [start]
        while q:
            rr, cc = q.popleft()
            for nr, nc in _neighbors4(rr, cc):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    q.append((nr, nc))
                    comp.append((nr, nc))
        components.append(comp)
    return components


def _bbox(coords: list[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(coords: list[Coord] | tuple[Coord, ...]) -> Shape:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_mask(shape: Shape) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _mask_to_shape(mask: np.ndarray) -> Shape:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return _normalize_shape(coords)


def _shape_variants(shape: Shape) -> list[Shape]:
    base = _shape_mask(shape)
    variants: set[Shape] = set()
    for flip in (False, True):
        work = np.fliplr(base) if flip else base
        for k in range(4):
            variants.add(_mask_to_shape(np.rot90(work, k)))
    return sorted(variants)


def _shape_dims(shape: Shape) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _shape_signature(shape: Shape) -> tuple[int, tuple[int, int]]:
    h, w = _shape_dims(shape)
    return len(shape), tuple(sorted((h, w)))


def _is_irregular(shape: Shape) -> bool:
    h, w = _shape_dims(shape)
    return h > 1 and w > 1 and h * w != len(shape)


def _enumerate_shapes() -> list[Shape]:
    all_cells = [(r, c) for r in range(3) for c in range(3)]
    shapes: set[Shape] = set()
    for size in range(3, 6):
        for combo in combinations(all_cells, size):
            if len(_components_from_cells(combo)) == 1:
                shapes.add(_normalize_shape(combo))
    return sorted(shapes)


ALL_SHAPES = _enumerate_shapes()


def _template_candidates() -> list[Shape]:
    candidates: list[Shape] = []
    for shape in ALL_SHAPES:
        variants = set(_shape_variants(shape))
        has_decoy = any(
            other not in variants and _shape_signature(other) == _shape_signature(shape)
            for other in ALL_SHAPES
        )
        if _is_irregular(shape) and len(variants) > 1 and has_decoy:
            candidates.append(shape)
    return candidates


TEMPLATE_SHAPES = _template_candidates()


def _extract_single_shape(grid: np.ndarray, chamber: list[Coord], color: int) -> Shape:
    cells = [(r, c) for r, c in chamber if int(grid[r, c]) == int(color)]
    assert 3 <= len(cells) <= 5, f"expected 3-5 cells of color {color}"
    comps = _components_from_cells(cells)
    assert len(comps) == 1, f"expected one connected component of color {color}"
    return _normalize_shape(comps[0])


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    chambers = _components_from_mask(g != WALL)
    assert chambers, "expected at least one chamber"

    template_indices = [i for i, chamber in enumerate(chambers) if any(g[r, c] == TEAL for r, c in chamber)]
    assert len(template_indices) == 1, "expected exactly one template chamber"
    template_idx = template_indices[0]

    template_chamber = chambers[template_idx]
    template_colors = {int(g[r, c]) for r, c in template_chamber}
    assert template_colors <= {BLACK, TEAL}, "template chamber must contain only black and teal"
    template_shape = _extract_single_shape(g, template_chamber, TEAL)
    template_variants = set(_shape_variants(template_shape))

    marker_colors_seen: set[int] = set()
    non_template = []
    for i, chamber in enumerate(chambers):
        if i == template_idx:
            continue
        marker_cells = [(r, c, int(g[r, c])) for r, c in chamber if int(g[r, c]) in MARKER_PALETTE]
        assert len(marker_cells) == 1, "each non-template chamber must have exactly one marker"
        marker_r, marker_c, marker_color = marker_cells[0]
        assert marker_color not in marker_colors_seen, "marker colors must be unique"
        marker_colors_seen.add(marker_color)

        colors = {int(g[r, c]) for r, c in chamber}
        assert colors <= {BLACK, GREEN, marker_color}, "unexpected color in non-template chamber"
        green_shape = _extract_single_shape(g, chamber, GREEN)
        non_template.append(
            {
                "index": i,
                "cells": chamber,
                "bbox": _bbox(chamber),
                "marker": (marker_r, marker_c),
                "marker_color": marker_color,
                "green_shape": green_shape,
                "matches": green_shape in template_variants,
            }
        )

    return {
        "grid": g,
        "chambers": chambers,
        "template_idx": template_idx,
        "template_chamber": template_chamber,
        "template_shape": template_shape,
        "template_variants": template_variants,
        "non_template": non_template,
    }


def _render_output(analysis: dict) -> np.ndarray:
    g = analysis["grid"]
    out = g.copy()
    for info in analysis["non_template"]:
        if info["matches"]:
            color = int(info["marker_color"])
            for r, c in info["cells"]:
                out[r, c] = color
        else:
            for r, c in info["cells"]:
                if int(out[r, c]) == GREEN or int(out[r, c]) in MARKER_PALETTE:
                    out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill or clear each chamber according to whether its green sprite matches the teal template.

    Input structure:
    - Gray walls partition the black background into several 4-connected chambers.
    - One chamber contains only a teal reference template.
    - Every other chamber contains one connected green sprite and one colored marker pixel.

    Rule:
    - Extract the teal template as a tight-bbox mask and enumerate its unique rotations and reflections.
    - For each non-template chamber, compare the chamber's green sprite mask against those template variants.
    - If the sprite matches, flood the entire chamber with that chamber's marker color, covering the black cells, the sprite, and the marker.
    - If the sprite does not match, erase the sprite and marker to black while leaving the chamber otherwise unchanged.

    Key insight:
    - The teal shape is a gate: it does not specify a recolor target, it selects between two different chamber-wide operations, fill versus erase.
    """
    analysis = _analyze_input(np.asarray(grid, dtype=int))
    return _render_output(analysis)


def _build_partition() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    split_r = random.randint(MIN_ROOM_SIDE, h - MIN_ROOM_SIDE - 1)
    split_c = random.randint(MIN_ROOM_SIDE, w - MIN_ROOM_SIDE - 1)
    grid[split_r, :] = WALL
    grid[:, split_c] = WALL

    want_five = random.random() < 0.55
    if want_five:
        base_rooms = _components_from_mask(grid != WALL)
        options: list[tuple[str, tuple[int, int, int, int]]] = []
        for room in base_rooms:
            r0, c0, r1, c1 = _bbox(room)
            room_h = r1 - r0 + 1
            room_w = c1 - c0 + 1
            if room_w >= 2 * MIN_ROOM_SIDE + 1:
                options.append(("v", (r0, c0, r1, c1)))
            if room_h >= 2 * MIN_ROOM_SIDE + 1:
                options.append(("h", (r0, c0, r1, c1)))
        if not options:
            return None
        orient, (r0, c0, r1, c1) = random.choice(options)
        if orient == "v":
            wall_c = random.randint(c0 + MIN_ROOM_SIDE, c1 - MIN_ROOM_SIDE)
            grid[r0 : r1 + 1, wall_c] = WALL
        else:
            wall_r = random.randint(r0 + MIN_ROOM_SIDE, r1 - MIN_ROOM_SIDE)
            grid[wall_r, c0 : c1 + 1] = WALL

    chambers = _components_from_mask(grid != WALL)
    if len(chambers) not in (4, 5):
        return None
    for chamber in chambers:
        r0, c0, r1, c1 = _bbox(chamber)
        if r1 - r0 + 1 < MIN_ROOM_SIDE or c1 - c0 + 1 < MIN_ROOM_SIDE:
            return None
    return grid


def _room_anchor(room_bbox: tuple[int, int, int, int], shape: Shape) -> Coord:
    r0, c0, r1, c1 = room_bbox
    shape_h, shape_w = _shape_dims(shape)
    row_choices = list(range(r0 + 1, r1 - shape_h + 1))
    col_choices = list(range(c0 + 1, c1 - shape_w + 1))
    assert row_choices and col_choices, "shape must fit with one-cell padding"
    return random.choice(row_choices), random.choice(col_choices)


def _place_shape(grid: np.ndarray, room_bbox: tuple[int, int, int, int], shape: Shape, color: int) -> set[Coord]:
    start_r, start_c = _room_anchor(room_bbox, shape)
    placed = {(start_r + dr, start_c + dc) for dr, dc in shape}
    for r, c in placed:
        grid[r, c] = int(color)
    return placed


def _marker_cell(room_bbox: tuple[int, int, int, int], occupied: set[Coord]) -> Coord:
    r0, c0, r1, c1 = room_bbox
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if (r, c) not in occupied]
    all_free = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in occupied]
    preferred = [
        (r, c)
        for r, c in interior
        if all(abs(r - sr) + abs(c - sc) >= 2 for sr, sc in occupied)
    ]
    choices = preferred or interior or all_free
    assert choices, "room must have a free marker cell"
    return random.choice(choices)


def _mismatch_shapes(template_shape: Shape) -> tuple[list[Shape], list[Shape]]:
    variants = set(_shape_variants(template_shape))
    signature = _shape_signature(template_shape)
    same_sig = [s for s in ALL_SHAPES if s not in variants and _shape_signature(s) == signature]
    other = [s for s in ALL_SHAPES if s not in variants]
    return same_sig, other


def _choose_mismatches(template_shape: Shape, count: int) -> list[Shape]:
    same_sig, other = _mismatch_shapes(template_shape)
    chosen: list[Shape] = []
    used: set[Shape] = set()
    if count > 0 and same_sig:
        first = random.choice(same_sig)
        chosen.append(first)
        used.add(first)
    pool = [s for s in other if s not in used]
    while len(chosen) < count:
        if not pool:
            pool = [s for s in other]
        pick = random.choice(pool)
        chosen.append(pick)
        if pick in pool:
            pool.remove(pick)
    random.shuffle(chosen)
    return chosen


def _manual_expected(grid: np.ndarray) -> np.ndarray:
    return _render_output(_analyze_input(grid))


def _validate_generated_case(inp: np.ndarray) -> bool:
    try:
        analysis = _analyze_input(inp)
    except AssertionError:
        return False

    non_template = analysis["non_template"]
    if len(analysis["chambers"]) not in (4, 5):
        return False
    if len(non_template) < 3:
        return False

    template_shape = analysis["template_shape"]
    template_sig = _shape_signature(template_shape)
    match_count = sum(info["matches"] for info in non_template)
    mismatch_count = len(non_template) - match_count
    if match_count < 1 or mismatch_count < 1:
        return False

    if not any(info["matches"] and info["green_shape"] != template_shape for info in non_template):
        return False
    if not any((not info["matches"]) and _shape_signature(info["green_shape"]) == template_sig for info in non_template):
        return False

    marker_colors = [info["marker_color"] for info in non_template]
    if len(marker_colors) != len(set(marker_colors)):
        return False

    template_room = analysis["template_chamber"]
    if len(template_room) - len(template_shape) < 4:
        return False

    out = _manual_expected(inp)
    for info in non_template:
        empty_space = len(info["cells"]) - len(info["green_shape"]) - 1
        if empty_space < 4:
            return False
        vals = {int(out[r, c]) for r, c in info["cells"]}
        if info["matches"]:
            if vals != {info["marker_color"]}:
                return False
        else:
            if vals != {BLACK}:
                return False

    template_out = out.copy()
    for info in non_template:
        for r, c in info["cells"]:
            template_out[r, c] = BLACK
    template_in = inp.copy()
    for info in non_template:
        for r, c in info["cells"]:
            template_in[r, c] = BLACK
    if not np.array_equal(template_in, template_out):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def _generate_case() -> np.ndarray | None:
    grid = _build_partition()
    if grid is None:
        return None

    chambers = _components_from_mask(grid != WALL)
    rooms = [{"cells": chamber, "bbox": _bbox(chamber)} for chamber in chambers]

    template_shape = random.choice(TEMPLATE_SHAPES)
    variants = _shape_variants(template_shape)
    alt_variants = [shape for shape in variants if shape != template_shape]
    if not alt_variants:
        return None

    template_room_idx = random.randrange(len(rooms))
    non_template_indices = [i for i in range(len(rooms)) if i != template_room_idx]
    match_count = random.randint(1, len(non_template_indices) - 1)
    match_rooms = set(random.sample(non_template_indices, match_count))

    out_grid = grid.copy()
    _place_shape(out_grid, rooms[template_room_idx]["bbox"], template_shape, TEAL)

    mismatch_shapes = _choose_mismatches(template_shape, len(non_template_indices) - match_count)
    mismatch_iter = iter(mismatch_shapes)

    colors = random.sample(list(MARKER_PALETTE), len(non_template_indices))
    rotated_match_room = random.choice(list(match_rooms))

    for color, room_idx in zip(colors, non_template_indices):
        room_bbox = rooms[room_idx]["bbox"]
        if room_idx in match_rooms:
            if room_idx == rotated_match_room:
                green_shape = random.choice(alt_variants)
            else:
                green_shape = random.choice(variants)
        else:
            green_shape = next(mismatch_iter)
        occupied = _place_shape(out_grid, room_bbox, green_shape, GREEN)
        mr, mc = _marker_cell(room_bbox, occupied)
        out_grid[mr, mc] = color

    if not _validate_generated_case(out_grid):
        return None
    return out_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chamber-matching input")
