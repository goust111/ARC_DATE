from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 800
PLACEMENT_ATTEMPTS = 300
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = DIR4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

ARROW_LAYOUTS = {
    (0, 1): {
        "shape": frozenset({(1, 1), (0, 0), (2, 0)}),
        "tip": (1, 1),
        "wing_offsets": ((-1, -1), (1, -1)),
    },
    (0, -1): {
        "shape": frozenset({(1, 0), (0, 1), (2, 1)}),
        "tip": (1, 0),
        "wing_offsets": ((-1, 1), (1, 1)),
    },
    (-1, 0): {
        "shape": frozenset({(0, 1), (1, 0), (1, 2)}),
        "tip": (0, 1),
        "wing_offsets": ((1, -1), (1, 1)),
    },
    (1, 0): {
        "shape": frozenset({(1, 1), (0, 0), (0, 2)}),
        "tip": (1, 1),
        "wing_offsets": ((-1, -1), (-1, 1)),
    },
}
ARROW_PATTERNS = {
    layout["shape"]: {"dir": direction, "tip": layout["tip"]}
    for direction, layout in ARROW_LAYOUTS.items()
}

BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
)


def _components_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    components: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue: deque[tuple[int, int]] = deque([(r, c)])
        cells: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            cells.append((cr, cc))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(cells)
    return components


def _normalize_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _match_arrow(cells: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(cells) != 3:
        return None
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    pattern = frozenset((r - min_r, c - min_c) for r, c in cells)
    info = ARROW_PATTERNS.get(pattern)
    if info is None:
        return None
    tip_rel = info["tip"]
    tip = (min_r + int(tip_rel[0]), min_c + int(tip_rel[1]))
    return {"dir": info["dir"], "tip": tip}


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows: list[dict[str, object]] = []
    arrow_cells: set[tuple[int, int]] = set()
    for color in np.unique(g):
        color = int(color)
        if color == BLACK:
            continue
        for comp in _components_mask(g == color, connectivity=8):
            info = _match_arrow(comp)
            if info is None:
                continue
            cells = tuple(sorted(comp))
            arrows.append(
                {
                    "cells": cells,
                    "cell_set": set(cells),
                    "color": color,
                    "tip": info["tip"],
                    "dir": info["dir"],
                }
            )
            arrow_cells.update(cells)

    assert arrows, "expected at least one arrow"
    arrows.sort(key=lambda item: (item["tip"][0], item["tip"][1], item["color"]))

    arrow_mask = np.zeros_like(g, dtype=bool)
    for r, c in arrow_cells:
        arrow_mask[r, c] = True

    objects: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BLACK:
            continue
        mask = (g == color) & ~arrow_mask
        for comp in _components_mask(mask, connectivity=4):
            cells = tuple(sorted(comp))
            objects.append(
                {
                    "cells": cells,
                    "cell_set": set(cells),
                    "color": color,
                }
            )

    assert objects, "expected at least one target object"
    objects.sort(key=lambda item: (item["cells"][0][0], item["cells"][0][1], item["color"]))
    for obj in objects:
        size = len(obj["cells"])
        assert 2 <= size <= 5, "objects must be 2-5 pixels"

    cell_to_object: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        for cell in obj["cells"]:
            cell_to_object[cell] = idx

    h, w = g.shape
    assignments: list[int] = []
    for arrow in arrows:
        r, c = arrow["tip"]
        dr, dc = arrow["dir"]
        target = None
        nr = r + dr
        nc = c + dc
        while 0 <= nr < h and 0 <= nc < w:
            target = cell_to_object.get((nr, nc))
            if target is not None:
                break
            nr += dr
            nc += dc
        assert target is not None, "each arrow must hit an object"
        assignments.append(int(target))

    assert len(arrows) == len(objects), "objects and arrows must pair one-to-one"
    assert len(set(assignments)) == len(assignments), "each object must be targeted once"
    return arrows, objects, assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each 3-pixel arrow's color with the color of the object it points to.

    Input structure:
    - The background is black.
    - Several small monochrome filled objects appear as 2-5 pixel 4-connected shapes.
    - Each object has one nearby monochrome 3-pixel arrow of a different color.
    - Every arrow is one tip pixel plus two diagonal wing pixels arranged as a left, right, up, or down arrow.

    Rule:
    - Detect every 3-pixel arrow from its geometry and recover its pointing direction from the tip/wing layout.
    - Cast a ray forward from the arrow tip and find the first filled object encountered on that line.
    - Recolor the whole target object to the arrow's color.
    - Recolor the three arrow pixels to the target object's original color, leaving all shapes and positions unchanged.

    Key insight:
    - The task requires three linked steps: recognize the disconnected 3-pixel arrow motif, infer its facing direction, then use line-of-sight to swap colors rather than merely copying one color onto the other.
    """
    g = np.asarray(grid, dtype=int)
    arrows, objects, assignments = _analyze_grid(g)

    out = g.copy()
    for arrow, object_idx in zip(arrows, assignments):
        target = objects[object_idx]
        arrow_color = int(arrow["color"])
        object_color = int(target["color"])
        for r, c in target["cells"]:
            out[r, c] = arrow_color
        for r, c in arrow["cells"]:
            out[r, c] = object_color
    return out


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = list(shape)
    for _ in range(random.randrange(4)):
        cells = [(c, -r) for r, c in cells]
    if random.random() < 0.5:
        cells = [(r, -c) for r, c in cells]
    return _normalize_cells(tuple(cells))


def _hit_candidates(shape: tuple[tuple[int, int], ...], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction[0] == 0:
        rows: dict[int, list[int]] = {}
        for r, c in shape:
            rows.setdefault(r, []).append(c)
        if direction == (0, 1):
            return [(r, min(cols)) for r, cols in sorted(rows.items())]
        return [(r, max(cols)) for r, cols in sorted(rows.items())]

    cols: dict[int, list[int]] = {}
    for r, c in shape:
        cols.setdefault(c, []).append(r)
    if direction == (-1, 0):
        return [(max(rows), c) for c, rows in sorted(cols.items())]
    return [(min(rows), c) for c, rows in sorted(cols.items())]


def _arrow_cells_from_tip(tip: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    wing_offsets = ARROW_LAYOUTS[direction]["wing_offsets"]
    return tuple(sorted((tip, *((tip[0] + dr, tip[1] + dc) for dr, dc in wing_offsets))))


def _tip_from_hit(hit: tuple[int, int], direction: tuple[int, int], gap: int) -> tuple[int, int]:
    return hit[0] - direction[0] * gap, hit[1] - direction[1] * gap


def _path_cells(tip: tuple[int, int], direction: tuple[int, int], gap: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(tip[0] + dr * step, tip[1] + dc * step) for step in range(1, gap)]


def _expanded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _in_bounds(shape: tuple[int, int], cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _pair_colors() -> tuple[int, int]:
    object_color = random.choice(COLORS)
    arrow_color = random.choice([color for color in COLORS if color != object_color])
    return object_color, arrow_color


def _render_expected(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for placement in placements:
        object_color = int(placement["object_color"])
        arrow_color = int(placement["arrow_color"])
        for r, c in placement["object_cells"]:
            out[r, c] = arrow_color
        for r, c in placement["arrow_cells"]:
            out[r, c] = object_color
    return out


def _try_place_pair(
    shape: tuple[int, int],
    forbidden_entities: set[tuple[int, int]],
    occupied_entities: set[tuple[int, int]],
    reserved_corridors: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    direction = random.choice(tuple(ARROW_LAYOUTS))
    sprite = _transform_shape(random.choice(BASE_SHAPES))
    candidates = _hit_candidates(sprite, direction)
    if not candidates:
        return None
    hit_rel = random.choice(candidates)
    sprite_h, sprite_w = _shape_dims(sprite)
    object_color, arrow_color = _pair_colors()

    for _ in range(PLACEMENT_ATTEMPTS):
        origin_r = random.randint(0, h - sprite_h)
        origin_c = random.randint(0, w - sprite_w)
        object_cells = tuple(sorted((origin_r + r, origin_c + c) for r, c in sprite))
        hit = (origin_r + hit_rel[0], origin_c + hit_rel[1])
        gap = random.randint(2, 6)
        tip = _tip_from_hit(hit, direction, gap)
        arrow_cells = _arrow_cells_from_tip(tip, direction)
        entity_cells = set(object_cells) | set(arrow_cells)
        if len(entity_cells) != len(object_cells) + len(arrow_cells):
            continue
        if not _in_bounds(shape, object_cells) or not _in_bounds(shape, arrow_cells):
            continue
        if any(cell in forbidden_entities for cell in entity_cells):
            continue
        if any(cell in reserved_corridors for cell in entity_cells):
            continue

        path = _path_cells(tip, direction, gap)
        if not _in_bounds(shape, path):
            continue
        if any(cell in occupied_entities for cell in path):
            continue

        return {
            "direction": direction,
            "object_color": object_color,
            "arrow_color": arrow_color,
            "object_cells": object_cells,
            "arrow_cells": arrow_cells,
            "path_cells": tuple(path),
        }
    return None


def _sample_grid() -> tuple[np.ndarray, list[dict[str, object]]] | None:
    height = random.randint(14, 24)
    width = random.randint(14, 24)
    max_pairs = 5 if height * width >= 320 else 4 if height * width >= 220 else 3
    pair_count = random.randint(2, max_pairs)

    grid = np.full((height, width), BLACK, dtype=int)
    occupied_entities: set[tuple[int, int]] = set()
    forbidden_entities: set[tuple[int, int]] = set()
    reserved_corridors: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    for _ in range(pair_count):
        placement = _try_place_pair(
            grid.shape,
            forbidden_entities=forbidden_entities,
            occupied_entities=occupied_entities,
            reserved_corridors=reserved_corridors,
        )
        if placement is None:
            return None

        for r, c in placement["object_cells"]:
            grid[r, c] = int(placement["object_color"])
        for r, c in placement["arrow_cells"]:
            grid[r, c] = int(placement["arrow_color"])

        entity_cells = set(placement["object_cells"]) | set(placement["arrow_cells"])
        occupied_entities.update(entity_cells)
        forbidden_entities.update(_expanded(entity_cells, height, width))
        reserved_corridors.update(placement["path_cells"])
        placements.append(placement)

    return grid, placements


def _validate_generated(grid: np.ndarray, placements: list[dict[str, object]]) -> bool:
    expected = _render_expected(grid, placements)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    try:
        arrows, objects, assignments = _analyze_grid(grid)
    except AssertionError:
        return False

    if len(arrows) != len(placements) or len(objects) != len(placements):
        return False

    detected_arrow_sets = {frozenset(arrow["cells"]) for arrow in arrows}
    detected_object_sets = {frozenset(obj["cells"]) for obj in objects}
    planned_arrow_sets = {frozenset(placement["arrow_cells"]) for placement in placements}
    planned_object_sets = {frozenset(placement["object_cells"]) for placement in placements}
    if detected_arrow_sets != planned_arrow_sets or detected_object_sets != planned_object_sets:
        return False

    assigned_pairs = {
        (frozenset(arrows[idx]["cells"]), frozenset(objects[obj_idx]["cells"]))
        for idx, obj_idx in enumerate(assignments)
    }
    planned_pairs = {
        (frozenset(placement["arrow_cells"]), frozenset(placement["object_cells"]))
        for placement in placements
    }
    if assigned_pairs != planned_pairs:
        return False

    return not np.array_equal(grid, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sampled = _sample_grid()
        if sampled is None:
            continue
        grid, placements = sampled
        if _validate_generated(grid, placements):
            return grid
    raise RuntimeError("failed to generate a valid input")
