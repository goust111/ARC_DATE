from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
TRAIL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

Coord = tuple[int, int]

ARROW_SPECS = {
    "up": {
        "dir": (-1, 0),
        "tip_local": (0, 1),
        "mask": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    },
    "right": {
        "dir": (0, 1),
        "tip_local": (1, 2),
        "mask": {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
    },
    "down": {
        "dir": (1, 0),
        "tip_local": (2, 1),
        "mask": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    },
    "left": {
        "dir": (0, -1),
        "tip_local": (1, 0),
        "mask": {(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)},
    },
}
ARROW_BY_MASK = {
    frozenset(spec["mask"]): {
        "name": name,
        "dir": spec["dir"],
        "tip_local": spec["tip_local"],
    }
    for name, spec in ARROW_SPECS.items()
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _normalize(cells: set[Coord]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_size(norm: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in norm) + 1, max(c for _, c in norm) + 1


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            cells: set[Coord] = set()

            while queue:
                cr, cc = queue.popleft()
                cells.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            r0, c0, r1, c1 = _bbox(cells)
            norm = _normalize(cells)
            coords = np.asarray(sorted(cells), dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "norm": norm,
                    "area": len(cells),
                }
            )

    components.sort(key=lambda comp: tuple(comp["bbox"]))  # deterministic processing
    return components


def _classify_arrow(comp: dict[str, object]) -> dict[str, object] | None:
    if int(comp["area"]) != 5:
        return None
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    spec = ARROW_BY_MASK.get(frozenset(comp["norm"]))
    if spec is None:
        return None

    tip_local = spec["tip_local"]
    tip = (r0 + tip_local[0], c0 + tip_local[1])
    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox_origin": (r0, c0),
        "tip": tip,
        "tip_local": tip_local,
        "dir": spec["dir"],
        "name": spec["name"],
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[Coord, int]]:
    arrows: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    target_by_cell: dict[Coord, int] = {}

    for comp in _find_components(grid):
        arrow = _classify_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
            continue

        assert int(comp["area"]) >= 2, "targets must have area >= 2"
        target_idx = len(targets)
        targets.append(
            {
                "color": int(comp["color"]),
                "norm": tuple(comp["norm"]),
                "bbox": comp["bbox"],
                "cells": set(comp["cells"]),
            }
        )
        for cell in comp["cells"]:
            target_by_cell[cell] = target_idx

    assert arrows, "expected at least one arrow"
    assert targets, "expected at least one target"
    return arrows, targets, target_by_cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each colored arrow straight toward its matching target, leaving a grey trail and turning the moved arrow into the target shape at the stop point.

    Input structure:
    - The background is black.
    - Some monochrome 4-connected components are 3x3 arrow sprites: a five-pixel
      cardinal arrow whose tip direction is encoded by the occupied cells in its 3x3 box.
    - The remaining monochrome components are non-arrow target shapes with area at
      least 2; each arrow has exactly one same-color target somewhere ahead on its ray.

    Rule:
    - Detect each arrow's color, 3x3 origin, and pointing direction, then erase the
      original arrow cells to black.
    - Starting from the cell immediately beyond the tip, march cell by cell in the
      pointing direction, painting every traversed black cell grey until the first
      same-color target cell is encountered.
    - When the tip reaches that target cell, paste a copy of the hit target's
      bbox-normalized shape at the arrow's translated resting position: align the
      copy's top-left with the moved arrow's translated 3x3 top-left.
    - Keep the original target component in place and process all arrows independently.

    Key insight:
    - The arrow sprite encodes motion direction, while the color and the first
      same-color component on that ray determine both where the motion stops and
      which shape gets duplicated at the destination.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows, targets, target_by_cell = _parse_scene(g)
    out = g.copy()

    for arrow in arrows:
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = arrow["dir"]
        tip_r, tip_c = arrow["tip"]
        step = 0

        while True:
            step += 1
            r = tip_r + dr * step
            c = tip_c + dc * step
            assert _in_bounds(g.shape, r, c), "ray must hit a target before leaving the grid"

            value = int(g[r, c])
            if value == BG:
                out[r, c] = TRAIL
                continue

            assert value == color, "first blocker on the ray must be the matching target color"
            target_idx = target_by_cell.get((r, c))
            assert target_idx is not None, "ray must stop on a target, not another arrow"
            target = targets[target_idx]
            assert int(target["color"]) == color

            origin_r = r - int(arrow["tip_local"][0])
            origin_c = c - int(arrow["tip_local"][1])
            for tr, tc in target["norm"]:
                rr = origin_r + tr
                cc = origin_c + tc
                assert _in_bounds(g.shape, rr, cc), "duplicated target must stay in bounds"
                out[rr, cc] = color
            break

    return out


def _rotate_shape(norm: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, w = _shape_size(norm)
    return _normalize({(c, h - 1 - r) for r, c in norm})


def _reflect_shape(norm: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _h, w = _shape_size(norm)
    return _normalize({(r, w - 1 - c) for r, c in norm})


def _random_target_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        target_size = random.randint(2, 6)
        seed = (random.randrange(4), random.randrange(4))
        cells = {seed}
        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_size:
            continue

        norm = _normalize(cells)
        transformed = norm
        for _ in range(random.randint(0, 3)):
            transformed = _rotate_shape(transformed)
        if random.random() < 0.5:
            transformed = _reflect_shape(transformed)

        if frozenset(transformed) in ARROW_BY_MASK:
            continue
        return transformed

    raise RuntimeError("failed to sample a non-arrow target shape")


def _contact_candidates(
    norm: tuple[tuple[int, int], ...],
    direction: str,
) -> list[Coord]:
    cells = set(norm)
    if direction == "right":
        return [(r, c) for r, c in norm if (r, c - 1) not in cells]
    if direction == "left":
        return [(r, c) for r, c in norm if (r, c + 1) not in cells]
    if direction == "up":
        return [(r, c) for r, c in norm if (r + 1, c) not in cells]
    if direction == "down":
        return [(r, c) for r, c in norm if (r - 1, c) not in cells]
    raise AssertionError(f"unknown direction: {direction}")


def _shifted_cells(norm: tuple[tuple[int, int], ...], origin: Coord) -> set[Coord]:
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r, c in norm}


def _trail_cells(tip: Coord, direction: Coord, distance: int) -> set[Coord]:
    dr, dc = direction
    return {(tip[0] + dr * step, tip[1] + dc * step) for step in range(1, distance)}


def _expanded(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _fits(cells: set[Coord], shape: tuple[int, int]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in cells)


def _render_input(
    shape: tuple[int, int],
    placements: list[dict[str, object]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rec in placements:
        color = int(rec["color"])
        for r, c in rec["arrow_cells"]:
            grid[r, c] = color
        for r, c in rec["target_cells"]:
            grid[r, c] = color
    return grid


def _render_expected(
    inp: np.ndarray,
    placements: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for rec in placements:
        for r, c in rec["arrow_cells"]:
            out[r, c] = BG
    for rec in placements:
        for r, c in rec["trail_cells"]:
            out[r, c] = TRAIL
        color = int(rec["color"])
        for r, c in rec["copy_cells"]:
            out[r, c] = color
    return out


def _direction_list(pair_count: int) -> list[str]:
    base = list(ARROW_SPECS)
    random.shuffle(base)
    directions = [base[i % 4] for i in range(pair_count)]
    random.shuffle(directions)
    return directions


def _sample_pair(
    grid_shape: tuple[int, int],
    direction: str,
    color: int,
    blocked: set[Coord],
) -> dict[str, object] | None:
    arrow_spec = ARROW_SPECS[direction]
    arrow_norm = tuple(sorted(arrow_spec["mask"]))
    dr, dc = arrow_spec["dir"]
    tip_local = arrow_spec["tip_local"]

    for _ in range(400):
        target_norm = _random_target_shape()
        contact = random.choice(_contact_candidates(target_norm, direction))
        target_h, target_w = _shape_size(target_norm)
        distance = random.randint(3, 11)
        arrow_origin = (
            random.randint(0, grid_shape[0] - 3),
            random.randint(0, grid_shape[1] - 3),
        )
        tip = (arrow_origin[0] + tip_local[0], arrow_origin[1] + tip_local[1])
        hit = (tip[0] + dr * distance, tip[1] + dc * distance)
        copy_origin = (hit[0] - tip_local[0], hit[1] - tip_local[1])
        target_origin = (hit[0] - contact[0], hit[1] - contact[1])

        arrow_cells = _shifted_cells(arrow_norm, arrow_origin)
        target_cells = _shifted_cells(target_norm, target_origin)
        copy_cells = _shifted_cells(target_norm, copy_origin)
        trail_cells = _trail_cells(tip, (dr, dc), distance)

        if not _fits(target_cells | copy_cells | trail_cells, grid_shape):
            continue
        if arrow_cells & target_cells:
            continue
        if arrow_cells & copy_cells:
            continue
        if target_cells & copy_cells:
            continue
        if trail_cells & (arrow_cells | target_cells | copy_cells):
            continue

        used = arrow_cells | target_cells | trail_cells | copy_cells
        if used & blocked:
            continue

        bbox_h, bbox_w = target_h, target_w
        # Keep the translated copy visibly away from the grid edge and the original target.
        if copy_origin[0] + bbox_h > grid_shape[0] or copy_origin[1] + bbox_w > grid_shape[1]:
            continue
        if target_origin[0] + bbox_h > grid_shape[0] or target_origin[1] + bbox_w > grid_shape[1]:
            continue

        return {
            "color": color,
            "direction": direction,
            "arrow_cells": arrow_cells,
            "target_cells": target_cells,
            "copy_cells": copy_cells,
            "trail_cells": trail_cells,
        }

    return None


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 4)
    h = random.randint(max(14, 8 + pair_count * 2), 30)
    w = random.randint(max(14, 8 + pair_count * 2), 30)
    colors = random.sample(COLOR_POOL, pair_count)
    directions = _direction_list(pair_count)

    placements: list[dict[str, object]] = []
    blocked: set[Coord] = set()

    for direction, color in zip(directions, colors, strict=False):
        rec = _sample_pair((h, w), direction, color, blocked)
        if rec is None:
            return None
        placements.append(rec)
        blocked |= _expanded(
            rec["arrow_cells"] | rec["target_cells"] | rec["trail_cells"] | rec["copy_cells"],
            (h, w),
        )

    inp = _render_input((h, w), placements)
    expected = _render_expected(inp, placements)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-and-target scene")
