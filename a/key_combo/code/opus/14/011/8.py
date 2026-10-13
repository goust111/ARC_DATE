from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
TARGET = 8
NON_TARGET_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
SIDES = ("top", "bottom", "left", "right")
MOVE_TO_TARGET = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
MOVE_AWAY = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
TARGET_NEIGHBOR_OFFSET = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GRID_H_RANGE = (13, 17)
GRID_W_RANGE = (13, 17)
TARGET_AREA_RANGE = (10, 15)
SPRITE_COUNT_RANGE = (2, 4)
INPUT_GAP_RANGE = (2, 6)
GEN_RETRIES = 500


SHAPE_LIBRARY = [
    {
        "cells": [(0, 0), (0, 1), (0, 2)],
        "inner": [(0, 1)],
    },
    {
        "cells": [(0, 0), (1, 0), (1, 1)],
        "inner": [(1, 0)],
    },
    {
        "cells": [(0, 0), (0, 1), (0, 2), (1, 1)],
        "inner": [(1, 1)],
    },
    {
        "cells": [(0, 0), (1, 0), (1, 1), (2, 1)],
        "inner": [(1, 0)],
    },
    {
        "cells": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "inner": [(0, 0)],
    },
    {
        "cells": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        "inner": [(0, 1), (0, 2)],
    },
    {
        "cells": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        "inner": [(1, 1)],
    },
    {
        "cells": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        "inner": [(0, 1), (1, 1)],
    },
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(pts), dtype=int))
    return comps


def _component_colors(grid: np.ndarray, coords: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid[coords[:, 0], coords[:, 1]]))


def _normalize_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _transform_cells(
    cells: list[tuple[int, int]], inner: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coords = list(cells)
    inner_set = set(inner)
    if random.random() < 0.5:
        coords = [(r, -c) for r, c in coords]
        inner_set = {(r, -c) for r, c in inner_set}
    turns = random.randrange(4)
    for _ in range(turns):
        coords = [(c, -r) for r, c in coords]
        inner_set = {(c, -r) for r, c in inner_set}
    norm_cells = _normalize_cells(coords)
    norm_inner = sorted(set(_normalize_cells(list(inner_set))))
    return norm_cells, norm_inner


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _render_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    inner: set[tuple[int, int]],
    outer_color: int,
    inner_color: int,
) -> None:
    for cell in cells:
        grid[cell] = inner_color if cell in inner else outer_color


def _landed_colors(
    cells: list[tuple[int, int]],
    inner: set[tuple[int, int]],
    outer_color: int,
) -> dict[tuple[int, int], int]:
    colors = {}
    for cell in cells:
        colors[cell] = outer_color if cell in inner else TARGET
    return colors


def _has_4_adjacent(cells: list[tuple[int, int]], target_set: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in target_set:
                return True
    return False


def _classify_side(
    sprite_bbox: tuple[int, int, int, int], target_bbox: tuple[int, int, int, int]
) -> str:
    sr0, sc0, sr1, sc1 = sprite_bbox
    tr0, tc0, tr1, tc1 = target_bbox
    if sr1 < tr0 and _spans_overlap(sc0, sc1, tc0, tc1):
        return "top"
    if sr0 > tr1 and _spans_overlap(sc0, sc1, tc0, tc1):
        return "bottom"
    if sc1 < tc0 and _spans_overlap(sr0, sr1, tr0, tr1):
        return "left"
    if sc0 > tc1 and _spans_overlap(sr0, sr1, tr0, tr1):
        return "right"
    raise AssertionError("sprite does not occupy a unique projection side")


def _slide_to_collision(
    coords: np.ndarray,
    side: str,
    target_set: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    dr, dc = MOVE_TO_TARGET[side]
    cells = [tuple(map(int, rc)) for rc in coords]
    assert not _has_4_adjacent(cells, target_set), "sprite is already touching the target"
    for step in range(1, max(shape) + 1):
        shifted = [(r + dr * step, c + dc * step) for r, c in cells]
        if any(not _in_bounds(shape, r, c) for r, c in shifted):
            break
        if any((r, c) in target_set for r, c in shifted):
            raise AssertionError("sprite reached overlap before adjacency")
        if _has_4_adjacent(shifted, target_set):
            return np.array(sorted(shifted), dtype=int), step
    raise AssertionError("sprite failed to collide with the target")


def _projected_output(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    target_components = _components_from_mask(g == TARGET)
    assert len(target_components) == 1, "expected exactly one teal target object"
    target = target_components[0]
    target_bbox = _bbox(target)
    target_set = {tuple(map(int, rc)) for rc in target}

    sprite_components = _components_from_mask((g != BG) & (g != TARGET))
    assert sprite_components, "expected at least one two-tone sprite"

    placements = []
    out = g.copy()
    out[(g != BG) & (g != TARGET)] = BG

    for comp in sprite_components:
        colors = _component_colors(g, comp)
        assert len(colors) == 2, "every sprite must be exactly two-tone"
        counts = Counter(int(g[r, c]) for r, c in comp)
        outer_color, outer_count = counts.most_common(1)[0]
        inner_colors = [color for color in colors if color != outer_color]
        assert len(inner_colors) == 1
        inner_color = inner_colors[0]
        assert outer_count > counts[inner_color], "outer color must be the strict majority"

        side = _classify_side(_bbox(comp), target_bbox)
        landed, distance = _slide_to_collision(comp, side, target_set, g.shape)
        landed_set = {tuple(map(int, rc)) for rc in landed}
        inner_set = {
            tuple(map(int, rc))
            for rc in comp
            if int(g[tuple(rc)]) == inner_color
        }
        dr = int(landed[0, 0] - comp[0, 0])
        dc = int(landed[0, 1] - comp[0, 1])
        shifted_inner = {(r + dr, c + dc) for r, c in inner_set}
        placements.append(
            {
                "side": side,
                "distance": distance,
                "outer_color": outer_color,
                "inner_color": inner_color,
                "landed": landed_set,
                "shifted_inner": shifted_inner,
            }
        )

    for item in sorted(placements, key=lambda x: x["distance"]):
        for r, c in item["landed"]:
            out[r, c] = item["outer_color"] if (r, c) in item["shifted_inner"] else TARGET

    return out, placements, target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each two-tone sprite onto the teal target, stopping at first edge contact and swapping shell/core colors.

    Input structure:
    - A black background contains one teal 4-connected target object near the center.
    - Several detached 4-connected two-tone sprites sit above, below, left, or right of the target.
    - Each sprite uses one majority outer color and one minority inner color, and its straight projection lane to the target is clear.

    Rule:
    - Find the unique teal target object and every non-teal connected sprite.
    - Determine each sprite's projection direction from which side of the target it occupies, then slide it one cell at a time toward the target.
    - Stop when the translated sprite first becomes 4-neighbor adjacent to the target, erase its original position, and recolor it so outer pixels become teal while inner pixels become the sprite's former outer color.

    Key insight:
    - The target is the only teal object in the input, and each sprite's location relative to that object determines both its motion direction and the collision point where the color swap must be applied.
    """
    out, _placements, _target = _projected_output(grid)
    return out


def _random_target_shape() -> list[tuple[int, int]]:
    for _ in range(200):
        box_h = random.randint(4, 5)
        box_w = random.randint(4, 5)
        area = random.randint(*TARGET_AREA_RANGE)
        if area >= box_h * box_w:
            continue
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        coords = np.array(sorted(cells), dtype=int)
        if len(coords) != area:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        norm = _normalize_cells(list(cells))
        norm_set = set(norm)
        nr0, nc0, nr1, nc1 = _bbox(np.array(norm, dtype=int))
        if nr0 != 0 or nc0 != 0:
            continue
        if any((0, c) not in norm_set for c in range(nc0, nc1 + 1)) and len(norm_set) == (nr1 + 1) * (nc1 + 1):
            continue
        if len(norm_set) == (nr1 + 1) * (nc1 + 1):
            continue
        top = any(r == 0 for r, _ in norm_set)
        bottom = any(r == nr1 for r, _ in norm_set)
        left = any(c == 0 for _, c in norm_set)
        right = any(c == nc1 for _, c in norm_set)
        if top and bottom and left and right:
            return norm
    raise RuntimeError("failed to build a suitable target shape")


def _make_sprite(outer_color: int, inner_color: int) -> dict:
    spec = random.choice(SHAPE_LIBRARY)
    cells, inner = _transform_cells(spec["cells"], spec["inner"])
    return {
        "rel_cells": cells,
        "rel_inner": set(inner),
        "outer_color": outer_color,
        "inner_color": inner_color,
    }


def _target_boundary_by_side(target_cells: set[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    boundary = {side: [] for side in SIDES}
    for r, c in sorted(target_cells):
        if (r - 1, c) not in target_cells:
            boundary["top"].append((r, c))
        if (r + 1, c) not in target_cells:
            boundary["bottom"].append((r, c))
        if (r, c - 1) not in target_cells:
            boundary["left"].append((r, c))
        if (r, c + 1) not in target_cells:
            boundary["right"].append((r, c))
    return boundary


def _frontier_cells(rel_cells: list[tuple[int, int]], side: str) -> list[tuple[int, int]]:
    if side == "top":
        edge = max(r for r, _ in rel_cells)
        return [(r, c) for r, c in rel_cells if r == edge]
    if side == "bottom":
        edge = min(r for r, _ in rel_cells)
        return [(r, c) for r, c in rel_cells if r == edge]
    if side == "left":
        edge = max(c for _, c in rel_cells)
        return [(r, c) for r, c in rel_cells if c == edge]
    edge = min(c for _, c in rel_cells)
    return [(r, c) for r, c in rel_cells if c == edge]


def _cells_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(cells, dtype=int)
    return _bbox(arr)


def _separated_from(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    require_gap: bool,
) -> bool:
    for cell in cells:
        if cell in occupied:
            return False
        if require_gap:
            for nbr in _neighbors4(*cell):
                if nbr in occupied:
                    return False
    return True


def _max_outward_gap(cells: list[tuple[int, int]], side: str, shape: tuple[int, int]) -> int:
    dr, dc = MOVE_AWAY[side]
    gap = 0
    while True:
        trial = [(r + dr * (gap + 1), c + dc * (gap + 1)) for r, c in cells]
        if any(not _in_bounds(shape, r, c) for r, c in trial):
            return gap
        gap += 1


def _travel_band(cells: list[tuple[int, int]], side: str, gap: int) -> set[tuple[int, int]]:
    dr, dc = MOVE_AWAY[side]
    band = set()
    for step in range(gap):
        for r, c in cells:
            band.add((r + dr * step, c + dc * step))
    return band


def _input_cells_for_gap(cells: list[tuple[int, int]], side: str, gap: int) -> list[tuple[int, int]]:
    dr, dc = MOVE_AWAY[side]
    return [(r + dr * gap, c + dc * gap) for r, c in cells]


def _candidate_landed_positions(
    sprite: dict,
    side: str,
    target_cells: set[tuple[int, int]],
    target_bbox: tuple[int, int, int, int],
    output_occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[dict]:
    candidates = []
    boundary = _target_boundary_by_side(target_cells)[side][:]
    frontier = _frontier_cells(sprite["rel_cells"], side)
    random.shuffle(boundary)
    random.shuffle(frontier)
    out_dr, out_dc = TARGET_NEIGHBOR_OFFSET[side]
    tr0, tc0, tr1, tc1 = target_bbox

    for target_cell in boundary:
        for front_cell in frontier:
            desired = (target_cell[0] + out_dr, target_cell[1] + out_dc)
            shift_r = desired[0] - front_cell[0]
            shift_c = desired[1] - front_cell[1]
            landed = _shift_cells(sprite["rel_cells"], shift_r, shift_c)
            inner = {_shift_cells([cell], shift_r, shift_c)[0] for cell in sprite["rel_inner"]}
            if any(not _in_bounds(shape, r, c) for r, c in landed):
                continue
            if any(cell in target_cells for cell in landed):
                continue
            lr0, lc0, lr1, lc1 = _cells_bbox(landed)
            if side == "top" and lr1 >= tr0:
                continue
            if side == "bottom" and lr0 <= tr1:
                continue
            if side == "left" and lc1 >= tc0:
                continue
            if side == "right" and lc0 <= tc1:
                continue
            if not _has_4_adjacent(landed, target_cells):
                continue
            away = _shift_cells(landed, *MOVE_AWAY[side])
            if any(not _in_bounds(shape, r, c) for r, c in away):
                continue
            if _has_4_adjacent(away, target_cells):
                continue
            if not _separated_from(landed, output_occupied, require_gap=False):
                continue
            max_gap = _max_outward_gap(landed, side, shape)
            low_gap, high_gap = INPUT_GAP_RANGE
            if max_gap < low_gap:
                continue
            candidates.append(
                {
                    "side": side,
                    "landed": landed,
                    "landed_inner": inner,
                    "max_gap": min(max_gap, high_gap),
                    "outer_color": sprite["outer_color"],
                    "inner_color": sprite["inner_color"],
                }
            )
    return candidates


def _assign_input_positions(items: list[dict]) -> list[dict] | None:
    order = list(range(len(items)))
    random.shuffle(order)
    chosen: dict[int, dict] = {}
    occupied: set[tuple[int, int]] = set()
    bands: dict[int, set[tuple[int, int]]] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        item = items[idx]
        gaps = list(range(INPUT_GAP_RANGE[0], item["max_gap"] + 1))
        random.shuffle(gaps)
        for gap in gaps:
            inp_cells = _input_cells_for_gap(item["landed"], item["side"], gap)
            inp_inner = {
                _input_cells_for_gap([cell], item["side"], gap)[0]
                for cell in item["landed_inner"]
            }
            if not _separated_from(inp_cells, occupied, require_gap=True):
                continue
            band = _travel_band(item["landed"], item["side"], gap)
            if any(cell in occupied for cell in band):
                continue
            if any(
                any(cell in bands.get(other_idx, set()) for cell in inp_cells)
                for other_idx in chosen
            ):
                continue
            chosen[idx] = {
                **item,
                "input": inp_cells,
                "input_inner": inp_inner,
                "gap": gap,
            }
            occupied.update(inp_cells)
            bands[idx] = band
            if backtrack(pos + 1):
                return True
            for cell in inp_cells:
                occupied.remove(cell)
            del chosen[idx]
            del bands[idx]
        return False

    if not backtrack(0):
        return None
    return [chosen[i] for i in range(len(items))]


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    target_rel = _random_target_shape()
    target_h = max(r for r, _ in target_rel) + 1
    target_w = max(c for _, c in target_rel) + 1
    target_r0 = (h - target_h) // 2
    target_c0 = (w - target_w) // 2
    target_cells = {(target_r0 + r, target_c0 + c) for r, c in target_rel}
    target_bbox = _cells_bbox(list(target_cells))

    sprite_count = random.randint(*SPRITE_COUNT_RANGE)
    sides = random.sample(list(SIDES), sprite_count)
    color_pool = random.sample(NON_TARGET_COLORS, sprite_count * 2)
    outer_colors = color_pool[:sprite_count]
    inner_colors = color_pool[sprite_count:]

    output_occupied = set(target_cells)
    landed_items = []
    for side, outer_color, inner_color in zip(sides, outer_colors, inner_colors, strict=False):
        sprite = _make_sprite(outer_color, inner_color)
        candidates = _candidate_landed_positions(
            sprite,
            side,
            target_cells,
            target_bbox,
            output_occupied,
            (h, w),
        )
        if not candidates:
            raise RuntimeError("no landed placement for sprite")
        choice = random.choice(candidates)
        output_occupied.update(choice["landed"])
        landed_items.append(choice)

    assigned = _assign_input_positions(landed_items)
    if assigned is None:
        raise RuntimeError("failed to place input sprites")

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for r, c in target_cells:
        inp[r, c] = TARGET
        out[r, c] = TARGET

    for item in assigned:
        _render_cells(
            inp,
            item["input"],
            item["input_inner"],
            item["outer_color"],
            item["inner_color"],
        )
        for cell, color in _landed_colors(
            item["landed"], item["landed_inner"], item["outer_color"]
        ).items():
            out[cell] = color

    meta = {
        "target_cells": target_cells,
        "sprites": assigned,
        "sides": set(sides),
    }
    return inp, out, meta


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    assert inp.shape == expected.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert expected.ndim == 2 and max(expected.shape) <= 30
    assert not np.array_equal(inp, expected), "input must differ from output"

    target_components = _components_from_mask(inp == TARGET)
    assert len(target_components) == 1
    target = target_components[0]
    assert TARGET_AREA_RANGE[0] <= len(target) <= TARGET_AREA_RANGE[1]
    target_set = {tuple(map(int, rc)) for rc in target}
    target_bbox = _bbox(target)

    sprite_components = _components_from_mask((inp != BG) & (inp != TARGET))
    assert len(sprite_components) == len(meta["sprites"])

    seen_outer = set()
    seen_sides = set()
    predicted_signatures = []
    for comp in sprite_components:
        counts = Counter(int(inp[r, c]) for r, c in comp)
        assert len(counts) == 2
        outer_color, outer_count = counts.most_common(1)[0]
        inner_color = next(color for color in counts if color != outer_color)
        assert outer_count > counts[inner_color]
        seen_outer.add(outer_color)
        side = _classify_side(_bbox(comp), target_bbox)
        seen_sides.add(side)
        landed, distance = _slide_to_collision(comp, side, target_set, inp.shape)
        assert distance >= INPUT_GAP_RANGE[0]
        inner_coords = {
            tuple(map(int, rc))
            for rc in comp
            if int(inp[tuple(rc)]) == inner_color
        }
        dr = int(landed[0, 0] - comp[0, 0])
        dc = int(landed[0, 1] - comp[0, 1])
        landed_inner = {(r + dr, c + dc) for r, c in inner_coords}
        predicted_signatures.append(
            frozenset(
                (r, c, outer_color if (r, c) in landed_inner else TARGET)
                for r, c in map(tuple, landed.tolist())
            )
        )

    assert len(seen_outer) == len(sprite_components), "outer colors must be unique"
    assert seen_sides == meta["sides"], "side coverage mismatch"
    expected_signatures = []
    for item in meta["sprites"]:
        expected_signatures.append(
            frozenset(
                (r, c, item["outer_color"] if (r, c) in item["landed_inner"] else TARGET)
                for r, c in item["landed"]
            )
        )
    assert set(predicted_signatures) == set(expected_signatures)

    solved = main(inp)
    assert np.array_equal(solved, expected), "solver must match constructed output"
    assert np.array_equal(solved[target[:, 0], target[:, 1]], np.full(len(target), TARGET)), "target changed"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected, meta = _build_case()
            _validate_case(inp, expected, meta)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
