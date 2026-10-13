from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


Coord = tuple[int, int]
Vec = tuple[int, int]


def _add(point: Coord, vec: Vec, k: int = 1) -> Coord:
    return point[0] + vec[0] * k, point[1] + vec[1] * k


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "cells": cells, "cell_set": set(cells), "area": len(cells)})
    return components


CARDINALS: dict[str, tuple[Vec, Vec]] = {
    "right": ((0, 1), (1, 0)),
    "left": ((0, -1), (1, 0)),
    "down": ((1, 0), (0, 1)),
    "up": ((-1, 0), (0, 1)),
}


def _build_arrow_templates() -> dict[frozenset[Coord], dict[str, object]]:
    templates: dict[frozenset[Coord], dict[str, object]] = {}
    for name, (direction, perp) in CARDINALS.items():
        tail = (0, 0)
        rod_end = _add(tail, direction, 2)
        tip = _add(tail, direction, 3)
        raw = {
            tail,
            _add(tail, direction, 1),
            rod_end,
            tip,
            _add(rod_end, perp, 1),
            _add(rod_end, perp, -1),
        }
        min_r = min(r for r, _ in raw)
        min_c = min(c for _, c in raw)
        norm = frozenset((r - min_r, c - min_c) for r, c in raw)
        templates[norm] = {
            "name": name,
            "direction": direction,
            "tail_offset": (tail[0] - min_r, tail[1] - min_c),
            "tip_offset": (tip[0] - min_r, tip[1] - min_c),
        }
    return templates


ARROW_TEMPLATES = _build_arrow_templates()


def _classify_arrow(cells: list[Coord]) -> dict[str, object] | None:
    if len(cells) != 6:
        return None
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = frozenset((r - min_r, c - min_c) for r, c in cells)
    template = ARROW_TEMPLATES.get(norm)
    if template is None:
        return None
    tail_offset = template["tail_offset"]
    tip_offset = template["tip_offset"]
    return {
        "name": template["name"],
        "direction": template["direction"],
        "tail": (min_r + int(tail_offset[0]), min_c + int(tail_offset[1])),
        "tip": (min_r + int(tip_offset[0]), min_c + int(tip_offset[1])),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each monochrome arrow, follow its pointing line to the nearest target object,
    optionally read a relay-color dot on that line, then stamp the target shape behind the
    arrow tail in the relay color (or arrow color if no relay dot exists).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    h, w = g.shape

    components = _find_components(g, bg)
    cell_to_component: dict[Coord, int] = {}
    arrows: list[dict[str, object]] = []
    for idx, component in enumerate(components):
        cells = component["cells"]
        assert isinstance(cells, list)
        for cell in cells:
            cell_to_component[cell] = idx
        arrow = _classify_arrow(cells)
        if arrow is not None:
            arrow["component_id"] = idx
            arrow["color"] = int(component["color"])
            arrow["cells"] = cells
            arrows.append(arrow)

    assert arrows, "expected at least one arrow"
    arrow_ids = {int(arrow["component_id"]) for arrow in arrows}

    erase: set[Coord] = set()
    paint_ops: list[tuple[list[Coord], int]] = []
    used_targets: set[int] = set()

    for arrow in arrows:
        dr, dc = arrow["direction"]
        tip_r, tip_c = arrow["tip"]
        arrow_color = int(arrow["color"])

        relay_color: int | None = None
        target_component: dict[str, object] | None = None
        anchor: Coord | None = None

        r, c = tip_r, tip_c
        while True:
            r += dr
            c += dc
            assert 0 <= r < h and 0 <= c < w, "arrow must hit a target before leaving the grid"
            value = int(g[r, c])
            if value == bg:
                continue

            comp_id = cell_to_component[(r, c)]
            assert comp_id not in arrow_ids, "arrow sightline cannot cross another arrow"
            component = components[comp_id]
            size = int(component["area"])

            if size == 1:
                assert relay_color is None, "expected at most one relay dot on the path"
                relay_color = int(component["color"])
                erase.update(component["cells"])
                continue

            assert size in (3, 4), "target object must be a 3- or 4-cell component"
            assert comp_id not in used_targets, "each target must belong to exactly one arrow"
            used_targets.add(comp_id)
            target_component = component
            anchor = (r, c)
            erase.update(component["cells"])
            break

        assert target_component is not None and anchor is not None
        tail = arrow["tail"]
        dest_anchor = (int(tail[0]) - dr, int(tail[1]) - dc)
        paint_color = relay_color if relay_color is not None else arrow_color

        translated: list[Coord] = []
        for tr, tc in target_component["cells"]:
            nr = dest_anchor[0] + (tr - anchor[0])
            nc = dest_anchor[1] + (tc - anchor[1])
            assert 0 <= nr < h and 0 <= nc < w, "copied target must fit behind the tail"
            translated.append((nr, nc))
        paint_ops.append((translated, paint_color))

    out = g.copy()
    for r, c in erase:
        out[r, c] = bg
    for cells, color in paint_ops:
        for r, c in cells:
            assert int(out[r, c]) == bg, "copied shape must land on empty cells"
            out[r, c] = color
    return out


TARGET_TEMPLATES: list[tuple[tuple[int, int], ...]] = [
    ((0, 0), (0, 1), (-1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (-1, 1), (-1, 2)),
    ((0, 0), (0, 1), (0, 2), (-1, 2)),
    ((0, 0), (0, 1), (-1, 1), (-2, 1)),
    ((0, 0), (0, 1), (0, 2), (-1, 1)),
    ((0, 0), (0, 1), (-1, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
]


def _target_cells(anchor: Coord, direction: Vec, perp: Vec, offsets: tuple[tuple[int, int], ...]) -> set[Coord]:
    cells = set()
    for along, cross in offsets:
        cells.add(
            (
                anchor[0] + along * direction[0] + cross * perp[0],
                anchor[1] + along * direction[1] + cross * perp[1],
            )
        )
    return cells


def _arrow_cells(tail: Coord, direction: Vec, perp: Vec) -> set[Coord]:
    rod_end = _add(tail, direction, 2)
    return {
        tail,
        _add(tail, direction, 1),
        rod_end,
        _add(tail, direction, 3),
        _add(rod_end, perp, 1),
        _add(rod_end, perp, -1),
    }


def _padding(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    padded: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    padded.add((nr, nc))
    return padded


def _scenario_spec(
    shape: tuple[int, int],
    tail: Coord,
    direction: Vec,
    perp: Vec,
    offsets: tuple[tuple[int, int], ...],
    distance: int,
    relay_step: int | None,
) -> dict[str, object] | None:
    tip = _add(tail, direction, 3)
    anchor = _add(tip, direction, distance)
    copy_anchor = _add(tail, direction, -1)

    arrow = _arrow_cells(tail, direction, perp)
    target = _target_cells(anchor, direction, perp, offsets)
    copy = _target_cells(copy_anchor, direction, perp, offsets)
    path = {_add(tip, direction, step) for step in range(1, distance)}
    relay = {_add(tip, direction, relay_step)} if relay_step is not None else set()

    relevant = arrow | target | copy | path | relay
    if not all(_in_bounds(shape, cell) for cell in relevant):
        return None

    if len(arrow | target | relay) != len(arrow) + len(target) + len(relay):
        return None
    if copy & (arrow | target | relay):
        return None
    if (path - relay) & (arrow | target | relay | copy):
        return None

    return {
        "arrow": arrow,
        "target": target,
        "copy": copy,
        "path": path,
        "relay": relay,
        "tail": tail,
        "tip": tip,
        "anchor": anchor,
        "blocked": _padding(relevant, shape),
    }


def _relay_flags(count: int, rng: random.Random) -> list[bool]:
    if count == 1:
        return [rng.random() < 0.5]
    flags = [True, False]
    while len(flags) < count:
        flags.append(rng.random() < 0.5)
    rng.shuffle(flags)
    return flags


def _grid_size(count: int, rng: random.Random) -> tuple[int, int]:
    ranges = {
        1: (14, 18),
        2: (18, 22),
        3: (21, 24),
    }
    lo, hi = ranges[count]
    return rng.randint(lo, hi), rng.randint(lo, hi)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    scenario_count = rng.choices([1, 2, 3], weights=[2, 4, 4], k=1)[0]
    h, w = _grid_size(scenario_count, rng)
    shape = (h, w)
    relay_flags = _relay_flags(scenario_count, rng)

    colors = COLORS[:]
    rng.shuffle(colors)
    color_idx = 0

    grid = np.full(shape, BG, dtype=int)
    blocked: set[Coord] = set()
    scenarios: list[dict[str, object]] = []

    for needs_relay in relay_flags:
        placed = False
        for _ in range(250):
            name = rng.choice(list(CARDINALS.keys()))
            direction, perp = CARDINALS[name]
            base_offsets = rng.choice(TARGET_TEMPLATES)
            mirror = -1 if rng.random() < 0.5 else 1
            offsets = tuple((along, cross * mirror) for along, cross in base_offsets)

            min_distance = 2 if needs_relay else 1
            max_distance = min_distance + 4
            distance = rng.randint(min_distance, max_distance)
            relay_step = rng.randint(1, distance - 1) if needs_relay else None

            tail = (rng.randrange(h), rng.randrange(w))
            scenario = _scenario_spec(shape, tail, direction, perp, offsets, distance, relay_step)
            if scenario is None:
                continue
            if scenario["blocked"] & blocked:
                continue

            arrow_color = colors[color_idx]
            target_color = colors[color_idx + 1]
            color_idx += 2
            relay_color = None
            if needs_relay:
                relay_color = colors[color_idx]
                color_idx += 1

            scenario["arrow_color"] = arrow_color
            scenario["target_color"] = target_color
            scenario["relay_color"] = relay_color
            scenarios.append(scenario)
            blocked |= scenario["blocked"]
            placed = True
            break
        if not placed:
            return None

    for scenario in scenarios:
        arrow_color = int(scenario["arrow_color"])
        target_color = int(scenario["target_color"])
        relay_color = scenario["relay_color"]

        for r, c in scenario["arrow"]:
            grid[r, c] = arrow_color
        for r, c in scenario["target"]:
            grid[r, c] = target_color
        if relay_color is not None:
            for r, c in scenario["relay"]:
                grid[r, c] = int(relay_color)

    if Counter(map(int, grid.ravel()))[BG] <= (h * w) // 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
