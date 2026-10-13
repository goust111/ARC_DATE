from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
AZURE = 8
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7)

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
PERPS = {
    (0, 1): ((-1, 0), (1, 0)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (-1, 0): ((0, -1), (0, 1)),
}
AXIS = {(0, 1): "h", (0, -1): "h", (1, 0): "v", (-1, 0): "v"}

TAIL_MIN = 1
TAIL_MAX = 5
GRID_MIN = 12
GRID_MAX = 22
ARROW_COUNT_MIN = 2
ARROW_COUNT_MAX = 5
MAX_GENERATE_ATTEMPTS = 400
MAX_PLACE_ATTEMPTS = 250


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _add(p: tuple[int, int], delta: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return int(p[0] + delta[0] * k), int(p[1] + delta[1] * k)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if not _in_bounds(mask.shape, nr, nc):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _line_endpoints(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(cells) == 1:
        return [next(iter(cells))]
    endpoints = []
    for r, c in cells:
        degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        if degree == 1:
            endpoints.append((r, c))
    return endpoints


def _parse_arrow(grid: np.ndarray, coords: set[tuple[int, int]]) -> dict[str, object]:
    azure = {p for p in coords if int(grid[p]) == AZURE}
    tail = coords - azure
    assert len(azure) == 3, "each arrow must have exactly three azure head pixels"
    assert tail, "each arrow must have a non-empty tail"

    tail_colors = {int(grid[p]) for p in tail}
    assert len(tail_colors) == 1, "tail must be monochrome"
    tail_color = tail_colors.pop()
    assert tail_color not in (BG, AZURE), "tail color must differ from bg/head"

    tail_rows = {r for r, _c in tail}
    tail_cols = {c for _r, c in tail}
    assert len(tail_rows) == 1 or len(tail_cols) == 1, "tail must be straight"
    if len(tail_rows) == 1:
        row = next(iter(tail_rows))
        cols = sorted(c for _r, c in tail)
        assert tail == {(row, c) for c in range(cols[0], cols[-1] + 1)}, "tail must be contiguous"
    else:
        col = next(iter(tail_cols))
        rows = sorted(r for r, _c in tail)
        assert tail == {(r, col) for r in range(rows[0], rows[-1] + 1)}, "tail must be contiguous"

    endpoints = _line_endpoints(tail)
    near_head: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for endpoint in endpoints:
        adj_azure = [(nr, nc) for nr, nc in _neighbors4(*endpoint) if (nr, nc) in azure]
        if len(adj_azure) == 1:
            near_head.append((endpoint, adj_azure[0]))
    assert len(near_head) == 1, "tail must have exactly one endpoint attached to the head"

    tail_end, hinge = near_head[0]
    direction = (int(hinge[0] - tail_end[0]), int(hinge[1] - tail_end[1]))
    assert direction in DIRS, "arrow direction must be cardinal"

    tail_len = len(tail)
    expected_tail = {_add(hinge, direction, -step) for step in range(1, tail_len + 1)}
    assert tail == expected_tail, "tail must extend directly away from the head"

    tip = _add(hinge, direction)
    assert tip in azure, "head must have a forward tip pixel"

    flank_cells = list(azure - {hinge, tip})
    assert len(flank_cells) == 1, "head must have one flank pixel"
    flank = flank_cells[0]
    flank_delta = (int(flank[0] - hinge[0]), int(flank[1] - hinge[1]))
    assert flank_delta in PERPS[direction], "head flank must be perpendicular to direction"

    return {
        "coords": set(coords),
        "hinge": hinge,
        "tip": tip,
        "flank": flank,
        "direction": direction,
        "axis": AXIS[direction],
        "tail_len": int(tail_len),
        "color": int(tail_color),
        "tail": set(tail),
        "head": set(azure),
    }


def _extract_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    assert comps, "expected at least one arrow"
    arrows = [_parse_arrow(g, comp) for comp in comps]
    return arrows


def _trace_rays(
    arrows: list[dict[str, object]],
    shape: tuple[int, int],
    entity_cells: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], list[tuple[str, int]]], list[dict[str, object]]]:
    contributions: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    traces: list[dict[str, object]] = []
    for arrow in arrows:
        tip = arrow["tip"]
        direction = arrow["direction"]
        color = int(arrow["color"])
        painted: list[tuple[int, int]] = []
        stop_reason: str | None = None
        for step in range(1, int(arrow["tail_len"]) + 1):
            cell = _add(tip, direction, step)
            if not _in_bounds(shape, *cell):
                stop_reason = "boundary"
                break
            if cell in entity_cells:
                stop_reason = "entity"
                break
            painted.append(cell)
            contributions[cell].append((str(arrow["axis"]), color))
        traces.append(
            {
                "painted": painted,
                "stop_reason": stop_reason,
                "axis": arrow["axis"],
                "color": color,
            }
        )
    return contributions, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect azure-headed arrows, then splash the tail color forward by tail length.

    Rays start just beyond the tip pixel, stop before any original arrow pixel,
    and horizontal/vertical crossings are recolored maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows = _extract_arrows(g)
    entity_cells = {tuple(map(int, p)) for p in np.argwhere(g != BG)}
    contributions, _traces = _trace_rays(arrows, g.shape, entity_cells)

    out = g.copy()
    for cell, hits in contributions.items():
        axes = {axis for axis, _color in hits}
        colors = {int(color) for _axis, color in hits}
        if len(axes) >= 2:
            out[cell] = MAROON
        elif len(colors) == 1:
            out[cell] = next(iter(colors))
        else:
            out[cell] = MAROON
    return out


def _build_arrow(
    shape: tuple[int, int],
    hinge: tuple[int, int],
    direction: tuple[int, int],
    flank_delta: tuple[int, int],
    tail_len: int,
    color: int,
) -> dict[str, object] | None:
    tip = _add(hinge, direction)
    flank = _add(hinge, flank_delta)
    tail = {_add(hinge, direction, -step) for step in range(1, tail_len + 1)}
    entity = {hinge, tip, flank} | tail
    if len(entity) != tail_len + 3:
        return None
    if any(not _in_bounds(shape, r, c) for r, c in entity):
        return None
    return {
        "hinge": hinge,
        "tip": tip,
        "flank": flank,
        "direction": direction,
        "axis": AXIS[direction],
        "tail_len": int(tail_len),
        "color": int(color),
        "tail": set(tail),
        "head": {hinge, tip, flank},
        "coords": set(entity),
    }


def _render_input(shape: tuple[int, int], arrows: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for arrow in arrows:
        for r, c in arrow["tail"]:
            grid[r, c] = int(arrow["color"])
        for r, c in arrow["head"]:
            grid[r, c] = AZURE
    return grid


def _entity_cells(arrows: list[dict[str, object]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for arrow in arrows:
        cells |= set(arrow["coords"])
    return cells


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _scene_stats(arrows: list[dict[str, object]], shape: tuple[int, int]) -> dict[str, object]:
    entities = _entity_cells(arrows)
    contributions, traces = _trace_rays(arrows, shape, entities)
    cross = False
    same_axis_conflict = False
    for hits in contributions.values():
        axes = {axis for axis, _color in hits}
        if len(axes) >= 2:
            cross = True
        elif len(hits) > 1:
            same_axis_conflict = True
    return {
        "contributions": contributions,
        "ray_cells": set(contributions),
        "traces": traces,
        "cross": cross,
        "entity_block": any(trace["stop_reason"] == "entity" for trace in traces),
        "boundary_block": any(trace["stop_reason"] == "boundary" for trace in traces),
        "same_axis_conflict": same_axis_conflict,
    }


def _cross_status(scenario: str, stats: dict[str, object]) -> bool:
    cross = bool(stats["cross"])
    block = bool(stats["entity_block"])
    boundary = bool(stats["boundary_block"])
    same_axis = bool(stats["same_axis_conflict"])
    if boundary or same_axis:
        return False
    if scenario == "plain":
        return not cross and not block
    if scenario == "cross":
        return cross and not block
    assert scenario == "block"
    return block and not cross


def _safe_with_existing(
    arrow: dict[str, object],
    arrows: list[dict[str, object]],
    shape: tuple[int, int],
) -> bool:
    existing_entities = _entity_cells(arrows)
    if set(arrow["coords"]) & _halo(existing_entities, shape):
        return False

    stats = _scene_stats(arrows, shape)
    existing_rays = set(stats["ray_cells"])
    if set(arrow["coords"]) & existing_rays:
        return False

    direction = arrow["direction"]
    tip = arrow["tip"]
    candidate_ray = set()
    for step in range(1, int(arrow["tail_len"]) + 1):
        cell = _add(tip, direction, step)
        if not _in_bounds(shape, *cell):
            return False
        if cell in existing_entities or cell in existing_rays:
            return False
        candidate_ray.add(cell)
    return len(candidate_ray) == int(arrow["tail_len"])


def _place_cross_pair(shape: tuple[int, int], colors: list[int]) -> list[dict[str, object]] | None:
    h, w = shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        cross_cell = (random.randint(3, h - 4), random.randint(3, w - 4))
        horiz_dir = random.choice(((0, 1), (0, -1)))
        vert_dir = random.choice(((1, 0), (-1, 0)))
        horiz_len = random.randint(2, TAIL_MAX)
        vert_len = random.randint(2, TAIL_MAX)
        horiz_dist = random.randint(1, horiz_len)
        vert_dist = random.randint(1, vert_len)

        horiz_tip = _add(cross_cell, horiz_dir, -horiz_dist)
        vert_tip = _add(cross_cell, vert_dir, -vert_dist)
        horiz_hinge = _add(horiz_tip, horiz_dir, -1)
        vert_hinge = _add(vert_tip, vert_dir, -1)

        horiz_flank = random.choice(PERPS[horiz_dir])
        vert_flank = random.choice(PERPS[vert_dir])
        a = _build_arrow(shape, horiz_hinge, horiz_dir, horiz_flank, horiz_len, colors[0])
        b = _build_arrow(shape, vert_hinge, vert_dir, vert_flank, vert_len, colors[1])
        if a is None or b is None:
            continue
        if set(a["coords"]) & _halo(set(b["coords"]), shape):
            continue
        stats = _scene_stats([a, b], shape)
        if _cross_status("cross", stats):
            return [a, b]
    return None


def _place_block_pair(shape: tuple[int, int], colors: list[int]) -> list[dict[str, object]] | None:
    h, w = shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        block_cell = (random.randint(3, h - 4), random.randint(3, w - 4))
        dir_a = random.choice(DIRS)
        len_a = random.randint(2, TAIL_MAX)
        painted_before_block = random.randint(1, len_a - 1)
        tip_a = _add(block_cell, dir_a, -(painted_before_block + 1))
        hinge_a = _add(tip_a, dir_a, -1)
        flank_a = random.choice(PERPS[dir_a])
        a = _build_arrow(shape, hinge_a, dir_a, flank_a, len_a, colors[0])
        if a is None:
            continue

        dir_b = random.choice(DIRS)
        flank_b = random.choice(PERPS[dir_b])
        len_b = random.randint(TAIL_MIN, TAIL_MAX)
        b = _build_arrow(shape, block_cell, dir_b, flank_b, len_b, colors[1])
        if b is None:
            continue
        if set(a["coords"]) & _halo(set(b["coords"]), shape):
            continue
        stats = _scene_stats([a, b], shape)
        if _cross_status("block", stats):
            return [a, b]
    return None


def _place_safe_arrow(
    shape: tuple[int, int],
    arrows: list[dict[str, object]],
    color: int,
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        direction = random.choice(DIRS)
        flank = random.choice(PERPS[direction])
        tail_len = random.randint(TAIL_MIN, TAIL_MAX)
        hinge = (random.randrange(h), random.randrange(w))
        arrow = _build_arrow(shape, hinge, direction, flank, tail_len, color)
        if arrow is None:
            continue
        if _safe_with_existing(arrow, arrows, shape):
            return arrow
    return None


def _attempt_scene() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    scenario = random.choice(("plain", "cross", "block"))
    total_arrows = random.randint(ARROW_COUNT_MIN, ARROW_COUNT_MAX)
    colors = random.sample(COLOR_POOL, total_arrows)
    arrows: list[dict[str, object]]

    if scenario == "plain":
        arrows = []
        for color in colors:
            arrow = _place_safe_arrow(shape, arrows, color)
            if arrow is None:
                return None
            arrows.append(arrow)
    elif scenario == "cross":
        pair = _place_cross_pair(shape, colors[:2])
        if pair is None:
            return None
        arrows = pair
        for color in colors[2:]:
            arrow = _place_safe_arrow(shape, arrows, color)
            if arrow is None:
                return None
            arrows.append(arrow)
    else:
        pair = _place_block_pair(shape, colors[:2])
        if pair is None:
            return None
        arrows = pair
        for color in colors[2:]:
            arrow = _place_safe_arrow(shape, arrows, color)
            if arrow is None:
                return None
            arrows.append(arrow)

    stats = _scene_stats(arrows, shape)
    if not _cross_status(scenario, stats):
        return None

    inp = _render_input(shape, arrows)
    try:
        out = main(inp)
    except Exception:
        return None
    assert out.shape == inp.shape
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        inp = _attempt_scene()
        if inp is not None:
            return inp
    raise RuntimeError(f"generate_input failed after {MAX_GENERATE_ATTEMPTS} attempts")
