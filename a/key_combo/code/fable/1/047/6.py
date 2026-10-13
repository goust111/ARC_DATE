from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NON_START_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BG = 0
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _rotate_point(point: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    r, c = map(int, point)
    for _ in range(int(quarter_turns) % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_dir(direction: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    return _rotate_point(direction, quarter_turns)


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = map(int, center)
    dr, dc = map(int, direction)
    pr, pc = dc, -dr
    return {
        (r, c),
        (r + pr, c + pc),
        (r - pr, c - pc),
        (r - dr, c - dc),
    }


def _parse_arrow(comp: dict) -> dict:
    coords = [tuple(map(int, cell)) for cell in comp["coords"]]
    coord_set = set(coords)
    assert int(comp["area"]) == 4, "expected 4-pixel arrow objects"

    degrees = {
        cell: sum((nbr in coord_set) for nbr in _neighbors4(*cell))
        for cell in coord_set
    }
    centers = [cell for cell, deg in degrees.items() if deg == 3]
    leaves = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(centers) == 1 and len(leaves) == 3, "expected T-shaped arrow"

    center = centers[0]
    cr, cc = center
    leaf_vectors = [(r - cr, c - cc) for r, c in leaves]
    control_vectors = [
        vec for vec in leaf_vectors if (-vec[0], -vec[1]) not in leaf_vectors
    ]
    assert len(control_vectors) == 1, "expected one control pixel"
    control_vec = control_vectors[0]
    direction = (-control_vec[0], -control_vec[1])
    assert direction in DIRS, "direction must be axis-aligned"
    assert coord_set == _arrow_cells(center, direction), "unexpected arrow geometry"

    return {
        "color": int(comp["color"]),
        "coords": np.array(sorted(coord_set), dtype=int),
        "coord_set": coord_set,
        "center": (int(center[0]), int(center[1])),
        "direction": (int(direction[0]), int(direction[1])),
    }


def _extract_arrows(grid: np.ndarray) -> tuple[int, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert comps, "expected at least one foreground component"
    arrows = [_parse_arrow(comp) for comp in comps]
    colors = [int(arrow["color"]) for arrow in arrows]
    assert len(set(colors)) == len(colors), "arrow colors must be unique"
    start_indices = [i for i, arrow in enumerate(arrows) if int(arrow["color"]) == Color.GREY]
    assert len(start_indices) == 1, "expected exactly one gray start arrow"
    return bg, arrows, int(start_indices[0])


def _ray_hit_distance(
    start: tuple[int, int],
    direction: tuple[int, int],
    coords: np.ndarray,
) -> int | None:
    sr, sc = map(int, start)
    dr, dc = map(int, direction)
    best: int | None = None

    if dr == 0:
        for r, c in coords:
            if int(r) != sr:
                continue
            dist = (int(c) - sc) * dc
            if dist <= 0:
                continue
            if best is None or dist < best:
                best = int(dist)
        return best

    for r, c in coords:
        if int(c) != sc:
            continue
        dist = (int(r) - sr) * dr
        if dist <= 0:
            continue
        if best is None or dist < best:
            best = int(dist)
    return best


def _activated_chain(arrows: list[dict], start_idx: int, *, removed_still_block: bool = False) -> list[int]:
    alive = set(range(len(arrows)))
    alive.remove(int(start_idx))
    removed = {int(start_idx)}
    order = [int(start_idx)]
    current = int(start_idx)

    while True:
        best_idx: int | None = None
        best_dist: int | None = None
        for idx, arrow in enumerate(arrows):
            idx = int(idx)
            if idx == current:
                continue
            if not removed_still_block and idx not in alive:
                continue
            dist = _ray_hit_distance(
                arrows[current]["center"],
                arrows[current]["direction"],
                arrow["coords"],
            )
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = int(dist)
                best_idx = idx
            elif dist == best_dist:
                raise AssertionError("ray-hit order must be unambiguous")

        if best_idx is None:
            break
        if removed_still_block and best_idx in removed:
            break

        order.append(int(best_idx))
        alive.remove(int(best_idx))
        removed.add(int(best_idx))
        current = int(best_idx)

    return order


def _erase_indices(grid: np.ndarray, arrows: list[dict], indices: list[int], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for idx in indices:
        coords = arrows[int(idx)]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase exactly the arrow chain triggered by successive first ray hits from the gray start arrow.

    Input structure:
    - The background contains several separated monochrome arrow objects.
    - Each arrow object is a three-pixel straight bar plus one control pixel touching the bar midpoint.
    - Exactly one arrow is gray and serves as the start arrow; every arrow has its own color.

    Rule:
    - Read each arrow's pointing direction as the direction from the bar midpoint away from its control pixel.
    - Starting from the gray arrow, cast a straight axis-aligned ray from the midpoint; the nearest still-present arrow hit by that ray is activated.
    - An activated arrow is erased completely and immediately emits its own ray by the same rule; already erased arrows no longer block later rays.
    - Continue until a ray hits no remaining arrow, then return the original grid with all activated arrows erased to background and every untouched arrow unchanged.

    Key insight:
    - The erased set is a causal first-hit chain, not the union of all arrows lying on the start ray or on the same lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, arrows, start_idx = _extract_arrows(g)
    activated = _activated_chain(arrows, start_idx)
    return _erase_indices(g, arrows, activated, bg)


def _build_turn_template() -> tuple[list[dict], bool]:
    d1 = random.randint(4, 6)
    d2 = random.randint(4, 6)
    d3 = random.randint(4, 6)
    d4 = random.randint(4, 6)

    specs = [
        {"center": (0, 0), "direction": (0, 1), "active": True},
        {"center": (0, d1), "direction": (1, 0), "active": True},
        {"center": (d2, d1), "direction": (0, -1), "active": True},
        {"center": (0, d1 + d3), "direction": random.choice(DIRS), "active": False},
        {"center": (d2 + d4, d1), "direction": random.choice(DIRS), "active": False},
    ]
    if random.random() < 0.7:
        specs.append(
            {
                "center": (d2 + random.randint(4, 6), d1 + random.randint(4, 6)),
                "direction": random.choice(DIRS),
                "active": False,
            }
        )
    return specs, False


def _build_bounce_template() -> tuple[list[dict], bool]:
    d1 = random.randint(4, 6)
    d2 = random.randint(4, 6)
    d3 = random.randint(4, 6)
    d4 = random.randint(4, 6)
    d5 = random.randint(4, 6)
    d6 = random.randint(4, 6)

    return [
        {"center": (0, 0), "direction": (0, 1), "active": True},
        {"center": (0, d1), "direction": (0, -1), "active": True},
        {"center": (0, -d2), "direction": (1, 0), "active": True},
        {"center": (d3, -d2), "direction": (0, 1), "active": True},
        {"center": (0, d1 + d4), "direction": random.choice(DIRS), "active": False},
        {"center": (0, -d2 - d5), "direction": random.choice(DIRS), "active": False},
        {"center": (d3 + d6, -d2), "direction": random.choice(DIRS), "active": False},
    ], True


def _choose_template() -> tuple[list[dict], bool]:
    if random.random() < 0.55:
        return _build_bounce_template()
    return _build_turn_template()


def _assign_colors(specs: list[dict]) -> list[dict]:
    colored: list[dict] = []
    palette = random.sample(list(NON_START_COLORS), len(specs) - 1)
    used = iter(palette)
    for i, spec in enumerate(specs):
        new_spec = dict(spec)
        new_spec["color"] = Color.GREY if i == 0 else int(next(used))
        colored.append(new_spec)
    return colored


def _transform_specs(specs: list[dict]) -> tuple[list[dict], tuple[int, int]]:
    quarter_turns = random.randrange(4)
    rotated: list[dict] = []
    for spec in specs:
        rotated.append(
            {
                **spec,
                "center": _rotate_point(spec["center"], quarter_turns),
                "direction": _rotate_dir(spec["direction"], quarter_turns),
            }
        )

    raw_cells = [cell for spec in rotated for cell in _arrow_cells(spec["center"], spec["direction"])]
    min_r = min(r for r, _ in raw_cells)
    max_r = max(r for r, _ in raw_cells)
    min_c = min(c for _, c in raw_cells)
    max_c = max(c for _, c in raw_cells)

    inner_h = max_r - min_r + 1
    inner_w = max_c - min_c + 1
    if inner_h > 28 or inner_w > 28:
        raise ValueError("scene too large")

    extra_h = 30 - inner_h
    extra_w = 30 - inner_w
    top = random.randint(1, min(4, extra_h - 1))
    bottom = random.randint(1, min(4, extra_h - top))
    left = random.randint(1, min(4, extra_w - 1))
    right = random.randint(1, min(4, extra_w - left))
    height = inner_h + top + bottom
    width = inner_w + left + right
    assert height <= 30 and width <= 30

    shift_r = top - min_r
    shift_c = left - min_c
    transformed: list[dict] = []
    for spec in rotated:
        center = (
            int(spec["center"][0] + shift_r),
            int(spec["center"][1] + shift_c),
        )
        direction = tuple(map(int, spec["direction"]))
        transformed.append(
            {
                **spec,
                "center": center,
                "direction": direction,
                "cells": _arrow_cells(center, direction),
            }
        )
    return transformed, (int(height), int(width))


def _objects_are_separated(specs: list[dict], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        cells = spec["cells"]
        for r, c in cells:
            if not (0 <= int(r) < h and 0 <= int(c) < w):
                return False
            if (int(r), int(c)) in occupied:
                return False
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (int(r) + dr, int(c) + dc) in occupied:
                        return False
        occupied.update((int(r), int(c)) for r, c in cells)
    return True


def _render_scene(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        rows, cols = zip(*sorted(spec["cells"]))
        grid[np.array(rows, dtype=int), np.array(cols, dtype=int)] = int(spec["color"])
    return grid


def _expected_from_specs(inp: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, list[int]]:
    out = np.array(inp, copy=True)
    active_colors = [int(spec["color"]) for spec in specs if bool(spec["active"])]
    for spec in specs:
        if not bool(spec["active"]):
            continue
        rows, cols = zip(*sorted(spec["cells"]))
        out[np.array(rows, dtype=int), np.array(cols, dtype=int)] = BG
    return out, active_colors


def _naive_initial_ray_output(grid: np.ndarray) -> np.ndarray:
    bg, arrows, start_idx = _extract_arrows(grid)
    start = arrows[start_idx]
    hits = [
        idx
        for idx, arrow in enumerate(arrows)
        if idx != start_idx
        and _ray_hit_distance(start["center"], start["direction"], arrow["coords"]) is not None
    ]
    return _erase_indices(grid, arrows, [start_idx] + hits, bg)


def _blocked_by_removed_output(grid: np.ndarray) -> np.ndarray:
    bg, arrows, start_idx = _extract_arrows(grid)
    activated = _activated_chain(arrows, start_idx, removed_still_block=True)
    return _erase_indices(grid, arrows, activated, bg)


def _gen_attempt() -> np.ndarray | None:
    specs, needs_removed_pass = _choose_template()
    specs = _assign_colors(specs)
    specs, shape = _transform_specs(specs)
    if not _objects_are_separated(specs, shape):
        return None

    inp = _render_scene(specs, shape)
    expected, planned_colors = _expected_from_specs(inp, specs)
    if len(planned_colors) < 2:
        return None

    try:
        bg, arrows, start_idx = _extract_arrows(inp)
    except Exception:
        return None
    activated = _activated_chain(arrows, start_idx)
    got_colors = [int(arrows[idx]["color"]) for idx in activated]
    if got_colors != planned_colors:
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(expected, inp):
        return None

    survivors = [arrow for arrow in arrows if int(arrow["color"]) not in set(planned_colors)]
    if len(survivors) < 2:
        return None

    if np.array_equal(_naive_initial_ray_output(inp), expected):
        return None
    if needs_removed_pass and np.array_equal(_blocked_by_removed_output(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-chain erase input")
