from __future__ import annotations

from collections import defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
in_bounds = _ARC_UTILS.in_bounds


BG = Color.BLACK
CENTER_COLOR = Color.MAROON
DOMINO_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
AXIAL_RAYS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_RAYS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ALL_RAYS = AXIAL_RAYS + DIAGONAL_RAYS
MAX_GENERATION_ATTEMPTS = 400


def _find_center_block(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    maroon = np.argwhere(grid == CENTER_COLOR)
    assert maroon.shape[0] == 9, "expected exactly nine center-block pixels"
    r0, c0 = maroon.min(axis=0)
    r1, c1 = maroon.max(axis=0)
    assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (3, 3), "center block must be 3x3"
    block_mask = np.zeros_like(grid, dtype=bool)
    block_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    assert np.all(grid[block_mask] == CENTER_COLOR), "center block must be solid"
    return (int(r0 + 1), int(c0 + 1)), block_mask


def _ray_direction(center: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int] | None:
    dr = int(cell[0] - center[0])
    dc = int(cell[1] - center[1])
    if dr == 0 and dc == 0:
        return None
    sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sdc = 0 if dc == 0 else (1 if dc > 0 else -1)
    if dr == 0 or dc == 0 or abs(dr) == abs(dc):
        return sdr, sdc
    return None


def _classify_domino(center: tuple[int, int], coords: np.ndarray) -> tuple[int, int] | None:
    dirs = [_ray_direction(center, (int(r), int(c))) for r, c in coords]
    if dirs[0] is None or dirs[1] is None:
        return None
    return dirs[0] if dirs[0] == dirs[1] else None


def _sorted_cells(coords: np.ndarray | list[tuple[int, int]]) -> list[tuple[int, int]]:
    if isinstance(coords, np.ndarray):
        return sorted((int(r), int(c)) for r, c in coords.tolist())
    return sorted((int(r), int(c)) for r, c in coords)


def _near_distance(center: tuple[int, int], coords: np.ndarray) -> int:
    return min(abs(int(r) - center[0]) + abs(int(c) - center[1]) for r, c in coords)


def _cells_on_ray(center: tuple[int, int], ray: tuple[int, int], near_dist: int) -> list[tuple[int, int]]:
    dr, dc = ray
    r, c = center
    return [
        (r + dr * near_dist, c + dc * near_dist),
        (r + dr * (near_dist + 1), c + dc * (near_dist + 1)),
    ]


def _ray_edge_distance(shape: tuple[int, int], center: tuple[int, int], ray: tuple[int, int]) -> int:
    h, w = map(int, shape)
    r, c = center
    dr, dc = ray
    steps = []
    if dr > 0:
        steps.append(h - 1 - r)
    elif dr < 0:
        steps.append(r)
    if dc > 0:
        steps.append(w - 1 - c)
    elif dc < 0:
        steps.append(c)
    return int(min(steps))


def _reserve(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _candidate_pinned_pairs(
    shape: tuple[int, int],
    center: tuple[int, int],
    blocked: np.ndarray,
    forbidden_output: set[tuple[int, int]],
    *,
    cross_ray: bool,
) -> list[list[tuple[int, int]]]:
    h, w = shape
    candidates: list[list[tuple[int, int]]] = []
    if cross_ray:
        for ray in DIAGONAL_RAYS:
            max_dist = _ray_edge_distance(shape, center, ray)
            for dist in range(3, max_dist + 1):
                a = (center[0] + ray[0] * dist, center[1] + ray[1] * dist)
                for off in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    b = (a[0] + off[0], a[1] + off[1])
                    if not in_bounds((h, w), b[0], b[1]):
                        continue
                    pair = [a, b]
                    dirs = [_ray_direction(center, cell) for cell in pair]
                    if dirs[0] is None or dirs[0] == dirs[1]:
                        continue
                    if any(blocked[r, c] for r, c in pair):
                        continue
                    if any((r, c) in forbidden_output for r, c in pair):
                        continue
                    candidates.append(pair)
    else:
        for r in range(h):
            for c in range(w):
                for dr, dc in ((0, 1), (1, 0)):
                    pair = [(r, c), (r + dr, c + dc)]
                    if not all(in_bounds((h, w), rr, cc) for rr, cc in pair):
                        continue
                    dirs = [_ray_direction(center, cell) for cell in pair]
                    if any(direction is not None for direction in dirs):
                        continue
                    if any(blocked[rr, cc] for rr, cc in pair):
                        continue
                    if any((rr, cc) in forbidden_output for rr, cc in pair):
                        continue
                    candidates.append(pair)
    random.shuffle(candidates)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the dominoes whose two cells share one center-emitted ray into the 3x3 maroon attractor block.

    Input structure:
    - Black background with one solid 3x3 maroon (9) block whose center defines eight compass-direction rays.
    - Every other foreground object is a monochrome 4-connected domino of color 1, 2, 3, 4, or 7.
    - Some dominoes have both cells on the same ray; others are off-ray or span different rays and are pinned.

    Rule:
    - Keep the maroon 3x3 block fixed.
    - Classify each domino by testing whether both of its cells lie on the same one of the eight virtual rays from the block center.
    - Pinned dominoes stay where they are. Ray-qualified dominoes slide rigidly toward the center along their own ray.
    - Settle ray-qualified dominoes from smaller center distance to larger, stopping a domino when its next step would enter the 3x3 block or a previously settled sliding domino.
    - Diagonal rays, if present, move by one row and one column per step.

    Key insight:
    - The 3x3 block is both the ray source and the stopper: only dominoes whose two cells are strictly co-linear on one center ray join that ray's independent gravity channel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    center, block_mask = _find_center_block(g)
    dominoes = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=list(DOMINO_COLORS),
        return_format="dict",
    )
    assert dominoes, "expected at least one domino"
    for domino in dominoes:
        assert int(domino["area"]) == 2, "every foreground non-center object must be a domino"

    movable: list[dict] = []
    pinned_cells: set[tuple[int, int]] = set()
    out = g.copy()
    for domino in dominoes:
        coords = domino["coords"].astype(int)
        direction = _classify_domino(center, coords)
        if direction is None:
            for cell in _sorted_cells(coords):
                pinned_cells.add(cell)
            continue
        movable.append(
            {
                "coords": coords.copy(),
                "color": int(domino["color"]),
                "direction": direction,
                "near": _near_distance(center, coords),
            }
        )
        out[coords[:, 0], coords[:, 1]] = BG

    occupied = block_mask.copy()
    for domino in sorted(
        movable,
        key=lambda item: (
            int(item["near"]),
            int(item["direction"][0]),
            int(item["direction"][1]),
            _sorted_cells(item["coords"]),
        ),
    ):
        step = np.array((-domino["direction"][0], -domino["direction"][1]), dtype=int)
        coords = domino["coords"].copy()
        while True:
            nxt = coords + step
            if np.any(block_mask[nxt[:, 0], nxt[:, 1]]) or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt
        final_cells = _sorted_cells(coords)
        assert not any(cell in pinned_cells for cell in final_cells), "pinned domino overlapped sliding result"
        out[coords[:, 0], coords[:, 1]] = int(domino["color"])
        occupied[coords[:, 0], coords[:, 1]] = True

    return out


def _build_expected(
    shape: tuple[int, int],
    center: tuple[int, int],
    movable_specs: list[dict],
    pinned_specs: list[dict],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    cr, cc = center
    out[cr - 1 : cr + 2, cc - 1 : cc + 2] = CENTER_COLOR
    for spec in pinned_specs:
        for r, c in spec["cells"]:
            out[r, c] = int(spec["color"])

    by_ray: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for spec in movable_specs:
        by_ray[tuple(spec["ray"])].append(spec)

    for ray, items in by_ray.items():
        for idx, spec in enumerate(sorted(items, key=lambda item: int(item["near_dist"]))):
            final_near = 2 + 2 * idx
            for r, c in _cells_on_ray(center, ray, final_near):
                assert out[r, c] == BG, "generated output must stay non-overlapping"
                out[r, c] = int(spec["color"])
    return out


def _render_input(
    shape: tuple[int, int],
    center: tuple[int, int],
    movable_specs: list[dict],
    pinned_specs: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    cr, cc = center
    grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = CENTER_COLOR
    for spec in movable_specs + pinned_specs:
        for r, c in spec["cells"]:
            grid[r, c] = int(spec["color"])
    return grid


def _sample_movable_specs(shape: tuple[int, int], center: tuple[int, int], movable_count: int) -> list[dict] | None:
    capacities = {
        ray: (_ray_edge_distance(shape, center, ray) - 1) // 3
        for ray in AXIAL_RAYS
    }
    usable = [ray for ray, cap in capacities.items() if cap >= 1]
    stackable = [ray for ray, cap in capacities.items() if cap >= 2]
    if len(usable) < 2 or not stackable:
        return None

    max_rays = min(4, movable_count - 1, len(usable))
    if max_rays < 2:
        return None
    ray_count = random.randint(2, max_rays)
    chosen = random.sample(usable, ray_count)
    if not any(ray in stackable for ray in chosen):
        forced = random.choice(stackable)
        chosen[random.randrange(ray_count)] = forced
        chosen = list(dict.fromkeys(chosen))
        while len(chosen) < ray_count:
            candidate = random.choice(usable)
            if candidate not in chosen:
                chosen.append(candidate)

    counts = {ray: 1 for ray in chosen}
    remaining = movable_count - ray_count
    while remaining > 0:
        options = [ray for ray in chosen if counts[ray] < capacities[ray]]
        if not options:
            return None
        pick = random.choice(options)
        counts[pick] += 1
        remaining -= 1
    if not any(count >= 2 for count in counts.values()):
        return None

    specs: list[dict] = []
    for ray in chosen:
        max_near = _ray_edge_distance(shape, center, ray) - 1
        last_near = 0
        for idx in range(counts[ray]):
            remaining_here = counts[ray] - idx - 1
            min_near = 3 if idx == 0 else last_near + 3
            max_here = max_near - 3 * remaining_here
            if min_near > max_here:
                return None
            near_dist = random.randint(min_near, max_here)
            cells = _cells_on_ray(center, ray, near_dist)
            specs.append(
                {
                    "cells": cells,
                    "ray": ray,
                    "near_dist": near_dist,
                    "color": random.choice(DOMINO_COLORS),
                }
            )
            last_near = near_dist
    return specs


def _sample_pinned_specs(
    shape: tuple[int, int],
    center: tuple[int, int],
    blocked: np.ndarray,
    forbidden_output: set[tuple[int, int]],
    pinned_count: int,
) -> list[dict] | None:
    specs: list[dict] = []
    required_cross = min(1, pinned_count)
    required_off = min(1, max(0, pinned_count - required_cross))
    kinds = ["cross"] * required_cross + ["off"] * required_off
    while len(kinds) < pinned_count:
        kinds.append(random.choice(["cross", "off"]))
    random.shuffle(kinds)

    for kind in kinds:
        candidates = _candidate_pinned_pairs(
            shape,
            center,
            blocked,
            forbidden_output,
            cross_ray=(kind == "cross"),
        )
        if not candidates:
            return None
        cells = candidates[0]
        specs.append({"cells": cells, "color": random.choice(DOMINO_COLORS), "kind": kind})
        _reserve(blocked, cells)
    return specs


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    movable_specs: list[dict],
    pinned_specs: list[dict],
    center: tuple[int, int],
) -> bool:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(inp, expected):
        return False

    dominoes = find_components(
        inp,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=list(DOMINO_COLORS),
        return_format="dict",
    )
    if len(dominoes) != len(movable_specs) + len(pinned_specs):
        return False
    if any(int(domino["area"]) != 2 for domino in dominoes):
        return False

    movable_count = 0
    pinned_count = 0
    rays_used: set[tuple[int, int]] = set()
    for domino in dominoes:
        direction = _classify_domino(center, domino["coords"])
        if direction is None:
            pinned_count += 1
        else:
            movable_count += 1
            rays_used.add(direction)
    if movable_count != len(movable_specs) or pinned_count != len(pinned_specs):
        return False
    if len(rays_used) < 2:
        return False
    if not any(spec["kind"] == "cross" for spec in pinned_specs):
        return False
    if not any(spec["kind"] == "off" for spec in pinned_specs):
        return False

    stacks = defaultdict(int)
    for spec in movable_specs:
        stacks[tuple(spec["ray"])] += 1
    if max(stacks.values()) < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    center = (random.randint(7, h - 8), random.randint(7, w - 8))

    domino_count = random.randint(6, 10)
    pinned_count = random.randint(2, min(4, domino_count - 2))
    movable_count = domino_count - pinned_count

    movable_specs = _sample_movable_specs((h, w), center, movable_count)
    if movable_specs is None:
        return None

    provisional_expected = _build_expected((h, w), center, movable_specs, [])
    forbidden_output = set(map(tuple, np.argwhere(provisional_expected != BG)))

    blocked = np.zeros((h, w), dtype=bool)
    center_cells = [(r, c) for r in range(center[0] - 1, center[0] + 2) for c in range(center[1] - 1, center[1] + 2)]
    _reserve(blocked, center_cells)
    for spec in movable_specs:
        _reserve(blocked, spec["cells"])

    pinned_specs = _sample_pinned_specs((h, w), center, blocked, forbidden_output, pinned_count)
    if pinned_specs is None:
        return None

    inp = _render_input((h, w), center, movable_specs, pinned_specs)
    expected = _build_expected((h, w), center, movable_specs, pinned_specs)
    if not _validate_instance(inp, expected, movable_specs, pinned_specs, center):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-channel domino input")
