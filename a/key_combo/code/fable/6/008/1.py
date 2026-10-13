from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, get_bg, in_bounds


_DIR_FAMILIES = (
    ((-1, 0), (1, 0)),
    ((0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 1), (1, -1)),
)
_GEN_BG_CHOICES = tuple(range(10))
_CENTER_RETRIES = 240
_SAMPLE_RETRIES = 360
_MAX_COLORS = 3
_MIN_SIZE = 18
_MAX_SIZE = 28
_MAX_RAY_POINTS = 4


def _normalize_direction(dr: int, dc: int) -> tuple[int, int] | None:
    dr = int(dr)
    dc = int(dc)
    if dr == 0 and dc == 0:
        return None
    if dr == 0:
        return 0, 1 if dc > 0 else -1
    if dc == 0:
        return 1 if dr > 0 else -1, 0
    if abs(dr) != abs(dc):
        return None
    return 1 if dr > 0 else -1, 1 if dc > 0 else -1


def _anchor_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _segment_cells(
    center: tuple[int, int], direction: tuple[int, int], count: int
) -> list[tuple[int, int]]:
    cr, cc = center
    dr, dc = direction
    return [(cr + (k + 2) * dr, cc + (k + 2) * dc) for k in range(int(count))]


def _max_distance(
    shape: tuple[int, int], center: tuple[int, int], direction: tuple[int, int]
) -> int:
    cr, cc = center
    dr, dc = direction
    dist = 0
    while in_bounds(shape, cr + (dist + 1) * dr, cc + (dist + 1) * dc):
        dist += 1
    return int(dist)


def _center_groups(
    coords: np.ndarray, center: tuple[int, int]
) -> dict[tuple[int, int], list[int]] | None:
    cr, cc = center
    groups: dict[tuple[int, int], list[int]] = {}
    for raw_r, raw_c in coords:
        r, c = int(raw_r), int(raw_c)
        dr = r - cr
        dc = c - cc
        direction = _normalize_direction(dr, dc)
        if direction is None:
            return None
        dist = max(abs(dr), abs(dc))
        if dist <= 1:
            return None
        groups.setdefault(direction, []).append(int(dist))

    if not (2 <= len(groups) <= 4):
        return None
    used_dirs = set(groups)
    for direction, distances in groups.items():
        if len(distances) < 2:
            return None
        if tuple(-v for v in direction) in used_dirs:
            return None
        distances.sort()
    return groups


def _find_color_spec(
    grid: np.ndarray, color: int, bg: int
) -> tuple[tuple[int, int], dict[tuple[int, int], list[int]]]:
    coords = np.argwhere(grid == int(color))
    assert len(coords) >= 4, "each color needs at least two points on two lines"
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], dict[tuple[int, int], list[int]]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == bg):
                continue
            groups = _center_groups(coords, (r, c))
            if groups is None:
                continue
            candidates.append(((r, c), groups))
    assert len(candidates) == 1, f"expected one hidden anchor for color {color}, found {len(candidates)}"
    return candidates[0]


def _solve_specs(
    grid: np.ndarray,
) -> list[tuple[int, tuple[int, int], dict[tuple[int, int], list[int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    specs = []
    for color in colors_present(g, exclude={bg}):
        center, groups = _find_color_spec(g, int(color), bg)
        specs.append((int(color), center, groups))
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild each hidden 3x3 anchor from concurrent sightlines and compress the ray pixels against it.

    Input structure:
    - The grid has one dominant background color and one or more foreground colors.
    - For each foreground color, all pixels lie on 2 to 4 distinct straight 8-direction rays.
    - Extending those rays backward makes them meet at one background cell whose full 3x3 neighborhood is still background in the input.

    Rule:
    - For each color, find the unique common intersection of its rays.
    - Draw that color as a solid 3x3 anchor block centered on the intersection.
    - For every ray of that color, count its scattered pixels and replace them by a consecutive segment on the same ray, starting immediately outside the anchor block.
    - Clear every other cell to the background color.

    Key insight:
    - The anchor is never visible directly; it must be reconstructed from the concurrency of several same-colored sightlines before the ray compression has a target.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    for color, center, groups in _solve_specs(g):
        for cell in _anchor_cells(center):
            assert in_bounds(g.shape, *cell)
            assert out[cell] == bg
            out[cell] = color

        for direction, distances in groups.items():
            for cell in _segment_cells(center, direction, len(distances)):
                assert in_bounds(g.shape, *cell)
                assert out[cell] == bg
                out[cell] = color

        assert int((out == color).sum()) == 9 + sum(len(v) for v in groups.values())

    return out


def _choose_directions(rng: random.Random, mode: str) -> list[tuple[int, int]]:
    if mode == "diag_pair":
        family_ids = [2, 3]
    elif mode == "four_ray":
        family_ids = [0, 1, 2, 3]
    elif mode == "triple_mix":
        family_ids = rng.sample(range(4), 3)
    elif mode == "orth_pair":
        family_ids = [0, 1]
    else:
        family_ids = rng.sample(range(4), rng.randint(2, 4))
    return [rng.choice(_DIR_FAMILIES[idx]) for idx in family_ids]


def _choose_scatter_distances(
    rng: random.Random, maxdist: int, count: int
) -> list[int] | None:
    universe = list(range(2, int(maxdist) + 1))
    target = list(range(2, int(count) + 2))
    if len(universe) < count or int(maxdist) < int(count) + 2:
        return None
    for _ in range(60):
        distances = sorted(rng.sample(universe, count))
        if distances != target:
            return distances
    return None


def _bbox_center_shortcut(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        center = (int((r0 + r1) // 2), int((c0 + c1) // 2))
        groups = _center_groups(coords, center)
        if groups is None:
            continue
        for cell in _anchor_cells(center):
            if in_bounds(g.shape, *cell):
                out[cell] = int(color)
        for direction, distances in groups.items():
            for cell in _segment_cells(center, direction, len(distances)):
                if in_bounds(g.shape, *cell):
                    out[cell] = int(color)
    return out


def _validate_generated_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if g.ndim != 2 or max(h, w) > 30:
        return False
    bg = int(get_bg(g))
    fg = colors_present(g, exclude={bg})
    if not fg or len(fg) > _MAX_COLORS:
        return False

    moved_ray_exists = False
    specs = _solve_specs(g)
    out = main(g)
    occupied_out: set[tuple[int, int]] = set()
    for color, center, groups in specs:
        anchor = _anchor_cells(center)
        if any(not in_bounds(g.shape, *cell) for cell in anchor):
            return False
        if not all(int(g[cell]) == bg for cell in anchor):
            return False
        if occupied_out & anchor:
            return False
        occupied_out |= anchor
        for direction, distances in groups.items():
            if len(distances) < 2:
                return False
            target = list(range(2, len(distances) + 2))
            if distances != target:
                moved_ray_exists = True
            seg = set(_segment_cells(center, direction, len(distances)))
            if any(not in_bounds(g.shape, *cell) for cell in seg):
                return False
            if occupied_out & seg:
                return False
            occupied_out |= seg
            for cell in seg:
                if int(out[cell]) != color:
                    return False
        if not np.all(out[center[0] - 1 : center[0] + 2, center[1] - 1 : center[1] + 2] == color):
            return False

    if not moved_ray_exists:
        return False
    if np.array_equal(_bbox_center_shortcut(g), out):
        return False
    return True


def _try_place_color(
    rng: random.Random,
    shape: tuple[int, int],
    color: int,
    mode: str,
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    directions = _choose_directions(rng, mode)
    for _ in range(_CENTER_RETRIES):
        center = (rng.randint(1, h - 2), rng.randint(1, w - 2))
        anchor = _anchor_cells(center)
        if anchor & occupied_in or anchor & occupied_out or anchor & reserved_bg:
            continue

        rays: dict[tuple[int, int], list[int]] = {}
        input_cells: set[tuple[int, int]] = set()
        output_cells = set(anchor)
        ok = True
        for direction in directions:
            maxdist = _max_distance(shape, center, direction)
            maxcount = min(_MAX_RAY_POINTS, maxdist - 2)
            if maxcount < 2:
                ok = False
                break
            count = rng.randint(2, maxcount)
            distances = _choose_scatter_distances(rng, maxdist, count)
            if distances is None:
                ok = False
                break

            ray_input = {
                (center[0] + dist * direction[0], center[1] + dist * direction[1])
                for dist in distances
            }
            ray_output = set(_segment_cells(center, direction, count))
            if (
                ray_input & occupied_in
                or ray_input & reserved_bg
                or ray_output & occupied_out
                or ray_output & reserved_bg
                or ray_input & input_cells
                or ray_output & output_cells
            ):
                ok = False
                break

            rays[direction] = distances
            input_cells |= ray_input
            output_cells |= ray_output

        if not ok:
            continue

        return {
            "color": int(color),
            "center": center,
            "rays": rays,
            "input_cells": input_cells,
            "output_cells": output_cells,
            "reserved_bg": anchor,
        }
    return None


def _mode_and_color_count(rng: random.Random) -> tuple[str, int]:
    pick = rng.choice(
        [
            ("diag_pair", rng.randint(1, 3)),
            ("four_ray", rng.randint(1, 2)),
            ("triple_mix", rng.randint(1, 3)),
            ("orth_pair", rng.randint(1, 3)),
            ("random", rng.randint(2, 3)),
        ]
    )
    return pick[0], pick[1]


def _build_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(_MIN_SIZE, _MAX_SIZE)
    w = rng.randint(_MIN_SIZE, _MAX_SIZE)
    bg = rng.choice(_GEN_BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)

    mode, n_colors = _mode_and_color_count(rng)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    colors = palette[:n_colors]

    occupied_in: set[tuple[int, int]] = set()
    occupied_out: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    specs = []

    for idx, color in enumerate(colors):
        local_mode = mode if idx == 0 else "random"
        spec = _try_place_color(
            rng,
            (h, w),
            int(color),
            local_mode,
            occupied_in,
            occupied_out,
            reserved_bg,
        )
        if spec is None:
            return None
        specs.append(spec)
        occupied_in |= spec["input_cells"]
        occupied_out |= spec["output_cells"]
        reserved_bg |= spec["reserved_bg"]

    for spec in specs:
        for cell in spec["input_cells"]:
            grid[cell] = spec["color"]

    if len(colors_present(grid, exclude={bg})) != len(colors):
        return None
    if not _validate_generated_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_SAMPLE_RETRIES):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid hidden-anchor ray sample")
