from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BG = 0
PLUS = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

# Generator controls.
GRID_MIN = 12
GRID_MAX = 18
PLUS_COUNT_CHOICES = (2, 3)
CENTER_SEPARATION = 5
MIN_TAIL_EXTENSION = 1
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _find_pluses(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    infos: list[dict] = []
    for comp in _components4(g == PLUS):
        if len(comp) != 5:
            continue
        comp_set = set(comp)
        centers = [p for p in comp if sum((nr, nc) in comp_set for nr, nc in _neighbors4(*p)) == 4]
        if len(centers) != 1:
            continue
        center = centers[0]
        expected = {center, *list(_neighbors4(*center))}
        if comp_set != expected:
            continue

        marker_colors: dict[tuple[int, int], int] = {}
        valid = True
        for dr, dc in DIRS:
            mr = center[0] + 2 * dr
            mc = center[1] + 2 * dc
            if not _in_bounds(g.shape, mr, mc):
                valid = False
                break
            color = int(g[mr, mc])
            if color in (BG, PLUS):
                valid = False
                break
            marker_colors[(dr, dc)] = color
        if not valid or len(marker_colors) != 4:
            continue

        counts = Counter(marker_colors.values())
        if sorted(counts.values()) != [1, 3]:
            continue
        odd_color = next(color for color, count in counts.items() if count == 1)
        same_color = next(color for color, count in counts.items() if count == 3)
        odd_dir = next(direction for direction, color in marker_colors.items() if color == odd_color)

        infos.append(
            {
                "center": center,
                "plus_cells": comp_set,
                "marker_colors": marker_colors,
                "odd_dir": odd_dir,
                "odd_color": odd_color,
                "same_color": same_color,
            }
        )

    infos.sort(key=lambda info: info["center"])
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the uniquely colored arm-tip marker on each grey cross to choose a ray direction and color.

    Input structure:
    - The grid has a black background, plus 2-3 separate grey '+' objects made of exactly five pixels:
      one center pixel with one grey arm pixel in each cardinal direction.
    - At distance two from each center, on the same row or column as an arm, there is one colored arm-tip marker.
    - For each '+', exactly three of the four arm-tip markers share a color and one arm-tip marker has a different color.

    Rule:
    - For each grey '+', compare the four arm-tip markers and identify the uniquely colored one.
    - From the center, extend a straight cardinal ray toward that unique marker and continue to the grid edge,
      coloring background cells on the ray with the unique marker's color; the grey '+' pixels stay grey.
    - Erase the three same-colored arm-tip markers to black, and keep the unique marker in place.

    Key insight:
    - The outlier arm-tip marker simultaneously encodes both which direction to fire and which color to use.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for info in _find_pluses(g):
        center_r, center_c = info["center"]
        odd_dir = info["odd_dir"]
        odd_color = int(info["odd_color"])
        odd_marker = (center_r + 2 * odd_dir[0], center_c + 2 * odd_dir[1])

        for direction in DIRS:
            mr = center_r + 2 * direction[0]
            mc = center_c + 2 * direction[1]
            if direction != odd_dir:
                out[mr, mc] = BG

        for rr, cc in _ray_cells(g.shape, (center_r + odd_dir[0], center_c + odd_dir[1]), odd_dir):
            if (rr, cc) in info["plus_cells"]:
                continue
            if out[rr, cc] == BG or (rr, cc) == odd_marker:
                out[rr, cc] = odd_color

    return out


def _sample_directions(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    if count == 2:
        picks = rng.choice(len(DIRS), size=2, replace=False)
        return [DIRS[int(i)] for i in picks]

    count_unique = 3 if rng.random() < 0.45 else 2
    unique_dirs = [DIRS[int(i)] for i in rng.choice(len(DIRS), size=count_unique, replace=False)]
    dirs = unique_dirs.copy()
    while len(dirs) < count:
        dirs.append(unique_dirs[int(rng.integers(0, len(unique_dirs)))])
    rng.shuffle(dirs)
    return dirs


def _sample_color_pairs(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    while len(pairs) < count:
        same, odd = rng.choice(PALETTE, size=2, replace=False)
        pair = (int(same), int(odd))
        if pair in used:
            continue
        pairs.append(pair)
        used.add(pair)
    odd_colors = {odd for _same, odd in pairs}
    if len(odd_colors) == 1:
        index = int(rng.integers(0, count))
        same = pairs[index][0]
        choices = [color for color in PALETTE if color != same and color != pairs[index][1]]
        pairs[index] = (same, int(choices[int(rng.integers(0, len(choices)))]))
    return pairs


def _center_candidates(
    shape: tuple[int, int],
    direction: tuple[int, int],
    centers: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked_rays: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if any(max(abs(r - rr), abs(c - cc)) < CENTER_SEPARATION for rr, cc in centers):
                continue
            tail_len = len(_ray_cells(shape, (r + 2 * direction[0], c + 2 * direction[1]), direction)) - 1
            if tail_len < MIN_TAIL_EXTENSION:
                continue

            plus_cells = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            marker_cells = {(r + 2 * dr, c + 2 * dc) for dr, dc in DIRS}
            ray_cells = set(_ray_cells(shape, (r + 2 * direction[0], c + 2 * direction[1]), direction))

            if plus_cells & occupied:
                continue
            if marker_cells & occupied:
                continue
            if plus_cells & blocked_rays:
                continue
            if marker_cells & blocked_rays:
                continue
            if ray_cells & occupied:
                continue
            if ray_cells & blocked_rays:
                continue

            candidates.append((r, c))
    return candidates


def _render_input(
    shape: tuple[int, int],
    specs: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        r, c = spec["center"]
        grid[r, c] = PLUS
        grid[r - 1, c] = PLUS
        grid[r + 1, c] = PLUS
        grid[r, c - 1] = PLUS
        grid[r, c + 1] = PLUS
        for direction in DIRS:
            mr = r + 2 * direction[0]
            mc = c + 2 * direction[1]
            if direction == spec["odd_dir"]:
                grid[mr, mc] = spec["odd_color"]
            else:
                grid[mr, mc] = spec["same_color"]
    return grid


def _validate_semantics(inp: np.ndarray, out: np.ndarray, expected_count: int) -> None:
    infos = _find_pluses(inp)
    assert len(infos) == expected_count
    assert len({info["odd_dir"] for info in infos}) >= 2
    assert len({(info["same_color"], info["odd_color"]) for info in infos}) >= 2

    allowed_changes: set[tuple[int, int]] = set()
    expected_values: dict[tuple[int, int], int] = {}

    for info in infos:
        center_r, center_c = info["center"]
        odd_dir = info["odd_dir"]
        odd_color = int(info["odd_color"])
        odd_marker = (center_r + 2 * odd_dir[0], center_c + 2 * odd_dir[1])

        for direction in DIRS:
            marker = (center_r + 2 * direction[0], center_c + 2 * direction[1])
            if direction == odd_dir:
                assert int(out[marker]) == odd_color
            else:
                allowed_changes.add(marker)
                expected_values[marker] = BG

        tail = _ray_cells(inp.shape, odd_marker, odd_dir)
        assert len(tail) >= 2
        for cell in tail:
            if cell == odd_marker:
                assert int(out[cell]) == odd_color
                continue
            allowed_changes.add(cell)
            expected_values[cell] = odd_color

        for cell in info["plus_cells"]:
            assert int(out[cell]) == PLUS

    diff_cells = {(int(r), int(c)) for r, c in np.argwhere(inp != out)}
    assert diff_cells
    assert diff_cells <= allowed_changes
    for cell in diff_cells:
        assert int(out[cell]) == expected_values[cell]
    for r, c in np.argwhere(inp == out):
        cell = (int(r), int(c))
        if cell not in allowed_changes:
            assert int(out[cell]) == int(inp[cell])


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_ATTEMPTS):
        h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        shape = (h, w)
        plus_count = int(PLUS_COUNT_CHOICES[int(rng.integers(0, len(PLUS_COUNT_CHOICES)))])
        directions = _sample_directions(rng, plus_count)
        color_pairs = _sample_color_pairs(rng, plus_count)

        centers: list[tuple[int, int]] = []
        occupied: set[tuple[int, int]] = set()
        blocked_rays: set[tuple[int, int]] = set()
        specs: list[dict] = []
        feasible = True

        order = list(rng.permutation(plus_count))
        for idx in order:
            direction = directions[idx]
            candidates = _center_candidates(shape, direction, centers, occupied, blocked_rays)
            if not candidates:
                feasible = False
                break

            center = candidates[int(rng.integers(0, len(candidates)))]
            same_color, odd_color = color_pairs[idx]
            r, c = center
            plus_cells = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            marker_cells = {(r + 2 * dr, c + 2 * dc) for dr, dc in DIRS}
            ray_cells = set(_ray_cells(shape, (r + 2 * direction[0], c + 2 * direction[1]), direction))

            occupied |= plus_cells | marker_cells
            blocked_rays |= ray_cells
            centers.append(center)
            specs.append(
                {
                    "center": center,
                    "odd_dir": direction,
                    "same_color": same_color,
                    "odd_color": odd_color,
                }
            )

        if not feasible:
            continue

        inp = _render_input(shape, specs)
        out = main(inp)
        if np.array_equal(inp, out):
            continue

        try:
            _validate_semantics(inp, out, plus_count)
        except AssertionError:
            continue

        return inp

    raise RuntimeError("failed to generate a valid cross-voting ray sample")
