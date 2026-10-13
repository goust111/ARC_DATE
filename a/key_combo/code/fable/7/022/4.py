from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]

UP: Direction = (-1, 0)
RIGHT: Direction = (0, 1)
DOWN: Direction = (1, 0)
LEFT: Direction = (0, -1)
DIRS: tuple[Direction, ...] = (UP, RIGHT, DOWN, LEFT)
CW_TURN = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )

    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _trace_spiral(shape: tuple[int, int], marker: Coord, direction: Direction, step: int) -> list[Coord]:
    current = (int(marker[0]), int(marker[1]))
    heading = direction
    segment_index = 0
    path: list[Coord] = []

    while True:
        segment_len = int(step + segment_index // 2)
        dr, dc = heading
        for _ in range(segment_len):
            nr = current[0] + dr
            nc = current[1] + dc
            if not in_bounds(shape, nr, nc):
                return path
            current = (nr, nc)
            path.append(current)
        heading = CW_TURN[heading]
        segment_index += 1


def _bar_cells(marker: Coord, direction: Direction, step: int) -> list[Coord]:
    dr, dc = direction
    return [(marker[0] - dr * k, marker[1] - dc * k) for k in range(1, int(step) + 1)]


def _object_from_component(comp: dict) -> dict:
    coords = [(int(r), int(c)) for r, c in comp["coords"]]
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    color = int(comp["color"])

    if len(rows) == 1:
        r = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert len(coords) == c1 - c0 + 1 and len(coords) >= 2, "horizontal control bar must be contiguous"
        return {
            "color": color,
            "step": len(coords) - 1,
            "candidates": [((r, c0), LEFT), ((r, c1), RIGHT)],
            "bbox": comp["bbox"],
        }

    if len(cols) == 1:
        c = cols[0]
        r0, r1 = rows[0], rows[-1]
        assert len(coords) == r1 - r0 + 1 and len(coords) >= 2, "vertical control bar must be contiguous"
        return {
            "color": color,
            "step": len(coords) - 1,
            "candidates": [((r0, c), UP), ((r1, c), DOWN)],
            "bbox": comp["bbox"],
        }

    raise AssertionError("each foreground object must be a single straight control bar with one endpoint marker")


def _find_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)
    objects = [_object_from_component(comp) for comp in _find_color_components(g, bg)]
    assert objects, "expected at least one control bar object"
    return bg, objects


def _build_candidate(shape: tuple[int, int], obj: dict, candidate_index: int) -> dict:
    marker, direction = obj["candidates"][candidate_index]
    path = _trace_spiral(shape, marker, direction, int(obj["step"]))
    cells = [marker, *path]
    assert len(cells) == len(set(cells)), "single spiral must not self-overlap"
    return {
        "marker": marker,
        "direction": direction,
        "path": path,
        "cells": set(cells),
    }


def _find_valid_solutions(shape: tuple[int, int], objects: list[dict]) -> list[tuple[tuple[int, ...], list[dict]]]:
    candidate_cache = [
        [_build_candidate(shape, obj, 0), _build_candidate(shape, obj, 1)]
        for obj in objects
    ]
    valid: list[tuple[tuple[int, ...], list[dict]]] = []

    for choice_bits in itertools.product((0, 1), repeat=len(objects)):
        occupied: set[Coord] = set()
        placements: list[dict] = []
        ok = True

        for obj_index, bit in enumerate(choice_bits):
            cand = candidate_cache[obj_index][bit]
            for cell in cand["cells"]:
                if cell in occupied:
                    ok = False
                    break
                if any(nei in occupied for nei in neighbors8(*cell)):
                    ok = False
                    break
            if not ok:
                break
            occupied.update(cand["cells"])
            placements.append(cand)

        if ok:
            valid.append((tuple(map(int, choice_bits)), placements))

    return valid


def _render_solution(grid: np.ndarray, bg: int, objects: list[dict], placements: list[dict]) -> np.ndarray:
    out = np.full_like(np.asarray(grid, dtype=int), int(bg))
    for obj, placement in zip(objects, placements):
        color = int(obj["color"])
        mr, mc = placement["marker"]
        out[mr, mc] = color
        for r, c in placement["path"]:
            out[r, c] = color
    return out


def _solve(grid: np.ndarray) -> tuple[int, list[dict], tuple[int, ...], list[dict]]:
    bg, objects = _find_objects(grid)
    solutions = _find_valid_solutions(np.asarray(grid).shape, objects)
    assert len(solutions) == 1, "control-bar endpoints must determine a unique non-touching spiral interpretation"
    bits, placements = solutions[0]
    return bg, objects, bits, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each endpoint-marked straight bar into an outward clockwise spiral and erase the bars.

    Input structure:
    - A black background contains several monochrome straight-line control objects.
    - Each object is a same-color contiguous horizontal or vertical line whose true marker is one endpoint.
    - The line length minus that endpoint gives the initial step size, and the scene is arranged so the intended spirals are uniquely the non-touching ones.

    Rule:
    - For each object, treat one endpoint as the marker and the rest of the same-color line as its attached bar.
    - The first segment starts from the marker and points away from the attached bar; then draw segment lengths N, N, N+1, N+1, N+2, N+2, ... while turning 90 degrees clockwise after every segment.
    - Stop a spiral as soon as its next step would leave the grid, paint it in the object's color, erase the original bar cells to background, and keep the marker pixel.

    Key insight:
    - The bar is a compound key: its orientation chooses the outward launch direction, its length chooses the initial step, and the globally unique non-touching spiral assignment disambiguates which endpoint is the true marker.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects, _bits, placements = _solve(g)
    return _render_solution(g, bg, objects, placements)


def _cells_touch(cells: set[Coord], occupied: set[Coord]) -> bool:
    for cell in cells:
        if cell in occupied:
            return True
        if any(nei in occupied for nei in neighbors8(*cell)):
            return True
    return False


def _place_objects(shape: tuple[int, int], specs: list[tuple[int, Direction, int]]) -> list[dict] | None:
    h, w = shape
    center_row = random.randint(max(4, h // 3), min(h - 5, (2 * h) // 3))
    center_col = random.randint(max(4, w // 3), min(w - 5, (2 * w) // 3))
    output_occupied: set[Coord] = set()
    input_occupied: set[Coord] = set()
    placed: list[dict] = []

    for color, direction, step in sorted(specs, key=lambda item: -item[2]):
        success = False
        for _ in range(300):
            if direction == UP:
                row_lo = int(step)
                row_hi = min(h // 3, int(step + 3))
                marker = (random.randint(row_lo, row_hi), center_col)
            elif direction == DOWN:
                row_lo = max((2 * h) // 3, h - 4 - step)
                row_hi = h - 1 - step
                marker = (random.randint(row_lo, row_hi), center_col)
            elif direction == LEFT:
                col_lo = int(step)
                col_hi = min(w // 3, int(step + 3))
                marker = (center_row, random.randint(col_lo, col_hi))
            else:
                col_lo = max((2 * w) // 3, w - 4 - step)
                col_hi = w - 1 - step
                marker = (center_row, random.randint(col_lo, col_hi))

            path = _trace_spiral(shape, marker, direction, step)
            bar = _bar_cells(marker, direction, step)
            output_cells = {marker, *path}
            input_cells = {marker, *bar}

            if len(path) < max(4, step + 1):
                continue
            if _cells_touch(output_cells, output_occupied):
                continue
            if _cells_touch(input_cells, input_occupied):
                continue

            placed.append(
                {
                    "color": int(color),
                    "step": int(step),
                    "marker": marker,
                    "direction": direction,
                    "path": path,
                    "output_cells": output_cells,
                    "input_cells": input_cells,
                }
            )
            output_occupied.update(output_cells)
            input_occupied.update(input_cells)
            success = True
            break

        if not success:
            return None

    return placed


def _draw_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in placed:
        color = int(obj["color"])
        for r, c in obj["input_cells"]:
            grid[r, c] = color
    return grid


def _draw_output(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in placed:
        color = int(obj["color"])
        mr, mc = obj["marker"]
        grid[mr, mc] = color
        for r, c in obj["path"]:
            grid[r, c] = color
    return grid


def _semantic_checks(inp: np.ndarray) -> None:
    bg, objects, bits, placements = _solve(inp)
    solved = _render_solution(inp, bg, objects, placements)
    assert np.array_equal(main(inp), solved)
    assert len(objects) >= 3, "generator must produce multiple bars so endpoint disambiguation matters"
    assert len(set(int(obj["step"]) for obj in objects)) >= 2, "samples should expose different initial step lengths"

    for obj, bit, placement in zip(objects, bits, placements):
        other = _build_candidate(inp.shape, obj, 1 - int(bit))
        assert other["marker"] != placement["marker"] or other["path"] != placement["path"], (
            "each object's attachment side must affect the result"
        )

    all_solutions = _find_valid_solutions(inp.shape, objects)
    assert len(all_solutions) == 1


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(16, 28)
        w = random.randint(16, 28)
        count = random.randint(3, 4)
        colors = random.sample(COLOR_POOL, count)
        directions = random.sample(DIRS, count)
        while True:
            steps = [random.randint(1, 5) for _ in range(count)]
            if len(set(steps)) >= 2:
                break

        placed = _place_objects((h, w), list(zip(colors, directions, steps)))
        if placed is None:
            continue

        inp = _draw_input((h, w), placed)
        expected = _draw_output((h, w), placed)

        try:
            actual = main(inp)
            _semantic_checks(inp)
        except AssertionError:
            continue

        if not np.array_equal(actual, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a valid spiral-control ARC input")
