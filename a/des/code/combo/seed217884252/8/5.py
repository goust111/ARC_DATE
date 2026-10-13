from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
OBJECT_COUNT_OPTIONS = (2, 3, 4, 5)
AREA_RANGE = (3, 10)
GRID_H_MIN = 14
GRID_H_MAX = 26
GRID_W_MIN = 14
GRID_W_MAX = 26
TOP_MARGIN_RANGE = (0, 2)
BOTTOM_MARGIN_RANGE = (0, 2)
ROW_GAP_RANGE = (1, 3)
EXTRA_WIDTH_RANGE = (4, 9)
SHAPE_TRIES = 120


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_once(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize({(c, -r) for r, c in cells})


def _rotate(cells: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    out = cells
    for _ in range(turns % 4):
        out = _rotate_once(out)
    return out


def _bbox(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _grow_polyomino(area: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    for _ in range(area - 1):
        choices = list(frontier)
        rng.shuffle(choices)
        picked = None
        for cell in choices:
            trial = cells | {cell}
            h, w = _bbox(_normalize(trial))
            if max(h, w) <= 6:
                picked = cell
                break
        if picked is None:
            picked = choices[0]
        cells.add(picked)
        frontier.discard(picked)
        for dr, dc in NEIGHBORS4:
            nxt = (picked[0] + dr, picked[1] + dc)
            if nxt not in cells:
                frontier.add(nxt)
    return _normalize(cells)


def _line_shape(area: int, vertical: bool) -> tuple[tuple[int, int], ...]:
    if vertical:
        return tuple((r, 0) for r in range(area))
    return tuple((0, c) for c in range(area))


def _rectangle_shape(area: int, rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    factors = [
        (h, area // h)
        for h in range(2, int(area**0.5) + 1)
        if area % h == 0 and area // h >= 2
    ]
    if not factors:
        return None
    h, w = rng.choice(factors)
    cells = {(r, c) for r in range(h) for c in range(w)}
    return _normalize(cells)


def _l_shape(area: int, rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    if area < 3:
        return None
    options = []
    for arm_h in range(2, area):
        arm_w = area - arm_h + 1
        if arm_w >= 2:
            options.append((arm_h, arm_w))
    if not options:
        return None
    arm_h, arm_w = rng.choice(options)
    cells = {(r, 0) for r in range(arm_h)}
    cells |= {(arm_h - 1, c) for c in range(arm_w)}
    return _normalize(cells)


def _t_shape(area: int, rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    if area < 4:
        return None
    options = []
    for bar_w in range(3, area):
        stem_h = area - bar_w + 1
        if stem_h >= 2:
            options.append((bar_w, stem_h))
    if not options:
        return None
    bar_w, stem_h = rng.choice(options)
    stem_c = bar_w // 2
    cells = {(0, c) for c in range(bar_w)}
    cells |= {(r, stem_c) for r in range(stem_h)}
    return _normalize(cells)


def _plus_shape(area: int, rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    if area < 5:
        return None
    options = []
    for vert_h in range(3, area):
        horiz_w = area - vert_h + 1
        if horiz_w >= 3:
            options.append((vert_h, horiz_w))
    if not options:
        return None
    vert_h, horiz_w = rng.choice(options)
    center_r = vert_h // 2
    center_c = horiz_w // 2
    cells = {(r, center_c) for r in range(vert_h)}
    cells |= {(center_r, c) for c in range(horiz_w)}
    return _normalize(cells)


def _make_shape(
    area: int,
    kind: str,
    rng: random.Random,
) -> tuple[tuple[int, int], ...] | None:
    if kind == "line":
        return _line_shape(area, vertical=bool(rng.randrange(2)))
    if kind == "rectangle":
        return _rectangle_shape(area, rng)
    if kind == "l":
        return _l_shape(area, rng)
    if kind == "t":
        return _t_shape(area, rng)
    if kind == "plus":
        return _plus_shape(area, rng)
    if kind == "random":
        return _grow_polyomino(area, rng)
    raise ValueError(f"unknown shape kind: {kind}")


def _sample_object_specs(
    count: int, rng: random.Random
) -> list[dict[str, object]] | None:
    complex_kinds = ("l", "t", "plus", "random")
    all_kinds = ("line", "rectangle", "l", "t", "plus", "random")

    forced_complex_idx = rng.randrange(count)
    specs: list[dict[str, object]] = []
    for idx in range(count):
        for _ in range(SHAPE_TRIES):
            area = rng.randint(*AREA_RANGE)
            if idx == forced_complex_idx:
                kind = rng.choice(complex_kinds)
            else:
                kind = rng.choice(all_kinds)
            cells = _make_shape(area, kind, rng)
            if cells is None:
                continue
            turns = rng.randrange(4)
            cells = _rotate(cells, turns)
            h, w = _bbox(cells)
            if h > 6 or w > 8:
                continue
            specs.append(
                {
                    "area": area,
                    "kind": kind,
                    "cells": cells,
                    "shape_h": h,
                    "shape_w": w,
                    "complex": area != h * w,
                }
            )
            break
        else:
            return None

    if not any(bool(spec["complex"]) for spec in specs):
        return None
    return specs


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            objects.append(
                {
                    "color": color,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground object with a left-aligned horizontal segment whose
    length equals the object's pixel count, ordered by original top row.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    if not objects:
        return np.zeros((1, 1), dtype=int)

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    out_h = 2 * len(objects) - 1
    out_w = max(int(obj["area"]) for obj in objects)
    out = np.zeros((out_h, out_w), dtype=int)

    for idx, obj in enumerate(objects):
        out[2 * idx, : int(obj["area"])] = int(obj["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice(OBJECT_COUNT_OPTIONS)
    colors = rng.sample(list(PALETTE), n_objects)
    specs = _sample_object_specs(n_objects, rng)
    if specs is None:
        return None
    rng.shuffle(specs)

    top_margin = rng.randint(*TOP_MARGIN_RANGE)
    bottom_margin = rng.randint(*BOTTOM_MARGIN_RANGE)
    row_gaps = [rng.randint(*ROW_GAP_RANGE) for _ in range(n_objects - 1)]

    top_rows: list[int] = []
    current_top = top_margin
    for idx, spec in enumerate(specs):
        top_rows.append(current_top)
        current_top += int(spec["shape_h"])
        if idx < n_objects - 1:
            current_top += row_gaps[idx]

    needed_h = current_top + bottom_margin
    if not (GRID_H_MIN <= needed_h <= GRID_H_MAX):
        return None
    grid_h = rng.randint(needed_h, GRID_H_MAX)

    max_shape_w = max(int(spec["shape_w"]) for spec in specs)
    extra_w = rng.randint(*EXTRA_WIDTH_RANGE)
    needed_w = max(GRID_W_MIN, max_shape_w + extra_w)
    if needed_w > GRID_W_MAX:
        return None
    grid_w = rng.randint(needed_w, GRID_W_MAX)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for color, spec, top in zip(colors, specs, top_rows):
        cells = spec["cells"]
        shape_w = int(spec["shape_w"])
        left = rng.randint(0, grid_w - shape_w)
        for dr, dc in cells:
            grid[top + dr, left + dc] = color

    objects = _extract_objects(grid)
    if len(objects) != n_objects:
        return None

    top_values = [int(obj["bbox"][0]) for obj in objects]
    if len(set(top_values)) != len(top_values):
        return None

    if not any(int(obj["area"]) != (obj["bbox"][2] - obj["bbox"][0] + 1) * (obj["bbox"][3] - obj["bbox"][1] + 1) for obj in objects):
        return None

    out = main(grid)
    if out.shape[0] != 2 * n_objects - 1:
        return None
    if out.shape[1] != max(int(obj["area"]) for obj in objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
