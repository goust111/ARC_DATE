from __future__ import annotations

import random

import numpy as np


BG = 0
SHADOW = 5
ANCHOR = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_HEIGHT_RANGE = (20, 28)
GRID_WIDTH_RANGE = (24, 32)
OBJECT_COUNT_RANGE = (2, 4)
SHADOW_LENGTH_RANGE = (2, 8)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _draw_anchor(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        grid[rr, cc] = ANCHOR


def _find_anchor_center(grid: np.ndarray) -> tuple[int, int]:
    maroon_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == ANCHOR)}
    centers = []
    for r, c in maroon_cells:
        plus = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        if plus <= maroon_cells:
            centers.append((r, c))
    assert len(centers) == 1, "expected exactly one maroon anchor cross"
    return centers[0]


def _find_objects(grid: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BG, SHADOW, ANCHOR):
            continue
        for comp in _connected_components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            assert center in set(comp), "object must contain its bbox center pixel"

            shadow_candidates = []
            for r, c in comp:
                if c != c1 or c + 1 >= grid.shape[1] or int(grid[r, c + 1]) != SHADOW:
                    continue
                seg: list[tuple[int, int]] = []
                cc = c + 1
                while cc < grid.shape[1] and int(grid[r, cc]) == SHADOW:
                    seg.append((r, cc))
                    cc += 1
                shadow_candidates.append((len(seg), abs(r - center[0]), seg))

            assert shadow_candidates, "every object must have a shadow"
            shadow_candidates.sort(key=lambda item: (-item[0], item[1], item[2][0][0], item[2][0][1]))
            shadow_len, _dist, shadow_cells = shadow_candidates[0]

            objects.append(
                {
                    "color": color,
                    "cells": comp,
                    "center": center,
                    "shadow_len": shadow_len,
                    "shadow_cells": shadow_cells,
                }
            )
    assert objects, "expected at least one object"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack colored objects onto the maroon anchor, using shadow length as layer depth.

    Objects with longer grey shadows are placed first (lower layers). Shorter-shadow
    objects are drawn later and therefore appear on top. The maroon anchor cross is
    preserved above the stack.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    anchor_center = _find_anchor_center(g)
    objects = _find_objects(g)

    out = np.full_like(g, bg)
    for obj in sorted(objects, key=lambda item: (-int(item["shadow_len"]), item["center"][0], item["center"][1])):
        cr, cc = obj["center"]
        dr = anchor_center[0] - int(cr)
        dc = anchor_center[1] - int(cc)
        for r, c in obj["cells"]:
            rr = int(r) + dr
            cc2 = int(c) + dc
            if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
                out[rr, cc2] = int(obj["color"])

    _draw_anchor(out, anchor_center)
    return out


def _make_shape() -> tuple[np.ndarray, tuple[int, int]]:
    h, w = random.choice(((3, 3), (3, 5), (5, 3), (5, 5)))
    cr, cc = h // 2, w // 2

    cells = {(cr, c) for c in range(w)} | {(r, cc) for r in range(h)}
    target_area = random.randint(len(cells), h * w)
    while len(cells) < target_area:
        frontier = []
        for r, c in list(cells):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    patch = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    return patch, (cr, cc)


def _footprint_clear(
    occupied: set[tuple[int, int]],
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    gap: int = 1,
) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _render_expected(
    shape_specs: list[dict[str, object]],
    grid_shape: tuple[int, int],
    anchor_center: tuple[int, int],
) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    for spec in sorted(shape_specs, key=lambda item: (-int(item["shadow_len"]), item["center"][0], item["center"][1])):
        cr, cc = spec["center"]
        dr = anchor_center[0] - int(cr)
        dc = anchor_center[1] - int(cc)
        for r, c in spec["cells"]:
            out[int(r) + dr, int(c) + dc] = int(spec["color"])
    _draw_anchor(out, anchor_center)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    anchor_center = (h - random.choice((3, 4)), w - random.choice((3, 4)))
    if anchor_center[0] < 4 or anchor_center[1] < 4:
        return None
    _draw_anchor(grid, anchor_center)

    occupied = {
        (r, c)
        for r in range(anchor_center[0] - 3, anchor_center[0] + 4)
        for c in range(anchor_center[1] - 3, anchor_center[1] + 4)
        if 0 <= r < h and 0 <= c < w
    }

    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(OBJECT_COLORS, n_objects)
    shadow_lengths = random.sample(range(SHADOW_LENGTH_RANGE[0], SHADOW_LENGTH_RANGE[1] + 1), n_objects)
    random.shuffle(shadow_lengths)

    specs: list[dict[str, object]] = []
    for color, shadow_len in zip(colors, shadow_lengths):
        patch, center_rel = _make_shape()
        ph, pw = patch.shape
        max_r0 = anchor_center[0] - ph - 3
        max_c0 = anchor_center[1] - pw - shadow_len - 4
        if max_r0 < 1 or max_c0 < 1:
            return None

        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            object_cells = {
                (r0 + int(r), c0 + int(c))
                for r, c in np.argwhere(patch)
            }
            shadow_row = r0 + center_rel[0]
            shadow_start = c0 + pw
            shadow_cells = {(shadow_row, shadow_start + d) for d in range(shadow_len)}
            footprint = object_cells | shadow_cells
            if not _footprint_clear(occupied, footprint, h, w):
                continue

            for r, c in object_cells:
                grid[r, c] = color
            for r, c in shadow_cells:
                grid[r, c] = SHADOW

            specs.append(
                {
                    "color": color,
                    "cells": sorted(object_cells),
                    "center": (r0 + center_rel[0], c0 + center_rel[1]),
                    "shadow_len": shadow_len,
                }
            )
            occupied |= footprint
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(specs, grid.shape, anchor_center)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
