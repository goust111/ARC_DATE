from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
DIR_VECTORS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
RETRY_LIMIT = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _decode_arrow(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, tuple[int, int], str]:
    assert len(coords) == 2, "arrow must contain exactly two pixels"
    (r0, c0), (r1, c1) = coords
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "arrow pixels must be orthogonally adjacent"
    v0 = int(grid[r0, c0])
    v1 = int(grid[r1, c1])
    assert {v0, v1} != {MAROON}, "arrow must include exactly one non-maroon color pixel"
    assert MAROON in (v0, v1), "arrow must include a maroon tip"
    if v0 == MAROON:
        color = v1
        dr = r0 - r1
        dc = c0 - c1
    else:
        color = v0
        dr = r1 - r0
        dc = c1 - c0
    assert color not in (BG, MAROON), "rectangle colors must differ from black and maroon"
    direction = {
        (-1, 0): "U",
        (1, 0): "D",
        (0, -1): "L",
        (0, 1): "R",
    }[(dr, dc)]
    return color, (min(r0, r1), min(c0, c1)), direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored solid rectangle according to its same-color arrow segments.

    Each arrow is a 2-pixel component made of one rectangle-colored pixel and one
    maroon pixel. The maroon endpoint is the arrow tip; the object moves by its
    own height for vertical arrows or by its own width for horizontal arrows.
    Multiple arrows of the same color are executed in row-major order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w > 0, "grid must be non-empty"
    assert np.all((0 <= g) & (g <= 9)), "ARC grids must use color ids 0..9"

    rectangles: dict[int, tuple[int, int, int, int]] = {}
    arrows: dict[int, list[tuple[tuple[int, int], str]]] = defaultdict(list)

    for comp in _components(g != BG):
        colors = {int(g[r, c]) for r, c in comp}
        if MAROON in colors:
            color, key, direction = _decode_arrow(g, comp)
            arrows[color].append((key, direction))
            continue

        assert len(colors) == 1, "non-arrow components must be monochrome"
        color = next(iter(colors))
        assert color != MAROON, "maroon is reserved for arrow tips"
        assert _is_solid_rectangle(comp), "foreground object must be a solid rectangle"
        r0, c0, r1, c1 = _bbox(comp)
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        assert obj_h <= 3 and obj_w <= 3, "rectangle objects must be at most 3x3"
        assert color not in rectangles, "each rectangle color must identify exactly one object"
        rectangles[color] = (r0, c0, obj_h, obj_w)

    assert rectangles, "expected at least one rectangle object"
    assert set(arrows) == set(rectangles), "every rectangle must have one or more matching arrows"

    out = np.full_like(g, BG)
    for color, (r, c, obj_h, obj_w) in rectangles.items():
        cur_r = int(r)
        cur_c = int(c)
        for _key, direction in sorted(arrows[color], key=lambda item: item[0]):
            dr, dc = DIR_VECTORS[direction]
            step = obj_h if dr != 0 else obj_w
            cur_r += dr * step
            cur_c += dc * step
            assert 0 <= cur_r <= h - obj_h, "movement must stay inside the grid"
            assert 0 <= cur_c <= w - obj_w, "movement must stay inside the grid"
        if np.any(out[cur_r : cur_r + obj_h, cur_c : cur_c + obj_w] != BG):
            raise AssertionError("moved rectangles may not overlap")
        out[cur_r : cur_r + obj_h, cur_c : cur_c + obj_w] = color
    return out


def _rect_cells(r: int, c: int, obj_h: int, obj_w: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for dr in range(obj_h) for dc in range(obj_w)}


def _moat(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _fits(cells: set[tuple[int, int]], blocked: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in blocked:
            return False
    return True


def _prefix_offsets(obj_h: int, obj_w: int, directions: list[str]) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    cur_r = 0
    cur_c = 0
    for direction in directions:
        dr, dc = DIR_VECTORS[direction]
        step = obj_h if dr != 0 else obj_w
        cur_r += dr * step
        cur_c += dc * step
        offsets.append((cur_r, cur_c))
    return offsets


def _arrow_cells(direction: str, base_r: int, base_c: int) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = DIR_VECTORS[direction]
    return (base_r, base_c), (base_r + dr, base_c + dc)


def _arrow_key(cells: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, int]:
    (r0, c0), (r1, c1) = cells
    return min(r0, r1), min(c0, c1)


def _expected_output(shape: tuple[int, int], objects: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for obj in objects:
        r, c = obj["final"]
        obj_h = int(obj["h"])
        obj_w = int(obj["w"])
        color = int(obj["color"])
        if np.any(out[r : r + obj_h, c : c + obj_w] != BG):
            raise AssertionError("generator produced overlapping final rectangles")
        out[r : r + obj_h, c : c + obj_w] = color
    return out


def _sample_object_specs() -> list[dict[str, object]] | None:
    num_objects = random.randint(2, 4)
    colors = random.sample(PALETTE, num_objects)
    multi_idx = random.randrange(num_objects) if random.random() < 0.75 else -1
    nonsquare_idx = random.randrange(num_objects)
    objects: list[dict[str, object]] = []
    all_dirs: list[str] = []
    for idx, color in enumerate(colors):
        if idx == nonsquare_idx:
            obj_h, obj_w = random.choice([(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)])
        else:
            obj_h = random.randint(1, 3)
            obj_w = random.randint(1, 3)
        count_low = 2 if idx == multi_idx else 1
        count_high = 4 if idx == multi_idx else 3
        directions = [random.choice(list(DIR_VECTORS)) for _ in range(random.randint(count_low, count_high))]
        all_dirs.extend(directions)
        objects.append(
            {
                "color": color,
                "h": obj_h,
                "w": obj_w,
                "directions": directions,
                "offsets": _prefix_offsets(obj_h, obj_w, directions),
            }
        )
    if not any(d in ("U", "D") for d in all_dirs):
        return None
    if not any(d in ("L", "R") for d in all_dirs):
        return None
    return objects


def _place_rectangles(shape: tuple[int, int], objects: list[dict[str, object]]) -> bool:
    h, w = shape
    input_blocked: set[tuple[int, int]] = set()
    final_blocked: set[tuple[int, int]] = set()
    order = sorted(
        range(len(objects)),
        key=lambda idx: (
            len(objects[idx]["directions"]),
            int(objects[idx]["h"]) * int(objects[idx]["w"]),
        ),
        reverse=True,
    )
    for idx in order:
        obj = objects[idx]
        obj_h = int(obj["h"])
        obj_w = int(obj["w"])
        offsets = list(obj["offsets"])
        min_dr = min(dr for dr, _ in offsets)
        max_dr = max(dr for dr, _ in offsets)
        min_dc = min(dc for _, dc in offsets)
        max_dc = max(dc for _, dc in offsets)
        r_lo = -min_dr
        r_hi = h - obj_h - max_dr
        c_lo = -min_dc
        c_hi = w - obj_w - max_dc
        if r_lo > r_hi or c_lo > c_hi:
            return False

        candidates: list[tuple[int, int]] = []
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                start_cells = _rect_cells(r, c, obj_h, obj_w)
                final_r = r + offsets[-1][0]
                final_c = c + offsets[-1][1]
                final_cells = _rect_cells(final_r, final_c, obj_h, obj_w)
                if _fits(start_cells, input_blocked, shape) and _fits(final_cells, final_blocked, shape):
                    candidates.append((r, c))
        if not candidates:
            return False

        start_r, start_c = random.choice(candidates)
        final_r = start_r + offsets[-1][0]
        final_c = start_c + offsets[-1][1]
        start_cells = _rect_cells(start_r, start_c, obj_h, obj_w)
        final_cells = _rect_cells(final_r, final_c, obj_h, obj_w)
        input_blocked |= _moat(start_cells, shape)
        final_blocked |= _moat(final_cells, shape)
        obj["start"] = (start_r, start_c)
        obj["final"] = (final_r, final_c)

    return True


def _place_arrows(shape: tuple[int, int], objects: list[dict[str, object]]) -> list[dict[str, object]] | None:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for obj in objects:
        start_r, start_c = obj["start"]
        cells = _rect_cells(start_r, start_c, int(obj["h"]), int(obj["w"]))
        blocked |= _moat(cells, shape)

    arrows: list[dict[str, object]] = []
    order = sorted(range(len(objects)), key=lambda idx: len(objects[idx]["directions"]), reverse=True)
    for idx in order:
        obj = objects[idx]
        prev_key = (-1, -1)
        for direction in obj["directions"]:
            candidates: list[tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]] = []
            for r in range(h):
                for c in range(w):
                    cells = _arrow_cells(direction, r, c)
                    key = _arrow_key(cells)
                    cell_set = {cells[0], cells[1]}
                    if key <= prev_key:
                        continue
                    if _fits(cell_set, blocked, shape):
                        candidates.append((key, cells))
            if not candidates:
                return None
            key, cells = random.choice(candidates)
            blocked |= _moat({cells[0], cells[1]}, shape)
            arrows.append(
                {
                    "color": int(obj["color"]),
                    "direction": direction,
                    "cells": cells,
                    "key": key,
                }
            )
            prev_key = key
    return arrows


def _render_input(shape: tuple[int, int], objects: list[dict[str, object]], arrows: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        r, c = obj["start"]
        obj_h = int(obj["h"])
        obj_w = int(obj["w"])
        grid[r : r + obj_h, c : c + obj_w] = int(obj["color"])
    for arrow in arrows:
        color = int(arrow["color"])
        direction = str(arrow["direction"])
        (base_r, base_c), (tip_r, tip_c) = arrow["cells"]
        if direction in ("U", "D", "L", "R"):
            grid[base_r, base_c] = color
            grid[tip_r, tip_c] = MAROON
        else:
            raise AssertionError("unknown direction")
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 17)
    w = random.randint(11, 17)
    shape = (h, w)

    objects = _sample_object_specs()
    if objects is None:
        return None
    if not _place_rectangles(shape, objects):
        return None
    arrows = _place_arrows(shape, objects)
    if arrows is None:
        return None

    grid = _render_input(shape, objects, arrows)
    expected = _expected_output(shape, objects)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {RETRY_LIMIT} attempts")
