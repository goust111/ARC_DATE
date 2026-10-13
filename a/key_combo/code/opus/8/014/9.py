from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIVIDER_COLOR = GREY
BARRIER_COLOR = RED
SPRITE_COLORS = [BLUE, GREEN, YELLOW, ORANGE, PINK, TEAL, MAROON]

WIDE_SHAPES = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=int),
]

TALL_SHAPES = [shape.T.copy() for shape in WIDE_SHAPES]

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (16, 22)
LEFT_W_RANGE = (7, 10)
SPRITE_COUNT_RANGE = (3, 5)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_divider_col(grid: np.ndarray) -> int:
    cols = np.where(np.all(grid == DIVIDER_COLOR, axis=0))[0]
    assert len(cols) == 1, "expected exactly one full-height grey divider column"
    return int(cols[0])


def _count_left_black_regions(grid: np.ndarray, divider_col: int) -> int:
    left = grid[:, :divider_col]
    return len(_components_from_mask(left == BLACK))


def _sprite_components(grid: np.ndarray, divider_col: int) -> list[dict]:
    right = grid[:, divider_col + 1 :]
    comps = []
    for coords in _components_from_mask((right != BLACK) & (right != DIVIDER_COLOR)):
        global_coords = coords.copy()
        global_coords[:, 1] += divider_col + 1
        colors = {int(grid[r, c]) for r, c in global_coords}
        assert len(colors) == 1, "each sprite must be monochrome"
        comps.append(
            {
                "coords": global_coords,
                "color": colors.pop(),
                "bbox": _bbox(global_coords),
            }
        )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each right-side sprite by a distance encoded by left-side background-region count.

    Input structure:
    - A black grid is split by one full-height grey divider column into a left barrier zone and a right sprite zone.
    - The left zone contains a connected red barrier that partitions the black background into 2-5 four-connected regions.
    - The right zone contains 3-5 separated monochrome sprites; every sprite has a non-square bounding box.

    Rule:
    - Find the grey divider and count the number N of four-connected black background regions in the left zone.
    - For each right-zone sprite, inspect its bounding-box orientation.
    - If the bbox is wider than tall, erase the sprite and redraw it shifted right by N cells; if taller than wide, shift it down by N cells.
    - Keep the entire left zone and the divider unchanged.

    Key insight:
    - The two slide parameters come from different places: the left barrier topology gives the distance, while each sprite's own geometry gives the movement axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_col = _find_divider_col(g)
    shift = _count_left_black_regions(g, divider_col)
    sprites = _sprite_components(g, divider_col)
    assert 2 <= shift <= 5
    assert 3 <= len(sprites) <= 5

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    h, w = g.shape
    occupied = np.zeros(g.shape, dtype=bool)
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert width != height, "sprite bbox must be non-square"
        delta = np.array([0, shift], dtype=int) if width > height else np.array([shift, 0], dtype=int)
        dst = sprite["coords"] + delta
        assert bool((0 <= dst[:, 0]).all() and (dst[:, 0] < h).all())
        assert bool((divider_col < dst[:, 1]).all() and (dst[:, 1] < w).all())
        assert not bool(occupied[dst[:, 0], dst[:, 1]].any()), "shifted sprites must not overlap"
        out[dst[:, 0], dst[:, 1]] = int(sprite["color"])
        occupied[dst[:, 0], dst[:, 1]] = True

    return out


def _shape_coords(shape: np.ndarray) -> np.ndarray:
    return np.argwhere(shape != 0).astype(int)


def _sample_barrier_rows(height: int, arms: int) -> list[int] | None:
    if arms == 0:
        return []
    candidates = list(range(2, height - 2))
    if len(candidates) < arms:
        return None
    for _ in range(100):
        rows = sorted(random.sample(candidates, arms))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _make_left_barrier(height: int, left_w: int, regions: int) -> np.ndarray | None:
    arms = regions - 2
    if left_w < 6 or height < 8:
        return None
    for _ in range(100):
        spine_col = random.randint(2, left_w - 3)
        rows = _sample_barrier_rows(height, arms)
        if rows is None:
            return None
        mask = np.zeros((height, left_w), dtype=bool)
        mask[:, spine_col] = True
        side = random.choice(["left", "right"])
        for row in rows:
            if side == "left":
                mask[row, : spine_col + 1] = True
            else:
                mask[row, spine_col:] = True
            side = "right" if side == "left" else "left"
        grid = np.zeros((height, left_w), dtype=int)
        grid[mask] = BARRIER_COLOR
        if _count_left_black_regions(np.hstack([grid, np.full((height, 1), DIVIDER_COLOR, dtype=int)]), left_w) == regions:
            return grid
    return None


def _chebyshev_separated(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> bool:
    for ar, ac in cells_a:
        for br, bc in cells_b:
            if max(abs(ar - br), abs(ac - bc)) <= 1:
                return False
    return True


def _can_place(
    orig_cells: list[tuple[int, int]],
    dst_cells: list[tuple[int, int]],
    existing_originals: list[list[tuple[int, int]]],
    occupied_all: set[tuple[int, int]],
) -> bool:
    orig_set = set(orig_cells)
    dst_set = set(dst_cells)
    if orig_set & occupied_all:
        return False
    if dst_set & occupied_all:
        return False
    for prev in existing_originals:
        if not _chebyshev_separated(orig_cells, prev):
            return False
    return True


def _place_sprites(height: int, width: int, divider_col: int, shift: int) -> tuple[np.ndarray, list[dict]] | None:
    right_w = width - divider_col - 1
    sprite_count = random.randint(*SPRITE_COUNT_RANGE)
    orientations = ["wide", "tall"]
    while len(orientations) < sprite_count:
        orientations.append(random.choice(["wide", "tall"]))
    random.shuffle(orientations)

    colors = random.sample(SPRITE_COLORS, sprite_count)
    canvas = np.zeros((height, width), dtype=int)
    specs = []
    existing_originals: list[list[tuple[int, int]]] = []
    occupied_all: set[tuple[int, int]] = set()

    for orientation, color in zip(orientations, colors):
        shapes = WIDE_SHAPES if orientation == "wide" else TALL_SHAPES
        placed = False
        shape_order = random.sample(shapes, len(shapes))
        for shape in shape_order:
            local = _shape_coords(shape)
            box_h, box_w = shape.shape
            max_r = height - box_h - (shift if orientation == "tall" else 0)
            max_c = right_w - box_w - (shift if orientation == "wide" else 0)
            if max_r < 0 or max_c < 0:
                continue
            for _ in range(80):
                r0 = random.randint(0, max_r)
                c0 = random.randint(0, max_c)
                global_orig = [(r0 + int(r), divider_col + 1 + c0 + int(c)) for r, c in local]
                if orientation == "wide":
                    global_dst = [(r, c + shift) for r, c in global_orig]
                else:
                    global_dst = [(r + shift, c) for r, c in global_orig]
                if not _can_place(global_orig, global_dst, existing_originals, occupied_all):
                    continue
                for r, c in global_orig:
                    canvas[r, c] = color
                specs.append(
                    {
                        "color": color,
                        "orientation": orientation,
                        "shape": shape.copy(),
                        "origin": (r0, divider_col + 1 + c0),
                        "orig_cells": global_orig,
                        "dst_cells": global_dst,
                    }
                )
                existing_originals.append(global_orig)
                occupied_all.update(global_orig)
                occupied_all.update(global_dst)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return canvas, specs


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec["orig_cells"]:
            out[r, c] = BLACK
    for spec in specs:
        for r, c in spec["dst_cells"]:
            out[r, c] = int(spec["color"])
    return out


def _validate_instance(inp: np.ndarray, out: np.ndarray, regions: int, specs: list[dict], divider_col: int) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not np.all(inp[:, divider_col] == DIVIDER_COLOR):
        return False
    if _count_left_black_regions(inp, divider_col) != regions:
        return False
    if not any(spec["orientation"] == "wide" for spec in specs):
        return False
    if not any(spec["orientation"] == "tall" for spec in specs):
        return False
    if not np.array_equal(out, _expected_output(inp, specs)):
        return False
    if not np.array_equal(main(inp), out):
        return False

    changed = np.argwhere(inp != out)
    if len(changed) == 0:
        return False
    if np.any(changed[:, 1] <= divider_col):
        return False
    return True


def _generate_one() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    left_w = random.randint(*LEFT_W_RANGE)
    if left_w >= width - 5:
        return None

    regions = random.randint(2, 5)
    left = _make_left_barrier(height, left_w, regions)
    if left is None:
        return None

    divider = np.full((height, 1), DIVIDER_COLOR, dtype=int)
    right, specs = _place_sprites(height, width, left_w, regions) or (None, None)
    if right is None:
        return None

    inp = np.hstack([left, divider, right[:, left_w + 1 :]])
    out = _expected_output(inp, specs)
    return inp if _validate_instance(inp, out, regions, specs, left_w) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
