from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 0
DOOR_COLOR = 5
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PAIR_COUNTS = [2, 3, 4]
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
MAX_DOOR_SIDE = 10
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    corner_counts = Counter(corners)
    color, count = corner_counts.most_common(1)[0]
    if count >= 3:
        return int(color)
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in cells]
    cols = [int(c) for _r, c in cells]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _normalize_shape(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [int(r) for r, _c in shape]
    cols = [int(c) for _r, c in shape]
    return int(max(rows) + 1), int(max(cols) + 1)


def _rotate90_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((int(c), int(-r)) for r, c in shape))


def _rotate180_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, w = _shape_size(shape)
    return _normalize_shape(tuple((int(h - 1 - r), int(w - 1 - c)) for r, c in shape))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    cur = _normalize_shape(shape)
    for _ in range(4):
        variants.add(cur)
        cur = _rotate90_shape(cur)
    return sorted(variants)


def _is_rot180_symmetric(shape: tuple[tuple[int, int], ...]) -> bool:
    return _rotate180_shape(shape) == shape


def _find_doors(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    door_mask = np.zeros((h, w), dtype=bool)
    doors: list[dict[str, object]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sh, sw = map(int, sub.shape)
            if sh < 3 or sw < 3:
                continue
            if np.any((sub != int(color)) & (sub != int(bg))):
                continue

            hole_comps = _components(sub == int(bg))
            if len(hole_comps) != 1:
                continue
            hole_rel = hole_comps[0]
            if not (3 <= len(hole_rel) <= 5):
                continue
            if any(rr in (0, sh - 1) or cc in (0, sw - 1) for rr, cc in hole_rel):
                continue
            if len(comp) + len(hole_rel) != sh * sw:
                continue

            hole_abs = [(int(r0 + rr), int(c0 + cc)) for rr, cc in hole_rel]
            hole_shape = _normalize_shape(hole_rel)
            doors.append(
                {
                    "color": int(color),
                    "cells": [(int(r), int(c)) for r, c in comp],
                    "hole": hole_abs,
                    "hole_shape": hole_shape,
                }
            )
            for r, c in comp:
                door_mask[int(r), int(c)] = True

    doors.sort(key=lambda d: _bbox(d["cells"]))
    return doors, door_mask


def _find_keys(grid: np.ndarray, bg: int, door_mask: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    key_mask = (g != int(bg)) & (~door_mask)
    keys: list[dict[str, object]] = []
    for comp in _components(key_mask):
        if not (3 <= len(comp) <= 5):
            continue
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        keys.append(
            {
                "color": int(color),
                "cells": [(int(r), int(c)) for r, c in comp],
                "shape": _normalize_shape(comp),
            }
        )
    keys.sort(key=lambda d: _bbox(d["cells"]))
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match rotated key shapes to door lockholes and recolor each door with its key.

    Input structure:
    - The background is a solid color, with empty space around all objects.
    - Several monochrome rectangular door panels contain a single connected background lockhole fully inside the rectangle.
    - Separate monochrome key shapes of other colors are scattered outside the doors.
    - Each key has 3 to 5 cells, and each lockhole has 3 to 5 cells.

    Rule:
    - Identify each door as a solid rectangle with one internal background hole.
    - For every key, rotate its shape by 180 degrees and match it to the unique door whose lockhole has that exact shape.
    - Remove the key from the grid.
    - Recolor all non-hole pixels of the matched door with the key's color, while keeping the lockhole as background.

    Key insight:
    - The pairing clue is not direct shape equality: every key must be mentally turned 180 degrees before it matches a lockhole, and that match determines the door's final fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    doors, door_mask = _find_doors(g, bg)
    keys = _find_keys(g, bg, door_mask)

    hole_to_door: dict[tuple[tuple[int, int], ...], int] = {}
    for idx, door in enumerate(doors):
        hole_shape = door["hole_shape"]
        assert hole_shape not in hole_to_door
        hole_to_door[hole_shape] = int(idx)

    out = g.copy()
    for key in keys:
        for r, c in key["cells"]:
            out[int(r), int(c)] = int(bg)

    used_doors: set[int] = set()
    for key in keys:
        rotated = _rotate180_shape(key["shape"])
        assert rotated in hole_to_door
        door_idx = hole_to_door[rotated]
        assert door_idx not in used_doors
        door = doors[door_idx]
        for r, c in door["cells"]:
            out[int(r), int(c)] = int(key["color"])
        for r, c in door["hole"]:
            out[int(r), int(c)] = int(bg)
        used_doors.add(int(door_idx))

    assert len(doors) == len(keys) == len(used_doors)
    return out


BASE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
]


ALL_SHAPES = sorted(
    {variant for base in BASE_SHAPES for variant in _shape_variants(base)},
    key=lambda shape: (len(shape), _shape_size(shape), shape),
)


def _boxes_clear(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in placed:
        separated = (
            int(r1) + int(gap) < int(or0)
            or int(or1) + int(gap) < int(r0)
            or int(c1) + int(gap) < int(oc0)
            or int(oc1) + int(gap) < int(c0)
        )
        if not separated:
            return False
    return True


def _paint_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    color: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr, dc in shape:
        r = int(top + dr)
        c = int(left + dc)
        grid[r, c] = int(color)
        cells.append((int(r), int(c)))
    return cells


def _sample_shapes(rng: random.Random, count: int) -> list[tuple[tuple[int, int], ...]]:
    asymmetric = [shape for shape in ALL_SHAPES if not _is_rot180_symmetric(shape)]
    assert asymmetric
    for _ in range(100):
        chosen = rng.sample(ALL_SHAPES, count)
        if any(shape in asymmetric for shape in chosen):
            return chosen
    return [rng.choice(asymmetric)] + rng.sample([s for s in ALL_SHAPES if s not in asymmetric], count - 1)


def _sample_door_spec(
    rng: random.Random,
    shape: tuple[tuple[int, int], ...],
    key_color: int,
) -> dict[str, object] | None:
    hole_h, hole_w = _shape_size(shape)
    min_h = max(5, int(hole_h + 2))
    min_w = max(5, int(hole_w + 2))
    max_h = min(MAX_DOOR_SIDE, int(hole_h + 5))
    max_w = min(MAX_DOOR_SIDE, int(hole_w + 5))
    if min_h > max_h or min_w > max_w:
        return None

    door_h = rng.randint(min_h, max_h)
    door_w = rng.randint(min_w, max_w)
    hole_top = rng.randint(1, int(door_h - hole_h - 1))
    hole_left = rng.randint(1, int(door_w - hole_w - 1))
    key_shape = _rotate180_shape(shape)

    return {
        "hole_shape": shape,
        "key_shape": key_shape,
        "color": int(key_color),
        "door_h": int(door_h),
        "door_w": int(door_w),
        "hole_top": int(hole_top),
        "hole_left": int(hole_left),
    }


def _render_scene(
    h: int,
    w: int,
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG_COLOR, dtype=int)
    out = np.full((h, w), BG_COLOR, dtype=int)

    for spec in specs:
        top = int(spec["door_top"])
        left = int(spec["door_left"])
        door_h = int(spec["door_h"])
        door_w = int(spec["door_w"])
        color = int(spec["color"])

        inp[top : top + door_h, left : left + door_w] = DOOR_COLOR
        out[top : top + door_h, left : left + door_w] = int(color)

        hole_shape = spec["hole_shape"]
        hole_top = int(top + spec["hole_top"])
        hole_left = int(left + spec["hole_left"])
        for dr, dc in hole_shape:
            inp[int(hole_top + dr), int(hole_left + dc)] = BG_COLOR
            out[int(hole_top + dr), int(hole_left + dc)] = BG_COLOR

        key_top = int(spec["key_top"])
        key_left = int(spec["key_left"])
        _paint_shape(inp, spec["key_shape"], key_top, key_left, color)

    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNTS)
    shapes = _sample_shapes(rng, pair_count)
    rng.shuffle(shapes)
    colors = rng.sample(KEY_COLORS, pair_count)

    specs: list[dict[str, object]] = []
    for shape, color in zip(shapes, colors):
        spec = _sample_door_spec(rng, shape, color)
        if spec is None:
            return None
        specs.append(spec)

    door_area = sum(int(spec["door_h"]) * int(spec["door_w"]) for spec in specs)
    min_h = max(GRID_H_RANGE[0], 10 + 2 * pair_count, max(int(spec["door_h"]) for spec in specs) + 3)
    min_w = max(GRID_W_RANGE[0], 10 + 2 * pair_count, max(int(spec["door_w"]) for spec in specs) + 3)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    dims: list[tuple[int, int]] = []
    for _ in range(20):
        h = rng.randint(min_h, GRID_H_RANGE[1])
        w = rng.randint(min_w, GRID_W_RANGE[1])
        if h * w >= 2 * door_area + 40:
            dims.append((int(h), int(w)))
    if not dims:
        return None
    h, w = rng.choice(dims)

    placed: list[tuple[int, int, int, int]] = []
    for spec in sorted(specs, key=lambda s: int(s["door_h"]) * int(s["door_w"]), reverse=True):
        door_h = int(spec["door_h"])
        door_w = int(spec["door_w"])
        success = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(1, int(h - door_h - 1))
            left = rng.randint(1, int(w - door_w - 1))
            box = (int(top), int(left), int(top + door_h - 1), int(left + door_w - 1))
            if _boxes_clear(box, placed, gap=1):
                spec["door_top"] = int(top)
                spec["door_left"] = int(left)
                placed.append(box)
                success = True
                break
        if not success:
            return None

    key_indices = list(range(pair_count))
    rng.shuffle(key_indices)
    for idx in key_indices:
        spec = specs[idx]
        key_h, key_w = _shape_size(spec["key_shape"])
        success = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(1, int(h - key_h - 1))
            left = rng.randint(1, int(w - key_w - 1))
            box = (int(top), int(left), int(top + key_h - 1), int(left + key_w - 1))
            if _boxes_clear(box, placed, gap=1):
                spec["key_top"] = int(top)
                spec["key_left"] = int(left)
                placed.append(box)
                success = True
                break
        if not success:
            return None

    inp, expected = _render_scene(h, w, specs)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
