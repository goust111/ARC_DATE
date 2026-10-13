from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _expansion_layers(
    bg_mask: np.ndarray, cells: list[tuple[int, int]], depth: int
) -> list[set[tuple[int, int]]]:
    h, w = bg_mask.shape
    seen = set(cells)
    q = deque((r, c, 0) for r, c in cells)
    layers = [set() for _ in range(depth)]
    while q:
        r, c, dist = q.popleft()
        if dist == depth:
            continue
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or not bg_mask[nr, nc]:
                continue
            seen.add((nr, nc))
            nd = dist + 1
            layers[nd - 1].add((nr, nc))
            q.append((nr, nc, nd))
    return layers


def _analyze_objects(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert colors, "expected at least one foreground color"
    grey = max(colors, key=lambda color: int(np.count_nonzero(g == color)))
    objects = []
    for comp in _components4(g != bg):
        seeds = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) != grey]
        assert 1 <= len(seeds) <= 4, "each object must contain 1 to 4 seed pixels"
        seeds.sort(key=lambda item: (item[0], item[1]))
        objects.append(
            {
                "cells": comp,
                "seeds": seeds,
                "seed_count": len(seeds),
                "top_left": min(comp),
            }
        )
    assert objects, "expected at least one object"
    return bg, grey, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each grey object outward by as many layers as its internal seed count.

    The k-th layer uses the color of the k-th highest seed inside the same object.
    Layers are computed against the original background, then written in descending
    seed-count priority so larger objects win expansion overlaps.
    """
    g = np.asarray(grid, dtype=int)
    bg, _grey, objects = _analyze_objects(g)
    bg_mask = g == bg
    out = g.copy()

    objects.sort(key=lambda obj: (-obj["seed_count"], obj["top_left"]))
    for obj in objects:
        layers = _expansion_layers(bg_mask, obj["cells"], obj["seed_count"])
        for layer_idx, layer in enumerate(layers):
            color = obj["seeds"][layer_idx][2]
            for r, c in layer:
                if out[r, c] == bg:
                    out[r, c] = color
    return out


def _make_object_spec(rng: random.Random, seed_count: int, palette: list[int]):
    core_h = rng.randint(seed_count + 2, seed_count + 4)
    core_w = rng.randint(4, 7)

    cells = {(r, c) for r in range(core_h) for c in range(core_w)}
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[: rng.randint(1, 3)]:
        if side in {"top", "bottom"}:
            span = rng.randint(max(2, core_w // 2), core_w)
            start = rng.randint(0, core_w - span)
            length = rng.randint(1, 3)
            rows = range(-length, 0) if side == "top" else range(core_h, core_h + length)
            for r in rows:
                for c in range(start, start + span):
                    cells.add((r, c))
        else:
            span = rng.randint(max(2, core_h // 2), core_h)
            start = rng.randint(0, core_h - span)
            length = rng.randint(1, 3)
            cols = range(-length, 0) if side == "left" else range(core_w, core_w + length)
            for r in range(start, start + span):
                for c in cols:
                    cells.add((r, c))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm_cells = {(r - min_r, c - min_c) for r, c in cells}
    row_offset = -min_r
    col_offset = -min_c
    height = max(r for r, _ in norm_cells) + 1
    width = max(c for _, c in norm_cells) + 1

    seed_rows = sorted(rng.sample(list(range(1, core_h - 1)), seed_count))
    seed_colors = rng.sample(palette, seed_count)
    seeds = []
    for row, color in zip(seed_rows, seed_colors):
        col = rng.randint(1, core_w - 2)
        seeds.append((row_offset + row, col_offset + col, color))

    return {
        "cells": norm_cells,
        "seeds": seeds,
        "seed_count": seed_count,
        "height": height,
        "width": width,
    }


def _place_specs(
    rng: random.Random, specs: list[dict], height: int, width: int
) -> list[dict] | None:
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []

    order = sorted(specs, key=lambda spec: len(spec["cells"]), reverse=True)
    for spec in order:
        if spec["height"] > height or spec["width"] > width:
            return None
        placed_spec = None
        for _ in range(300):
            r0 = rng.randint(0, height - spec["height"])
            c0 = rng.randint(0, width - spec["width"])
            translated = {(r0 + r, c0 + c) for r, c in spec["cells"]}
            if translated & blocked:
                continue
            placed_spec = {
                "cells": [(r0 + r, c0 + c) for r, c in sorted(spec["cells"])],
                "seeds": [(r0 + r, c0 + c, color) for r, c, color in spec["seeds"]],
                "seed_count": spec["seed_count"],
            }
            break
        if placed_spec is None:
            return None
        placed.append(placed_spec)
        for r, c in placed_spec["cells"]:
            occupied.add((r, c))
            blocked.add((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width:
                    blocked.add((nr, nc))

    return placed


def _has_overlap(grid: np.ndarray) -> bool:
    bg, _grey, objects = _analyze_objects(grid)
    bg_mask = grid == bg
    expansions = []
    for obj in objects:
        cells = set()
        for layer in _expansion_layers(bg_mask, obj["cells"], obj["seed_count"]):
            cells |= layer
        expansions.append(cells)
    for i in range(len(expansions)):
        for j in range(i + 1, len(expansions)):
            if expansions[i] & expansions[j]:
                return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.randint(2, 4)
    want_overlap = rng.random() < 0.5

    colors = list(range(10))
    bg = rng.choice(colors)
    remaining = [color for color in colors if color != bg]
    grey = rng.choice(remaining)
    palette = [color for color in remaining if color != grey]

    seed_counts = rng.sample([1, 2, 3, 4], num_objects)
    specs = [_make_object_spec(rng, seed_count, palette) for seed_count in seed_counts]

    if want_overlap:
        height = rng.randint(18, 24)
        width = rng.randint(18, 24)
    else:
        height = rng.randint(24, 30)
        width = rng.randint(24, 30)

    placed = _place_specs(rng, specs, height, width)
    if placed is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for obj in placed:
        for r, c in obj["cells"]:
            grid[r, c] = grey
        for r, c, color in obj["seeds"]:
            grid[r, c] = color

    try:
        bg2, grey2, objects = _analyze_objects(grid)
    except AssertionError:
        return None
    if bg2 != bg or grey2 != grey or len(objects) != num_objects:
        return None
    if sorted(obj["seed_count"] for obj in objects) != sorted(seed_counts):
        return None
    if _has_overlap(grid) != want_overlap:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
