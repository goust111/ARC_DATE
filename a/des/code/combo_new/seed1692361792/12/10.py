from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _components_by_color(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, c0 = min(rows), min(cols)
            key = tuple(sorted((rr - r0, cc - c0) for rr, cc in coords))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "top": r0,
                    "left": c0,
                    "key": key,
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each shape-equivalence class to the color of its topmost instance.

    Input structure:
    - The background is black.
    - The foreground contains several disjoint monochrome 4-connected objects.
    - Objects that have the same cropped occupancy mask belong to the same shape class even if their colors differ.
    - Every shape class appears at least twice, and its topmost instance uniquely determines that class's standard color.

    Rule:
    - Split the grid into monochrome connected components.
    - Normalize each component by cropping it to its bounding box and recording the translated pixel mask.
    - Group components whose normalized masks are identical.
    - In each group, find the component whose bounding-box top row is smallest and recolor every component in that group to that component's color.

    Key insight:
    - Color is not the grouping cue: shape identity comes from translation-normalized masks, while the winning color is chosen purely by vertical position.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components_by_color(g)
    groups: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = {}
    for comp in comps:
        groups.setdefault(comp["key"], []).append(comp)

    out = g.copy()
    for group in groups.values():
        standard = min(group, key=lambda comp: (int(comp["top"]), int(comp["left"])))
        standard_color = int(standard["color"])
        for comp in group:
            for r, c in comp["coords"]:
                out[r, c] = standard_color
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        target = rng.randint(3, 6)
        cells = {(0, 0)}
        frontier = [(0, 0)]
        while len(cells) < target:
            base_r, base_c = frontier[rng.randrange(len(frontier))]
            dr, dc = DIRS4[rng.randrange(4)]
            cand = (base_r + dr, base_c + dc)
            cells.add(cand)
            frontier = list(cells)
        norm = _normalize_shape(cells)
        rows = [r for r, _ in norm]
        cols = [c for _, c in norm]
        if max(rows) <= 3 and max(cols) <= 3 and len({r for r, _ in norm}) >= 2 and len({c for _, c in norm}) >= 2:
            return norm
    raise RuntimeError("failed to build shape")


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(12, 22)
        w = rng.randint(12, 22)
        grid = np.zeros((h, w), dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        class_count = rng.randint(2, 4)
        shape_keys: set[tuple[tuple[int, int], ...]] = set()
        palette = list(COLORS)
        rng.shuffle(palette)
        palette_idx = 0

        ok = True
        for _class_id in range(class_count):
            shape = _random_shape(rng)
            if shape in shape_keys:
                ok = False
                break
            shape_keys.add(shape)
            shape_h = 1 + max(r for r, _ in shape)
            shape_w = 1 + max(c for _, c in shape)
            inst_count = rng.randint(2, 3)
            if palette_idx + inst_count > len(palette):
                ok = False
                break
            colors = palette[palette_idx : palette_idx + inst_count]
            palette_idx += inst_count

            placements: list[tuple[int, int]] = []
            for inst_i in range(inst_count):
                placed = False
                for _try in range(200):
                    top = rng.randint(0, h - shape_h)
                    left = rng.randint(0, w - shape_w)
                    rs = max(0, top - 1)
                    re = min(h, top + shape_h + 1)
                    cs = max(0, left - 1)
                    ce = min(w, left + shape_w + 1)
                    if occ[rs:re, cs:ce].any():
                        continue
                    if any(top == prev_top for prev_top, _ in placements):
                        continue
                    placements.append((top, left))
                    for dr, dc in shape:
                        occ[top + dr, left + dc] = True
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok:
                break

            topmost_index = min(range(inst_count), key=lambda idx: placements[idx][0])
            ordered_colors = colors[:]
            if rng.random() < 0.5:
                rng.shuffle(ordered_colors)
            if ordered_colors[topmost_index] != colors[0]:
                ordered_colors[topmost_index], ordered_colors[0] = ordered_colors[0], ordered_colors[topmost_index]

            for (top, left), color in zip(placements, ordered_colors):
                for dr, dc in shape:
                    grid[top + dr, left + dc] = int(color)

        if ok:
            return grid

    raise RuntimeError("failed to generate input")
