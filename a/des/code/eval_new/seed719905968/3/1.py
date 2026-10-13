from __future__ import annotations

import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    banned = exclude or set()
    return [int(v) for v in np.unique(grid) if int(v) not in banned]


def _find_wall_and_regions(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    h, w = grid.shape
    best: tuple[tuple[int, int, int], int, list[dict[str, object]]] | None = None

    for color in _colors_present(grid, exclude={bg}):
        regions: list[dict[str, object]] = []
        for coords in _components_mask(grid != color, connectivity=4):
            if _touches_edge(coords, h, w):
                continue

            values = grid[coords[:, 0], coords[:, 1]]
            non_bg_values = values[values != bg]
            if non_bg_values.size != 1:
                continue

            seed_color = int(non_bg_values[0])
            seed_pos = coords[values != bg]
            if seed_pos.shape != (1, 2):
                continue

            regions.append(
                {
                    "cells": coords.copy(),
                    "seed": (int(seed_pos[0, 0]), int(seed_pos[0, 1])),
                    "seed_color": seed_color,
                }
            )

        if not regions:
            continue

        score = (
            len(regions),
            sum(int(region["cells"].shape[0]) for region in regions),
            int(np.count_nonzero(grid == color)),
        )
        if best is None or score > best[0]:
            best = (score, color, regions)

    assert best is not None, "expected one wall color that encloses seed regions"
    return best[1], best[2]


def _extract_palette_textures(
    grid: np.ndarray, palette_mask: np.ndarray, seed_colors: set[int], bg: int
) -> dict[int, np.ndarray]:
    textures: dict[int, np.ndarray] = {}

    for coords in _components_mask(palette_mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        marker_color = int(sub[0, 0])
        if marker_color not in seed_colors:
            continue

        texture_mask = sub != bg
        texture_mask[0, 0] = False
        pattern_coords = np.argwhere(texture_mask)
        rows = sorted({int(r) for r, _c in pattern_coords})
        cols = sorted({int(c) for _r, c in pattern_coords})
        size = len(rows)

        assert size in (2, 3), "palette texture must be 2x2 or 3x3"
        assert rows == list(range(1, 1 + size)), "texture rows must sit directly below the marker"
        assert cols == list(range(1, 1 + size)), "texture cols must sit directly right of the marker"

        pattern = sub[1 : 1 + size, 1 : 1 + size].copy()
        assert np.all(pattern != bg), "palette texture block must be solid"
        textures[marker_color] = pattern

    assert textures, "expected at least one palette texture"
    return textures


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each enclosed region with the palette texture keyed by its seed color.

    Input structure:
    - bg is the most common color. Several irregular one-color wall loops surround interior regions.
    - Each enclosed region contains exactly one singleton seed pixel whose color also appears as a marker in one palette cell.
    - A palette strip touches one grid edge; every cell has the seed-colored marker in its top-left corner and a 2x2 or 3x3 checker texture offset down-right from it.

    Rule:
    - Detect the common wall color by finding which color encloses non-edge regions.
    - For each enclosed region, read its singleton seed color, locate the palette cell with the matching marker, and extract that cell's texture block.
    - Replace every pixel inside the region, including the seed, by periodic tiling of the texture anchored so the seed position uses texture[0, 0].
    - Leave the wall loops unchanged and erase the whole palette strip to background.

    Key insight:
    - The seed color is only a lookup key into the edge palette; the actual fill is a periodic texture whose phase is determined by the seed position, not by the region boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall_color, regions = _find_wall_and_regions(g, bg)

    region_mask = np.zeros_like(g, dtype=bool)
    seed_colors: set[int] = set()
    for region in regions:
        coords = region["cells"]
        region_mask[coords[:, 0], coords[:, 1]] = True
        seed_colors.add(int(region["seed_color"]))

    palette_mask = (g != bg) & (g != wall_color) & ~region_mask
    textures = _extract_palette_textures(g, palette_mask, seed_colors, bg)
    assert seed_colors <= set(textures), "every seed must have a palette texture"

    out = g.copy()
    out[palette_mask] = bg
    for region in regions:
        coords = region["cells"]
        seed_r, seed_c = region["seed"]
        pattern = textures[int(region["seed_color"])]
        ph, pw = pattern.shape
        for r, c in coords:
            out[r, c] = int(pattern[(int(r) - seed_r) % ph, (int(c) - seed_c) % pw])
    return out


def _checkerboard(size: int, a: int, b: int, phase: int) -> np.ndarray:
    pattern = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            pattern[r, c] = a if (r + c + phase) % 2 == 0 else b
    return pattern


def _random_region_cells(rng: random.Random) -> set[tuple[int, int]]:
    base_h = rng.randint(3, 5)
    base_w = rng.randint(3, 6)
    cells = {(r, c) for r in range(base_h) for c in range(base_w)}

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[: rng.randint(1, 2)]:
        if side in ("top", "bottom"):
            span = rng.randint(1, max(1, base_w - 1))
            start = rng.randint(0, base_w - span)
            depth = rng.randint(1, 2)
            rows = range(-depth, 0) if side == "top" else range(base_h, base_h + depth)
            for r in rows:
                for c in range(start, start + span):
                    cells.add((r, c))
        else:
            span = rng.randint(1, max(1, base_h - 1))
            start = rng.randint(0, base_h - span)
            depth = rng.randint(1, 2)
            cols = range(-depth, 0) if side == "left" else range(base_w, base_w + depth)
            for r in range(start, start + span):
                for c in cols:
                    cells.add((r, c))

    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _render_object_patch(
    region_cells: set[tuple[int, int]], wall_color: int, bg: int, seed_color: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    shifted = {(r + 1, c + 1) for r, c in region_cells}
    max_r = max(r for r, _c in shifted)
    max_c = max(c for _r, c in shifted)
    patch = np.full((max_r + 2, max_c + 2), bg, dtype=int)

    wall_cells: set[tuple[int, int]] = set()
    for r, c in shifted:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in shifted:
                wall_cells.add((nr, nc))
    for r, c in wall_cells:
        patch[r, c] = wall_color

    seed = rng.choice(sorted(shifted))
    patch[seed] = seed_color
    coords = np.array(sorted(shifted), dtype=int)
    return patch, coords, seed


def _make_palette_cell(seed_color: int, pattern: np.ndarray, bg: int) -> np.ndarray:
    size = int(pattern.shape[0])
    cell = np.full((size + 1, size + 1), bg, dtype=int)
    cell[0, 0] = seed_color
    cell[1:, 1:] = pattern
    return cell


def _compose_palette(cells: list[np.ndarray], side: str, bg: int) -> np.ndarray:
    gap = 1
    if side in ("left", "right"):
        height = sum(int(cell.shape[0]) for cell in cells) + gap * (len(cells) - 1)
        width = max(int(cell.shape[1]) for cell in cells)
        strip = np.full((height, width), bg, dtype=int)
        cursor = 0
        for cell in cells:
            ch, cw = cell.shape
            col = 0 if side == "left" else width - cw
            strip[cursor : cursor + ch, col : col + cw] = cell
            cursor += ch + gap
        return strip

    width = sum(int(cell.shape[1]) for cell in cells) + gap * (len(cells) - 1)
    height = max(int(cell.shape[0]) for cell in cells)
    strip = np.full((height, width), bg, dtype=int)
    cursor = 0
    for cell in cells:
        ch, cw = cell.shape
        row = 0 if side == "top" else height - ch
        strip[row : row + ch, cursor : cursor + cw] = cell
        cursor += cw + gap
    return strip


def _can_place(occupied: np.ndarray, top: int, left: int, patch_shape: tuple[int, int]) -> bool:
    ph, pw = patch_shape
    h, w = occupied.shape
    if top < 1 or left < 1 or top + ph > h - 1 or left + pw > w - 1:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + ph + 1)
    c1 = min(w, left + pw + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_palette(grid: np.ndarray, palette: np.ndarray, side: str, rng: random.Random) -> None:
    h, w = grid.shape
    ph, pw = palette.shape
    if side == "left":
        top = rng.randint(0, h - ph)
        grid[top : top + ph, 0:pw] = palette
    elif side == "right":
        top = rng.randint(0, h - ph)
        grid[top : top + ph, w - pw : w] = palette
    elif side == "top":
        left = rng.randint(0, w - pw)
        grid[0:ph, left : left + pw] = palette
    else:
        left = rng.randint(0, w - pw)
        grid[h - ph : h, left : left + pw] = palette


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice([2, 3, 4])
    side = rng.choice(["top", "bottom", "left", "right"])

    all_colors = list(range(10))
    bg = rng.choice(all_colors)
    remaining = [c for c in all_colors if c != bg]
    rng.shuffle(remaining)

    wall_color = remaining.pop()
    seed_colors = [remaining.pop() for _ in range(n_objects)]
    texture_colors = remaining
    if len(texture_colors) < 2:
        return None

    sizes = [2, 3]
    while len(sizes) < n_objects:
        sizes.append(rng.choice([2, 3]))
    rng.shuffle(sizes)

    used_patterns: set[bytes] = set()
    objects: list[dict[str, object]] = []
    for seed_color, size in zip(seed_colors, sizes):
        region = _random_region_cells(rng)

        pattern = None
        for _ in range(50):
            a, b = rng.sample(texture_colors, 2)
            candidate = _checkerboard(size, a, b, rng.randint(0, 1))
            sig = candidate.tobytes()
            if sig not in used_patterns:
                used_patterns.add(sig)
                pattern = candidate
                break
        if pattern is None:
            return None

        patch, coords, seed = _render_object_patch(region, wall_color, bg, seed_color, rng)
        objects.append(
            {
                "patch": patch,
                "region_cells": coords,
                "seed": seed,
                "seed_color": seed_color,
                "pattern": pattern,
            }
        )

    palette_cells = [_make_palette_cell(int(obj["seed_color"]), obj["pattern"], bg) for obj in objects]
    rng.shuffle(palette_cells)
    palette = _compose_palette(palette_cells, side, bg)

    scene_h = rng.randint(16, 20)
    scene_w = rng.randint(16, 20)
    sep = 2
    if side in ("left", "right"):
        h = max(scene_h, int(palette.shape[0]) + rng.randint(0, 4))
        w = scene_w + int(palette.shape[1]) + sep
        scene_rows = (0, h)
        if side == "left":
            scene_cols = (int(palette.shape[1]) + sep, w)
        else:
            scene_cols = (0, w - int(palette.shape[1]) - sep)
    else:
        h = scene_h + int(palette.shape[0]) + sep
        w = max(scene_w, int(palette.shape[1]) + rng.randint(0, 4))
        scene_cols = (0, w)
        if side == "top":
            scene_rows = (int(palette.shape[0]) + sep, h)
        else:
            scene_rows = (0, h - int(palette.shape[0]) - sep)

    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    rng.shuffle(objects)
    for obj in objects:
        patch = obj["patch"]
        ph, pw = patch.shape
        r_lo = max(scene_rows[0], 1)
        r_hi = min(scene_rows[1] - ph, h - ph - 1)
        c_lo = max(scene_cols[0], 1)
        c_hi = min(scene_cols[1] - pw, w - pw - 1)
        if r_lo > r_hi or c_lo > c_hi:
            return None

        placed = False
        for _ in range(200):
            top = rng.randint(r_lo, r_hi)
            left = rng.randint(c_lo, c_hi)
            if not _can_place(occupied, top, left, patch.shape):
                continue
            grid[top : top + ph, left : left + pw][patch != bg] = patch[patch != bg]
            occupied[top : top + ph, left : left + pw] = True
            placed = True
            break
        if not placed:
            return None

    _place_palette(grid, palette, side, rng)

    non_bg_colors = _colors_present(grid, exclude={bg})
    if not non_bg_colors:
        return None
    if np.count_nonzero(grid == bg) <= max(np.count_nonzero(grid == c) for c in non_bg_colors):
        return None

    seed_counts = {color: int(np.count_nonzero(grid == color)) for color in seed_colors}
    if any(count != 2 for count in seed_counts.values()):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
