from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, RIGHT, DOWN, LEFT)
DIR_NAMES = {
    UP: "up",
    RIGHT: "right",
    DOWN: "down",
    LEFT: "left",
}

# The 7-pixel directional sprite is a 3x3 perimeter ring with the center absent
# and one additional missing corner. The missing-corner quadrant determines one
# of the four cardinal directions in clockwise order.
SPRITE_PATTERNS = {
    UP: {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)},
    RIGHT: {(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)},
    DOWN: {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)},
    LEFT: {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)},
}
SIDE_ORIGIN = {
    UP: (0, 1),
    RIGHT: (1, 2),
    DOWN: (2, 1),
    LEFT: (1, 0),
}
EDGE_KEY_BY_DIR = {
    UP: "top",
    RIGHT: "right",
    DOWN: "bottom",
    LEFT: "left",
}

GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (12, 19)
PAIR_COUNT_RANGE = (1, 2)
EXTRA_COUNT_RANGE = (1, 3)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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


def _edge_colors(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 5 and w >= 5
    top = g[0, 1:-1]
    bottom = g[h - 1, 1:-1]
    left = g[1:-1, 0]
    right = g[1:-1, w - 1]
    assert top.size and bottom.size and left.size and right.size
    assert np.all(top == top[0])
    assert np.all(bottom == bottom[0])
    assert np.all(left == left[0])
    assert np.all(right == right[0])
    colors = {
        "top": int(top[0]),
        "bottom": int(bottom[0]),
        "left": int(left[0]),
        "right": int(right[0]),
    }
    assert len(set(colors.values())) == 4
    assert BLACK not in colors.values()
    return colors


def _sprite_from_component(
    grid: np.ndarray, comp: list[tuple[int, int]]
) -> dict[str, object] | None:
    if len(comp) != 7:
        return None
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    rel = {(r - r0, c - c0) for r, c in comp}
    for direction, pattern in SPRITE_PATTERNS.items():
        if rel == pattern:
            origin_dr, origin_dc = SIDE_ORIGIN[direction]
            return {
                "cells": comp,
                "cell_set": set(comp),
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "origin": (r0 + origin_dr, c0 + origin_dc),
                "color": int(grid[comp[0]]),
            }
    return None


def _extract_sprites(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    grey_mask = g == GREY
    sprites = []
    for comp in _find_components(grey_mask):
        sprite = _sprite_from_component(g, comp)
        assert sprite is not None, "all grey components must be 7-pixel directional sprites"
        sprites.append(sprite)
    assert sprites, "expected at least one directional sprite"
    return sprites


def _ray_result(
    sprite: dict[str, object],
    occupancy: np.ndarray,
    sprite_at: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], int | None]:
    h, w = occupancy.shape
    direction = sprite["direction"]
    dr, dc = direction
    start_r, start_c = sprite["origin"]
    r = int(start_r + dr)
    c = int(start_c + dc)
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if occupancy[r, c]:
            return path, sprite_at.get((r, c))
        path.append((r, c))
        r += dr
        c += dc
    return path, None


def _analyze_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = _edge_colors(g)

    border = np.zeros_like(g, dtype=bool)
    border[0, :] = True
    border[h - 1, :] = True
    border[:, 0] = True
    border[:, w - 1] = True
    interior_non_bg = (g != BLACK) & ~border
    assert np.all((g == GREY) | ~interior_non_bg), "interior non-background pixels must be grey sprites"

    sprites = _extract_sprites(g)
    sprite_at: dict[tuple[int, int], int] = {}
    for idx, sprite in enumerate(sprites):
        for cell in sprite["cells"]:
            assert cell not in sprite_at
            sprite_at[cell] = idx

    occupancy = g != BLACK
    hits_by_target = Counter()
    rays = []
    for idx, sprite in enumerate(sprites):
        path, target = _ray_result(sprite, occupancy, sprite_at)
        if target is not None and target != idx:
            hits_by_target[target] += 1
        rays.append(
            {
                "sprite_index": idx,
                "path": path,
                "target": target if target != idx else None,
                "ray_color": colors[EDGE_KEY_BY_DIR[sprite["direction"]]],
            }
        )
    return {
        "sprites": sprites,
        "rays": rays,
        "edge_colors": colors,
        "hits_by_target": hits_by_target,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire colored rays from directional grey sprites using the border strips as the
    ray-color legend. Rays travel straight through background, stop on the first
    non-background cell or the boundary, and recolor any hit sprite.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_scene(g)
    out = g.copy()

    for ray in analysis["rays"]:
        ray_color = int(ray["ray_color"])
        for r, c in ray["path"]:
            out[r, c] = ray_color
        target = ray["target"]
        if target is not None:
            for r, c in analysis["sprites"][target]["cells"]:
                out[r, c] = ray_color

    return out


def _paint_edges(grid: np.ndarray, edge_colors: dict[str, int]) -> None:
    h, w = grid.shape
    grid[0, 1:-1] = edge_colors["top"]
    grid[h - 1, 1:-1] = edge_colors["bottom"]
    grid[1:-1, 0] = edge_colors["left"]
    grid[1:-1, w - 1] = edge_colors["right"]
    grid[0, 0] = edge_colors["top"]
    grid[0, w - 1] = edge_colors["top"]
    grid[h - 1, 0] = edge_colors["bottom"]
    grid[h - 1, w - 1] = edge_colors["bottom"]


def _sprite_patch(direction: tuple[int, int]) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    for r, c in SPRITE_PATTERNS[direction]:
        patch[r, c] = GREY
    return patch


def _paste_sprite(grid: np.ndarray, top: int, left: int, direction: tuple[int, int]) -> None:
    patch = _sprite_patch(direction)
    grid[top : top + 3, left : left + 3][patch == GREY] = GREY


def _clear_with_margin(grid: np.ndarray, top: int, left: int, margin: int = 1) -> bool:
    h, w = grid.shape
    r0 = max(1, top - margin)
    c0 = max(1, left - margin)
    r1 = min(h - 2, top + 2 + margin)
    c1 = min(w - 2, left + 2 + margin)
    return np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK)


def _choose_edge_colors() -> dict[str, int]:
    palette = [c for c in range(1, 10) if c != GREY]
    top, bottom, left, right = random.sample(palette, 4)
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _pair_positions(
    shape: tuple[int, int], direction: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    if direction == RIGHT:
        row = random.randint(1, h - 4)
        c1 = random.randint(1, max(1, w - 9))
        c2 = random.randint(c1 + 4, w - 4)
        return (row, c1), (row, c2)
    if direction == LEFT:
        row = random.randint(1, h - 4)
        c2 = random.randint(1, max(1, w - 9))
        c1 = random.randint(c2 + 4, w - 4)
        return (row, c1), (row, c2)
    if direction == DOWN:
        col = random.randint(1, w - 4)
        r1 = random.randint(1, max(1, h - 9))
        r2 = random.randint(r1 + 4, h - 4)
        return (r1, col), (r2, col)
    col = random.randint(1, w - 4)
    r2 = random.randint(1, max(1, h - 9))
    r1 = random.randint(r2 + 4, h - 4)
    return (r1, col), (r2, col)


def _ray_path_cells(top: int, left: int, direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    origin_r, origin_c = SIDE_ORIGIN[direction]
    dr, dc = direction
    r = top + origin_r + dr
    c = left + origin_c + dc
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _target_hit_cell(top: int, left: int, direction: tuple[int, int]) -> tuple[int, int]:
    if direction == RIGHT:
        return (top + 1, left)
    if direction == LEFT:
        return (top + 1, left + 2)
    if direction == DOWN:
        return (top, left + 1)
    return (top + 2, left + 1)


def _try_place_pair(
    grid: np.ndarray,
    specs: list[tuple[int, int, tuple[int, int]]],
    direction: tuple[int, int],
) -> bool:
    shape = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        shooter, target = _pair_positions(shape, direction)
        if shooter is None:
            continue
        st, sl = shooter
        tt, tl = target
        if not _clear_with_margin(grid, st, sl) or not _clear_with_margin(grid, tt, tl):
            continue
        path = _ray_path_cells(st, sl, direction, shape)
        hit_pos = _target_hit_cell(tt, tl, direction)
        if hit_pos not in path:
            continue
        path_before_hit = []
        for cell in path:
            if cell == hit_pos:
                break
            path_before_hit.append(cell)
        if any(grid[r, c] != BLACK for r, c in path_before_hit):
            continue

        _paste_sprite(grid, st, sl, direction)
        target_dir = random.choice([d for d in DIRS if d != (-direction[0], -direction[1])])
        _paste_sprite(grid, tt, tl, target_dir)
        specs.append((st, sl, direction))
        specs.append((tt, tl, target_dir))
        return True
    return False


def _try_place_extra(
    grid: np.ndarray,
    specs: list[tuple[int, int, tuple[int, int]]],
) -> bool:
    h, w = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        direction = random.choice(DIRS)
        top = random.randint(1, h - 4)
        left = random.randint(1, w - 4)
        if not _clear_with_margin(grid, top, left):
            continue
        path = _ray_path_cells(top, left, direction, grid.shape)
        if not path:
            continue
        if any(grid[r, c] != BLACK for r, c in path[:-1]):
            continue
        _paste_sprite(grid, top, left, direction)
        specs.append((top, left, direction))
        return True
    return False


def _good_scene(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_scene(grid)
    except AssertionError:
        return False

    hit_counts = analysis["hits_by_target"]
    if any(v > 1 for v in hit_counts.values()):
        return False

    ray_hits = sum(ray["target"] is not None for ray in analysis["rays"])
    ray_misses = sum(ray["target"] is None for ray in analysis["rays"])
    if ray_hits < 1 or ray_misses < 1:
        return False

    directions = {ray["sprite_index"]: analysis["sprites"][ray["sprite_index"]]["direction"] for ray in analysis["rays"]}
    if len(set(directions.values())) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    _paint_edges(grid, _choose_edge_colors())

    specs: list[tuple[int, int, tuple[int, int]]] = []
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    extra_count = random.randint(*EXTRA_COUNT_RANGE)

    chosen_dirs = random.sample(DIRS, k=min(pair_count, len(DIRS)))
    for i in range(pair_count):
        direction = chosen_dirs[i] if i < len(chosen_dirs) else random.choice(DIRS)
        if not _try_place_pair(grid, specs, direction):
            return None

    for _ in range(extra_count):
        if not _try_place_extra(grid, specs):
            return None

    if not _good_scene(grid):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
