from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]

BG_COLOR = 0
FRAME_COLOR = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7)

PLAY_H = 9
PLAY_W = 13
SAFE_CORE_TRANSFORMS = (
    (False, False),
    (True, False),
    (True, True),
)

BASE_CORE_LAYOUT: dict[str, tuple[Coord, Coord]] = {
    "S1": ((1, 1), (1, 1)),
    "S2": ((0, 12), (1, -1)),
    "S3": ((2, 6), (1, -1)),
    "T1": ((6, 3), (1, -1)),
    "T2": ((6, 7), (1, -1)),
}
BASE_DISTRACTOR_LAYOUT: tuple[Coord, Coord] = ((10, 1), (1, 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _connected_components(mask: np.ndarray, *, conn8: bool) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []
    if conn8:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _bbox(cells: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == FRAME_COLOR
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _connected_components(mask, conn8=False):
        r0, c0, r1, c1 = _bbox(comp)
        if c1 - c0 != 3 or r1 - r0 < 4:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) != border:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[1] != 2 or inner.shape[0] not in (3, 4):
            continue
        if np.any(inner == BG_COLOR) or np.any(inner == FRAME_COLOR):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected a unique grey-framed 2-column legend"
    return candidates[0]


def _legend_pairs_and_mask(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    r0, c0, r1, c1 = _find_legend_bbox(grid)
    pairs = [(int(grid[r, c0 + 1]), int(grid[r, c0 + 2])) for r in range(r0 + 1, r1)]
    mask = np.zeros(grid.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return pairs, mask


def _sprite_vertex_and_direction(cells: list[Coord]) -> tuple[Coord, Coord]:
    assert len(cells) == 3, "each sprite must have exactly three cells"
    r0, c0, r1, c1 = _bbox(cells)
    assert r1 - r0 == 1 and c1 - c0 == 1, "sprite must occupy three corners of a 2x2 box"
    all_corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    occupied = set(cells)
    missing = list(all_corners - occupied)
    assert len(missing) == 1, "sprite must have one missing corner"
    mr, mc = missing[0]
    vr = r0 + r1 - mr
    vc = c0 + c1 - mc
    assert (vr, vc) in occupied
    return (vr, vc), (mr - vr, mc - vc)


def _extract_sprites(grid: np.ndarray, legend_mask: np.ndarray, bg: int) -> list[dict[str, object]]:
    mask = (grid != bg) & (~legend_mask)
    components = _connected_components(mask, conn8=True)
    sprites: list[dict[str, object]] = []
    for comp in components:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each sprite must be monochrome"
        color = next(iter(colors))
        vertex, direction = _sprite_vertex_and_direction(comp)
        sprites.append(
            {
                "cells": tuple(sorted(comp)),
                "color": color,
                "vertex": vertex,
                "direction": direction,
            }
        )
    assert 4 <= len(sprites) <= 6
    return sprites


def _trace_first_hit(
    vertex: Coord,
    direction: Coord,
    shooter_id: int,
    cell_to_sprite: dict[Coord, int],
    shape: tuple[int, int],
) -> tuple[int | None, int | None, list[Coord]]:
    h, w = shape
    vr, vc = vertex
    dr, dc = direction
    r, c = vr + dr, vc + dc
    path: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        path.append((r, c))
        target_id = cell_to_sprite.get((r, c))
        if target_id is not None and target_id != shooter_id:
            return target_id, abs(r - vr) + abs(c - vc), path
        r += dr
        c += dc
    return None, None, path


def _collect_hits(
    sprites: list[dict[str, object]],
    shape: tuple[int, int],
) -> tuple[dict[int, list[int]], dict[int, list[Coord]]]:
    cell_to_sprite: dict[Coord, int] = {}
    for sprite_id, sprite in enumerate(sprites):
        for cell in sprite["cells"]:
            cell_to_sprite[cell] = sprite_id

    hit_distances = {sprite_id: [] for sprite_id in range(len(sprites))}
    ray_paths: dict[int, list[Coord]] = {}
    for shooter_id, sprite in enumerate(sprites):
        target_id, distance, path = _trace_first_hit(
            sprite["vertex"],
            sprite["direction"],
            shooter_id,
            cell_to_sprite,
            shape,
        )
        ray_paths[shooter_id] = path
        if target_id is not None and distance is not None:
            hit_distances[target_id].append(distance)
    return hit_distances, ray_paths


def _apply_one_round(color: int, pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    for src, dst in pairs:
        if current == src:
            current = dst
    return current


def _apply_rounds(color: int, pairs: list[tuple[int, int]], rounds: int) -> int:
    current = int(color)
    for _ in range(rounds):
        current = _apply_one_round(current, pairs)
    return current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire diagonal rays from 3-pixel L sprites and recolor only the sprites they hit using a framed substitution legend.

    Input structure:
    - The grid has black background, 4-6 well-separated monochrome 3-pixel L sprites, and one grey-framed vertical 2-column legend near the lower-right corner.
    - Each sprite occupies exactly three corners of a 2x2 box, so one diagonal corner is missing; the legend interior has 3-4 rows of source->target color pairs.

    Rule:
    - For each sprite, find the occupied corner opposite the missing cell and emit a ray diagonally from that vertex toward the missing corner.
    - The ray is invisible and keeps advancing until it first reaches a cell of another sprite or leaves the grid; each hit contributes one replacement round to the struck sprite, and hit rounds are ordered by increasing source-to-hit Manhattan distance.
    - In one replacement round, scan the legend from top to bottom, carrying the sprite's current color; whenever it matches a row's left color, replace it with that row's right color and continue scanning lower rows. Recolor each hit sprite uniformly after all of its rounds; unhit sprites and the legend stay unchanged.

    Key insight:
    - The missing corner determines each ray direction, while the number of successful ray hits determines how many full top-to-bottom chained legend scans a target sprite undergoes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    pairs, legend_mask = _legend_pairs_and_mask(g)
    sprites = _extract_sprites(g, legend_mask, bg)
    hit_distances, _ = _collect_hits(sprites, g.shape)

    out = g.copy()
    for sprite_id, distances in hit_distances.items():
        if not distances:
            continue
        color = int(sprites[sprite_id]["color"])
        for _distance in sorted(distances):
            color = _apply_one_round(color, pairs)
        for r, c in sprites[sprite_id]["cells"]:
            out[r, c] = color
    return out


def _l_cells(corner: Coord, direction: Coord) -> tuple[Coord, Coord, Coord]:
    r, c = corner
    dr, dc = direction
    return ((r, c), (r + dr, c), (r, c + dc))


def _transform_core_placement(
    corner: Coord,
    direction: Coord,
    *,
    vflip: bool,
    hflip: bool,
) -> tuple[Coord, Coord]:
    r, c = corner
    dr, dc = direction
    if vflip:
        r = PLAY_H - 1 - r
        dr = -dr
    if hflip:
        c = PLAY_W - 1 - c
        dc = -dc
    return (r, c), (dr, dc)


def _build_scene_layout(
    *,
    with_distractor: bool,
    vflip: bool,
    hflip: bool,
    row_shift: int,
    col_shift: int,
) -> dict[str, tuple[Coord, Coord]]:
    scene: dict[str, tuple[Coord, Coord]] = {}
    for name, (corner, direction) in BASE_CORE_LAYOUT.items():
        if with_distractor:
            tc, td = corner, direction
        else:
            tc, td = _transform_core_placement(corner, direction, vflip=vflip, hflip=hflip)
        scene[name] = ((tc[0] + row_shift, tc[1] + col_shift), td)

    if with_distractor:
        corner, direction = BASE_DISTRACTOR_LAYOUT
        scene["U"] = ((corner[0] + row_shift, corner[1] + col_shift), direction)
    return scene


def _paint_legend(grid: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    h, w = grid.shape
    inner_rows = len(pairs)
    r0 = h - (inner_rows + 2)
    c0 = w - 4
    grid[r0 : r0 + inner_rows + 2, c0 : c0 + 4] = FRAME_COLOR
    for i, (src, dst) in enumerate(pairs):
        grid[r0 + 1 + i, c0 + 1] = src
        grid[r0 + 1 + i, c0 + 2] = dst


def _scene_role_colors() -> tuple[dict[str, int], list[tuple[int, int]]]:
    a, b, c, d, e, f = random.sample(list(SPRITE_COLORS), len(SPRITE_COLORS))
    role_colors = {
        "T2": a,
        "S1": b,
        "S2": c,
        "T1": d,
        "S3": e,
        "U": f,
    }
    if random.random() < 0.5:
        pairs = [(b, c), (d, e), (a, b)]
    else:
        pairs = [(b, c), (d, e), (a, b), (f, a)]
    return role_colors, pairs


def _paint_scene(grid: np.ndarray, scene: dict[str, tuple[Coord, Coord]], role_colors: dict[str, int]) -> dict[str, tuple[Coord, ...]]:
    role_cells: dict[str, tuple[Coord, ...]] = {}
    occupied: set[Coord] = set()
    for name, (corner, direction) in scene.items():
        cells = _l_cells(corner, direction)
        assert all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in cells)
        assert not (occupied & set(cells))
        occupied.update(cells)
        role_cells[name] = cells
        for r, c in cells:
            grid[r, c] = role_colors[name]
    return role_cells


def _validate_generated_scene(
    grid: np.ndarray,
    role_cells: dict[str, tuple[Coord, ...]],
    role_colors: dict[str, int],
    pairs: list[tuple[int, int]],
) -> bool:
    try:
        legend_pairs, legend_mask = _legend_pairs_and_mask(grid)
        if legend_pairs != pairs:
            return False
        sprites = _extract_sprites(grid, legend_mask, BG_COLOR)
        hit_distances, ray_paths = _collect_hits(sprites, grid.shape)
    except AssertionError:
        return False

    cell_to_sprite: dict[Coord, int] = {}
    for sprite_id, sprite in enumerate(sprites):
        for cell in sprite["cells"]:
            cell_to_sprite[cell] = sprite_id

    hit_count_by_role: dict[str, int] = {}
    for name, cells in role_cells.items():
        sprite_id = cell_to_sprite[cells[0]]
        hit_count_by_role[name] = len(hit_distances[sprite_id])

    if hit_count_by_role.get("T1") != 1 or hit_count_by_role.get("T2") != 2:
        return False
    for shooter_name in ("S1", "S2", "S3", "U"):
        if shooter_name in hit_count_by_role and hit_count_by_role[shooter_name] != 0:
            return False

    for path in ray_paths.values():
        for r, c in path:
            if legend_mask[r, c]:
                return False

    one_round = _apply_rounds(role_colors["T2"], pairs, 1)
    two_rounds = _apply_rounds(role_colors["T2"], pairs, 2)
    if one_round == two_rounds:
        return False
    if _apply_rounds(role_colors["T1"], pairs, 1) == role_colors["T1"]:
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    for name in ("T1", "T2"):
        sprite_id = cell_to_sprite[role_cells[name][0]]
        final_color = int(out[role_cells[name][0]])
        expected = _apply_rounds(role_colors[name], pairs, len(hit_distances[sprite_id]))
        if final_color != expected:
            return False
    if "U" in role_cells and int(out[role_cells["U"][0]]) != role_colors["U"]:
        return False

    counts = sorted(hit_count_by_role.values())
    return counts.count(1) == 1 and counts.count(2) == 1


def generate_input() -> np.ndarray:
    for _ in range(400):
        with_distractor = random.random() < 0.45
        role_colors, pairs = _scene_role_colors()

        if with_distractor:
            h = random.choice((15, 16))
            w = random.choice((15, 16))
            row_shift = random.randint(0, 1)
            col_shift = 0 if w == 15 else random.randint(0, 1)
            scene = _build_scene_layout(
                with_distractor=True,
                vflip=False,
                hflip=False,
                row_shift=row_shift,
                col_shift=col_shift,
            )
        else:
            vflip, hflip = random.choice(SAFE_CORE_TRANSFORMS)
            h = 16 if vflip else random.choice((15, 16))
            w = random.choice((15, 16))
            row_shift = 0 if vflip else (0 if h == 15 else random.randint(0, 1))
            col_shift = 0 if w == 15 else random.randint(0, 1)
            scene = _build_scene_layout(
                with_distractor=False,
                vflip=vflip,
                hflip=hflip,
                row_shift=row_shift,
                col_shift=col_shift,
            )

        grid = np.full((h, w), BG_COLOR, dtype=int)
        role_cells = _paint_scene(grid, scene, role_colors)
        _paint_legend(grid, pairs)

        if _validate_generated_scene(grid, role_cells, role_colors, pairs):
            return grid

    raise RuntimeError("failed to generate a valid ARC input")
