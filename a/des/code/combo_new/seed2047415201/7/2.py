from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0

DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GAP_BY_OFFSET = {offset: name for name, offset in DIRECTIONS.items()}
HIT_SIDE_BY_STEP: dict[tuple[int, int], str] = {
    (-1, 0): "down",
    (1, 0): "up",
    (0, -1): "right",
    (0, 1): "left",
}
SIDE_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "up": ((-1, -1), (-1, 0), (-1, 1)),
    "down": ((1, -1), (1, 0), (1, 1)),
    "left": ((-1, -1), (0, -1), (1, -1)),
    "right": ((-1, 1), (0, 1), (1, 1)),
}
DIR_NAMES = tuple(DIRECTIONS)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in DIRECTIONS.values():
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _find_sprites(g: np.ndarray) -> list[dict]:
    comps = _components(g != BLACK)
    assert comps, "expected at least one sprite"

    sprites: list[dict] = []
    used_colors: set[int] = set()
    for coords in comps:
        values = {int(g[r, c]) for r, c in coords}
        assert len(values) == 1, "each sprite must be monochrome in the input"
        color = values.pop()
        assert color != BLACK
        assert color not in used_colors, "sprite colors must be distinct"
        used_colors.add(color)

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        assert r1 - r0 == 2 and c1 - c0 == 2, "each sprite must fit a 3x3 box"

        block = g[r0 : r1 + 1, c0 : c1 + 1]
        assert int(np.count_nonzero(block == color)) == 8, "sprite must have eight colored pixels"
        assert set(map(int, np.unique(block))) <= {BLACK, color}, "sprite box must contain only its color and black"
        zeros = np.argwhere(block == BLACK)
        assert zeros.shape == (1, 2), "sprite must have exactly one missing side pixel"
        zr, zc = map(int, zeros[0])
        gap_offset = (zr - 1, zc - 1)
        assert gap_offset in GAP_BY_OFFSET, "missing pixel must be the middle of one side"
        assert int(block[1, 1]) == color, "sprite center must be present"

        sprites.append(
            {
                "id": len(sprites),
                "color": color,
                "gap": GAP_BY_OFFSET[gap_offset],
                "center": (r0 + 1, c0 + 1),
                "bbox": (r0, c0, r1, c1),
            }
        )
    return sprites


def _sprite_indexes(g: np.ndarray, sprites: list[dict]) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    colored_owner: dict[tuple[int, int], int] = {}
    box_owner: dict[tuple[int, int], int] = {}
    for sprite in sprites:
        sid = int(sprite["id"])
        color = int(sprite["color"])
        r0, c0, r1, c1 = sprite["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                assert (r, c) not in box_owner, "sprite boxes must not overlap"
                box_owner[(r, c)] = sid
                if int(g[r, c]) != BLACK:
                    assert int(g[r, c]) == color
                    colored_owner[(r, c)] = sid
    return colored_owner, box_owner


def _face_cells(sprite: dict, side: str) -> list[tuple[int, int]]:
    cr, cc = sprite["center"]
    return [(cr + dr, cc + dc) for dr, dc in SIDE_OFFSETS[side]]


def _put_color(target: dict[tuple[int, int], int], cell: tuple[int, int], color: int) -> None:
    previous = target.get(cell)
    assert previous is None or previous == color, "ambiguous overlapping ray or receiver coloring"
    target[cell] = int(color)


def _transform(grid: np.ndarray, with_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must not exceed 30x30"

    sprites = _find_sprites(g)
    colored_owner, box_owner = _sprite_indexes(g, sprites)

    ray_paint: dict[tuple[int, int], int] = {}
    face_paint: dict[tuple[int, int], int] = {}
    hit_count = 0

    for sprite in sprites:
        sid = int(sprite["id"])
        color = int(sprite["color"])
        dr, dc = DIRECTIONS[sprite["gap"]]
        r, c = sprite["center"][0] + dr, sprite["center"][1] + dc
        while 0 <= r < h and 0 <= c < w:
            owner = colored_owner.get((r, c))
            if owner is not None and owner != sid:
                hit_count += 1
                target = sprites[owner]
                side = HIT_SIDE_BY_STEP[(dr, dc)]
                for cell in _face_cells(target, side):
                    if colored_owner.get(cell) == owner:
                        _put_color(face_paint, cell, color)
                break

            if (r, c) not in box_owner and int(g[r, c]) == BLACK:
                _put_color(ray_paint, (r, c), color)
            r += dr
            c += dc

    out = g.copy()
    for (r, c), color in ray_paint.items():
        out[r, c] = color
    for (r, c), color in face_paint.items():
        out[r, c] = color

    if with_stats:
        return out, {
            "hits": hit_count,
            "ray_cells": len(ray_paint),
            "directions": {sprite["gap"] for sprite in sprites},
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw colored signal rays from each sprite gap and recolor receiver faces.

    Input structure:
    - The background is black.
    - Each foreground object is a distinct-color 3x3 sprite with exactly one
      missing middle pixel on its top, bottom, left, or right side.
    - Sprite boxes do not overlap, and the missing side pixel marks that
      sprite's outgoing ray direction.

    Rule:
    - From every sprite center, cast a one-cell-wide straight ray through its
      missing side, painting only black background cells outside all sprite boxes.
    - The ray stops at the first colored pixel of another sprite and does not
      continue through that sprite.
    - The hit side of the receiving sprite is recolored with the ray color, but
      any missing gap pixel on that side remains black.

    Key insight:
    - The side gap is both a direction marker and a protected hole; the visible
      transformation combines path drawing with a second receiver-side recolor.
    """
    return _transform(grid)


# --------------- generator ---------------

GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
SPRITE_COUNT_RANGE = (3, 7)
COLOR_VALUES = tuple(range(1, 10))
PAIR_DISTANCE_CAP = 16
PLACE_RETRIES = 600
GENERATE_RETRIES = 500


def _separated(center: tuple[int, int], centers: list[tuple[int, int]]) -> bool:
    r, c = center
    return all(abs(r - rr) >= 4 or abs(c - cc) >= 4 for rr, cc in centers)


def _make_required_pair(h: int, w: int) -> tuple[list[tuple[int, int]], str] | None:
    for _ in range(PLACE_RETRIES):
        direction = random.choice(DIR_NAMES)
        dr, dc = DIRECTIONS[direction]
        if dr == 0:
            max_distance = min(PAIR_DISTANCE_CAP, w - 3)
            if max_distance < 4:
                continue
            distance = random.randint(4, max_distance)
            r = random.randint(1, h - 2)
            if dc > 0:
                c = random.randint(1, w - 2 - distance)
                return [(r, c), (r, c + distance)], direction
            c = random.randint(1 + distance, w - 2)
            return [(r, c), (r, c - distance)], direction

        max_distance = min(PAIR_DISTANCE_CAP, h - 3)
        if max_distance < 4:
            continue
        distance = random.randint(4, max_distance)
        c = random.randint(1, w - 2)
        if dr > 0:
            r = random.randint(1, h - 2 - distance)
            return [(r, c), (r + distance, c)], direction
        r = random.randint(1 + distance, h - 2)
        return [(r, c), (r - distance, c)], direction
    return None


def _place_extra_centers(h: int, w: int, centers: list[tuple[int, int]], target_count: int) -> bool:
    while len(centers) < target_count:
        placed = False
        for _ in range(PLACE_RETRIES):
            candidate = (random.randint(1, h - 2), random.randint(1, w - 2))
            if _separated(candidate, centers):
                centers.append(candidate)
                placed = True
                break
        if not placed:
            return False
    return True


def _render_grid(h: int, w: int, centers: list[tuple[int, int]], gaps: list[str], colors: list[int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for (cr, cc), gap, color in zip(centers, gaps, colors):
        gr, gc = DIRECTIONS[gap]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (gr, gc):
                    continue
                grid[cr + dr, cc + dc] = color
    return grid


def _receiver_gaps_are_clear(grid: np.ndarray) -> bool:
    sprites = _find_sprites(grid)
    colored_owner, _box_owner = _sprite_indexes(grid, sprites)
    h, w = grid.shape
    for sprite in sprites:
        sid = int(sprite["id"])
        dr, dc = DIRECTIONS[sprite["gap"]]
        r, c = sprite["center"][0] + dr, sprite["center"][1] + dc
        while 0 <= r < h and 0 <= c < w:
            owner = colored_owner.get((r, c))
            if owner is not None and owner != sid:
                incoming_side = HIT_SIDE_BY_STEP[(dr, dc)]
                if sprites[owner]["gap"] == incoming_side:
                    return False
                break
            r += dr
            c += dc
    return True


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if not _receiver_gaps_are_clear(grid):
        return False
    try:
        out, stats = _transform(grid, with_stats=True)
    except AssertionError:
        return False
    if stats["hits"] < 1:
        return False
    if stats["ray_cells"] < 2:
        return False
    if len(stats["directions"]) < 2:
        return False
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    count = random.randint(*SPRITE_COUNT_RANGE)
    pair = _make_required_pair(h, w)
    if pair is None:
        return None
    centers, source_gap = pair
    if not _place_extra_centers(h, w, centers, count):
        return None

    colors = random.sample(COLOR_VALUES, count)
    random.shuffle(centers)
    source_center = pair[0][0]
    target_center = pair[0][1]
    source_index = centers.index(source_center)
    target_index = centers.index(target_center)

    gaps = [random.choice(DIR_NAMES) for _ in range(count)]
    gaps[source_index] = source_gap
    incoming_to_target = HIT_SIDE_BY_STEP[DIRECTIONS[source_gap]]
    gaps[target_index] = random.choice([name for name in DIR_NAMES if name != incoming_to_target])

    grid = _render_grid(h, w, centers, gaps, colors)
    if _valid_generated_grid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
