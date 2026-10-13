from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
ANCHOR = 5
BODY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ANCHOR_POS = (0, 0)
CORE_BODY = {(0, 1), (-1, 0), (1, 0)}
POOL = {
    (r, c)
    for r in range(-3, 4)
    for c in range(-3, 4)
    if (r, c) != ANCHOR_POS
}
GRID_RANGE = range(23, 30)
OBJECT_COUNT_RANGE = range(3, 6)
AREA_CHOICES = tuple(range(5, 13))
MAX_SHAPE_ATTEMPTS = 200
MAX_PLACEMENT_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(coord: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = coord
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _find_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _component_to_relative_cells(
    grid: np.ndarray, coords: list[tuple[int, int]]
) -> dict[str, object]:
    colors = {int(grid[r, c]) for r, c in coords}
    anchor_coords = [(r, c) for r, c in coords if int(grid[r, c]) == ANCHOR]
    assert len(anchor_coords) == 1, "each sprite must have exactly one gray anchor"
    body_colors = colors - {ANCHOR}
    assert len(body_colors) == 1, "each sprite must use exactly one body color"
    ar, ac = anchor_coords[0]
    rel_cells = [(int(r - ar), int(c - ac), int(grid[r, c])) for r, c in coords]
    rel_non_anchor = {(dr, dc) for dr, dc, color in rel_cells if color != ANCHOR}
    return {
        "area": len(coords),
        "body_color": int(next(iter(body_colors))),
        "rel_cells": rel_cells,
        "rel_non_anchor": rel_non_anchor,
    }


def _overlay_relative_sprites(sprites: list[dict[str, object]]) -> np.ndarray:
    rel_points = [
        (dr, dc)
        for sprite in sprites
        for dr, dc, color in sprite["rel_cells"]
        if color != BG
    ]
    min_r = min(dr for dr, _dc in rel_points)
    max_r = max(dr for dr, _dc in rel_points)
    min_c = min(dc for _dr, dc in rel_points)
    max_c = max(dc for _dr, dc in rel_points)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for sprite in sprites:
        for dr, dc, color in sprite["rel_cells"]:
            out[dr - min_r, dc - min_c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay anchored sprites in ascending pixel-count order and crop the composite.

    Input structure:
    - The grid has black background and 3-5 separated 4-connected multicolor sprites.
    - Each sprite contains exactly one gray anchor pixel and all other pixels share one unique body color.
    - Sprite areas are all distinct, and the sprites are meant to overlap when their gray anchors are aligned.

    Rule:
    - Split the non-black cells into connected sprites.
    - Count each sprite's total non-background pixels, including its gray anchor.
    - Align all gray anchors to the same point and overlay the sprites from smallest area to largest area, so later non-black pixels cover earlier ones.
    - Crop the final composite to the smallest bounding box containing all non-black pixels.

    Key insight:
    - The gray pixel is only a registration mark; the decisive information is each sprite's pixel count, because that count chooses the overlay depth and therefore the visible colors in overlap regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert components, "expected at least one sprite"

    sprites = [_component_to_relative_cells(g, coords) for coords in components]
    areas = [int(sprite["area"]) for sprite in sprites]
    assert len(set(areas)) == len(areas), "sprite areas must be unique"

    ordered = sorted(sprites, key=lambda sprite: int(sprite["area"]))
    return _crop_nonzero(_overlay_relative_sprites(ordered))


def _shortest_body_path(
    target: tuple[int, int], forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    if target in forbidden or target in CORE_BODY:
        return None

    start_nodes = {ANCHOR_POS, *CORE_BODY}
    allowed = (POOL - forbidden) | start_nodes
    queue = deque(start_nodes)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {node: None for node in start_nodes}

    while queue:
        cur = queue.popleft()
        if cur == target:
            path: list[tuple[int, int]] = []
            while parent[cur] is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for nxt in _neighbors4(cur):
            if nxt in allowed and nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return None


def _build_shape(
    total_area: int, forbidden: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    extra_slots = total_area - 1 - len(CORE_BODY)
    assert extra_slots >= 1

    candidates = list(POOL - forbidden - CORE_BODY)
    random.shuffle(candidates)
    for signature in candidates:
        path = _shortest_body_path(signature, forbidden)
        if path is None or len(path) > extra_slots:
            continue

        body = set(CORE_BODY)
        body.update(path)
        frontier = {
            nbr
            for cell in body | {ANCHOR_POS}
            for nbr in _neighbors4(cell)
            if nbr in POOL and nbr not in forbidden and nbr not in body
        }

        while len(body) < total_area - 1 and frontier:
            nxt = random.choice(tuple(frontier))
            body.add(nxt)
            frontier.remove(nxt)
            for nbr in _neighbors4(nxt):
                if nbr in POOL and nbr not in forbidden and nbr not in body:
                    frontier.add(nbr)

        if len(body) != total_area - 1:
            continue

        all_cells = body | {ANCHOR_POS}
        degrees = {
            cell: sum(nei in all_cells for nei in _neighbors4(cell))
            for cell in all_cells
        }
        if max(degrees.values()) < 3:
            continue

        r0, c0, r1, c1 = _bbox(all_cells)
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if bbox_area == len(all_cells):
            continue

        return body, signature
    return None


def _render_relative_sprite(body: set[tuple[int, int]], color: int) -> dict[str, object]:
    rel_cells = [(0, 0, ANCHOR)] + [(r, c, color) for r, c in sorted(body)]
    return {
        "area": len(rel_cells),
        "body_color": color,
        "rel_cells": rel_cells,
        "rel_non_anchor": set(body),
    }


def _place_sprites(sprites: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.full(shape, BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    halo: set[tuple[int, int]] = set()

    order = list(sprites)
    random.shuffle(order)
    for sprite in order:
        rel_cells = [(dr, dc, color) for dr, dc, color in sprite["rel_cells"]]
        rel_coords = [(dr, dc) for dr, dc, _color in rel_cells]
        min_r, min_c, max_r, max_c = _bbox(rel_coords)
        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            ar = random.randint(row_lo, row_hi)
            ac = random.randint(col_lo, col_hi)
            abs_cells = [(ar + dr, ac + dc, color) for dr, dc, color in rel_cells]
            abs_coords = {(r, c) for r, c, _color in abs_cells}
            if abs_coords & halo:
                continue
            for r, c, color in abs_cells:
                grid[r, c] = color
            occupied |= abs_coords
            for r, c in abs_coords:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if 0 <= rr < h and 0 <= cc < w:
                            halo.add((rr, cc))
            placed = True
            break
        if not placed:
            return None
    return grid


def _validate_semantics(grid: np.ndarray, sprites: list[dict[str, object]]) -> bool:
    components = _find_components(grid)
    if len(components) != len(sprites):
        return False

    parsed = [_component_to_relative_cells(grid, coords) for coords in components]
    parsed.sort(key=lambda sprite: int(sprite["area"]))
    expected_areas = sorted(int(sprite["area"]) for sprite in sprites)
    if [int(sprite["area"]) for sprite in parsed] != expected_areas:
        return False

    pairwise_overlap = True
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            if not (parsed[i]["rel_non_anchor"] & parsed[j]["rel_non_anchor"]):
                pairwise_overlap = False
    if not pairwise_overlap:
        return False

    for i, sprite in enumerate(parsed[:-1]):
        later_union: set[tuple[int, int]] = set()
        for later in parsed[i + 1 :]:
            later_union |= later["rel_non_anchor"]
        if not (sprite["rel_non_anchor"] - later_union):
            return False

    forward = _crop_nonzero(_overlay_relative_sprites(parsed))
    backward = _crop_nonzero(_overlay_relative_sprites(list(reversed(parsed))))
    if np.array_equal(forward, backward):
        return False
    if not np.array_equal(main(grid), forward):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choice(tuple(OBJECT_COUNT_RANGE))
    areas = sorted(random.sample(AREA_CHOICES, object_count))
    colors = random.sample(BODY_COLORS, object_count)

    sprites: list[dict[str, object]] = []
    forbidden_signatures: set[tuple[int, int]] = set()
    for area, color in zip(areas, colors):
        built = None
        for _ in range(MAX_SHAPE_ATTEMPTS):
            built = _build_shape(area, forbidden_signatures)
            if built is not None:
                break
        if built is None:
            return None
        body, signature = built
        sprites.append(_render_relative_sprite(body, color))
        forbidden_signatures.add(signature)

    grid_shape = (random.choice(tuple(GRID_RANGE)), random.choice(tuple(GRID_RANGE)))
    grid = _place_sprites(sprites, grid_shape)
    if grid is None:
        return None
    if not _validate_semantics(grid, sprites):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored overlay puzzle")
