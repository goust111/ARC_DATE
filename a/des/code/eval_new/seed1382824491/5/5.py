from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
START_COLOR = 9
STEP = 3
GEN_RETRIES = 500
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)

DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}

SPRITE_BASES = [
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (0, 2), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _rotate90(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_r = max(r for r, _ in cells)
    return {(c, max_r - r) for r, c in cells}


def _flip_h(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _, c in cells)
    return {(r, max_c - c) for r, c in cells}


def _shape_variants(base: set[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    variants: set[frozenset[tuple[int, int]]] = set()
    current = set(base)
    for _ in range(4):
        variants.add(_normalize(current))
        variants.add(_normalize(_flip_h(current)))
        current = _rotate90(current)
    return list(variants)


SPRITE_SHAPES = [
    variant
    for base in SPRITE_BASES
    for variant in _shape_variants(base)
]


def _component_adjacent(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _component_overlap(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    return not a.isdisjoint(b)


def _component_boxes(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r for r, _ in cells),
        min(c for _, c in cells),
        max(r for r, _ in cells),
        max(c for _, c in cells),
    )


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    components: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        mask = grid == int(color)
        visited = np.zeros_like(mask, dtype=bool)
        for r in range(h):
            for c in range(w):
                if not bool(mask[r, c]) or bool(visited[r, c]):
                    continue
                visited[r, c] = True
                queue = deque([(r, c)])
                cells = {(int(r), int(c))}
                while queue:
                    rr, cc = queue.popleft()
                    for nr, nc in _neighbors4(int(rr), int(cc)):
                        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            cells.add((int(nr), int(nc)))
                components.append({"color": int(color), "cells": cells})
    return components


def _detect_arrow(cells: set[tuple[int, int]]) -> str | None:
    if len(cells) != 5:
        return None
    normalized = _normalize(cells)
    for direction, pattern in ARROW_PATTERNS.items():
        if normalized == pattern:
            return direction
    return None


def _arrow_front(cells: set[tuple[int, int]], direction: str) -> int:
    if direction == "up":
        return min(r for r, _ in cells)
    if direction == "down":
        return max(r for r, _ in cells)
    if direction == "left":
        return min(c for _, c in cells)
    return max(c for _, c in cells)


def _sprite_front(cells: set[tuple[int, int]], direction: str) -> int:
    if direction == "up":
        return min(r for r, _ in cells)
    if direction == "down":
        return max(r for r, _ in cells)
    if direction == "left":
        return min(c for _, c in cells)
    return max(c for _, c in cells)


def _movement_delta(sprite_cells: set[tuple[int, int]], arrow_cells: set[tuple[int, int]], direction: str) -> tuple[int, int]:
    shift = _arrow_front(arrow_cells, direction) - _sprite_front(sprite_cells, direction)
    if direction == "up":
        assert shift < 0
        return int(shift), 0
    if direction == "down":
        assert shift > 0
        return int(shift), 0
    if direction == "left":
        assert shift < 0
        return 0, int(shift)
    assert shift > 0
    return 0, int(shift)


def _paint_cells(canvas: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    h, w = canvas.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            canvas[r, c] = int(color)


def _start_candidates(
    sprite_cells: set[tuple[int, int]],
    arrows: list[dict[str, object]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    arrow_by_cell: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            arrow_by_cell[cell] = idx

    candidates: list[tuple[int, int]] = []
    for r, c in sprite_cells:
        for cell in _neighbors4(r, c):
            if cell in occupied:
                continue
            arrow_neighbors = [nb for nb in _neighbors4(*cell) if nb in arrow_by_cell]
            if len(arrow_neighbors) != 1:
                continue
            if arrow_by_cell[arrow_neighbors[0]] != 0:
                continue
            candidates.append(cell)
    return sorted(set(candidates))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode scattered arrow glyphs into a movement chain and leave arrow-colored footprints at each departure position.

    Input structure:
    - The background is black, with one monochrome multi-pixel active sprite,
      several monochrome arrow glyphs, and one singleton start marker.
    - Each arrow glyph is a 4-connected 5-pixel shape pointing up, down, left,
      or right; arrow colors are distinct from the active sprite color.
    - The start marker sits 4-neighbor to the initial sprite and also touches
      exactly one pixel of the first arrow glyph.

    Rule:
    - Identify the first arrow from the start marker, then repeatedly move the
      active sprite in that arrow's direction until the sprite's leading edge is
      aligned with the arrow's front tip row or column.
    - After each move, the sprite's previous position becomes a footprint in the
      triggering arrow's color; the moved sprite keeps its original color.
    - At every new stop, exactly one unused arrow glyph is 4-neighbor to the
      sprite; this defines the next move. Stop when no unused adjacent arrow remains.
    - Remove the start marker and all arrow glyphs from the output.

    Key insight:
    - The arrows encode both order and color: their geometry gives movement
      direction, local adjacency reveals the chain, and each used arrow supplies
      the footprint color for the position the sprite just vacated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    components = _monochrome_components(g, bg=bg)
    arrows: list[dict[str, object]] = []
    singleton_components: list[dict[str, object]] = []
    other_components: list[dict[str, object]] = []

    for comp in components:
        cells = comp["cells"]
        direction = _detect_arrow(cells)
        if direction is not None:
            arrows.append({"color": int(comp["color"]), "cells": set(cells), "direction": direction})
        elif len(cells) == 1:
            singleton_components.append(comp)
        else:
            other_components.append(comp)

    assert arrows, "expected at least one arrow"
    assert other_components, "expected a sprite component"

    start_candidates = []
    for comp in singleton_components:
        cell = next(iter(comp["cells"]))
        adj_arrow_ids = [idx for idx, arrow in enumerate(arrows) if _component_adjacent({cell}, arrow["cells"])]
        adj_sprite_ids = [idx for idx, sprite in enumerate(other_components) if _component_adjacent({cell}, sprite["cells"])]
        arrow_pixel_neighbors = sum(1 for nb in _neighbors4(*cell) if any(nb in arrow["cells"] for arrow in arrows))
        if len(adj_arrow_ids) == 1 and len(adj_sprite_ids) == 1 and arrow_pixel_neighbors == 1:
            start_candidates.append((comp, adj_sprite_ids[0]))

    if len(start_candidates) != 1:
        fallback = []
        for comp in singleton_components:
            cell = next(iter(comp["cells"]))
            adj_arrow_ids = [idx for idx, arrow in enumerate(arrows) if _component_adjacent({cell}, arrow["cells"])]
            adj_sprite_ids = [idx for idx, sprite in enumerate(other_components) if _component_adjacent({cell}, sprite["cells"])]
            if len(adj_arrow_ids) == 1 and len(adj_sprite_ids) == 1:
                fallback.append((comp, adj_sprite_ids[0]))
        start_candidates = fallback

    assert len(start_candidates) == 1, "expected exactly one start marker"
    start_comp, sprite_idx = start_candidates[0]
    start_cell = next(iter(start_comp["cells"]))
    sprite = other_components[int(sprite_idx)]
    sprite_cells = set(sprite["cells"])
    active_color = int(sprite["color"])

    out = np.full_like(g, bg)
    used: set[int] = set()

    current_arrow_ids = [idx for idx, arrow in enumerate(arrows) if _component_adjacent({start_cell}, arrow["cells"])]
    assert len(current_arrow_ids) == 1
    current_arrow_idx = int(current_arrow_ids[0])

    while True:
        arrow = arrows[current_arrow_idx]
        _paint_cells(out, sprite_cells, int(arrow["color"]))
        dr, dc = _movement_delta(sprite_cells, arrow["cells"], str(arrow["direction"]))
        sprite_cells = _translate(sprite_cells, int(dr), int(dc))
        used.add(current_arrow_idx)
        next_arrow_ids = [
            idx
            for idx, candidate in enumerate(arrows)
            if idx not in used and _component_adjacent(sprite_cells, candidate["cells"])
        ]
        assert len(next_arrow_ids) <= 1, "sprite must have at most one unused adjacent arrow"
        if not next_arrow_ids:
            break
        current_arrow_idx = int(next_arrow_ids[0])

    _paint_cells(out, sprite_cells, active_color)
    return out


def _arrow_from_boundary(direction: str, cell: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = cell
    if direction == "right":
        return {(r, c + 1), (r, c + 2), (r, c + 3), (r - 1, c + 3), (r + 1, c + 3)}
    if direction == "left":
        return {(r, c - 1), (r, c - 2), (r, c - 3), (r - 1, c - 3), (r + 1, c - 3)}
    if direction == "down":
        return {(r + 1, c), (r + 2, c), (r + 3, c), (r + 3, c - 1), (r + 3, c + 1)}
    return {(r - 1, c), (r - 2, c), (r - 3, c), (r - 3, c - 1), (r - 3, c + 1)}


def _boundary_cells(cells: set[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    if direction == "right":
        edge = max(c for _, c in cells)
        return [(r, c) for r, c in cells if c == edge]
    if direction == "left":
        edge = min(c for _, c in cells)
        return [(r, c) for r, c in cells if c == edge]
    if direction == "down":
        edge = max(r for r, _ in cells)
        return [(r, c) for r, c in cells if r == edge]
    edge = min(r for r, _ in cells)
    return [(r, c) for r, c in cells if r == edge]


def _build_offsets(shape: frozenset[tuple[int, int]], steps: int) -> tuple[list[str], list[tuple[int, int]], list[set[tuple[int, int]]]] | None:
    base = set(shape)
    offsets = [(0, 0)]
    positions = [base]
    directions: list[str] = []

    for _ in range(steps):
        choices = list(DIR_VECTORS)
        random.shuffle(choices)
        placed = False
        for direction in choices:
            dr, dc = DIR_VECTORS[direction]
            new_offset = (offsets[-1][0] + STEP * dr, offsets[-1][1] + STEP * dc)
            new_cells = _translate(base, new_offset[0], new_offset[1])
            if any(_component_overlap(new_cells, prev) for prev in positions):
                continue
            directions.append(direction)
            offsets.append(new_offset)
            positions.append(new_cells)
            placed = True
            break
        if not placed:
            return None

    return directions, offsets, positions


def _place_arrows(
    positions: list[set[tuple[int, int]]],
    directions: list[str],
) -> list[dict[str, object]] | None:
    initial_sprite = positions[0]
    placed: list[dict[str, object]] = []

    def rec(idx: int) -> bool:
        if idx == len(directions):
            return True
        direction = directions[idx]
        candidates = _boundary_cells(positions[idx], direction)
        random.shuffle(candidates)
        for cell in candidates:
            arrow_cells = _arrow_from_boundary(direction, cell)
            if _component_overlap(arrow_cells, initial_sprite):
                continue
            conflict = False
            for other in placed:
                other_cells = other["cells"]
                if _component_overlap(arrow_cells, other_cells) or _component_adjacent(arrow_cells, other_cells):
                    conflict = True
                    break
            if conflict:
                continue
            placed.append({"direction": direction, "cells": arrow_cells})
            if rec(idx + 1):
                return True
            placed.pop()
        return False

    return placed if rec(0) else None


def _validate_chain(
    positions: list[set[tuple[int, int]]],
    arrows: list[dict[str, object]],
    start_cell: tuple[int, int],
) -> bool:
    arrow_by_cell = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            arrow_by_cell[cell] = idx

    start_arrow_pixels = [nb for nb in _neighbors4(*start_cell) if nb in arrow_by_cell]
    if len(start_arrow_pixels) != 1 or arrow_by_cell[start_arrow_pixels[0]] != 0:
        return False

    if not _component_adjacent({start_cell}, positions[0]):
        return False

    for idx, pos in enumerate(positions[:-1]):
        reachable = [
            arrow_idx
            for arrow_idx in range(idx, len(arrows))
            if _component_adjacent(pos, arrows[arrow_idx]["cells"])
        ]
        if reachable != [idx]:
            return False

    # The final sprite is expected to touch or overlap the last arrow's front:
    # that is exactly how the stopping row/column is defined.  Only unused
    # arrows would continue the chain, and there are none left here.
    return True


def _place_on_canvas(
    sprite_cells: set[tuple[int, int]],
    arrows: list[dict[str, object]],
    start_cell: tuple[int, int],
    positions: list[set[tuple[int, int]]],
) -> np.ndarray | None:
    all_cells = set(sprite_cells) | {start_cell}
    for arrow in arrows:
        all_cells |= set(arrow["cells"])
    for pos in positions[1:]:
        all_cells |= set(pos)

    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    if span_h > 30 or span_w > 30:
        return None

    extra_h = 30 - span_h
    extra_w = 30 - span_w
    top = random.randint(1, min(4, extra_h)) if extra_h >= 1 else 0
    left = random.randint(1, min(4, extra_w)) if extra_w >= 1 else 0
    remaining_h = extra_h - top
    remaining_w = extra_w - left
    bottom = random.randint(0, min(4, remaining_h)) if remaining_h >= 0 else 0
    right = random.randint(0, min(4, remaining_w)) if remaining_w >= 0 else 0

    shift_r = top - min_r
    shift_c = left - min_c

    height = span_h + top + bottom
    width = span_w + left + right
    grid = np.full((height, width), BG, dtype=int)

    active_color = random.choice(COLOR_POOL)
    arrow_colors = random.sample([color for color in COLOR_POOL if color != active_color], len(arrows))

    for r, c in _translate(sprite_cells, shift_r, shift_c):
        grid[r, c] = int(active_color)
    sr, sc = start_cell
    grid[sr + shift_r, sc + shift_c] = START_COLOR
    for color, arrow in zip(arrow_colors, arrows):
        for r, c in _translate(arrow["cells"], shift_r, shift_c):
            grid[r, c] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = set(random.choice(SPRITE_SHAPES))
    steps = random.randint(1, 5)
    built = _build_offsets(frozenset(shape), steps)
    if built is None:
        return None
    directions, _offsets, positions = built

    arrows = _place_arrows(positions, directions)
    if arrows is None:
        return None

    occupied = set(positions[0])
    for arrow in arrows:
        occupied |= set(arrow["cells"])
    starts = _start_candidates(positions[0], arrows, occupied)
    if not starts:
        return None
    start_cell = random.choice(starts)

    if not _validate_chain(positions, arrows, start_cell):
        return None

    grid = _place_on_canvas(positions[0], arrows, start_cell, positions)
    if grid is None:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
