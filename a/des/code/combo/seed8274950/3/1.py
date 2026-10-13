from __future__ import annotations

from collections import deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
QUADRANTS = ("tl", "tr", "bl", "br")
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(values.argmax())


def _normalize(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    arr = np.array(sorted((r - min_r, c - min_c) for r, c in cells), dtype=int)
    return arr


def _component_cells(grid: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> np.ndarray:
    color = int(grid[start])
    q = deque([start])
    seen[start] = True
    coords: list[tuple[int, int]] = []

    while q:
        r, c = q.popleft()
        coords.append((int(r), int(c)))
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if not bool(seen[nr, nc]) and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))

    return np.array(coords, dtype=int)


def _quadrant_of(coords: np.ndarray, center_r: int, center_c: int) -> str:
    rows = coords[:, 0]
    cols = coords[:, 1]
    top = bool(np.all(rows < center_r))
    bottom = bool(np.all(rows > center_r))
    left = bool(np.all(cols < center_c))
    right = bool(np.all(cols > center_c))
    assert top ^ bottom, "sprite rows must stay within one half"
    assert left ^ right, "sprite cols must stay within one half"
    if top and left:
        return "tl"
    if top and right:
        return "tr"
    if bottom and left:
        return "bl"
    return "br"


def _extract_groups(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int, dict[str, list[tuple[int, np.ndarray]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "grid must have a unique center"

    bg = _get_bg(g)
    center_r, center_c = h // 2, w // 2
    cross = int(g[center_r, center_c])
    assert cross != bg, "center cross must differ from background"
    assert np.all(g[center_r, :] == cross), "center row must be the cross"
    assert np.all(g[:, center_c] == cross), "center col must be the cross"

    groups: dict[str, list[tuple[int, np.ndarray]]] = {quadrant: [] for quadrant in QUADRANTS}
    seen = np.zeros_like(g, dtype=bool)

    for r in range(h):
        for c in range(w):
            if bool(seen[r, c]):
                continue
            value = int(g[r, c])
            if value == bg or value == cross:
                continue
            coords = _component_cells(g, (r, c), seen)
            quadrant = _quadrant_of(coords, center_r, center_c)
            groups[quadrant].append((value, coords))

    assert any(groups.values()), "at least one sprite is required"
    return g, bg, center_r, center_c, cross, groups


def _group_shift(group: list[tuple[int, np.ndarray]], quadrant: str, shape: tuple[int, int]) -> tuple[int, int]:
    coords = np.concatenate([coords for _, coords in group], axis=0)
    min_r = int(coords[:, 0].min())
    max_r = int(coords[:, 0].max())
    min_c = int(coords[:, 1].min())
    max_c = int(coords[:, 1].max())

    if quadrant in ("tl", "tr"):
        dr = -min_r
    else:
        dr = int(shape[0] - 1 - max_r)

    if quadrant in ("tl", "bl"):
        dc = -min_c
    else:
        dc = int(shape[1] - 1 - max_c)

    return dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group monochrome sprites by their position relative to a central cross, then slide
    each quadrant's combined layout to the matching corner.

    Input structure:
    - Background-filled odd-sized grid.
    - A full center row and full center column form a special-color cross.
    - Small monochrome 4-connected sprites lie strictly inside one of the four quadrants.

    Rule:
    - Collect all sprites in each quadrant.
    - Preserve the relative offsets within each quadrant, but translate the whole quadrant
      group so its bounding box is corner-aligned: top-left, top-right, bottom-left, or
      bottom-right.
    - Erase the cross to background in the output.
    - If translated quadrant groups overlap, keep pixels by priority:
      top-left > top-right > bottom-left > bottom-right.
    """
    g, bg, _center_r, _center_c, _cross, groups = _extract_groups(grid)
    out = np.full_like(g, bg)

    placements: dict[str, list[tuple[int, np.ndarray]]] = {}
    for quadrant in QUADRANTS:
        group = groups[quadrant]
        if not group:
            continue
        dr, dc = _group_shift(group, quadrant, g.shape)
        translated: list[tuple[int, np.ndarray]] = []
        for color, coords in group:
            shifted = coords + np.array([dr, dc], dtype=int)
            assert np.all(shifted[:, 0] >= 0) and np.all(shifted[:, 0] < g.shape[0])
            assert np.all(shifted[:, 1] >= 0) and np.all(shifted[:, 1] < g.shape[1])
            translated.append((color, shifted))
        placements[quadrant] = translated

    for quadrant in reversed(QUADRANTS):
        for color, coords in placements.get(quadrant, []):
            out[coords[:, 0], coords[:, 1]] = color

    return out


def _flip_h(coords: np.ndarray) -> np.ndarray:
    width = int(coords[:, 1].max()) + 1
    cells = {(int(r), int(width - 1 - c)) for r, c in coords}
    return _normalize(cells)


def _flip_v(coords: np.ndarray) -> np.ndarray:
    height = int(coords[:, 0].max()) + 1
    cells = {(int(height - 1 - r), int(c)) for r, c in coords}
    return _normalize(cells)


def _transpose(coords: np.ndarray) -> np.ndarray:
    cells = {(int(c), int(r)) for r, c in coords}
    return _normalize(cells)


def _random_shape(rng: np.random.Generator, side_h: int, side_w: int) -> np.ndarray:
    max_box_h = min(4, side_h)
    max_box_w = min(4, side_w)
    for _ in range(100):
        box_h = int(rng.integers(1, max_box_h + 1))
        box_w = int(rng.integers(1, max_box_w + 1))
        max_area = min(box_h * box_w, 6)
        area = int(rng.integers(1, max_area + 1))

        start = (int(rng.integers(box_h)), int(rng.integers(box_w)))
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])

        if len(cells) != area:
            continue

        coords = _normalize(cells)
        if rng.random() < 0.5:
            coords = _flip_h(coords)
        if rng.random() < 0.5:
            coords = _flip_v(coords)
        if rng.random() < 0.5:
            coords = _transpose(coords)
        return coords

    raise RuntimeError("failed to sample a connected shape")


def _can_place(coords: np.ndarray, occupied: np.ndarray) -> bool:
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if occupied[rr, cc]:
            return False
        r0 = max(0, rr - 1)
        r1 = min(occupied.shape[0], rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(occupied.shape[1], cc + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _place_quadrant(
    rng: np.random.Generator, side_h: int, side_w: int, sprite_count: int
) -> list[np.ndarray] | None:
    occupied = np.zeros((side_h, side_w), dtype=bool)
    placed: list[np.ndarray] = []

    for _ in range(sprite_count):
        success = False
        for _ in range(120):
            shape = _random_shape(rng, side_h, side_w)
            shape_h = int(shape[:, 0].max()) + 1
            shape_w = int(shape[:, 1].max()) + 1
            if shape_h > side_h or shape_w > side_w:
                continue
            top = int(rng.integers(0, side_h - shape_h + 1))
            left = int(rng.integers(0, side_w - shape_w + 1))
            coords = shape + np.array([top, left], dtype=int)
            if not _can_place(coords, occupied):
                continue
            occupied[coords[:, 0], coords[:, 1]] = True
            placed.append(coords)
            success = True
            break
        if not success:
            return None

    return placed


def _to_global(coords: np.ndarray, quadrant: str, side_h: int, side_w: int) -> np.ndarray:
    row_base = 0 if quadrant in ("tl", "tr") else side_h + 1
    col_base = 0 if quadrant in ("tl", "bl") else side_w + 1
    return coords + np.array([row_base, col_base], dtype=int)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    side_h = int(rng.integers(5, 10))
    side_w = int(rng.integers(5, 10))
    h = 2 * side_h + 1
    w = 2 * side_w + 1

    colors = list(rng.permutation(PALETTE))
    bg = int(colors[0])
    cross = int(colors[1])
    sprite_pool = [int(color) for color in colors[2:]]

    occupied_count = int(rng.integers(1, 5))
    occupied_quadrants = list(rng.choice(QUADRANTS, size=occupied_count, replace=False))

    quadrant_sprites: dict[str, list[np.ndarray]] = {}
    total_sprites = 0
    for quadrant in occupied_quadrants:
        sprite_count = int(rng.integers(1, 4 if min(side_h, side_w) >= 6 else 3))
        placed = _place_quadrant(rng, side_h, side_w, sprite_count)
        if placed is None:
            return None
        quadrant_sprites[quadrant] = placed
        total_sprites += len(placed)

    sprite_palette_size = int(rng.integers(2, min(len(sprite_pool), 6) + 1))
    sprite_palette = list(rng.choice(sprite_pool, size=sprite_palette_size, replace=False))
    sprite_colors = [int(sprite_palette[int(rng.integers(len(sprite_palette)))]) for _ in range(total_sprites)]
    if total_sprites > 1 and len(set(sprite_colors)) == 1 and len(sprite_palette) > 1:
        sprite_colors[-1] = int(sprite_palette[1])

    grid = np.full((h, w), bg, dtype=int)
    center_r, center_c = h // 2, w // 2
    grid[center_r, :] = cross
    grid[:, center_c] = cross

    color_idx = 0
    for quadrant in QUADRANTS:
        for coords in quadrant_sprites.get(quadrant, []):
            global_coords = _to_global(coords, quadrant, side_h, side_w)
            color = int(sprite_colors[color_idx])
            color_idx += 1
            grid[global_coords[:, 0], global_coords[:, 1]] = color

    try:
        _, _, _, _, _, groups = _extract_groups(grid)
        moved = any(
            groups[quadrant] and _group_shift(groups[quadrant], quadrant, grid.shape) != (0, 0)
            for quadrant in QUADRANTS
        )
        if not moved:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
