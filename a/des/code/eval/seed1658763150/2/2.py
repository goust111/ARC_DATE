import random
from collections import Counter, deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PALETTE = (1, 2, 3, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _line_orientation(coords: list[tuple[int, int]]) -> str | None:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        return "horizontal" if cols == list(range(cols[0], cols[-1] + 1)) else None
    if len(cols) == 1:
        return "vertical" if rows == list(range(rows[0], rows[-1] + 1)) else None
    return None


def _find_wall(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        wall_color = int(grid[0, c])
        if wall_color == bg:
            continue
        if not np.all(grid[:, c] == wall_color):
            continue
        if not np.all(grid[:, c + 1 :] == bg):
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        candidates.append((c, wall_color))
    assert len(candidates) == 1, "expected a unique wall column"
    return candidates[0]


def _extract_objects(
    grid: np.ndarray, bg: int, wall_col: int
) -> tuple[tuple[int, int], tuple[list[tuple[int, int]], str], list[list[tuple[int, int]]]]:
    mask = grid != bg
    mask[:, wall_col] = False
    comps = _components(mask)
    assert comps, "expected non-wall foreground objects"

    light_candidates = [coords for coords in comps if len(coords) == 1]
    assert len(light_candidates) == 1, "expected a unique light source"
    light = light_candidates[0][0]

    filter_candidates: list[tuple[list[tuple[int, int]], str]] = []
    for coords in comps:
        if len(coords) < 2:
            continue
        orientation = _line_orientation(coords)
        if orientation is None:
            continue
        colors = [int(grid[r, c]) for r, c in coords]
        if len(set(colors)) != len(colors):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if orientation == "vertical":
            if not (light[1] < c0 < wall_col):
                continue
        else:
            if not (light[1] < c0 and c1 < wall_col):
                continue
        filter_candidates.append((coords, orientation))

    assert len(filter_candidates) == 1, "expected a unique filter strip"
    filter_coords, orientation = filter_candidates[0]

    filter_set = set(filter_coords)
    sprites: list[list[tuple[int, int]]] = []
    for coords in comps:
        if coords == light_candidates[0] or set(coords) == filter_set:
            continue
        colors = {int(grid[r, c]) for r, c in coords}
        assert 2 <= len(coords) <= 5, "sprite size must stay in the intended range"
        assert len(colors) == 1, "sprites must be monochrome components"
        sprites.append(coords)

    assert sprites, "expected at least one sprite"
    return light, (filter_coords, orientation), sprites


def _vertical_filter_lookup(grid: np.ndarray, filter_coords: list[tuple[int, int]]) -> dict[int, int]:
    return {int(r): int(grid[r, c]) for r, c in filter_coords}


def _horizontal_filter_lookup(
    grid: np.ndarray,
    filter_coords: list[tuple[int, int]],
    light: tuple[int, int],
    sprites: list[list[tuple[int, int]]],
) -> dict[int, int]:
    filter_row = filter_coords[0][0]
    filter_by_col = {int(c): int(grid[r, c]) for r, c in filter_coords}
    ordered_cols = sorted(filter_by_col)
    top_rows = sorted({min(r for r, _ in sprite) for sprite in sprites})

    lookup: dict[int, int] = {}
    for top_row in top_rows:
        candidates = [top_row]
        for sprite in sprites:
            if min(r for r, _ in sprite) == top_row:
                top_cells = [(r, c) for r, c in sprite if r == top_row]
                anchor_r, anchor_c = min(top_cells, key=lambda rc: rc[1])
                if anchor_r != light[0]:
                    raw_col = light[1] + (filter_row - light[0]) * (anchor_c - light[1]) / (anchor_r - light[0])
                    rounded = int(round(raw_col))
                    if rounded in filter_by_col and abs(raw_col - rounded) < 1e-6:
                        candidates.append(rounded)
        col = candidates[-1] if len(candidates) > 1 else ordered_cols[len(lookup) % len(ordered_cols)]
        if col not in filter_by_col:
            col = ordered_cols[len(lookup) % len(ordered_cols)]
        lookup[top_row] = filter_by_col[col]
    return lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each left-side sprite onto the wall by preserving its row offsets from the light source.

    The wall is the unique full-height non-background column with only background to its right.
    The light source is the unique single-pixel component. The filter strip is the unique straight
    multicolor component between the light and the wall. Each sprite projects to the wall column
    on the same set of rows it occupies; the projection color comes from the filter entry aligned
    with the sprite's top row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)

    wall_col, _wall_color = _find_wall(g, bg)
    light, (filter_coords, filter_orientation), sprites = _extract_objects(g, bg, wall_col)

    if filter_orientation == "vertical":
        filter_lookup = _vertical_filter_lookup(g, filter_coords)
    else:
        filter_lookup = _horizontal_filter_lookup(g, filter_coords, light, sprites)

    out = g.copy()
    for sprite in sprites:
        top_row = min(r for r, _ in sprite)
        color = filter_lookup[top_row]
        shadow_rows = {light[0] + (r - light[0]) for r, _ in sprite}
        for row in shadow_rows:
            if 0 <= row < g.shape[0]:
                out[row, wall_col] = color
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _reserve(blocked: set[tuple[int, int]], cells: list[tuple[int, int]], h: int, w: int) -> None:
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))


def _can_place(cells: list[tuple[int, int]], blocked: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w and (r, c) not in blocked for r, c in cells)


def _sample_layout() -> np.ndarray | None:
    h = random.randint(9, 13)
    w = random.randint(14, 18)
    grid = np.full((h, w), BLACK, dtype=int)

    wall_col = random.randint(w - 4, w - 2)
    filter_col = random.randint(max(4, wall_col - 4), wall_col - 2)
    light_col = random.randint(1, min(3, filter_col - 4))

    filter_len = random.randint(4, min(7, h - 2))
    light_row = random.randint(1, h - 2)
    start_lo = max(1, light_row - filter_len + 1)
    start_hi = min(light_row, h - filter_len)
    if start_lo > start_hi:
        return None
    filter_r0 = random.randint(start_lo, start_hi)
    filter_rows = list(range(filter_r0, filter_r0 + filter_len))

    grid[:, wall_col] = GREY
    grid[light_row, light_col] = YELLOW

    filter_colors = random.sample(PALETTE, filter_len)
    random.shuffle(filter_colors)
    for r, color in zip(filter_rows, filter_colors):
        grid[r, filter_col] = color

    blocked: set[tuple[int, int]] = set()
    _reserve(blocked, [(light_row, light_col)], h, w)
    _reserve(blocked, [(r, filter_col) for r in filter_rows], h, w)
    _reserve(blocked, [(r, wall_col) for r in range(h)], h, w)

    sprite_count = random.choice((2, 3))
    sprite_region_left = light_col + 2
    sprite_region_right = filter_col - 2
    if sprite_region_left > sprite_region_right:
        return None

    used_shadow_rows: set[int] = set()
    top_rows: list[int] = []
    for _ in range(sprite_count):
        placed = False
        for _attempt in range(200):
            shape = random.choice(SHAPES)
            shape_h, shape_w = _shape_dims(shape)
            if shape_w > sprite_region_right - sprite_region_left + 1:
                continue

            top_row = random.choice(filter_rows)
            shadow_rows = {top_row + r for r, _ in shape}
            if max(shadow_rows) >= h:
                continue
            if shadow_rows & used_shadow_rows:
                continue

            col0 = random.randint(sprite_region_left, sprite_region_right - shape_w + 1)
            cells = [(top_row + r, col0 + c) for r, c in shape]
            if not _can_place(cells, blocked, h, w):
                continue

            sprite_color = random.choice(PALETTE)
            for r, c in cells:
                grid[r, c] = sprite_color
            _reserve(blocked, cells, h, w)
            used_shadow_rows |= shadow_rows
            top_rows.append(top_row)
            placed = True
            break
        if not placed:
            return None

    if len(top_rows) < 2:
        return None
    if not any(row < light_row for row in top_rows):
        return None
    if not any(row > light_row for row in top_rows):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_layout()
        if grid is None:
            continue
        out = main(grid)
        if np.array_equal(out[:, -1], np.full(grid.shape[0], BLACK, dtype=int)):
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection input")
