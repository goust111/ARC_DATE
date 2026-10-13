from __future__ import annotations

import random

import numpy as np


BG = 0
LIGHT = 4
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OBJECT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 300
GRID_MIN_BY_COUNT = {2: 10, 3: 13, 4: 16}
GRID_MAX = 20


def _source_delta(light_pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = light_pos
    h, w = shape
    touches = [
        r == 0,
        r == h - 1,
        c == 0,
        c == w - 1,
    ]
    assert sum(touches) == 1, "light source must lie on exactly one edge"
    if r == 0:
        return (-1, 0)
    if r == h - 1:
        return (1, 0)
    if c == 0:
        return (0, -1)
    return (0, 1)


def _components_by_color(grid: np.ndarray, blocked: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in blocked or seen[r, c]:
                continue

            cells: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append((color, cells))
    return components


def _expand_shadow(cells: set[tuple[int, int]], source_delta: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = source_delta
    out = set(cells)
    for r, c in cells:
        if (r + dr, c + dc) not in cells:
            out.add((r + dr, c + dc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each shadow object by regrowing one missing layer on the side facing the border light source.

    Input structure:
    - Black background with exactly one yellow light-source pixel on one outer edge.
    - Several small monochrome 4-connected objects sit in the interior.
    - Each object is a shadowed remnant of a fuller shape, and each also contains a
      same-color single-pixel tail on the side opposite the light.

    Rule:
    - Use the yellow border pixel to determine which side the light comes from.
    - For every non-black, non-yellow 4-connected object, find each pixel that has no
      same-color neighbor in the source-facing direction.
    - Add one same-color pixel one step closer to the light for every such exposed
      pixel, leaving all existing object pixels unchanged.
    - Remove the yellow light-source pixel to black.

    Key insight:
    - The light source fixes a global direction, so every object's missing front layer
      is recovered by extending only its source-facing exposed edge by one cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    light_coords = np.argwhere(g == LIGHT)
    assert light_coords.shape == (1, 2), "expected exactly one yellow light source"
    light_r, light_c = map(int, light_coords[0])
    source_delta = _source_delta((light_r, light_c), g.shape)

    components = _components_by_color(g, blocked={BG, LIGHT})
    assert components, "expected at least one colored object"

    out = g.copy()
    out[light_r, light_c] = BG
    dr, dc = source_delta

    for color, cells in components:
        cell_set = set(cells)
        for r, c in cells:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_set:
                continue
            assert 0 <= nr < h and 0 <= nc < w, "restored pixels must stay inside the grid"
            assert int(out[nr, nc]) in (BG, color), "restored pixels must land on empty cells"
            out[nr, nc] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_connected_shape(area: int) -> set[tuple[int, int]]:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if h * w < area:
            continue

        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) == area:
            return _normalize(cells)
    raise RuntimeError("failed to sample a connected shadow core")


def _build_object(source_delta: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    rear_delta = (-source_delta[0], -source_delta[1])

    for _ in range(120):
        core = _random_connected_shape(random.randint(3, 5))
        anchors = [cell for cell in core if (cell[0] + rear_delta[0], cell[1] + rear_delta[1]) not in core]
        random.shuffle(anchors)
        for ar, ac in anchors:
            tail = (ar + rear_delta[0], ac + rear_delta[1])
            visible_raw = set(core)
            visible_raw.add(tail)
            full_raw = _expand_shadow(visible_raw, source_delta)

            min_r = min(r for r, _ in full_raw)
            min_c = min(c for _, c in full_raw)
            shift = (-min_r, -min_c)
            visible = {(r + shift[0], c + shift[1]) for r, c in visible_raw}
            full = {(r + shift[0], c + shift[1]) for r, c in full_raw}

            if not (4 <= len(visible) <= 6):
                continue
            if len(full) <= len(visible):
                continue
            if max(r for r, _ in full) >= 6 or max(c for _, c in full) >= 6:
                continue
            return visible, full

    raise RuntimeError("failed to build a valid shadow object")


def _place_bbox(reserved: np.ndarray, patch_shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = reserved.shape
    ph, pw = patch_shape
    candidates: list[tuple[int, int]] = []

    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if np.any(reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1]):
                continue
            candidates.append((r0, c0))

    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
    return r0, c0


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int, offset: tuple[int, int]) -> None:
    r0, c0 = offset
    for r, c in cells:
        grid[r0 + r, c0 + c] = color


def _edge_light_position(shape: tuple[int, int], source_delta: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if source_delta == (-1, 0):
        return 0, random.randint(1, w - 2)
    if source_delta == (1, 0):
        return h - 1, random.randint(1, w - 2)
    if source_delta == (0, -1):
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _gen_attempt() -> np.ndarray | None:
    source_delta = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    object_count = random.randint(2, 4)
    min_side = GRID_MIN_BY_COUNT[object_count]
    h = random.randint(min_side, GRID_MAX)
    w = random.randint(min_side, GRID_MAX)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    colors = random.sample(OBJECT_COLORS, object_count)
    for color in colors:
        visible, full = _build_object(source_delta)
        ph = max(r for r, _ in full) + 1
        pw = max(c for _, c in full) + 1
        placement = _place_bbox(reserved, (ph, pw))
        if placement is None:
            return None
        _paint_cells(inp, visible, color, placement)
        _paint_cells(out, full, color, placement)

    light_r, light_c = _edge_light_position((h, w), source_delta)
    if inp[light_r, light_c] != BG:
        return None
    inp[light_r, light_c] = LIGHT

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
