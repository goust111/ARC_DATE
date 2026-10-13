from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (BLUE, GREEN, YELLOW)
SIDES = ("top", "bottom", "left", "right")

PAIR_COUNT_RANGE = (2, 3)
HEIGHT_RANGE_BY_PAIRS = {2: (18, 24), 3: (22, 28)}
WIDTH_RANGE = (19, 26)
INNER_MARGIN = 4
ZONE_MIN_HEIGHT = 4
ZONE_GAP = 1
MIN_ROW_DROP = 2
MIN_HORIZONTAL_SPAN = 5
MAX_SPRITE_EXTENT = 2
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_nonblack(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _extract_objects(grid: np.ndarray) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    markers: dict[int, list[tuple[int, int]]] = {}
    sprites: dict[int, list[tuple[int, int]]] = {}
    for comp in _components_nonblack(grid):
        colors = {int(grid[r, c]) for r, c in comp}
        if RED in colors:
            non_red = [(r, c) for r, c in comp if int(grid[r, c]) != RED]
            assert len(non_red) == 1, "each sprite must have exactly one non-red anchor pixel"
            anchor_r, anchor_c = non_red[0]
            anchor_color = int(grid[anchor_r, anchor_c])
            assert anchor_color in MARKER_COLORS, "sprite anchor must use a marker color"
            assert colors <= {RED, anchor_color}, "sprite may contain only red plus its anchor color"
            red_offsets = sorted(
                (r - anchor_r, c - anchor_c) for r, c in comp if int(grid[r, c]) == RED
            )
            assert 3 <= len(comp) <= 5 and 2 <= len(red_offsets) <= 4, "sprite must be a 3-5 pixel component"
            assert anchor_color not in sprites, "each marker color must map to exactly one sprite"
            sprites[anchor_color] = red_offsets
        else:
            assert len(comp) == 1, "non-sprite objects must be isolated single-pixel markers"
            r, c = comp[0]
            color = int(grid[r, c])
            assert color in MARKER_COLORS, "marker colors must be blue, green, or yellow"
            markers.setdefault(color, []).append((r, c))
    return markers, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair with a grey L path, then stamp its matching external sprite at the path elbow.

    Input structure:
    - The background is black.
    - There are 2-3 marker colors chosen from blue, green, and yellow; each chosen color appears as exactly two isolated single-pixel markers.
    - There is one external 4-connected sprite for each chosen marker color; each sprite contains red pixels plus exactly one anchor pixel of that marker color.

    Rule:
    - Pair the markers by color.
    - For each pair, order the two marker pixels lexicographically by `(row, col)` and call them `(r1, c1)` and `(r2, c2)`.
    - Draw a grey L-shaped Manhattan path from `(r1, c1)` horizontally first to column `c2`, then vertically downward to row `r2`, painting only background cells.
    - Take the external sprite whose anchor color matches that pair, and copy only its red pixels to the elbow `(r1, c2)` using anchor-relative offsets; do not copy the anchor pixel and do not overwrite any marker.

    Key insight:
    - The marker color chooses which sprite to use, while the two marker positions jointly determine the elbow where that sprite must be stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    markers, sprites = _extract_objects(g)
    assert 2 <= len(markers) <= 3, "expected 2-3 marker pairs"
    assert set(markers) == set(sprites), "marker colors and sprite colors must match exactly"

    out = g.copy()
    h, w = g.shape
    marker_cells = {coord for coords in markers.values() for coord in coords}

    for color in sorted(markers):
        pair = sorted(markers[color])
        assert len(pair) == 2, "each chosen color must appear exactly twice as markers"
        (r1, c1), (r2, c2) = pair
        assert r1 < r2 and c1 != c2, "paired markers must be in different rows and columns"

        c_step = 1 if c2 > c1 else -1
        for c in range(c1 + c_step, c2 + c_step, c_step):
            assert int(out[r1, c]) == BG, "horizontal path cells must be background in the input"
            out[r1, c] = GREY
        for r in range(r1 + 1, r2):
            assert int(out[r, c2]) == BG, "vertical path cells must be background in the input"
            out[r, c2] = GREY

        elbow = (r1, c2)
        for dr, dc in sprites[color]:
            rr = elbow[0] + dr
            cc = elbow[1] + dc
            assert 0 <= rr < h and 0 <= cc < w, "stamped sprite must stay in bounds"
            assert (rr, cc) not in marker_cells, "stamping must not overwrite any marker"
            assert int(out[rr, cc]) in {BG, GREY}, "stamping may only cover background or grey path cells"
            out[rr, cc] = RED

    return out


def _random_partition(total: int, parts: int, min_size: int) -> list[int] | None:
    if total < parts * min_size:
        return None
    sizes = [min_size] * parts
    extra = total - parts * min_size
    for _ in range(extra):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _random_sprite_offsets() -> tuple[tuple[int, int], ...]:
    target_cells = random.randint(3, 5)
    cells = {(0, 0)}
    while len(cells) < target_cells:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if max(abs(nr), abs(nc)) <= MAX_SPRITE_EXTENT and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    offsets = tuple(sorted(cell for cell in cells if cell != (0, 0)))
    assert offsets
    return offsets


def _sprite_cells(anchor: tuple[int, int], offsets: tuple[tuple[int, int], ...], color: int) -> list[tuple[int, int, int]]:
    cells = [(anchor[0], anchor[1], color)]
    cells.extend((anchor[0] + dr, anchor[1] + dc, RED) for dr, dc in offsets)
    return cells


def _cells_clear(occupied: np.ndarray, cells: list[tuple[int, int, int]]) -> bool:
    h, w = occupied.shape
    for r, c, _ in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_sprite(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    offsets: tuple[tuple[int, int], ...],
    side: str,
) -> bool:
    h, w = grid.shape
    all_offsets = ((0, 0),) + offsets
    min_dr = min(dr for dr, _ in all_offsets)
    max_dr = max(dr for dr, _ in all_offsets)
    min_dc = min(dc for _, dc in all_offsets)
    max_dc = max(dc for _, dc in all_offsets)

    candidates: list[tuple[int, int]] = []
    if side == "top":
        r = -min_dr
        for c in range(-min_dc, w - max_dc):
            candidates.append((r, c))
    elif side == "bottom":
        r = h - 1 - max_dr
        for c in range(-min_dc, w - max_dc):
            candidates.append((r, c))
    elif side == "left":
        c = -min_dc
        for r in range(-min_dr, h - max_dr):
            candidates.append((r, c))
    else:
        assert side == "right"
        c = w - 1 - max_dc
        for r in range(-min_dr, h - max_dr):
            candidates.append((r, c))
    random.shuffle(candidates)

    for anchor in candidates:
        cells = _sprite_cells(anchor, offsets, color)
        if not _cells_clear(occupied, cells):
            continue
        for r, c, cell_color in cells:
            grid[r, c] = cell_color
            occupied[r, c] = True
        return True
    return False


def _choose_pair_rows(h: int, pair_count: int) -> list[tuple[int, int]]:
    inner_rows = h - 2 * INNER_MARGIN
    zone_sizes = _random_partition(inner_rows - ZONE_GAP * (pair_count - 1), pair_count, ZONE_MIN_HEIGHT)
    assert zone_sizes is not None

    zones: list[tuple[int, int]] = []
    cur = INNER_MARGIN
    for size in zone_sizes:
        zones.append((cur, cur + size - 1))
        cur += size + ZONE_GAP

    rows: list[tuple[int, int]] = []
    for z0, z1 in zones:
        r1_choices = list(range(z0, z1 - MIN_ROW_DROP + 1))
        random.shuffle(r1_choices)
        picked = None
        for r1 in r1_choices:
            r2_lo = r1 + MIN_ROW_DROP
            if r2_lo <= z1:
                picked = (r1, random.randint(r2_lo, z1))
                break
        assert picked is not None
        rows.append(picked)
    return rows


def _choose_pair_columns(w: int, directions: list[int]) -> list[tuple[int, int]]:
    low = INNER_MARGIN + MAX_SPRITE_EXTENT
    high = w - 1 - INNER_MARGIN - MAX_SPRITE_EXTENT
    assert high - low + 1 >= MIN_HORIZONTAL_SPAN + 2

    cols: list[tuple[int, int]] = []
    for direction in directions:
        if direction > 0:
            c2 = random.randint(low + MIN_HORIZONTAL_SPAN, high)
            c1 = random.randint(low, c2 - MIN_HORIZONTAL_SPAN)
        else:
            c2 = random.randint(low, high - MIN_HORIZONTAL_SPAN)
            c1 = random.randint(c2 + MIN_HORIZONTAL_SPAN, high)
        cols.append((c1, c2))
    return cols


def _relative_path_cells(c1: int, c2: int, r1: int, r2: int) -> set[tuple[int, int]]:
    cells = set()
    if c2 > c1:
        for c in range(c1 + 1, c2 + 1):
            cells.add((0, c - c2))
    else:
        for c in range(c2, c1):
            cells.add((0, c - c2))
    for r in range(r1 + 1, r2):
        cells.add((r - r1, 0))
    return cells


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        markers, sprites = _extract_objects(grid)
    except AssertionError:
        return False
    if not (2 <= len(markers) <= 3) or set(markers) != set(sprites):
        return False

    path_cells: set[tuple[int, int]] = set()
    stamp_cells: set[tuple[int, int]] = set()
    horizontal_signs: set[int] = set()
    has_path_overlap = False

    for color, pair in markers.items():
        if len(pair) != 2:
            return False
        (r1, c1), (r2, c2) = sorted(pair)
        if not (r1 < r2 and c1 != c2):
            return False
        if abs(r2 - r1) + abs(c2 - c1) <= 6:
            return False

        horizontal_signs.add(1 if c2 > c1 else -1)
        c_step = 1 if c2 > c1 else -1
        for c in range(c1 + c_step, c2 + c_step, c_step):
            cell = (r1, c)
            if cell in path_cells or (int(grid[cell]) != BG and cell != (r1, c2)):
                return False
            path_cells.add(cell)
        for r in range(r1 + 1, r2):
            cell = (r, c2)
            if cell in path_cells or int(grid[cell]) != BG:
                return False
            path_cells.add(cell)

        rel_path = _relative_path_cells(c1, c2, r1, r2)
        offsets = sprites[color]
        if not any(offset in rel_path for offset in offsets):
            continue
        has_path_overlap = True

    if len(markers) >= 2 and len(horizontal_signs) < 2:
        return False
    if not has_path_overlap:
        return False

    marker_cells = {coord for pair in markers.values() for coord in pair}
    for color, pair in markers.items():
        (r1, c1), (r2, c2) = sorted(pair)
        elbow = (r1, c2)
        offsets = sprites[color]
        rel_path = _relative_path_cells(c1, c2, r1, r2)
        if not any(offset not in rel_path for offset in offsets):
            return False
        for dr, dc in offsets:
            cell = (elbow[0] + dr, elbow[1] + dc)
            if cell in marker_cells or cell in stamp_cells:
                return False
            stamp_cells.add(cell)
    try:
        out = main(grid)
    except AssertionError:
        return False
    return (
        out.shape == grid.shape
        and np.count_nonzero(out == GREY) > 0
        and np.count_nonzero(out == RED) > np.count_nonzero(grid == RED)
    )


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    h = random.randint(*HEIGHT_RANGE_BY_PAIRS[pair_count])
    w = random.randint(*WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = random.sample(MARKER_COLORS, pair_count)
    directions = [1, -1] if pair_count == 2 else [1, -1, random.choice((1, -1))]
    random.shuffle(directions)
    row_pairs = _choose_pair_rows(h, pair_count)
    col_pairs = _choose_pair_columns(w, directions)

    sprite_offsets_by_color: dict[int, tuple[tuple[int, int], ...]] = {}
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for color in colors:
        for _ in range(50):
            offsets = _random_sprite_offsets()
            if offsets not in used_shapes:
                used_shapes.add(offsets)
                sprite_offsets_by_color[color] = offsets
                break
        else:
            return None

    for color, (r1, r2), (c1, c2) in zip(colors, row_pairs, col_pairs):
        grid[r1, c1] = color
        grid[r2, c2] = color
        occupied[r1, c1] = True
        occupied[r2, c2] = True

        rel_path = _relative_path_cells(c1, c2, r1, r2)
        offsets = sprite_offsets_by_color[color]
        if any(offset == (r2 - r1, 0) or offset == (0, c1 - c2) for offset in offsets):
            return None
        if not any(offset in rel_path for offset in offsets):
            continue

    if pair_count == 2 and all(
        not any(offset in _relative_path_cells(c1, c2, r1, r2) for offset in sprite_offsets_by_color[color])
        for color, (r1, r2), (c1, c2) in zip(colors, row_pairs, col_pairs)
    ):
        return None

    if pair_count == 3 and not any(
        any(offset in _relative_path_cells(c1, c2, r1, r2) for offset in sprite_offsets_by_color[color])
        for color, (r1, r2), (c1, c2) in zip(colors, row_pairs, col_pairs)
    ):
        return None

    sides = random.sample(SIDES, pair_count)
    order = list(colors)
    random.shuffle(order)
    for color, side in zip(order, sides):
        if not _place_sprite(grid, occupied, color, sprite_offsets_by_color[color], side):
            return None

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
