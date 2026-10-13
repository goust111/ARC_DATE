from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


OUTPUT_BY_COUNT = {1: 5, 2: 6, 3: 7, 4: 8}
INPUT_COLORS = (1, 2, 3, 4, 9)
TOP_HEIGHT_RANGE = (5, 7)
LOWER_HEIGHT_RANGE = (8, 12)
WIDTH_RANGE = (18, 24)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_same_color(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _is_square_block(component: dict[str, object]) -> bool:
    coords = component["coords"]
    r0, c0, r1, c1 = component["bbox"]
    return len(coords) == 4 and r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2


def _extract_blocks_and_sprite(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    block_counts: Counter[int] = Counter()
    block_mask = np.zeros(g.shape, dtype=bool)
    lower_top = 0
    for comp in _components_same_color(g):
        if _is_square_block(comp):
            color = int(comp["color"])
            block_counts[color] += 1
            for r, c in comp["coords"]:
                block_mask[r, c] = True
                lower_top = max(lower_top, r + 1)

    assert block_counts, "expected at least one 2x2 block"
    counts = list(block_counts.values())
    assert len(counts) == len(set(counts)), "block counts must be distinct"
    assert all(count in OUTPUT_BY_COUNT for count in counts), "block counts must stay in the 1..4 code range"

    sprite_mask = (g != 0) & ~block_mask
    sprite_components = _components_mask(sprite_mask)
    assert len(sprite_components) == 1, "expected one non-block foreground sprite"
    sprite_coords = sprite_components[0]
    r0, c0, r1, c1 = _bbox(sprite_coords)
    sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    sprite_colors = {int(v) for v in sprite.ravel() if int(v) != 0}
    assert sprite_colors, "sprite must be nonempty"
    assert sprite_colors.issubset(block_counts.keys()), "every sprite color must appear in the block codebook"
    return dict(block_counts), sprite, (r0, c0), lower_top


def _stamp_positions(anchor: int, step: int, tile_size: int, limit: int) -> list[int]:
    positions = [anchor]

    pos = anchor - step
    while pos + tile_size > 0:
        positions.append(pos)
        pos -= step

    pos = anchor + step
    while pos < limit:
        positions.append(pos)
        pos += step

    positions.sort()
    return positions


def _render_output_from_spec(
    shape: tuple[int, int],
    block_counts: dict[int, int],
    sprite: np.ndarray,
    anchor: tuple[int, int],
    lower_top: int,
) -> np.ndarray:
    h, w = shape
    sprite_h, sprite_w = sprite.shape
    recolored = np.zeros_like(sprite)
    for color, count in block_counts.items():
        recolored[sprite == int(color)] = OUTPUT_BY_COUNT[int(count)]

    out = np.zeros((h, w), dtype=int)
    row_step = sprite_h + 1
    col_step = sprite_w + 1
    row_positions = _stamp_positions(anchor[0], row_step, sprite_h, h)
    col_positions = _stamp_positions(anchor[1], col_step, sprite_w, w)

    for top in row_positions:
        for left in col_positions:
            for dr, dc in np.argwhere(recolored != 0):
                rr = top + int(dr)
                cc = left + int(dc)
                if 0 <= rr < h and 0 <= cc < w and rr >= lower_top and out[rr, cc] == 0:
                    out[rr, cc] = int(recolored[dr, dc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the lower sprite from upper 2x2 block counts, then stamp that recolored sprite on an anchored periodic lattice.

    Input structure:
    - Black background with two separated foreground zones.
    - The upper zone contains isolated monochrome 2x2 solid blocks; several blocks may share a color.
    - The lower zone contains one small 4-connected multicolor sprite whose colors all appear among the upper blocks.

    Rule:
    - Count how many 2x2 blocks belong to each color in the upper zone.
    - Convert each sprite color by the fixed lookup on those counts: 1->5, 2->6, 3->7, 4->8.
    - Erase the upper blocks and use the recolored sprite as a stamp whose phase anchor is the sprite bbox's top-left corner.
    - Stamp on every row/column lattice position spaced by sprite_height+1 and sprite_width+1 in both directions, but only paint background cells that stay inside the grid and on or below the row immediately under the lowest block row.

    Key insight:
    - The upper 2x2 blocks are a count-coded color dictionary: their multiplicities determine the recoloring of the lower sprite before the periodic layout can be applied.
    """
    block_counts, sprite, anchor, lower_top = _extract_blocks_and_sprite(np.asarray(grid, dtype=int))
    return _render_output_from_spec(grid.shape, block_counts, sprite, anchor, lower_top)


def _shape_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


SPRITE_SHAPES = (
    _shape_from_rows((".##", ".#.", ".#.")),
    _shape_from_rows(("###", ".#.", ".#.")),
    _shape_from_rows((".##", "##.", ".#.")),
    _shape_from_rows(("###.", ".###", "..#.")),
    _shape_from_rows(("####", ".#..", ".##.")),
    _shape_from_rows(("###.", ".##.", "..##")),
)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_sprite_mask() -> np.ndarray:
    mask = random.choice(SPRITE_SHAPES).copy()
    k = random.randrange(4)
    mask = np.rot90(mask, k)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return _trim_mask(mask)


def _has_monochrome_square(sprite: np.ndarray) -> bool:
    for r in range(sprite.shape[0] - 1):
        for c in range(sprite.shape[1] - 1):
            patch = sprite[r : r + 2, c : c + 2]
            vals = {int(v) for v in patch.ravel() if int(v) != 0}
            if patch.shape == (2, 2) and len(vals) == 1 and int(patch[0, 0]) != 0 and np.all(patch == patch[0, 0]):
                return True
    return False


def _color_sprite(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    for _ in range(200):
        sprite = np.zeros(mask.shape, dtype=int)
        order = coords[:]
        random.shuffle(order)
        for color, cell in zip(colors, order):
            sprite[cell] = color
        for cell in order[len(colors) :]:
            sprite[cell] = random.choice(colors)
        if {int(v) for v in sprite.ravel() if int(v) != 0} != set(colors):
            continue
        if not _has_monochrome_square(sprite):
            return sprite
    raise RuntimeError("failed to color sprite without creating a fake 2x2 block")


def _choose_palette() -> tuple[list[int], dict[int, int]]:
    sprite_color_count = random.choice((2, 3))
    sprite_colors = random.sample(list(INPUT_COLORS), sprite_color_count)
    remaining = [color for color in INPUT_COLORS if color not in sprite_colors]

    extra_count_choices = [0]
    if remaining:
        extra_count_choices.append(1)
    if len(remaining) >= 2 and sprite_color_count == 2:
        extra_count_choices.append(2)

    if len(remaining) >= 1 and random.random() < 0.6:
        extra_count = min(4 - sprite_color_count, len(remaining))
    else:
        extra_count = random.choice(extra_count_choices)

    all_colors = sprite_colors + random.sample(remaining, extra_count)
    if len(all_colors) == 4:
        count_values = [1, 2, 3, 4]
    else:
        count_values = random.sample([1, 2, 3, 4], len(all_colors))
    random.shuffle(count_values)
    block_counts = {color: count for color, count in zip(all_colors, count_values)}
    return sprite_colors, block_counts


def _block_slots(width: int, top_height: int) -> list[tuple[int, int]]:
    row_slots = [0, top_height - 2]
    col_slots = list(range(0, width - 1, 3))
    return [(r, c) for r in row_slots for c in col_slots]


def _place_blocks(grid: np.ndarray, block_counts: dict[int, int], top_height: int) -> tuple[list[tuple[int, int, int]], int]:
    slots = _block_slots(grid.shape[1], top_height)
    total_blocks = sum(block_counts.values())
    assert len(slots) >= total_blocks, "not enough block slots"

    bottom_slots = [slot for slot in slots if slot[0] == top_height - 2]
    top_slots = [slot for slot in slots if slot[0] == 0]
    chosen: list[tuple[int, int]] = []
    chosen.append(random.choice(bottom_slots))
    if total_blocks > 1:
        chosen.append(random.choice(top_slots))

    remaining = [slot for slot in slots if slot not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: total_blocks - len(chosen)])
    random.shuffle(chosen)

    placed: list[tuple[int, int, int]] = []
    idx = 0
    for color, count in block_counts.items():
        for _ in range(count):
            r, c = chosen[idx]
            idx += 1
            grid[r : r + 2, c : c + 2] = int(color)
            placed.append((r, c, int(color)))
    return placed, top_height


def _pick_anchor(lower_top: int, grid_h: int, grid_w: int, sprite_h: int, sprite_w: int) -> tuple[int, int]:
    row_step = sprite_h + 1
    col_step = sprite_w + 1

    row_modes: list[str] = ["edge"]
    if lower_top + sprite_h <= grid_h - sprite_h:
        row_modes.append("partial_up")
    if lower_top + row_step <= grid_h - sprite_h:
        row_modes.append("full_up")
    row_mode = random.choice(row_modes)

    if row_mode == "full_up":
        row_lo = lower_top + row_step
        row_hi = grid_h - sprite_h
    elif row_mode == "partial_up":
        row_lo = lower_top + 1
        row_hi = min(lower_top + sprite_h, grid_h - sprite_h)
    else:
        row_lo = row_hi = lower_top
    sprite_r = random.randint(row_lo, row_hi)

    max_leftish = max(0, grid_w // 2 - sprite_w)
    col_modes: list[str] = ["edge"]
    if 1 <= max_leftish:
        col_modes.append("partial_left")
    if col_step <= max_leftish:
        col_modes.append("full_left")
    col_mode = random.choice(col_modes)

    if col_mode == "full_left":
        col_lo = col_step
        col_hi = max_leftish
    elif col_mode == "partial_left":
        col_lo = 1
        col_hi = min(col_step - 1, max_leftish)
    else:
        col_lo = col_hi = 0
    sprite_c = random.randint(col_lo, col_hi)
    return sprite_r, sprite_c


def _validate_structure(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        block_counts, sprite, anchor, lower_top = _extract_blocks_and_sprite(grid)
    except AssertionError:
        return False

    if not np.array_equal(expected, _render_output_from_spec(grid.shape, block_counts, sprite, anchor, lower_top)):
        return False

    sprite_h, sprite_w = sprite.shape
    row_positions = _stamp_positions(anchor[0], sprite_h + 1, sprite_h, grid.shape[0])
    col_positions = _stamp_positions(anchor[1], sprite_w + 1, sprite_w, grid.shape[1])

    visible_anchors = 0
    for top in row_positions:
        for left in col_positions:
            if top < grid.shape[0] and top + sprite_h > lower_top and left < grid.shape[1] and left + sprite_w > 0:
                visible_anchors += 1
    if visible_anchors < 3:
        return False

    output_colors = {int(v) for v in expected.ravel() if int(v) != 0}
    valid_outputs = set(OUTPUT_BY_COUNT.values())
    if not output_colors or not output_colors.issubset(valid_outputs):
        return False

    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    top_height = random.randint(*TOP_HEIGHT_RANGE)
    lower_height = random.randint(*LOWER_HEIGHT_RANGE)
    height = top_height + lower_height
    width = random.randint(*WIDTH_RANGE)

    sprite_colors, block_counts = _choose_palette()
    sprite = _color_sprite(_random_sprite_mask(), sprite_colors)

    grid = np.zeros((height, width), dtype=int)
    _place_blocks(grid, block_counts, top_height)

    anchor = _pick_anchor(top_height, height, width, sprite.shape[0], sprite.shape[1])
    ar, ac = anchor
    sprite_mask = sprite != 0
    grid[ar : ar + sprite.shape[0], ac : ac + sprite.shape[1]][sprite_mask] = sprite[sprite_mask]

    expected = _render_output_from_spec(grid.shape, block_counts, sprite, anchor, top_height)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected = _generate_case()
        if not _validate_structure(grid, expected):
            continue
        try:
            produced = main(grid)
        except AssertionError:
            continue
        if np.array_equal(produced, expected):
            return grid
    raise RuntimeError("failed to generate a valid task instance")
