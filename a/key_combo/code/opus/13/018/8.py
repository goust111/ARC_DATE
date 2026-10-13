import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
MAGENTA = 6

MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _is_connected(coords: set[tuple[int, int]], h: int, w: int) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _bbox_spans(coords: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _grow_connected_subset(
    allowed: set[tuple[int, int]],
    size: int,
    h: int,
    w: int,
    rng: random.Random,
    *,
    required: tuple[int, int] | None = None,
) -> set[tuple[int, int]] | None:
    if size < 1 or len(allowed) < size:
        return None
    if required is not None and required not in allowed:
        return None

    start = required if required is not None else rng.choice(tuple(allowed))
    chosen = {start}
    frontier = {p for p in _neighbors4(start[0], start[1], h, w) if p in allowed}

    while len(chosen) < size and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in chosen:
            continue
        chosen.add(cell)
        for nb in _neighbors4(cell[0], cell[1], h, w):
            if nb in allowed and nb not in chosen:
                frontier.add(nb)

    return chosen if len(chosen) == size else None


def _find_divider_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    divider_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(divider_cols) == 1, "expected exactly one full-height grey divider column"
    divider_col = divider_cols[0]
    assert 0 < divider_col < w - 1, "divider must split left and right regions"
    return divider_col


def _parse_layout(grid: np.ndarray) -> tuple[int, int, int, list[int], int]:
    h, w = grid.shape
    divider_col = _find_divider_col(grid)
    left = grid[:, :divider_col]
    right = grid[:, divider_col + 1 :]

    assert left.shape[1] == right.shape[1], "left and right blocks must share the same width"
    panel_w = left.shape[1]
    assert panel_w % 2 == 1, "slot width must be odd to define a center marker"

    left_sep_rows = [r for r in range(h) if np.all(left[r, :] == GREY)]
    assert len(left_sep_rows) == 1, "expected exactly one grey separator row inside the left block"
    left_sep = left_sep_rows[0]

    right_sep_rows = [r for r in range(h) if np.all(right[r, :] == GREY)]
    assert 1 <= len(right_sep_rows) <= 3, "right block must contain 2..4 slots"

    slot_count = len(right_sep_rows) + 1
    if len(right_sep_rows) >= 2:
        step = right_sep_rows[1] - right_sep_rows[0]
        assert step >= 2 and all(
            right_sep_rows[i + 1] - right_sep_rows[i] == step for i in range(len(right_sep_rows) - 1)
        ), "right separators must be evenly spaced"
        panel_h = step - 1
    else:
        panel_h = right_sep_rows[0]

    assert panel_h % 2 == 1, "slot height must be odd to define a center marker"
    expected_h = slot_count * panel_h + (slot_count - 1)
    assert expected_h == h, "right slots must tile the full grid height"

    slot_rows = [i * (panel_h + 1) for i in range(slot_count)]
    expected_right_seps = [r0 + panel_h for r0 in slot_rows[:-1]]
    assert right_sep_rows == expected_right_seps, "right separator rows do not match the slot geometry"

    top_r0 = left_sep - panel_h
    bottom_r0 = left_sep + 1
    assert top_r0 >= 0 and bottom_r0 + panel_h <= h, "left separator must sit between two full panels"

    return divider_col, panel_h, panel_w, slot_rows, left_sep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Derive a magenta sprite from the two left panels' common background and stamp it into marked right slots.

    Input structure:
    - A full-height grey divider column splits the grid into a left comparison block and a right slot block.
    - The left block contains two equal-size black-background panels separated by one grey row: the upper panel has a blue connected shape, the lower panel has a red connected shape.
    - The right block is partitioned into 2..4 equal-height slots by grey horizontal separators; marked slots contain exactly one centered magenta pixel and unmarked slots are otherwise black.

    Rule:
    - Compare the two left panels cell by cell in shared local coordinates.
    - Build a sprite mask of the same size: paint magenta where both panels are black, otherwise keep black.
    - Copy that derived sprite into every marked right slot, replacing the single marker pixel; unmarked slots and all grey divider structure stay unchanged.

    Key insight:
    The stamped sprite is not present directly in the input: it must be inferred as the common empty-space mask of the two stacked panels before being distributed to the marked slots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    divider_col, panel_h, panel_w, slot_rows, left_sep = _parse_layout(g)
    top = g[left_sep - panel_h : left_sep, :panel_w]
    bottom = g[left_sep + 1 : left_sep + 1 + panel_h, :panel_w]

    assert set(np.unique(top)).issubset({BG, BLUE}), "top panel should use only black and blue"
    assert set(np.unique(bottom)).issubset({BG, RED}), "bottom panel should use only black and red"
    assert np.any(top == BLUE) and np.any(bottom == RED), "both panels must contain foreground"
    assert not np.any((top != BG) & (bottom != BG)), "panel foregrounds must not overlap in local coordinates"

    sprite = np.full((panel_h, panel_w), BG, dtype=int)
    sprite[(top == BG) & (bottom == BG)] = MAGENTA

    out = g.copy()
    right_c0 = divider_col + 1
    center = (panel_h // 2, panel_w // 2)

    for r0 in slot_rows:
        slot = g[r0 : r0 + panel_h, right_c0 : right_c0 + panel_w]
        non_bg = np.argwhere(slot != BG)
        if slot[center] == MAGENTA:
            assert non_bg.shape == (1, 2), "marked slots must contain only a centered marker"
            assert tuple(map(int, non_bg[0])) == center, "marker must be centered in its slot"
            out[r0 : r0 + panel_h, right_c0 : right_c0 + panel_w] = sprite
        else:
            assert non_bg.size == 0, "unmarked slots must be pure black"

    return out


def _choose_geometry(rng: random.Random) -> tuple[int, int, int, int]:
    slot_count = rng.choice((2, 3, 4))
    if slot_count == 2:
        panel_h = 5
    elif slot_count == 3:
        panel_h = rng.choice((3, 5))
    else:
        panel_h = 3

    total_h = slot_count * panel_h + (slot_count - 1)
    min_panel_w = total_h // 2 + 1
    first_odd_w = max(5, min_panel_w)
    if first_odd_w % 2 == 0:
        first_odd_w += 1
    width_choices = [w for w in range(first_odd_w, 15, 2) if 2 * w + 1 <= 30]
    assert width_choices
    panel_w = rng.choice(width_choices)
    return slot_count, panel_h, panel_w, total_h


def _generate_panel_masks(
    panel_h: int,
    panel_w: int,
    rng: random.Random,
    sprite_has_center: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    area = panel_h * panel_w
    center = (panel_h // 2, panel_w // 2)
    valid_counts = [
        (top_n, bottom_n)
        for top_n in range(8, 16)
        for bottom_n in range(8, 16)
        if 4 <= area - top_n - bottom_n < area
    ]
    if not valid_counts:
        return None

    all_cells = {(r, c) for r in range(panel_h) for c in range(panel_w)}
    for _ in range(240):
        top_n, bottom_n = rng.choice(valid_counts)
        sprite_n = area - top_n - bottom_n

        required_sprite = center if sprite_has_center else None
        sprite = _grow_connected_subset(all_cells, sprite_n, panel_h, panel_w, rng, required=required_sprite)
        if sprite is None:
            continue
        if (center in sprite) != sprite_has_center:
            continue
        if not _is_connected(all_cells - sprite, panel_h, panel_w):
            continue
        if min(_bbox_spans(sprite)) < 2:
            continue

        remaining = all_cells - sprite
        center_owner = None
        if not sprite_has_center:
            center_owner = rng.choice(("top", "bottom"))

        top_required = center if center_owner == "top" else None
        top = _grow_connected_subset(remaining, top_n, panel_h, panel_w, rng, required=top_required)
        if top is None:
            continue
        if center_owner == "top" and center not in top:
            continue

        bottom = remaining - top
        if len(bottom) != bottom_n or not _is_connected(bottom, panel_h, panel_w):
            continue
        if center_owner == "bottom" and center not in bottom:
            continue
        if min(_bbox_spans(top)) < 2 or min(_bbox_spans(bottom)) < 2:
            continue

        return sprite, top, bottom

    return None


def _paint_panel(coords: set[tuple[int, int]], shape: tuple[int, int], color: int) -> np.ndarray:
    panel = np.full(shape, BG, dtype=int)
    for r, c in coords:
        panel[r, c] = color
    return panel


def _expected_output(
    grid: np.ndarray,
    sprite: np.ndarray,
    divider_col: int,
    panel_h: int,
    panel_w: int,
    slot_rows: list[int],
) -> np.ndarray:
    out = grid.copy()
    right_c0 = divider_col + 1
    center = (panel_h // 2, panel_w // 2)
    for r0 in slot_rows:
        slot = grid[r0 : r0 + panel_h, right_c0 : right_c0 + panel_w]
        if slot[center] == MAGENTA:
            out[r0 : r0 + panel_h, right_c0 : right_c0 + panel_w] = sprite
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    slot_count, panel_h, panel_w, total_h = _choose_geometry(rng)
    panel_data = _generate_panel_masks(panel_h, panel_w, rng, sprite_has_center=bool(rng.getrandbits(1)))
    if panel_data is None:
        return None
    sprite_coords, top_coords, bottom_coords = panel_data

    divider_col = panel_w
    total_w = panel_w * 2 + 1
    grid = np.full((total_h, total_w), BG, dtype=int)
    grid[:, divider_col] = GREY

    slot_rows = [i * (panel_h + 1) for i in range(slot_count)]
    for r in slot_rows[:-1]:
        grid[r + panel_h, divider_col + 1 :] = GREY

    left_extra = total_h - (2 * panel_h + 1)
    top_margin = rng.randint(0, left_extra)
    left_sep = top_margin + panel_h
    bottom_r0 = left_sep + 1

    grid[left_sep, :panel_w] = GREY
    top_panel = _paint_panel(top_coords, (panel_h, panel_w), BLUE)
    bottom_panel = _paint_panel(bottom_coords, (panel_h, panel_w), RED)
    grid[top_margin : top_margin + panel_h, :panel_w] = top_panel
    grid[bottom_r0 : bottom_r0 + panel_h, :panel_w] = bottom_panel

    marked_count = rng.randint(1, slot_count - 1)
    marked_slots = set(rng.sample(range(slot_count), marked_count))
    center = (panel_h // 2, panel_w // 2)
    for i, r0 in enumerate(slot_rows):
        if i in marked_slots:
            grid[r0 + center[0], divider_col + 1 + center[1]] = MAGENTA

    sprite = np.full((panel_h, panel_w), BG, dtype=int)
    for r, c in sprite_coords:
        sprite[r, c] = MAGENTA

    if not _is_connected(sprite_coords, panel_h, panel_w):
        return None

    expected = _expected_output(grid, sprite, divider_col, panel_h, panel_w, slot_rows)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
