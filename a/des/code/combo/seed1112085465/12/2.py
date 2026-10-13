from __future__ import annotations

import random

import numpy as np


ORANGE = 7
GREY = 5
RED = 2
GREEN = 3

OBJECT_COLORS = [0, 1, 2, 3, 4, 6, 8, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MAX_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (2, 5)
BOX_HEIGHT_RANGE = (2, 4)
BOX_WIDTH_RANGE = (2, 6)
GAP_RANGE = (1, 2)
LEFT_WIDTH_RANGE = (4, 8)
RIGHT_WIDTH_RANGE = (8, 16)
ACCENT_PROBABILITY = 0.3


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _sort_key(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return r0, c0


def _is_filled_2x2(coords: np.ndarray) -> bool:
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0, c1 - c0) != (1, 1):
        return False
    cells = {(int(r), int(c)) for r, c in coords}
    return cells == {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def main(grid: np.ndarray) -> np.ndarray:
    """Filter right-side objects according to left-side red/green 2x2 switches."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode_color(g)
    separator_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(separator_cols) == 1, "expected exactly one full-height grey separator"
    sep = separator_cols[0]
    assert 0 < sep < g.shape[1] - 1, "separator must split the grid into two non-empty parts"

    control = g[:, :sep]
    main_region = g[:, sep + 1 :]

    switch_comps = _components(control != bg)
    assert switch_comps, "expected at least one control block"

    switches: list[tuple[int, np.ndarray]] = []
    for coords in switch_comps:
        values = {int(control[r, c]) for r, c in coords}
        assert len(values) == 1, "each control block must be monochrome"
        color = values.pop()
        assert color in {RED, GREEN}, "control blocks must be red or green"
        assert _is_filled_2x2(coords), "control blocks must be filled 2x2 squares"
        switches.append((color, coords))
    switches.sort(key=lambda item: _sort_key(item[1]))

    objects = _components(main_region != bg)
    assert len(objects) == len(switches), "switch count must match object count"
    objects.sort(key=_sort_key)

    out = main_region.copy()
    for (switch_color, _coords), obj in zip(switches, objects):
        if switch_color == RED:
            out[obj[:, 0], obj[:, 1]] = bg
        else:
            assert switch_color == GREEN

    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    assert 1 <= area <= height * width

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < area:
        choices = list(frontier)
        assert choices, "frontier unexpectedly empty"
        if random.random() < 0.7:
            choices.sort(
                key=lambda rc: sum((rc[0] + dr, rc[1] + dc) in cells for dr, dc in NEIGHBORS4),
                reverse=True,
            )
            cell = random.choice(choices[: max(1, min(4, len(choices)))])
        else:
            cell = random.choice(choices)
        frontier.discard(cell)
        cells.add(cell)
        add_frontier(*cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return mask


def _paint_patch(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, ORANGE, dtype=int)
    base = random.choice(OBJECT_COLORS)
    patch[mask] = base

    if random.random() < ACCENT_PROBABILITY and int(mask.sum()) >= 4:
        accent_choices = [c for c in OBJECT_COLORS if c != base]
        accent = random.choice(accent_choices)
        coords = np.argwhere(mask)
        count = random.randint(1, max(1, len(coords) // 3))
        for idx in random.sample(range(len(coords)), count):
            r, c = coords[idx]
            patch[int(r), int(c)] = accent
    return patch


def _switch_states(count: int) -> list[bool]:
    mode = random.choice(["mixed", "all_on", "all_off"])
    if mode == "all_on":
        return [True] * count
    if mode == "all_off":
        return [False] * count

    on_count = random.randint(1, count - 1)
    states = [False] * count
    for idx in random.sample(range(count), on_count):
        states[idx] = True
    return states


def _attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    box_heights = [random.randint(*BOX_HEIGHT_RANGE) for _ in range(count)]
    box_widths = [random.randint(*BOX_WIDTH_RANGE) for _ in range(count)]
    gaps = [random.randint(*GAP_RANGE) for _ in range(count + 1)]

    height = sum(box_heights) + sum(gaps)
    left_width = random.randint(*LEFT_WIDTH_RANGE)
    min_right = max(max(box_widths) + 2, RIGHT_WIDTH_RANGE[0])
    max_right = max(min_right, RIGHT_WIDTH_RANGE[1])
    right_width = random.randint(min_right, max_right)

    control = np.full((height, left_width), ORANGE, dtype=int)
    main_region = np.full((height, right_width), ORANGE, dtype=int)
    states = _switch_states(count)

    row = gaps[0]
    object_coords: list[np.ndarray] = []
    object_tops: list[int] = []

    for i in range(count):
        box_h = box_heights[i]
        box_w = box_widths[i]
        area = random.randint(3, box_h * box_w)
        mask = _random_connected_mask(box_h, box_w, area)
        patch = _paint_patch(mask)

        col_low = 1 if right_width - box_w >= 2 else 0
        col_high = right_width - box_w - 1 if right_width - box_w >= 2 else right_width - box_w
        col = random.randint(col_low, col_high)

        view = main_region[row : row + box_h, col : col + box_w]
        assert np.all(view == ORANGE)
        write_mask = patch != ORANGE
        view[write_mask] = patch[write_mask]

        coords = np.argwhere(write_mask) + np.array([row, col], dtype=int)
        object_coords.append(coords)
        object_tops.append(row)

        switch_row = row + random.randint(0, box_h - 2)
        switch_col = random.randint(0, left_width - 2)
        control[switch_row : switch_row + 2, switch_col : switch_col + 2] = GREEN if states[i] else RED

        row += box_h + gaps[i + 1]

    inp = np.full((height, left_width + 1 + right_width), ORANGE, dtype=int)
    inp[:, :left_width] = control
    inp[:, left_width] = GREY
    inp[:, left_width + 1 :] = main_region

    expected = main_region.copy()
    for keep, coords in zip(states, object_coords):
        if not keep:
            expected[coords[:, 0], coords[:, 1]] = ORANGE

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_ATTEMPTS} attempts")
