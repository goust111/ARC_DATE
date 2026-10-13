from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
INPUT_COLORS = (1, 2, 3, 4, 6, 8, 9)

ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_H_RANGE = (4, 7)
ROOM_W_RANGE = (4, 7)
MAX_TEMPLATE_COUNT = 3
MAX_GENERATION_ATTEMPTS = 300


def _room_spans(length: int, wall_lines: list[int]) -> list[tuple[int, int]]:
    boundaries = [-1] + sorted(set(int(x) for x in wall_lines)) + [int(length)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left > 1:
            spans.append((left + 1, right))
    return spans


def _room_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    wall_rows = [r for r in range(h) if np.all(grid[r] == GREY)]
    wall_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert wall_rows and wall_cols, "expected full grey separator rows and columns"

    row_spans = _room_spans(h, wall_rows)
    col_spans = _room_spans(w, wall_cols)
    assert row_spans and col_spans, "expected at least one room"

    room_heights = {r1 - r0 for r0, r1 in row_spans}
    room_widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(room_heights) == 1 and len(room_widths) == 1, "rooms must share one size"

    return row_spans, col_spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each template pattern into same-colored seed rooms and recolor the copied pixels orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_spans, col_spans = _room_layout(g)
    out = g.copy()

    templates: dict[int, np.ndarray] = {}
    seeds: list[tuple[int, int, int, int, int, tuple[int, int]]] = []

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            room = g[r0:r1, c0:c1]
            assert not np.any(room == GREY), "room interiors must not contain wall pixels"

            colors = [int(color) for color in np.unique(room) if int(color) != BLACK]
            if not colors:
                continue

            assert len(colors) == 1, "each non-empty room must use a single non-black color"
            color = colors[0]
            coords = np.argwhere(room == color)

            if len(coords) == 1:
                sr, sc = map(int, coords[0])
                seeds.append((r0, r1, c0, c1, color, (sr, sc)))
            else:
                assert color not in templates, "each template color must appear in at most one template room"
                templates[color] = room == color

    for r0, r1, c0, c1, color, seed_pos in seeds:
        assert color in templates, "every seed color must match a template room"
        mask = templates[color]
        room_out = np.full(mask.shape, BLACK, dtype=int)
        room_out[mask] = ORANGE
        room_out[seed_pos] = ORANGE
        out[r0:r1, c0:c1] = room_out

    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _make_pattern(room_h: int, room_w: int) -> np.ndarray:
    max_area = min(room_h * room_w - 1, max(5, (room_h * room_w * 2) // 5))
    target_area = random.randint(2, max_area)

    mask = np.zeros((room_h, room_w), dtype=bool)
    start = (random.randrange(room_h), random.randrange(room_w))
    mask[start] = True
    cells = [start]

    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c, room_h, room_w):
                if not mask[rr, cc]:
                    frontier.append((rr, cc))
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
        cells.append((rr, cc))

    return mask


def _room_bounds(ri: int, ci: int, room_h: int, room_w: int) -> tuple[int, int, int, int]:
    r0 = 1 + ri * (room_h + 1)
    c0 = 1 + ci * (room_w + 1)
    return r0, r0 + room_h, c0, c0 + room_w


def _sample_seed_position(mask: np.ndarray, want_on_pattern: bool) -> tuple[int, int]:
    candidates = np.argwhere(mask if want_on_pattern else ~mask)
    idx = random.randrange(len(candidates))
    return tuple(map(int, candidates[idx]))


def _expected_output(
    grid: np.ndarray,
    room_h: int,
    room_w: int,
    patterns: dict[int, np.ndarray],
    seed_specs: list[tuple[int, int, int, tuple[int, int]]],
) -> np.ndarray:
    out = grid.copy()
    for ri, ci, color, seed_pos in seed_specs:
        r0, r1, c0, c1 = _room_bounds(ri, ci, room_h, room_w)
        room_out = np.full((room_h, room_w), BLACK, dtype=int)
        room_out[patterns[color]] = ORANGE
        room_out[seed_pos] = ORANGE
        out[r0:r1, c0:c1] = room_out
    return out


def _gen_attempt() -> np.ndarray:
    room_rows = random.randint(*ROOM_ROWS_RANGE)
    room_cols = random.randint(*ROOM_COLS_RANGE)
    room_h = random.randint(*ROOM_H_RANGE)
    room_w = random.randint(*ROOM_W_RANGE)
    total_rooms = room_rows * room_cols

    template_count = random.randint(1, min(MAX_TEMPLATE_COUNT, total_rooms - 1))
    remaining = total_rooms - template_count
    if remaining >= 2 and random.random() < 0.65:
        seed_count = random.randint(1, remaining - 1)
    else:
        seed_count = random.randint(1, remaining)

    template_colors = random.sample(INPUT_COLORS, template_count)
    patterns = {color: _make_pattern(room_h, room_w) for color in template_colors}

    room_ids = [(ri, ci) for ri in range(room_rows) for ci in range(room_cols)]
    random.shuffle(room_ids)
    template_rooms = room_ids[:template_count]
    seed_rooms = room_ids[template_count : template_count + seed_count]

    if seed_count >= template_count and random.random() < 0.5:
        seed_colors = template_colors.copy()
        random.shuffle(seed_colors)
        while len(seed_colors) < seed_count:
            seed_colors.append(random.choice(template_colors))
        random.shuffle(seed_colors)
    else:
        seed_colors = [random.choice(template_colors) for _ in range(seed_count)]

    seed_modes = [bool(random.getrandbits(1)) for _ in range(seed_count)]
    if seed_count > 1 and all(seed_modes):
        seed_modes[random.randrange(seed_count)] = False
    if seed_count > 1 and not any(seed_modes):
        seed_modes[random.randrange(seed_count)] = True

    grid_h = room_rows * room_h + room_rows + 1
    grid_w = room_cols * room_w + room_cols + 1
    grid = np.full((grid_h, grid_w), GREY, dtype=int)

    for ri in range(room_rows):
        for ci in range(room_cols):
            r0, r1, c0, c1 = _room_bounds(ri, ci, room_h, room_w)
            grid[r0:r1, c0:c1] = BLACK

    for (ri, ci), color in zip(template_rooms, template_colors):
        r0, r1, c0, c1 = _room_bounds(ri, ci, room_h, room_w)
        room = grid[r0:r1, c0:c1]
        room[patterns[color]] = color

    seed_specs: list[tuple[int, int, int, tuple[int, int]]] = []
    for (ri, ci), color, want_on_pattern in zip(seed_rooms, seed_colors, seed_modes):
        seed_pos = _sample_seed_position(patterns[color], want_on_pattern)
        r0, r1, c0, c1 = _room_bounds(ri, ci, room_h, room_w)
        grid[r0 + seed_pos[0], c0 + seed_pos[1]] = color
        seed_specs.append((ri, ci, color, seed_pos))

    expected = _expected_output(grid, room_h, room_w, patterns, seed_specs)
    actual = main(grid)
    assert np.array_equal(actual, expected), "generator produced a sample inconsistent with main()"
    assert not np.array_equal(grid, expected), "sample must require a non-trivial transformation"

    row_spans, col_spans = _room_layout(grid)
    assert len(row_spans) == room_rows and len(col_spans) == room_cols
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
