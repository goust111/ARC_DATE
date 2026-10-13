from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Vector = tuple[int, int]


BG = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
VECTOR_POOL: tuple[Vector, ...] = (
    (-2, -1), (-2, 0), (-2, 1),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
    (2, -1), (2, 0), (2, 1),
)

HEIGHT_RANGE = (12, 18)
LEGEND_COUNT_RANGE = (3, 5)
RIGHT_WIDTH_MAX = 12
PROGRAM_EXTRA_MAX = 4
PAIR_PADDING = 3
GEN_ATTEMPTS = 300
RIGHT_START_ATTEMPTS = 120
LEFT_PLACE_ATTEMPTS = 240


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.argmax(np.bincount(values)))


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != bg and np.all(col == col[0]):
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, "expected exactly one full-height divider column"
    return candidates[0]


def _infer_gray_color(left: np.ndarray, right: np.ndarray, bg: int) -> int:
    assert right.shape[0] >= 2, "right panel must have a body below the program row"
    left_colors = set(map(int, np.unique(left))) - {bg}
    body_colors = set(map(int, np.unique(right[1:, :]))) - {bg}
    candidates = sorted(left_colors & body_colors)
    assert len(candidates) == 1, "expected one shared legend/start color"
    return int(candidates[0])


def _extract_mapping(left: np.ndarray, bg: int, gray: int, divider_color: int) -> dict[int, Vector]:
    legend_colors = [
        int(color)
        for color in sorted(np.unique(left))
        if int(color) not in {bg, gray, divider_color}
    ]
    gray_coords = [tuple(map(int, rc)) for rc in np.argwhere(left == gray)]
    assert legend_colors, "expected at least one legend entry"
    assert len(gray_coords) == len(legend_colors), "legend pairs must be one colored pixel and one gray pixel each"

    used_gray: set[int] = set()
    mapping: dict[int, Vector] = {}
    for color in legend_colors:
        color_coords = np.argwhere(left == color)
        assert len(color_coords) == 1, "each legend color must appear once"
        r, c = map(int, color_coords[0])
        distances = [abs(gr - r) + abs(gc - c) for gr, gc in gray_coords]
        best = min(distances)
        matches = [idx for idx, dist in enumerate(distances) if dist == best]
        assert len(matches) == 1, "legend pairing must be unambiguous"
        gray_idx = matches[0]
        assert gray_idx not in used_gray, "legend pairs must be one-to-one"
        used_gray.add(gray_idx)
        gr, gc = gray_coords[gray_idx]
        mapping[color] = (int(gr - r), int(gc - c))

    assert len(mapping) == len(gray_coords)
    return mapping


def _program_colors(right: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in right[0, :] if int(color) != bg]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the left-panel color-to-vector legend and replay the program on the right panel.

    Input structure:
    - A full-height non-background divider column splits the grid into a left legend panel and a right execution panel.
    - In the left panel, each legend entry is one colored pixel paired with one nearby gray pixel; different legend colors are distinct.
    - In the right panel, the top row contains a left-to-right program of colored pixels, and exactly one gray start marker sits below it.

    Rule:
    - Learn one displacement vector per legend color by subtracting each colored legend pixel from its uniquely nearest gray partner.
    - Drop the divider and keep only the right panel; erase the program row contents.
    - Starting from the gray marker, execute the top-row colors in order, moving by the learned vector each time and coloring only the landing cell of that step.

    Key insight:
    - The left panel is not decoration: every colored dot teaches the move associated with that color by pointing to its nearby gray example endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2 and g.shape[1] >= 3, "grid is too small for two panels and a program row"

    bg = _background_color(g)
    divider_c, divider_color = _find_divider(g, bg)
    left = g[:, :divider_c]
    right = g[:, divider_c + 1 :]
    assert left.size > 0 and right.size > 0, "both panels must be non-empty"

    gray = _infer_gray_color(left, right, bg)
    mapping = _extract_mapping(left, bg, gray, divider_color)

    start_coords = np.argwhere(right == gray)
    assert len(start_coords) == 1, "expected one gray start marker in the right panel"
    start_r, start_c = map(int, start_coords[0])
    assert start_r > 0, "start marker must be below the program row"

    program = _program_colors(right, bg)
    assert program, "expected at least one program color on the top row"

    out = np.full(right.shape, bg, dtype=int)
    start = (start_r, start_c)
    out[start] = gray

    r_cur, c_cur = start
    for color in program:
        assert color in mapping, "every program color must appear in the legend"
        dr, dc = mapping[color]
        r_cur += dr
        c_cur += dc
        assert 0 <= r_cur < out.shape[0] and 0 <= c_cur < out.shape[1], "program exits the right panel"
        if (r_cur, c_cur) != start:
            out[r_cur, c_cur] = color

    return out


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _simulate_positions(
    height: int,
    width: int,
    start: Coord,
    program: list[int],
    mapping: dict[int, Vector],
) -> list[tuple[int, int, int]] | None:
    seen = {start}
    r_cur, c_cur = start
    positions: list[tuple[int, int, int]] = []
    for color in program:
        dr, dc = mapping[color]
        r_cur += dr
        c_cur += dc
        if not (1 <= r_cur < height and 0 <= c_cur < width):
            return None
        if (r_cur, c_cur) in seen:
            return None
        seen.add((r_cur, c_cur))
        positions.append((r_cur, c_cur, color))
    return positions


def _build_program(colors: list[int]) -> list[int]:
    length = random.randint(len(colors) + 1, len(colors) + PROGRAM_EXTRA_MAX)
    program = colors[:]
    program.extend(random.choice(colors) for _ in range(length - len(colors)))
    random.shuffle(program)
    return program


def _try_right_panel(mapping: dict[int, Vector]) -> tuple[int, int, list[int], list[int], Coord, list[tuple[int, int, int]]] | None:
    colors = list(mapping)
    program = _build_program(colors)
    height = random.randint(*HEIGHT_RANGE)
    min_width = max(8, len(program) + 1)
    if min_width > RIGHT_WIDTH_MAX:
        return None
    width = random.randint(min_width, RIGHT_WIDTH_MAX)
    program_cols = sorted(random.sample(range(width), len(program)))

    for _ in range(RIGHT_START_ATTEMPTS):
        start = (random.randint(2, height - 2), random.randint(1, width - 2))
        positions = _simulate_positions(height, width, start, program, mapping)
        if positions is not None:
            return height, width, program, program_cols, start, positions
    return None


def _place_legend_pairs(height: int, width: int, mapping: dict[int, Vector]) -> dict[int, tuple[Coord, Coord]] | None:
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: dict[int, tuple[Coord, Coord]] = {}

    items = list(mapping.items())
    random.shuffle(items)
    for color, (dr, dc) in items:
        row_lo = max(0, -dr)
        row_hi = min(height - 1, height - 1 - dr)
        col_lo = max(0, -dc)
        col_hi = min(width - 1, width - 1 - dc)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(LEFT_PLACE_ATTEMPTS):
            r = random.randint(row_lo, row_hi)
            c = random.randint(col_lo, col_hi)
            gr, gc = r + dr, c + dc
            box = (
                min(r, gr) - PAIR_PADDING,
                min(c, gc) - PAIR_PADDING,
                max(r, gr) + PAIR_PADDING,
                max(c, gc) + PAIR_PADDING,
            )
            if any(_overlap(box, other) for other in placed_boxes):
                continue
            placements[color] = ((r, c), (gr, gc))
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return placements


def _expected_output(height: int, width: int, start: Coord, positions: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    out[start] = GRAY
    for r, c, color in positions:
        if (r, c) != start:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.randint(*LEGEND_COUNT_RANGE)
    chosen = random.sample(PALETTE, legend_count + 1)
    divider_color = chosen[0]
    legend_colors = chosen[1:]
    vectors = random.sample(VECTOR_POOL, legend_count)
    mapping = {color: vector for color, vector in zip(legend_colors, vectors)}

    right_panel = _try_right_panel(mapping)
    if right_panel is None:
        return None
    height, right_w, program, program_cols, start, positions = right_panel

    left_w_min = 14
    left_w_max = min(18, 29 - right_w)
    if left_w_min > left_w_max:
        return None
    left_w = random.randint(left_w_min, left_w_max)

    legend = _place_legend_pairs(height, left_w, mapping)
    if legend is None:
        return None

    grid = np.full((height, left_w + 1 + right_w), BG, dtype=int)
    grid[:, left_w] = divider_color

    for color, ((r, c), (gr, gc)) in legend.items():
        grid[r, c] = color
        grid[gr, gc] = GRAY

    for color, col in zip(program, program_cols):
        grid[0, left_w + 1 + col] = color
    grid[start[0], left_w + 1 + start[1]] = GRAY

    expected = _expected_output(height, right_w, start, positions)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
