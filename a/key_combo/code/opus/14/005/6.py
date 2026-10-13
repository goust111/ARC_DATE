from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
BLOCK_SIZE = 3
BLOCK_GRID = 3
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

CORNER_POSITIONS = ((0, 0), (0, 2), (2, 2), (2, 0))  # TL, TR, BR, BL
SIDE_POSITIONS = ((0, 1), (1, 2), (2, 1), (1, 0))  # T, R, B, L
RING_POSITIONS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)

GEN_ATTEMPTS = 400


def _is_separator_line(line: np.ndarray) -> bool:
    return set(map(int, line)).issubset({BLACK, GREY})


def _non_separator_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(flags):
        while i < len(flags) and flags[i]:
            i += 1
        start = i
        while i < len(flags) and not flags[i]:
            i += 1
        if start < i:
            runs.append((start, i))
    return runs


def _parse_block_intervals(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_runs = _non_separator_runs([_is_separator_line(grid[r, :]) for r in range(grid.shape[0])])
    col_runs = _non_separator_runs([_is_separator_line(grid[:, c]) for c in range(grid.shape[1])])
    assert len(row_runs) == BLOCK_GRID and len(col_runs) == BLOCK_GRID
    assert all(r1 - r0 == BLOCK_SIZE for r0, r1 in row_runs)
    assert all(c1 - c0 == BLOCK_SIZE for c0, c1 in col_runs)
    return row_runs, col_runs


def _iter_blocks(
    grid: np.ndarray, row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]]
):
    for br, (r0, r1) in enumerate(row_runs):
        for bc, (c0, c1) in enumerate(col_runs):
            yield br, bc, r0, r1, c0, c1, grid[r0:r1, c0:c1]


def _is_clean_ring(block: np.ndarray) -> bool:
    if block.shape != (BLOCK_SIZE, BLOCK_SIZE):
        return False
    if np.any(block == GREY):
        return False
    if int(block[1, 1]) != BLACK:
        return False
    outer = [int(block[r, c]) for r, c in RING_POSITIONS]
    if any(color in {BLACK, GREY} for color in outer):
        return False
    return 2 <= len(set(outer)) <= 4


def _infer_template(grid: np.ndarray) -> np.ndarray:
    row_runs, col_runs = _parse_block_intervals(grid)
    templates = [
        block.copy()
        for *_prefix, block in _iter_blocks(grid, row_runs, col_runs)
        if _is_clean_ring(block)
    ]
    assert templates, "expected at least one fully visible ring block"
    template = templates[0]
    for candidate in templates[1:]:
        assert np.array_equal(candidate, template), "clean ring blocks must all match"
    return template


def _majority_color(template: np.ndarray) -> int:
    counts = Counter(int(template[r, c]) for r, c in RING_POSITIONS)
    top = counts.most_common()
    assert len(top) >= 2 and top[0][1] > top[1][1], "ring needs a unique majority color"
    return int(top[0][0])


def _rotate_ring(template: np.ndarray) -> np.ndarray:
    out = template.copy()

    corners = [int(template[r, c]) for r, c in CORNER_POSITIONS]
    new_corners = [corners[3], corners[0], corners[1], corners[2]]
    for (r, c), color in zip(CORNER_POSITIONS, new_corners):
        out[r, c] = color

    sides = [int(template[r, c]) for r, c in SIDE_POSITIONS]
    new_sides = [sides[1], sides[2], sides[3], sides[0]]
    for (r, c), color in zip(SIDE_POSITIONS, new_sides):
        out[r, c] = color

    out[1, 1] = BLACK
    return out


def _matches_template(block: np.ndarray, template: np.ndarray) -> bool:
    known = block != GREY
    assert known.any(), "every block must expose at least one non-grey cell"
    return bool(np.all(block[known] == template[known]))


def _uniform_blank_color(block: np.ndarray, palette: set[int]) -> int | None:
    known = [int(v) for v in block[block != GREY]]
    if not known:
        return None
    if len(set(known)) != 1:
        return None
    color = known[0]
    if color not in palette:
        return None
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """Invert the ring-vs-solid block code, then rotate colors on the newly created rings.

    Input structure:
    - A 3x3 array of 3x3 blocks is separated by one-pixel black lines, optionally with an outer separator border.
    - Ring blocks are 3x3 loops with a black center and colored outer cells; some cells may be grey erasures.
    - Blank blocks are 3x3 solid-color blocks using one of the ring palette colors; separator pixels may also be grey-erased.

    Rule:
    - Infer the full ring template from clean mixed-color blocks that show an unerased 3x3 ring.
    - Replace every block matching that template, treating grey as a wildcard, with a solid block of the ring's majority color.
    - Replace every uniform blank block with the full ring template after one step of color cycling: corners move clockwise, side-centers move counterclockwise.
    - Restore every separator row and column pixel to black.

    Key insight:
    - The same 3x3 ring is both the code symbol and the structure to transform: after code inversion, only the rings created from blank blocks should receive the corner/side color cycle.
    """
    g = np.asarray(grid, dtype=int)
    row_runs, col_runs = _parse_block_intervals(g)
    template = _infer_template(g)
    rotated_template = _rotate_ring(template)
    majority = _majority_color(template)
    palette = {int(template[r, c]) for r, c in RING_POSITIONS}

    out = np.zeros_like(g)
    for _br, _bc, r0, r1, c0, c1, block in _iter_blocks(g, row_runs, col_runs):
        matches_sprite = _matches_template(block, template)
        blank_color = _uniform_blank_color(block, palette)
        matches_blank = blank_color is not None
        assert matches_sprite != matches_blank, "each block must be unambiguously ring or blank"
        out[r0:r1, c0:c1] = majority if matches_sprite else rotated_template

    return out


def _layout(with_border: bool) -> tuple[int, list[int], list[int]]:
    if with_border:
        return 13, [1, 5, 9], [1, 5, 9]
    return 11, [0, 4, 8], [0, 4, 8]


def _ring_palette_and_template() -> tuple[np.ndarray, list[int], int]:
    for _ in range(200):
        color_count = random.randint(2, 4)
        palette = random.sample(PAYLOAD_COLORS, color_count)
        outer = [random.choice(palette) for _ in range(8)]
        counts = Counter(outer)
        if set(counts) != set(palette):
            continue
        ranked = counts.most_common()
        if ranked[0][1] == ranked[1][1]:
            continue
        corners = [outer[0], outer[2], outer[7], outer[5]]
        sides = [outer[1], outer[4], outer[6], outer[3]]
        if len(set(corners)) < 2 or len(set(sides)) < 2:
            continue

        template = np.zeros((3, 3), dtype=int)
        for (r, c), color in zip(RING_POSITIONS, outer):
            template[r, c] = color
        if np.array_equal(_rotate_ring(template), template):
            continue
        return template, palette, int(ranked[0][0])
    raise RuntimeError("failed to generate a non-degenerate ring template")


def _blank_color_map(state: np.ndarray, palette: list[int], majority: int) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    for br in range(BLOCK_GRID):
        for bc in range(BLOCK_GRID):
            if state[br, bc]:
                continue
            if random.random() < 0.5:
                colors[(br, bc)] = majority
            else:
                colors[(br, bc)] = random.choice(palette)
    return colors


def _render_clean_input(
    with_border: bool, state: np.ndarray, template: np.ndarray, blank_colors: dict[tuple[int, int], int]
) -> np.ndarray:
    side, row_starts, col_starts = _layout(with_border)
    grid = np.zeros((side, side), dtype=int)
    for br, r0 in enumerate(row_starts):
        for bc, c0 in enumerate(col_starts):
            if state[br, bc]:
                grid[r0 : r0 + 3, c0 : c0 + 3] = template
            else:
                grid[r0 : r0 + 3, c0 : c0 + 3] = blank_colors[(br, bc)]
    return grid


def _render_expected_output(
    with_border: bool, state: np.ndarray, rotated_template: np.ndarray, majority: int
) -> np.ndarray:
    side, row_starts, col_starts = _layout(with_border)
    out = np.zeros((side, side), dtype=int)
    for br, r0 in enumerate(row_starts):
        for bc, c0 in enumerate(col_starts):
            out[r0 : r0 + 3, c0 : c0 + 3] = majority if state[br, bc] else rotated_template
    return out


def _protect_mask(clean: np.ndarray, with_border: bool, state: np.ndarray, clean_ring: tuple[int, int]) -> np.ndarray:
    side, row_starts, col_starts = _layout(with_border)
    protected = np.zeros((side, side), dtype=bool)

    cr, cc = clean_ring
    r0, c0 = row_starts[cr], col_starts[cc]
    protected[r0 : r0 + 3, c0 : c0 + 3] = True

    for br, rr in enumerate(row_starts):
        for bc, cc0 in enumerate(col_starts):
            protected[rr + 1, cc0 + 1] = True

    for r in range(side):
        if _is_separator_line(clean[r, :]):
            continue
        colored = [c for c in range(side) if clean[r, c] not in {BLACK, GREY}]
        assert colored
        protected[r, random.choice(colored)] = True

    for c in range(side):
        if _is_separator_line(clean[:, c]):
            continue
        colored = [r for r in range(side) if clean[r, c] not in {BLACK, GREY}]
        assert colored
        protected[random.choice(colored), c] = True

    return protected


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _grow_blob(allowed: np.ndarray, target: int) -> set[tuple[int, int]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if len(coords) < target:
        return None
    seed = random.choice(coords)
    blob = {seed}
    frontier = {cell for cell in _neighbors4(*seed, allowed.shape) if allowed[cell]}
    while len(blob) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        for nb in _neighbors4(*cell, allowed.shape):
            if allowed[nb] and nb not in blob:
                frontier.add(nb)
    return blob


def _make_grey_mask(allowed: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(allowed, dtype=bool)
    allowed_count = int(np.count_nonzero(allowed))
    if allowed_count == 0 or random.random() < 0.2:
        return mask

    max_grey = max(1, min(allowed_count, int(round(allowed.size * 0.18))))
    target = random.randint(1, max_grey)

    if target >= 4 and random.random() < 0.4:
        first_size = random.randint(1, target - 1)
        first = _grow_blob(allowed, first_size)
        if first is not None:
            blocked = allowed.copy()
            for cell in first:
                blocked[cell] = False
                for nb in _neighbors4(*cell, allowed.shape):
                    blocked[nb] = False
            second = _grow_blob(blocked, target - first_size)
            if second is not None:
                for r, c in first | second:
                    mask[r, c] = True
                return mask

    blob = _grow_blob(allowed, target)
    if blob is None:
        return mask
    for r, c in blob:
        mask[r, c] = True
    return mask


def _sample_state() -> tuple[np.ndarray, tuple[int, int]]:
    ring_count = random.randint(3, 6)
    state = np.zeros((BLOCK_GRID, BLOCK_GRID), dtype=bool)
    ring_cells = random.sample([(r, c) for r in range(3) for c in range(3)], ring_count)
    for rc in ring_cells:
        state[rc] = True
    return state, random.choice(ring_cells)


def _attempt_generate() -> np.ndarray | None:
    with_border = random.random() < 0.5
    template, palette, majority = _ring_palette_and_template()
    rotated = _rotate_ring(template)
    state, clean_ring = _sample_state()
    blank_colors = _blank_color_map(state, palette, majority)

    clean = _render_clean_input(with_border, state, template, blank_colors)
    expected = _render_expected_output(with_border, state, rotated, majority)
    protected = _protect_mask(clean, with_border, state, clean_ring)
    grey_mask = _make_grey_mask(~protected)

    inp = clean.copy()
    inp[grey_mask] = GREY

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
