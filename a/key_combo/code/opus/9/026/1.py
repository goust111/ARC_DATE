from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SOURCE_COLORS = (1, 2, 3, 4)
PIVOT_COLOR = 5
TEMPLATE_COLOR = 9
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
GENERATION_RETRIES = 500


def _in_bounds(n: int, r: int, c: int) -> bool:
    return 0 <= r < n and 0 <= c < n


def _rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, k=(-quarter_turns) % 4)


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == PIVOT_COLOR)]
    assert len(divider_cols) == 1, "expected exactly one full grey divider column"
    divider = divider_cols[0]

    source = g[:, :divider]
    template = g[:, divider + 1 :]
    assert source.shape == template.shape, "expected equal left and right panels"
    n = int(source.shape[0])
    assert source.shape == (n, n), "expected square panels"
    return source, template


def _template_rotations_distinct(mask: np.ndarray) -> bool:
    return len({np.rot90(mask, k=k).tobytes() for k in range(4)}) == 4


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(mask.shape[0], nr, nc) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _source_cells(source: np.ndarray) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((source != BACKGROUND) & (source != PIVOT_COLOR)):
        cells.append((int(r), int(c), int(source[r, c])))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a rotated copy of the template into each output block using grey pivot direction as the rotation code.

    Input structure:
    - A single full grey column splits the grid into equal square left and right panels on a black background.
    - The left panel is a sparse source grid with 2-4 colored pixels, each paired with exactly one 4-neighbor grey pivot pixel.
    - The right panel is a black template panel containing one maroon mask shape whose four 90-degree rotations are all distinct.

    Rule:
    - Read each colored source pixel at position (r, c) and find the unique grey pivot adjacent to it.
    - Decode pivot direction relative to the source pixel: above -> 0 degrees, right -> 90 clockwise, below -> 180, left -> 270.
    - Rotate the full template panel by that amount, recolor all maroon mask pixels to the source color, and stamp the result into output block (r, c).
    - The output is an n x n block grid of size n^2 x n^2, where each block has the same size as one input panel and empty blocks stay black.

    Key insight:
    - The grey pivot is not a symmetry center; it is a direction pointer whose side relative to the source pixel determines the template rotation before stamping.
    """
    source, template = _split_panels(np.asarray(grid, dtype=int))
    n = int(source.shape[0])

    template_mask = template == TEMPLATE_COLOR
    assert 3 <= int(template_mask.sum()) <= 5, "template mask must contain 3-5 cells"
    assert _connected4(template_mask), "template mask must be 4-connected"

    source_items = _source_cells(source)
    assert 2 <= len(source_items) <= 4, "expected 2-4 source pixels"
    assert len({color for _, _, color in source_items}) == len(source_items), "source colors must be distinct"

    pivot_positions = {(int(r), int(c)) for r, c in np.argwhere(source == PIVOT_COLOR)}
    assert len(pivot_positions) == len(source_items), "expected one grey pivot per source pixel"

    out = np.full((n * n, n * n), BACKGROUND, dtype=int)
    used_pivots: set[tuple[int, int]] = set()

    for r, c, color in source_items:
        choices: list[tuple[int, int, int]] = []
        for turn, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if _in_bounds(n, nr, nc) and source[nr, nc] == PIVOT_COLOR:
                choices.append((turn, nr, nc))
        assert len(choices) == 1, "each source pixel must have exactly one adjacent grey pivot"
        turn, pr, pc = choices[0]
        used_pivots.add((pr, pc))

        rotated_mask = _rotate_cw(template_mask, turn)
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        block[rotated_mask] = color

    assert used_pivots == pivot_positions, "every grey pivot must belong to exactly one source pixel"
    return out


def _grow_connected_mask(n: int, cell_count: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    r = random.randrange(n)
    c = random.randrange(n)
    mask[r, c] = True
    cells = {(r, c)}

    while len(cells) < cell_count:
        frontier = set()
        for cr, cc in cells:
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if _in_bounds(n, nr, nc) and not mask[nr, nc]:
                    frontier.add((nr, nc))
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
        cells.add((nr, nc))
    return mask


def _generate_template_panel(n: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        mask = _grow_connected_mask(n, random.randint(3, 5))
        if not _template_rotations_distinct(mask):
            continue
        panel = np.full((n, n), BACKGROUND, dtype=int)
        panel[mask] = TEMPLATE_COLOR
        return panel, mask
    raise RuntimeError("failed to generate a non-symmetric template mask")


def _candidate_pairs(n: int, turn: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    dr, dc = DIRS[turn]
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            pr, pc = r + dr, c + dc
            if _in_bounds(n, pr, pc):
                pairs.append(((r, c), (pr, pc)))
    random.shuffle(pairs)
    return pairs


def _pair_layout(n: int, turns: list[int]) -> list[tuple[int, tuple[int, int], tuple[int, int]]] | None:
    candidates = {turn: _candidate_pairs(n, turn) for turn in turns}

    def compatible(occupied: set[tuple[int, int]], pair_cells: set[tuple[int, int]]) -> bool:
        for r1, c1 in pair_cells:
            for r2, c2 in occupied:
                if abs(r1 - r2) + abs(c1 - c2) <= 1:
                    return False
        return True

    def search(idx: int, occupied: set[tuple[int, int]]) -> list[tuple[int, tuple[int, int], tuple[int, int]]] | None:
        if idx == len(turns):
            return []
        turn = turns[idx]
        for source_pos, pivot_pos in candidates[turn]:
            pair_cells = {source_pos, pivot_pos}
            if not compatible(occupied, pair_cells):
                continue
            rest = search(idx + 1, occupied | pair_cells)
            if rest is not None:
                return [(turn, source_pos, pivot_pos)] + rest
        return None

    return search(0, set())


def _oracle_output(
    n: int,
    template_mask: np.ndarray,
    placements: list[tuple[int, tuple[int, int], tuple[int, int], int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), BACKGROUND, dtype=int)
    for turn, (r, c), _pivot_pos, color in placements:
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        block[_rotate_cw(template_mask, turn)] = color
    return out


def _unrotated_output(
    n: int,
    template_mask: np.ndarray,
    placements: list[tuple[int, tuple[int, int], tuple[int, int], int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), BACKGROUND, dtype=int)
    for _turn, (r, c), _pivot_pos, color in placements:
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        block[template_mask] = color
    return out


def _validate_input(grid: np.ndarray) -> None:
    source, template = _split_panels(grid)
    n = int(source.shape[0])

    template_mask = template == TEMPLATE_COLOR
    assert 3 <= int(template_mask.sum()) <= 5
    assert _connected4(template_mask)
    assert _template_rotations_distinct(template_mask)

    source_items = _source_cells(source)
    assert 2 <= len(source_items) <= 4
    assert len({color for _, _, color in source_items}) == len(source_items)

    pivot_cells = {(int(r), int(c)) for r, c in np.argwhere(source == PIVOT_COLOR)}
    assert len(pivot_cells) == len(source_items)

    turns = set()
    occupied = []
    allowed_contacts: set[frozenset[tuple[int, int]]] = set()
    for r, c, _color in source_items:
        adjacent = []
        for turn, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if _in_bounds(n, nr, nc) and source[nr, nc] == PIVOT_COLOR:
                adjacent.append((turn, nr, nc))
        assert len(adjacent) == 1
        turns.add(adjacent[0][0])
        allowed_contacts.add(frozenset({(r, c), (adjacent[0][1], adjacent[0][2])}))
        occupied.append((r, c))
        occupied.append((adjacent[0][1], adjacent[0][2]))

    assert len(turns) >= 2, "need multiple directions so rotation is necessary"
    for i, (r1, c1) in enumerate(occupied):
        for r2, c2 in occupied[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) <= 1:
                assert frozenset({(r1, c1), (r2, c2)}) in allowed_contacts, "different source-pivot pairs must not touch"


def _attempt_generate() -> np.ndarray | None:
    n = random.choice([4, 5])
    panel, template_mask = _generate_template_panel(n)

    if n == 4:
        source_count = random.choices([2, 3], weights=[2, 3], k=1)[0]
    else:
        source_count = random.choices([2, 3, 4], weights=[1, 2, 3], k=1)[0]

    turns = random.sample([0, 1, 2, 3], source_count)
    random.shuffle(turns)
    layout = _pair_layout(n, turns)
    if layout is None:
        return None

    colors = random.sample(list(SOURCE_COLORS), source_count)
    placements: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    source = np.full((n, n), BACKGROUND, dtype=int)
    for (turn, source_pos, pivot_pos), color in zip(layout, colors):
        sr, sc = source_pos
        pr, pc = pivot_pos
        source[sr, sc] = color
        source[pr, pc] = PIVOT_COLOR
        placements.append((turn, source_pos, pivot_pos, color))

    divider = np.full((n, 1), PIVOT_COLOR, dtype=int)
    grid = np.concatenate([source, divider, panel], axis=1)

    try:
        _validate_input(grid)
    except AssertionError:
        return None

    expected = _oracle_output(n, template_mask, placements)
    if np.array_equal(expected, _unrotated_output(n, template_mask, placements)):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
