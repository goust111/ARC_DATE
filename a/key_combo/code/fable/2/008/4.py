import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
MACRO = 3
NON_GREY_RING_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GAP_BY_DIR = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _ring_mask(direction: str) -> np.ndarray:
    mask = np.ones((MACRO, MACRO), dtype=bool)
    mask[1, 1] = False
    mask[GAP_BY_DIR[direction]] = False
    return mask


RING_MASKS = {direction: _ring_mask(direction) for direction in DIRS}


def _make_ring_block(color: int, direction: str) -> np.ndarray:
    block = np.full((MACRO, MACRO), BLACK, dtype=int)
    block[RING_MASKS[direction]] = int(color)
    return block


def _ray_cells(pos: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = pos
    h, w = shape
    cells = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br > ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("segment endpoints must share a row or column")


def _detect_rings(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, str]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] % MACRO == 0 and g.shape[1] % MACRO == 0, "grid must decompose into 3x3 macro-blocks"
    bg = _get_bg(g)
    rings: dict[tuple[int, int], tuple[int, str]] = {}
    mh, mw = g.shape[0] // MACRO, g.shape[1] // MACRO
    for mr in range(mh):
        for mc in range(mw):
            block = g[mr * MACRO : (mr + 1) * MACRO, mc * MACRO : (mc + 1) * MACRO]
            colors = sorted(set(map(int, block.ravel())) - {bg})
            if not colors:
                assert np.all(block == bg), "empty macro-blocks must be pure background"
                continue
            assert len(colors) == 1, "each glyph macro-block must be monochrome"
            color = int(colors[0])
            mask = block == color
            matches = [direction for direction, ring_mask in RING_MASKS.items() if np.array_equal(mask, ring_mask)]
            assert len(matches) == 1, "foreground macro-blocks must be gapped 3x3 rings"
            assert np.all(block[~mask] == bg), "ring interior and gap must be background"
            rings[(mr, mc)] = (color, matches[0])
    return rings


def _solve_macro(rings: dict[tuple[int, int], tuple[int, str]], shape: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    start_positions = [pos for pos, (color, _direction) in rings.items() if color == GREY]
    assert len(start_positions) == 1, "expected exactly one grey start ring"
    start = start_positions[0]

    out = np.full(shape, BLACK, dtype=int)
    visited_order = [start]
    seen = {start}
    out[start] = GREY
    cur = start

    while True:
        color, direction = rings[cur]
        ray = _ray_cells(cur, direction, shape)
        successor = None
        for cell in ray:
            if cell in rings:
                successor = cell
                break

        if successor is None:
            for cell in ray:
                out[cell] = int(color)
            break

        for cell in _segment_cells(cur, successor)[1:-1]:
            out[cell] = int(color)

        assert successor not in seen, "ring walk must terminate instead of looping"
        out[successor] = int(rings[successor][0])
        seen.add(successor)
        visited_order.append(successor)
        cur = successor

    return out, visited_order


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the grey gapped-ring ray chain and return its coarse macro-grid path.

    Input structure:
    - The canvas size is a multiple of three, partitioned into aligned 3x3 macro-blocks.
    - Some macro-blocks contain one monochrome hollow 3x3 square ring with exactly one missing side-middle pixel.
    - Exactly one ring is grey and marks the start; all other rings have distinct non-grey colors, and some may be distractors.

    Rule:
    - Read each ring's missing side-middle pixel as a direction instruction: up, down, left, or right.
    - Starting from the grey ring, ray-cast through macro-block space in that direction to the nearest macro-block that contains another ring, then continue from that successor ring.
    - If a ring points toward no further ring, keep walking in that direction until the ray exits the canvas.
    - Output the downsampled macro-grid: visited ring cells keep their ring colors, traversed empty macro-cells take the departure ring color, and every untouched cell is black.

    Key insight:
    - The 3x3 glyphs are direction symbols on a coarser grid: each gap chooses a ray, and only the nearest ring hit by that ray continues the chain.
    """
    g = np.asarray(grid, dtype=int)
    rings = _detect_rings(g)
    macro_shape = (g.shape[0] // MACRO, g.shape[1] // MACRO)
    out, _visited = _solve_macro(rings, macro_shape)
    return out


def _clear_prefix(ray: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    clear = []
    for cell in ray:
        if cell in used:
            break
        clear.append(cell)
    return clear


def _has_future_option(pos: tuple[int, int], used: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    for direction in DIRS:
        if _clear_prefix(_ray_cells(pos, direction, shape), used):
            return True
    return False


def _choose_exit_direction(
    pos: tuple[int, int], used: set[tuple[int, int]], shape: tuple[int, int], rng: random.Random
) -> tuple[str, list[tuple[int, int]]] | None:
    options = []
    for direction in DIRS:
        ray = _ray_cells(pos, direction, shape)
        clear = _clear_prefix(ray, used)
        if len(clear) == len(ray):
            options.append((direction, clear))
        elif not ray:
            options.append((direction, clear))
    if not options:
        return None
    rng.shuffle(options)
    options.sort(key=lambda item: (len(item[1]) > 0, len(item[1])), reverse=True)
    return options[0]


def _build_path(
    shape: tuple[int, int], ring_count: int, rng: random.Random
) -> tuple[list[tuple[int, int]], list[str], set[tuple[int, int]]] | None:
    h, w = shape

    def dfs(
        cur: tuple[int, int],
        rings_left_after_cur: int,
        used: set[tuple[int, int]],
        positions: list[tuple[int, int]],
        dirs: list[str],
    ) -> tuple[list[tuple[int, int]], list[str], set[tuple[int, int]]] | None:
        if rings_left_after_cur == 0:
            exit_choice = _choose_exit_direction(cur, used, shape, rng)
            if exit_choice is None:
                return None
            exit_dir, exit_cells = exit_choice
            return positions, dirs + [exit_dir], used | set(exit_cells)

        candidates = []
        for direction in DIRS:
            clear = _clear_prefix(_ray_cells(cur, direction, shape), used)
            for distance, nxt in enumerate(clear, start=1):
                future_used = used | set(_segment_cells(cur, nxt)[1:])
                if rings_left_after_cur > 1 and not _has_future_option(nxt, future_used, shape):
                    continue
                prefer_interior = 1 if 0 < nxt[0] < h - 1 and 0 < nxt[1] < w - 1 else 0
                prefer_gap = 1 if distance >= 2 else 0
                candidates.append((prefer_gap, prefer_interior, rng.random(), direction, nxt))

        candidates.sort(reverse=True)
        for _gap_ok, _inner_ok, _noise, direction, nxt in candidates:
            new_used = used | set(_segment_cells(cur, nxt)[1:])
            result = dfs(nxt, rings_left_after_cur - 1, new_used, positions + [nxt], dirs + [direction])
            if result is not None:
                return result
        return None

    start_pool = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if ring_count == 1 or (0 < r < h - 1 and 0 < c < w - 1)
    ]
    rng.shuffle(start_pool)
    for start in start_pool:
        result = dfs(start, ring_count - 1, {start}, [start], [])
        if result is not None:
            return result
    return None


def _expected_from_path(
    shape: tuple[int, int], positions: list[tuple[int, int]], dirs: list[str], colors: list[int]
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for pos, color in zip(positions, colors):
        out[pos] = int(color)
    for idx in range(len(positions) - 1):
        for cell in _segment_cells(positions[idx], positions[idx + 1])[1:-1]:
            out[cell] = int(colors[idx])
    for cell in _ray_cells(positions[-1], dirs[-1], shape):
        out[cell] = int(colors[-1])
    return out


def _place_distractors(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    count: int,
    rng: random.Random,
) -> list[tuple[tuple[int, int], str]]:
    h, w = shape
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]
    if count <= 0 or not free:
        return []
    path_rows = {r for r, _c in occupied}
    path_cols = {c for _r, c in occupied}
    aligned = [pos for pos in free if pos[0] in path_rows or pos[1] in path_cols]
    others = [pos for pos in free if pos not in aligned]
    rng.shuffle(aligned)
    rng.shuffle(others)
    chosen = (aligned + others)[:count]
    return [(pos, rng.choice(tuple(DIRS))) for pos in chosen]


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    macro_h = rng.randint(6, 10)
    macro_w = rng.randint(6, 10)
    shape = (macro_h, macro_w)

    roll = rng.random()
    if roll < 0.18:
        path_ring_count = 1
    elif roll < 0.43:
        path_ring_count = 2
    elif roll < 0.74:
        path_ring_count = 3
    elif roll < 0.92:
        path_ring_count = 4
    else:
        path_ring_count = 5

    path_data = _build_path(shape, path_ring_count, rng)
    if path_data is None:
        return None
    positions, dirs, occupied = path_data

    remaining_palette = NON_GREY_RING_COLORS.copy()
    rng.shuffle(remaining_palette)
    path_colors = [GREY] + remaining_palette[: path_ring_count - 1]
    remaining_palette = remaining_palette[path_ring_count - 1 :]

    max_distractors = min(4, len(remaining_palette), macro_h * macro_w - len(occupied))
    if max_distractors <= 0:
        distractor_count = 0
    else:
        distractor_choices = [0, 1, 2, 3, 4][: max_distractors + 1]
        distractor_weights = [1, 3, 3, 2, 1][: len(distractor_choices)]
        distractor_count = rng.choices(distractor_choices, weights=distractor_weights, k=1)[0]
    distractors = _place_distractors(shape, occupied, distractor_count, rng)

    rings: dict[tuple[int, int], tuple[int, str]] = {}
    for pos, color, direction in zip(positions, path_colors, dirs):
        rings[pos] = (int(color), direction)
    for (pos, direction), color in zip(distractors, remaining_palette[:distractor_count]):
        rings[pos] = (int(color), direction)

    expected = _expected_from_path(shape, positions, dirs, path_colors)
    solved, visited = _solve_macro(rings, shape)
    if visited != positions:
        return None
    if not np.array_equal(expected, solved):
        return None

    grid = np.full((macro_h * MACRO, macro_w * MACRO), BLACK, dtype=int)
    for (mr, mc), (color, direction) in rings.items():
        grid[mr * MACRO : (mr + 1) * MACRO, mc * MACRO : (mc + 1) * MACRO] = _make_ring_block(color, direction)

    reparsed = _detect_rings(grid)
    if reparsed != rings:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-ring ray puzzle")
