from __future__ import annotations

import random
from itertools import product

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

SELECTOR_COLORS = (RED, BLUE, YELLOW, ORANGE)
LEGEND_SLOTS = ((0, 0), (0, 4), (4, 0), (4, 4))
LEGEND_LABEL_POS = ((7, 0), (7, 2), (7, 4), (7, 6))
LEGEND_ORDER = (RED, BLUE, YELLOW, ORANGE)
LEGEND_ERASE_SHAPE = (8, 7)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TURN_RETRIES = 350
GEN_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rot_k_for_incoming(direction: tuple[int, int]) -> int:
    mapping = {
        (0, 1): 0,   # from left
        (1, 0): 3,   # from top -> clockwise 90
        (0, -1): 2,  # from right
        (-1, 0): 1,  # from bottom -> counter-clockwise 90
    }
    return mapping[direction]


def _rotate_mask(mask: np.ndarray, direction: tuple[int, int], *, apply_rotation: bool = True) -> np.ndarray:
    if not apply_rotation:
        return mask
    return np.rot90(mask, _rot_k_for_incoming(direction))


def _extract_patterns(grid: np.ndarray) -> dict[int, np.ndarray]:
    patterns: dict[int, np.ndarray] = {}
    for color, (lr, lc), (r0, c0) in zip(LEGEND_ORDER, LEGEND_LABEL_POS, LEGEND_SLOTS):
        assert int(grid[lr, lc]) == color, "legend label row is malformed"
        patch = np.asarray(grid[r0 : r0 + 3, c0 : c0 + 3], dtype=int)
        assert patch.shape == (3, 3)
        assert set(np.unique(patch)).issubset({GREY, MAROON}), "legend patch must use grey and maroon only"
        patterns[color] = patch == MAROON
    return patterns


def _find_start(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert len(coords) == 2, "input must contain exactly two green start pixels"
    a, b = coords
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "start pixels must be 4-connected"
    border = [p for p in coords if p[1] == 0]
    assert len(border) == 1, "exactly one start pixel must lie on the left border"
    border_pixel = border[0]
    inner = b if a == border_pixel else a
    assert inner[1] == 1 and inner[0] == border_pixel[0]
    return border_pixel, inner


def _trace_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, tuple[int, int]]]]:
    _, inner = _find_start(grid)
    path_cells: list[tuple[int, int]] = []
    turns: list[tuple[int, int, int, tuple[int, int]]] = []
    r, c = inner
    dr, dc = 0, 1
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path walk should not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            break

        if int(grid[nr, nc]) not in SELECTOR_COLORS:
            path_cells.append((nr, nc))
            r, c = nr, nc
            continue

        choices: list[tuple[int, int]] = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = r + tdr, c + tdc
            if _in_bounds(grid.shape, tr, tc) and int(grid[tr, tc]) not in SELECTOR_COLORS:
                choices.append((tdr, tdc))
        assert len(choices) == 1, "every blocker collision must force a unique turn"
        turns.append((r, c, int(grid[nr, nc]), (dr, dc)))
        dr, dc = choices[0]

    return path_cells, turns


def _render_output(
    grid: np.ndarray,
    path_cells: list[tuple[int, int]],
    turns: list[tuple[int, int, int, tuple[int, int]]],
    patterns: dict[int, np.ndarray],
    *,
    apply_rotation: bool = True,
    stamp_colors: dict[int, int] | None = None,
    do_stamps: bool = True,
    erase_legend: bool = True,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in path_cells:
        if int(out[r, c]) == BG:
            out[r, c] = GREEN

    if do_stamps:
        for r, c, selector_color, direction in turns:
            lookup = selector_color if stamp_colors is None else stamp_colors[selector_color]
            mask = _rotate_mask(patterns[lookup], direction, apply_rotation=apply_rotation)
            for dr in range(3):
                for dc in range(3):
                    if not bool(mask[dr, dc]):
                        continue
                    rr = r + dr - 1
                    cc = c + dc - 1
                    if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) in {BG, GREEN}:
                        out[rr, cc] = GREEN

    if erase_legend:
        h, w = LEGEND_ERASE_SHAPE
        out[:h, :w] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the corridor from the left-edge start bar and stamp rotated legend glyphs at each forced turn.

    Input structure:
    - The background is black.
    - The top-left corner contains a compact legend: four grey-backed 3x3 glyph patches drawn with maroon pixels, plus one label row whose colors are red, blue, yellow, and orange.
    - Outside the legend there is a 2-pixel green start bar on the left border and a sparse field of red/blue/yellow/orange obstacle pixels that define a one-cell-wide corridor.

    Rule:
    - Start at the border pixel of the green bar, move through the inner green pixel, and keep advancing straight while the next cell is not an obstacle.
    - When the next cell is an obstacle, turn to the unique perpendicular non-obstacle neighbor, record that turn, and use the obstacle color to choose the corresponding 3x3 legend glyph.
    - Stamp the chosen glyph centered on the last path cell before the turn, rotating it by the incoming direction: from left = 0 degrees, from top = clockwise 90, from right = 180, from bottom = counter-clockwise 90. Maroon glyph cells paint green onto black/green cells only, and the legend area is erased to black after all path drawing and stamping finish.

    Key insight:
    - Each turn is controlled twice by the same blocker: its position forces the corridor turn, while its color selects which legend glyph to print, and the arrival direction determines that glyph's orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    patterns = _extract_patterns(g)
    path_cells, turns = _trace_path(g)
    return _render_output(g, path_cells, turns, patterns)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mask_orbit(mask: np.ndarray) -> set[bytes]:
    return {np.rot90(mask, k).tobytes() for k in range(4)}


def _pattern_pool() -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for bits in product((0, 1), repeat=9):
        arr = np.array(bits, dtype=bool).reshape(3, 3)
        count = int(arr.sum())
        if count < 3 or count > 5:
            continue
        if not arr[1, 1]:
            continue
        orbit = _mask_orbit(arr)
        if len(orbit) != 4:
            continue
        if any(not arr[r].any() for r in range(3)):
            continue
        if any(not arr[:, c].any() for c in range(3)):
            continue
        pool.append(arr)
    return pool


PATTERN_POOL = _pattern_pool()


def _sample_patterns() -> dict[int, np.ndarray]:
    for _ in range(TURN_RETRIES):
        chosen = random.sample(PATTERN_POOL, 4)
        orbits = [_mask_orbit(mask) for mask in chosen]
        if any(orbits[i] & orbits[j] for i in range(4) for j in range(i + 1, 4)):
            continue
        return {color: mask.copy() for color, mask in zip(LEGEND_ORDER, chosen)}
    raise RuntimeError("failed to sample four distinct glyph patterns")


def _paint_legend(grid: np.ndarray, patterns: dict[int, np.ndarray]) -> None:
    for color, (lr, lc), (r0, c0) in zip(LEGEND_ORDER, LEGEND_LABEL_POS, LEGEND_SLOTS):
        grid[r0 : r0 + 3, c0 : c0 + 3] = GREY
        grid[r0 : r0 + 3, c0 : c0 + 3][patterns[color]] = MAROON
        grid[lr, lc] = color


def _is_free(grid: np.ndarray, path_cells: set[tuple[int, int]], r: int, c: int) -> bool:
    return _in_bounds(grid.shape, r, c) and int(grid[r, c]) == BG and (r, c) not in path_cells


def _free_run(grid: np.ndarray, path_cells: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> int:
    steps = 0
    while True:
        r += dr
        c += dc
        if not _is_free(grid, path_cells, r, c):
            return steps
        steps += 1


def _extend_segment(
    grid: np.ndarray,
    path_list: list[tuple[int, int]],
    path_set: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
    seg_len: int | None,
) -> tuple[int, int] | None:
    steps = 0
    while True:
        if seg_len is not None and steps == seg_len:
            return r, c
        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            return (r, c) if seg_len is None else None
        if not _is_free(grid, path_set, nr, nc):
            return None
        path_list.append((nr, nc))
        path_set.add((nr, nc))
        r, c = nr, nc
        steps += 1


def _choose_turn_colors(n_turns: int) -> list[int]:
    base = list(SELECTOR_COLORS)
    random.shuffle(base)
    if n_turns <= 4:
        return base[:n_turns]
    out = base[:]
    out.extend(random.choice(SELECTOR_COLORS) for _ in range(n_turns - 4))
    return out


def _scatter_distractors(grid: np.ndarray, path_set: set[tuple[int, int]]) -> None:
    target = random.randint(0, 2)
    free: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BG or (r, c) in path_set:
                continue
            if r < LEGEND_ERASE_SHAPE[0] and c < LEGEND_ERASE_SHAPE[1]:
                continue
            if any((nr, nc) in path_set for nr, nc in _neighbors4(r, c)):
                continue
            free.append((r, c))
    random.shuffle(free)
    for cell in free[:target]:
        grid[cell] = random.choice(SELECTOR_COLORS)


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(19, 26)
    width = random.randint(19, 28)
    grid = np.zeros((height, width), dtype=int)

    patterns = _sample_patterns()
    _paint_legend(grid, patterns)

    start_row = random.randint(9, height - 2)
    grid[start_row, 0] = GREEN
    grid[start_row, 1] = GREEN
    path_list: list[tuple[int, int]] = []
    path_set: set[tuple[int, int]] = {(start_row, 0), (start_row, 1)}

    r, c = start_row, 1
    dr, dc = 0, 1
    n_turns = random.randint(4, 6)
    turn_colors = _choose_turn_colors(n_turns)

    for turn_index in range(n_turns):
        free_run = _free_run(grid, path_set, r, c, dr, dc)
        if free_run < 2:
            return None

        max_seg = min(6, free_run - 1)
        min_seg = 1 if turn_index > 0 else max(1, 7 - c)
        if min_seg > max_seg:
            return None
        seg_len = random.randint(min_seg, max_seg)
        pos = _extend_segment(grid, path_list, path_set, r, c, dr, dc, seg_len)
        if pos is None:
            return None
        r, c = pos

        ahead = (r + dr, c + dc)
        if not _is_free(grid, path_set, *ahead):
            return None

        perps = [(-dc, dr), (dc, -dr)]
        random.shuffle(perps)
        chosen: tuple[int, int] | None = None
        wrong: tuple[int, int] | None = None
        for tdr, tdc in perps:
            tr, tc = r + tdr, c + tdc
            wr, wc = r - tdr, c - tdc
            if not _is_free(grid, path_set, tr, tc):
                continue
            if _in_bounds(grid.shape, wr, wc) and int(grid[wr, wc]) not in SELECTOR_COLORS:
                if (wr, wc) in path_set or int(grid[wr, wc]) != BG:
                    continue
                wrong = (wr, wc)
            else:
                wrong = None

            needed = 2 if turn_index < n_turns - 1 else 1
            if _free_run(grid, path_set, r, c, tdr, tdc) < needed:
                continue
            chosen = (tdr, tdc)
            break
        if chosen is None:
            return None

        grid[ahead] = turn_colors[turn_index]
        if wrong is not None:
            grid[wrong] = random.choice(SELECTOR_COLORS)
        dr, dc = chosen

    pos = _extend_segment(grid, path_list, path_set, r, c, dr, dc, None)
    if pos is None:
        return None

    _scatter_distractors(grid, path_set)

    inp = grid.copy()
    for pr, pc in path_list:
        inp[pr, pc] = BG

    expected = _render_output(inp, path_list, _trace_path(inp)[1], patterns)
    if not np.array_equal(main(inp), expected):
        return None

    path_only = _render_output(inp, path_list, _trace_path(inp)[1], patterns, do_stamps=False)
    if np.array_equal(path_only, expected):
        return None

    unrotated = _render_output(inp, path_list, _trace_path(inp)[1], patterns, apply_rotation=False)
    if np.array_equal(unrotated, expected):
        return None

    collapsed = {color: RED for color in SELECTOR_COLORS}
    wrong_mapping = _render_output(inp, path_list, _trace_path(inp)[1], patterns, stamp_colors=collapsed)
    if np.array_equal(wrong_mapping, expected):
        return None

    no_erase = _render_output(inp, path_list, _trace_path(inp)[1], patterns, erase_legend=False)
    if np.array_equal(no_erase, expected):
        return None

    turns = _trace_path(inp)[1]
    used_colors = {color for _r, _c, color, _d in turns}
    used_dirs = {direction for _r, _c, _color, direction in turns}
    if len(turns) < 4 or len(used_colors) < 3 or len(used_dirs) < 3:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid corridor-stamping input")
