from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
TEAL = 8

ENCODE_COLORS = (BLUE, RED, YELLOW, ORANGE)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _perpendiculars(dr: int, dc: int) -> list[tuple[int, int]]:
    return [(-dc, dr), (dc, -dr)]


def _find_start_bar(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert len(pts) == 2, "expected exactly two green start pixels"
    (r0, c0), (r1, c1) = pts
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "start bar must be 4-connected"
    border = [p for p in pts if _is_border(grid.shape, *p)]
    assert len(border) == 1, "start bar needs exactly one border pixel"
    border_pt = border[0]
    inner_pt = pts[1] if pts[0] == border_pt else pts[0]
    direction = (inner_pt[0] - border_pt[0], inner_pt[1] - border_pt[1])
    return border_pt, inner_pt, direction


def _trace_path_encoding(grid: np.ndarray) -> tuple[list[int], list[int]]:
    border_pt, inner_pt, (dr, dc) = _find_start_bar(grid)
    _ = border_pt  # first segment length counts both start-bar pixels

    total_encoded = int(np.count_nonzero(np.isin(grid, ENCODE_COLORS)))
    assert 3 <= total_encoded <= 4, "generator uses 3-4 encoded turn blockers"

    r, c = inner_pt
    seg_len = 2
    lengths: list[int] = []
    colors: list[int] = []
    seen_states: set[tuple[int, int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc, len(colors))
        assert state not in seen_states, "path trace should not loop"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            lengths.append(seg_len)
            break

        cell = int(grid[nr, nc])
        if cell in (BG, GREEN):
            r, c = nr, nc
            seg_len += 1
            continue

        assert cell in (*ENCODE_COLORS, TEAL), "unexpected non-background color"
        assert cell in ENCODE_COLORS, "the forward blocker must be a layer-encoding color"
        lengths.append(seg_len)
        colors.append(cell)

        choices: list[tuple[int, int]] = []
        for tdr, tdc in _perpendiculars(dr, dc):
            tr, tc = r + tdr, c + tdc
            if _in_bounds(grid.shape, tr, tc) and int(grid[tr, tc]) == BG:
                choices.append((tdr, tdc))
        assert len(choices) == 1, "each turn must have exactly one local 90-degree continuation"
        dr, dc = choices[0]
        seg_len = 1

    assert len(colors) == total_encoded, "all encoded blockers must be encountered"
    return lengths, colors


def _paint_square(out: np.ndarray, center: tuple[int, int], half: int, color: int) -> None:
    cr, cc = center
    r0 = max(0, cr - half)
    r1 = min(out.shape[0] - 1, cr + half)
    c0 = max(0, cc - half)
    c1 = min(out.shape[1] - 1, cc + half)
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _render_nested(shape: tuple[int, int], lengths: list[int], colors: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    center = (shape[0] // 2, shape[1] // 2)

    half = lengths[0] - 1
    _paint_square(out, center, half, GREEN)
    for thickness, color in zip(lengths[1:], colors):
        outer_half = half + thickness
        for rr in range(center[0] - outer_half, center[0] + outer_half + 1):
            for cc in range(center[1] - outer_half, center[1] + outer_half + 1):
                if not _in_bounds(shape, rr, cc):
                    continue
                if max(abs(rr - center[0]), abs(cc - center[1])) > half:
                    out[rr, cc] = int(color)
        half = outer_half

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the start-bar corridor and convert its segment lengths into nested square layers.

    Input structure:
    - Black background with a 2-pixel green start bar; exactly one start pixel touches the border.
    - Colored single-pixel turn blockers use blue/red/yellow/orange, and optional teal guard blockers
      may occupy the forbidden side of a turn so the corridor stays locally unambiguous.
    - The colored blockers appear in the order the path meets them; the center of the grid is otherwise clear.

    Rule:
    - Start at the border pixel, move through the other green pixel, and continue straight through black cells.
    - When the next cell is a colored blocker, record the current segment length, record that blocker color,
      and turn into the unique free perpendicular direction; stop when the next step would leave the grid.
    - Use the first segment length to draw a centered solid green square of side `2*L1-1`.
    - Wrap one square frame per recorded blocker: frame thickness = next segment length, frame color = blocker color.
    - Erase the corridor instruction path and blockers; keep only the centered nested construction.

    Key insight:
    The corridor is only an instruction tape: its straight-run lengths encode layer thicknesses while the
    encountered blocker colors encode layer colors, so the path must be fully traced before any square can be built.
    """
    g = np.asarray(grid, dtype=int)
    lengths, colors = _trace_path_encoding(g)
    return _render_nested(g.shape, lengths, colors)


def _reserve_mask(shape: tuple[int, int], lengths: Iterable[int]) -> np.ndarray:
    half = int(sum(int(x) for x in lengths)) - 1
    cr, cc = shape[0] // 2, shape[1] // 2
    mask = np.zeros(shape, dtype=bool)
    r0 = max(0, cr - half)
    r1 = min(shape[0] - 1, cr + half)
    c0 = max(0, cc - half)
    c1 = min(shape[1] - 1, cc + half)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _sample_lengths(turns: int) -> list[int] | None:
    for _ in range(200):
        lengths = [random.randint(2, 5) for _ in range(turns + 1)]
        if turns == 4 and max(lengths) > 4:
            continue
        total = sum(lengths)
        if 8 <= total <= 12:
            return lengths
    return None


def _sample_shape(lengths: list[int]) -> tuple[int, int]:
    final_side = 2 * sum(lengths) - 1
    min_dim = max(21, final_side + 6)
    max_dim = min(29, final_side + 10)
    choices = list(range(min_dim, max_dim + 1))
    dim = random.choice(choices)
    if random.random() < 0.35:
        other = random.choice(choices)
        return dim, other
    return dim, dim


def _random_start(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = shape
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        c = random.randint(1, w - 2)
        return (0, c), (1, c), (1, 0)
    if side == "bottom":
        c = random.randint(1, w - 2)
        return (h - 1, c), (h - 2, c), (-1, 0)
    if side == "left":
        r = random.randint(1, h - 2)
        return (r, 0), (r, 1), (0, 1)
    r = random.randint(1, h - 2)
    return (r, w - 1), (r, w - 2), (0, -1)


def _steps_for_segment(lengths: list[int], seg_index: int) -> int:
    return lengths[0] - 2 if seg_index == 0 else lengths[seg_index] - 1


def _can_walk_cell(
    shape: tuple[int, int],
    cell: tuple[int, int],
    reserve: np.ndarray,
    blockers: dict[tuple[int, int], int],
    fixed_path: set[tuple[int, int]],
) -> bool:
    r, c = cell
    return (
        _in_bounds(shape, r, c)
        and not reserve[r, c]
        and cell not in blockers
        and cell not in fixed_path
    )


def _can_place_blocker(
    shape: tuple[int, int],
    cell: tuple[int, int],
    reserve: np.ndarray,
    path_cells: set[tuple[int, int]],
    blockers: dict[tuple[int, int], int],
) -> bool:
    r, c = cell
    return (
        _in_bounds(shape, r, c)
        and not reserve[r, c]
        and cell not in blockers
        and cell not in path_cells
    )


def _build_corridor(
    shape: tuple[int, int],
    lengths: list[int],
    encoded_colors: list[int],
    reserve: np.ndarray,
    border_pt: tuple[int, int],
    inner_pt: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]] | None:
    path_cells = {border_pt, inner_pt}
    fixed_path = {border_pt, inner_pt}
    blockers: dict[tuple[int, int], int] = {}
    turns = len(encoded_colors)

    def rec(
        seg_index: int,
        cur: tuple[int, int],
        dr: int,
        dc: int,
        path_cells: set[tuple[int, int]],
        blockers: dict[tuple[int, int], int],
    ) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]] | None:
        new_path = set(path_cells)
        new_blockers = dict(blockers)
        r, c = cur

        for _ in range(_steps_for_segment(lengths, seg_index)):
            nr, nc = r + dr, c + dc
            if not _can_walk_cell(shape, (nr, nc), reserve, new_blockers, fixed_path):
                return None
            new_path.add((nr, nc))
            r, c = nr, nc

        if seg_index == turns:
            exit_r, exit_c = r + dr, c + dc
            if _in_bounds(shape, exit_r, exit_c):
                return None
            return new_path, new_blockers

        ahead = (r + dr, c + dc)
        if not _can_place_blocker(shape, ahead, reserve, new_path, new_blockers):
            return None
        new_blockers[ahead] = int(encoded_colors[seg_index])

        choices = _perpendiculars(dr, dc)
        random.shuffle(choices)
        for idx, (ndr, ndc) in enumerate(choices):
            chosen = (r + ndr, c + ndc)
            if not _can_walk_cell(shape, chosen, reserve, new_blockers, fixed_path):
                continue

            odr, odc = choices[1 - idx]
            wrong = (r + odr, c + odc)
            branch_blockers = dict(new_blockers)
            if _in_bounds(shape, *wrong):
                if not _can_place_blocker(shape, wrong, reserve, new_path, branch_blockers):
                    continue
                branch_blockers[wrong] = TEAL

            result = rec(seg_index + 1, (r, c), ndr, ndc, new_path, branch_blockers)
            if result is not None:
                return result
        return None

    return rec(0, inner_pt, direction[0], direction[1], path_cells, blockers)


def _check_local_turns(grid: np.ndarray) -> bool:
    try:
        lengths, colors = _trace_path_encoding(grid)
    except AssertionError:
        return False
    return len(colors) == int(np.count_nonzero(np.isin(grid, ENCODE_COLORS))) and len(lengths) == len(colors) + 1


def _gen_attempt() -> np.ndarray | None:
    turns = random.choice([3, 4])
    lengths = _sample_lengths(turns)
    if lengths is None:
        return None

    shape = _sample_shape(lengths)
    reserve = _reserve_mask(shape, lengths)
    border_pt, inner_pt, direction = _random_start(shape)
    if reserve[border_pt] or reserve[inner_pt]:
        return None

    encoded_colors = random.sample(list(ENCODE_COLORS), turns)
    built = _build_corridor(shape, lengths, encoded_colors, reserve, border_pt, inner_pt, direction)
    if built is None:
        return None
    path_cells, blockers = built

    grid = np.zeros(shape, dtype=int)
    grid[border_pt] = GREEN
    grid[inner_pt] = GREEN
    for (r, c), color in blockers.items():
        grid[r, c] = int(color)

    if not _check_local_turns(grid):
        return None

    traced_lengths, traced_colors = _trace_path_encoding(grid)
    if traced_lengths != lengths or traced_colors != encoded_colors:
        return None

    out = main(grid)
    expected = _render_nested(shape, lengths, encoded_colors)
    if not np.array_equal(out, expected):
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(path_cells) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 800 attempts")
