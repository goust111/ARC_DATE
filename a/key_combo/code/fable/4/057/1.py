from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
PALETTE: tuple[int, ...] = tuple(range(10))
MAX_PERIOD_SHIFT_MISMATCHES = 12
MAX_GENERATION_ATTEMPTS = 400
MIN_RUN_BETWEEN_TURNS = 2
MIN_FINAL_RUN = 2


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _expected_color(tile: np.ndarray, r: int, c: int) -> int:
    return int(tile[r % tile.shape[0], c % tile.shape[1]])


def _axis_mismatch_count(grid: np.ndarray, shift: int, axis: int) -> int:
    if axis == 0:
        return int(np.count_nonzero(grid[:-shift, :] != grid[shift:, :]))
    return int(np.count_nonzero(grid[:, :-shift] != grid[:, shift:]))


def _infer_axis_period(grid: np.ndarray, axis: int) -> int:
    length = int(grid.shape[axis])
    candidates: list[int] = []
    scored: list[tuple[float, int]] = []
    for shift in range(1, length):
        mismatches = _axis_mismatch_count(grid, shift, axis)
        compared = (grid.shape[0] - shift) * grid.shape[1] if axis == 0 else grid.shape[0] * (grid.shape[1] - shift)
        rate = mismatches / compared
        if mismatches <= MAX_PERIOD_SHIFT_MISMATCHES:
            candidates.append(shift)
        scored.append((rate + 0.001 * shift, shift))
    if candidates:
        return min(candidates)
    return min(scored)[1]


def _majority_tile(grid: np.ndarray, pr: int, pc: int) -> np.ndarray:
    tile = np.zeros((pr, pc), dtype=int)
    for rr in range(pr):
        for cc in range(pc):
            values = [int(v) for v in grid[rr::pr, cc::pc].ravel()]
            tile[rr, cc] = Counter(values).most_common(1)[0][0]
    return tile


def _reconstruct_tile(grid: np.ndarray) -> np.ndarray:
    pr = _infer_axis_period(grid, axis=0)
    pc = _infer_axis_period(grid, axis=1)
    return _majority_tile(grid, pr, pc)


def _entry_direction(shape: tuple[int, int], marker: tuple[int, int]) -> int:
    h, w = shape
    r, c = marker
    if r == 0 and 0 < c < w - 1:
        return 2
    if r == h - 1 and 0 < c < w - 1:
        return 0
    if c == 0 and 0 < r < h - 1:
        return 1
    if c == w - 1 and 0 < r < h - 1:
        return 3
    raise AssertionError("entry marker must lie on a non-corner border cell")


def _find_marker(grid: np.ndarray, tile: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    expected = np.fromfunction(lambda r, c: tile[r.astype(int) % tile.shape[0], c.astype(int) % tile.shape[1]], grid.shape, dtype=int)
    mismatch_mask = grid != expected
    border_coords = [
        (int(r), int(c))
        for r, c in np.argwhere(mismatch_mask)
        if _is_border((h, w), int(r), int(c))
    ]
    if len(border_coords) == 1:
        return border_coords[0]

    tile_colors = {int(v) for v in tile.ravel()}
    uncommon = [
        (r, c)
        for r, c in border_coords
        if int(grid[r, c]) not in tile_colors
    ]
    if len(uncommon) == 1:
        return uncommon[0]
    raise AssertionError("failed to isolate the unique border entry marker")


def _defect_turn(tile: np.ndarray, shape: tuple[int, int], grid: np.ndarray, pos: tuple[int, int], dir_idx: int) -> int:
    r, c = pos
    shown = int(grid[r, c])
    own = _expected_color(tile, r, c)
    left_dir = (dir_idx - 1) % 4
    right_dir = (dir_idx + 1) % 4
    lr, lc = r + DIRS[left_dir][0], c + DIRS[left_dir][1]
    rr, rc = r + DIRS[right_dir][0], c + DIRS[right_dir][1]
    assert _in_bounds(shape, lr, lc) and _in_bounds(shape, rr, rc), "defect must have both side neighbors"
    left_color = _expected_color(tile, lr, lc)
    right_color = _expected_color(tile, rr, rc)
    if shown == left_color and shown != own and shown != right_color:
        return left_dir
    if shown == right_color and shown != own and shown != left_color:
        return right_dir
    raise AssertionError("defect color does not uniquely encode a turn")


def _trace_path(grid: np.ndarray, tile: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    marker = _find_marker(grid, tile)
    dir_idx = _entry_direction((h, w), marker)

    mismatches = {
        (int(r), int(c))
        for r, c in np.argwhere(
            grid
            != np.fromfunction(
                lambda rr, cc: tile[rr.astype(int) % tile.shape[0], cc.astype(int) % tile.shape[1]],
                grid.shape,
                dtype=int,
            )
        )
    }
    defects = mismatches - {marker}

    path = [marker]
    r, c = marker
    seen_states: set[tuple[int, int, int]] = {(r, c, dir_idx)}
    for _ in range(h * w * 4):
        if (r, c) in defects:
            dir_idx = _defect_turn(tile, (h, w), grid, (r, c), dir_idx)
        nr = r + DIRS[dir_idx][0]
        nc = c + DIRS[dir_idx][1]
        if not _in_bounds((h, w), nr, nc):
            return path
        state = (nr, nc, dir_idx)
        if state in seen_states:
            raise AssertionError("path should exit the grid instead of looping")
        seen_states.add(state)
        r, c = nr, nc
        path.append((r, c))
    raise AssertionError("path tracing exceeded safety bound")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the entry-colored path through a periodic tiling by turning at defect cells that borrow a side neighbor's expected color.

    Input structure:
    - The whole grid is a repetition of one small rectangular color tile.
    - One border cell uses a marker color that does not belong to the tile palette and marks the ray entry.
    - A few path cells are defects: their visible color differs from the tile and instead copies the expected color from the left or right side neighbor relative to the incoming direction.

    Rule:
    - Recover the smallest row and column periods of the underlying tile and reconstruct the expected color at every position.
    - Start from the border marker and move inward perpendicular to that edge; when a visited defect cell shows the expected left-neighbor color, turn left, and when it shows the expected right-neighbor color, turn right.
    - Recolor every visited path cell, including the entry marker and all defect cells, with the marker color; keep all other cells unchanged.

    Key insight:
    - The turn instruction is hidden inside the periodic background: only after restoring the tile can you tell which mismatched cells are defects and whether each one borrowed its color from the left side or the right side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    tile = _reconstruct_tile(g)
    marker = _find_marker(g, tile)
    marker_color = int(g[marker])
    path = _trace_path(g, tile)

    out = g.copy()
    for r, c in path:
        out[r, c] = marker_color
    return out


def _has_smaller_period(tile: np.ndarray) -> bool:
    pr, pc = tile.shape
    for rr in range(1, pr + 1):
        for cc in range(1, pc + 1):
            if rr == pr and cc == pc:
                continue
            ok = True
            for r in range(pr):
                for c in range(pc):
                    if int(tile[r, c]) != int(tile[r % rr, c % cc]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _make_tile(pr: int, pc: int) -> np.ndarray | None:
    color_count = random.randint(3, min(5, len(PALETTE) - 2))
    colors = random.sample(PALETTE, color_count)
    tile = np.array([[random.choice(colors) for _ in range(pc)] for _ in range(pr)], dtype=int)
    for color in colors:
        r = random.randrange(pr)
        c = random.randrange(pc)
        tile[r, c] = color
    if len({int(v) for v in tile.ravel()}) < 3:
        return None
    if _has_smaller_period(tile):
        return None
    return tile


def _line_to_border(shape: tuple[int, int], pos: tuple[int, int], dir_idx: int) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIRS[dir_idx]
    r, c = pos
    cells: list[tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _turn_options(tile: np.ndarray, shape: tuple[int, int], pos: tuple[int, int], dir_idx: int) -> list[tuple[int, int]]:
    r, c = pos
    own = _expected_color(tile, r, c)
    left_dir = (dir_idx - 1) % 4
    right_dir = (dir_idx + 1) % 4
    lr, lc = r + DIRS[left_dir][0], c + DIRS[left_dir][1]
    rr, rc = r + DIRS[right_dir][0], c + DIRS[right_dir][1]
    if not (_in_bounds(shape, lr, lc) and _in_bounds(shape, rr, rc)):
        return []
    left_color = _expected_color(tile, lr, lc)
    right_color = _expected_color(tile, rr, rc)
    if left_color == right_color:
        return []

    options: list[tuple[int, int]] = []
    if left_color != own:
        options.append((left_dir, left_color))
    if right_color != own:
        options.append((right_dir, right_color))
    random.shuffle(options)
    return options


def _build_path(
    shape: tuple[int, int],
    tile: np.ndarray,
    pos: tuple[int, int],
    dir_idx: int,
    turns_left: int,
    visited: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], int]]] | None:
    line = _line_to_border(shape, pos, dir_idx)
    if turns_left == 0:
        if len(line) < MIN_FINAL_RUN or any(cell in visited for cell in line):
            return None
        return line, []

    if len(line) < MIN_RUN_BETWEEN_TURNS + 1:
        return None

    candidate_indices = list(range(MIN_RUN_BETWEEN_TURNS - 1, len(line)))
    random.shuffle(candidate_indices)
    for idx in candidate_indices:
        segment = line[: idx + 1]
        turn_cell = segment[-1]
        if _is_border(shape, *turn_cell):
            continue
        if any(cell in visited for cell in segment):
            continue
        visited2 = visited | set(segment)
        for new_dir, defect_color in _turn_options(tile, shape, turn_cell, dir_idx):
            nr = turn_cell[0] + DIRS[new_dir][0]
            nc = turn_cell[1] + DIRS[new_dir][1]
            if not _in_bounds(shape, nr, nc) or (nr, nc) in visited2:
                continue
            rest = _build_path(shape, tile, turn_cell, new_dir, turns_left - 1, visited2)
            if rest is None:
                continue
            rest_cells, defects = rest
            return segment + rest_cells, [(turn_cell, defect_color)] + defects
    return None


def _choose_entry(shape: tuple[int, int]) -> tuple[tuple[int, int], int]:
    h, w = shape
    options = [
        ((0, c), 2) for c in range(1, w - 1)
    ] + [
        ((h - 1, c), 0) for c in range(1, w - 1)
    ] + [
        ((r, 0), 1) for r in range(1, h - 1)
    ] + [
        ((r, w - 1), 3) for r in range(1, h - 1)
    ]
    return random.choice(options)


def _render_expected(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None] % tile.shape[0]
    cc = np.arange(w)[None, :] % tile.shape[1]
    return tile[rr, cc].copy()


def _gen_attempt() -> np.ndarray | None:
    pr = random.randint(2, 5)
    pc = random.randint(2, 5)
    row_repeats = random.randint(3, min(5, 30 // pr))
    col_repeats = random.randint(3, min(5, 30 // pc))
    shape = (pr * row_repeats, pc * col_repeats)

    tile = None
    for _ in range(60):
        tile = _make_tile(pr, pc)
        if tile is not None:
            break
    if tile is None:
        return None

    expected = _render_expected(tile, shape)
    tile_colors = {int(v) for v in tile.ravel()}
    marker_choices = [color for color in PALETTE if color not in tile_colors]
    if not marker_choices:
        return None
    marker_color = random.choice(marker_choices)

    turn_count = random.choice([1, 1, 2, 2, 3, 4])
    for _ in range(80):
        entry, dir_idx = _choose_entry(shape)
        built = _build_path(shape, tile, entry, dir_idx, turn_count, {entry})
        if built is None:
            continue
        path_cells, defect_specs = built
        defect_map = dict(defect_specs)
        if len(defect_map) != turn_count:
            continue

        inp = expected.copy()
        inp[entry] = marker_color
        for cell, color in defect_map.items():
            inp[cell] = color

        out = expected.copy()
        for r, c in [entry, *path_cells]:
            out[r, c] = marker_color

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-turning sample")
