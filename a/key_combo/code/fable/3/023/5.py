from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
COLORS = tuple(range(1, 10))
MAX_GEN_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_start(grid: np.ndarray) -> tuple[int, int, int]:
    h, _w = grid.shape
    bottom_cols = np.argwhere(grid[h - 1] != BG).ravel()
    assert bottom_cols.size == 1, "expected exactly one colored start pixel on the bottom row"
    c = int(bottom_cols[0])
    return h - 1, c, int(grid[h - 1, c])


def _trace(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[Coord, Coord, int]], list[Coord]]:
    """Simulate the colored bouncing ray and return output plus path metadata."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 2, "grid must be at least 2x2"

    sr, sc, color = _find_start(g)
    blockers: dict[Coord, int] = {}
    for r, c in np.argwhere(g != BG):
        rr, cc = int(r), int(c)
        if (rr, cc) != (sr, sc):
            blockers[(rr, cc)] = int(g[rr, cc])

    out = g.copy()
    r, c = sr, sc
    dc = 1
    out[r, c] = color
    path: list[tuple[int, int, int]] = [(int(r), int(c), int(color))]
    hits: list[tuple[Coord, Coord, int]] = []
    wall_bounces: list[Coord] = []

    event_limit = h * w * 4
    events = 0
    while r > 0:
        events += 1
        assert events <= event_limit, "ray got trapped between blockers and a wall"

        if c + dc < 0 or c + dc >= w:
            dc = -dc
            wall_bounces.append((int(r), int(c)))

        nr, nc = int(r - 1), int(c + dc)
        assert 0 <= nr < h and 0 <= nc < w, "reflected step should remain in bounds"

        if (nr, nc) in blockers:
            color = int(blockers[(nr, nc)])
            hits.append(((int(r), int(c)), (int(nr), int(nc)), int(color)))
            dc = -dc
            continue

        r, c = nr, nc
        out[r, c] = int(color)
        path.append((int(r), int(c), int(color)))

    return out, path, hits, wall_bounces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a diagonally bouncing light ray that changes color whenever it strikes a blocker pixel.

    Input structure:
    - A black background contains one colored start pixel on the bottom row.
    - Every other colored cell is an isolated one-pixel blocker inside the grid.
    - Most blockers are distractors that are not struck by the final ray.

    Rule:
    - Start at the bottom pixel, moving one row upward and one column to the right each step.
    - Paint each entered cell with the current color, initially the start color.
    - If the next step would leave the left or right edge, reverse the horizontal direction before moving.
    - If the next diagonal cell is a blocker, do not enter it; reverse horizontally and use that blocker color from then on.
    - Stop after painting a cell in the top row, while preserving every blocker pixel.

    Key insight:
    The ray path and the color sequence must be simulated in time order because each struck blocker both redirects the future trajectory and supplies the future paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert _get_bg(g) == BG, "expected a black background"
    out, _path, _hits, _walls = _trace(g)
    return out


# --------------- generator ---------------


def _cell_clear_for_blocker(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (0 < int(r) < h - 1 and 0 <= int(c) < w):
        return False
    if int(grid[r, c]) != BG:
        return False
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[rr, cc]) != BG:
                return False
    return True


def _trace_signature(grid: np.ndarray) -> tuple[tuple[tuple[int, int, int], ...], tuple[Coord, ...], int]:
    _out, path, hits, walls = _trace(grid)
    return (
        tuple((int(r), int(c), int(col)) for r, c, col in path),
        tuple((int(br), int(bc)) for _corner, (br, bc), _col in hits),
        len(walls),
    )


def _wall_only_path(shape: tuple[int, int], start_col: int, start_color: int) -> list[tuple[int, int, int]]:
    h, w = shape
    r, c = h - 1, int(start_col)
    dc = 1
    path = [(int(r), int(c), int(start_color))]
    while r > 0:
        if c + dc < 0 or c + dc >= w:
            dc = -dc
        r -= 1
        c += dc
        path.append((int(r), int(c), int(start_color)))
    return path


def _place_path_blockers(
    grid: np.ndarray,
    target_hits: int,
    palette: list[int],
) -> bool:
    h, w = grid.shape
    start_r, start_c, color = _find_start(grid)
    r, c = start_r, start_c
    dc = 1
    hit_count = 0
    moves_since_hit = 2
    events = 0

    while r > 0 and events < h * w * 3:
        events += 1
        if c + dc < 0 or c + dc >= w:
            dc = -dc

        nr, nc = int(r - 1), int(c + dc)
        opposite_c = int(c - dc)
        can_hit = (
            hit_count < target_hits
            and moves_since_hit >= 2
            and 0 < nr < h - 1
            and 0 <= opposite_c < w
            and int(grid[nr, opposite_c]) == BG
            and _cell_clear_for_blocker(grid, nr, nc)
        )

        remaining_rows = int(r)
        remaining_hits = int(target_hits - hit_count)
        force = can_hit and remaining_hits > 0 and remaining_rows <= remaining_hits * 4 + 3
        if can_hit and (force or random.random() < 0.34):
            choices = [col for col in palette if int(col) != int(color)]
            blocker_color = int(random.choice(choices))
            grid[nr, nc] = blocker_color
            color = blocker_color
            dc = -dc
            hit_count += 1
            moves_since_hit = 0
            continue

        blocker_positions = {
            (int(rr), int(cc))
            for rr, cc in np.argwhere(grid != BG)
            if (int(rr), int(cc)) != (start_r, start_c)
        }
        if (nr, nc) in blocker_positions:
            return False

        r, c = nr, nc
        moves_since_hit += 1

    return hit_count == target_hits


def _try_add_distractor(
    grid: np.ndarray,
    r: int,
    c: int,
    color: int,
    base_sig: tuple[tuple[tuple[int, int, int], ...], tuple[Coord, ...], int],
) -> bool:
    if not _cell_clear_for_blocker(grid, int(r), int(c)):
        return False
    trial = grid.copy()
    trial[int(r), int(c)] = int(color)
    try:
        sig = _trace_signature(trial)
    except AssertionError:
        return False
    if sig != base_sig:
        return False
    grid[int(r), int(c)] = int(color)
    return True


def _add_distractors(grid: np.ndarray, palette: list[int], min_count: int, max_count: int) -> bool:
    base_sig = _trace_signature(grid)
    path_cells = {(int(r), int(c)) for r, c, _col in base_sig[0]}
    _start_r, start_c, start_color = _find_start(grid)
    wall_path = _wall_only_path(grid.shape, start_c, start_color)
    wall_only_decoys = [
        (r, c)
        for r, c, _col in wall_path
        if 0 < int(r) < grid.shape[0] - 1
        and (int(r), int(c)) not in path_cells
        and _cell_clear_for_blocker(grid, int(r), int(c))
    ]
    random.shuffle(wall_only_decoys)

    added = 0
    for r, c in wall_only_decoys[:4]:
        if _try_add_distractor(grid, r, c, random.choice(palette), base_sig):
            added += 1
            break

    target = random.randint(int(min_count), int(max_count))
    attempts = 0
    while added < target and attempts < 500:
        attempts += 1
        r = random.randint(1, grid.shape[0] - 2)
        c = random.randint(0, grid.shape[1] - 1)
        if (int(r), int(c)) in path_cells:
            continue
        if _try_add_distractor(grid, r, c, random.choice(palette), base_sig):
            added += 1

    return added >= min_count


def _segments(path: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    last = int(path[0][2])
    count = 0
    for _r, _c, color in path:
        color = int(color)
        if color == last:
            count += 1
        else:
            runs.append((last, count))
            last = color
            count = 1
    runs.append((last, count))
    return runs


def _semantic_valid(grid: np.ndarray) -> bool:
    try:
        out, path, hits, walls = _trace(grid)
    except AssertionError:
        return False

    h, w = grid.shape
    if max(h, w) > 30 or _get_bg(grid) != BG:
        return False
    if len(hits) < 2 or len(walls) < 1:
        return False
    if len(_segments(path)) < 3:
        return False

    start = _find_start(grid)[:2]
    blocker_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(grid != BG)
        if (int(r), int(c)) != start
    }
    hit_cells = {(int(br), int(bc)) for _corner, (br, bc), _col in hits}
    if not hit_cells <= blocker_cells:
        return False
    if len(blocker_cells - hit_cells) <= len(hit_cells):
        return False

    path_cells = {(int(r), int(c)) for r, c, _col in path}
    if path_cells & blocker_cells:
        return False

    for cell in blocker_cells:
        if int(out[cell]) != int(grid[cell]):
            return False

    # Each hit must both reflect the next move and supply the next visible path color.
    path_by_cell = {(int(r), int(c)): int(col) for r, c, col in path}
    for corner, blocked, hit_color in hits:
        cr, cc = corner
        br, bc = blocked
        reflected_c = int(cc - (bc - cc))
        next_cell = (int(cr - 1), reflected_c)
        if next_cell in path_by_cell and path_by_cell[next_cell] != int(hit_color):
            return False

    bottom_nonzero = int(np.count_nonzero(grid[h - 1] != BG))
    top_nonzero = int(np.count_nonzero(grid[0] != BG))
    return bottom_nonzero == 1 and top_nonzero == 0


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 23)
    w = random.randint(6, 13)
    start_col = random.randint(0, w - 1)
    palette = random.sample(list(COLORS), random.randint(4, 7))
    start_color = int(palette[0])
    target_hits = random.choice([2, 2, 3, 3, 4])

    grid = np.zeros((h, w), dtype=int)
    grid[h - 1, start_col] = start_color

    if not _place_path_blockers(grid, target_hits, palette):
        return None

    try:
        _out, _path, hits, walls = _trace(grid)
    except AssertionError:
        return None
    if len(hits) != target_hits or len(walls) < 1:
        return None

    min_distractors = len(hits) + 1
    max_distractors = min_distractors + random.randint(2, 6)
    if not _add_distractors(grid, palette, min_distractors, max_distractors):
        return None

    if not _semantic_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dyed pinball input")
