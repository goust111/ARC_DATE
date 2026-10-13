from __future__ import annotations

from collections import Counter
import random

import numpy as np


GEN_BG = 0
GEN_GREY = 5
GEN_PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _infer_starts(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    starts = [(0, c, int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) != bg]
    assert starts, "expected at least one top-edge path start"
    start_colors = [color for _r, _c, color in starts]
    assert len(set(start_colors)) == len(start_colors), "top-edge path colors must be unique"
    return starts


def _infer_grey(grid: np.ndarray, bg: int, start_colors: list[int]) -> int | None:
    colors = {int(color) for color in np.unique(grid)}
    candidates = sorted(colors - {bg} - set(start_colors))
    if not candidates:
        return None
    assert len(candidates) == 1, f"expected one intersection color, got {candidates}"
    return candidates[0]


def _grey_supports_color(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    horizontal = 0 < c < w - 1 and int(grid[r, c - 1]) == color and int(grid[r, c + 1]) == color
    vertical = 0 < r < h - 1 and int(grid[r - 1, c]) == color and int(grid[r + 1, c]) == color
    return horizontal or vertical


def _trace_path(grid: np.ndarray, start: tuple[int, int], color: int, grey: int | None) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        r, c = cur
        candidates: list[tuple[int, int]] = []
        for nr, nc in _neighbors4((h, w), r, c):
            if prev is not None and (nr, nc) == prev:
                continue
            value = int(grid[nr, nc])
            if value == color:
                candidates.append((nr, nc))
                continue
            if grey is not None and value == grey and _grey_supports_color(grid, nr, nc, color):
                candidates.append((nr, nc))

        if not candidates:
            break

        assert len(candidates) == 1, f"path of color {color} branches or is ambiguous at {cur}: {candidates}"
        nxt = candidates[0]
        assert nxt not in visited, f"path of color {color} forms a loop at {nxt}"
        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt
        assert len(path) <= h * w, "path trace exceeded grid area"

    end_r, end_c = path[-1]
    assert end_r in {0, h - 1} or end_c in {0, w - 1}, "path must terminate on the grid edge"
    return path


def _trace_all_paths(grid: np.ndarray) -> tuple[int, int | None, list[int], list[int], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    starts = _infer_starts(g, bg)
    start_cols = [c for _r, c, _color in starts]
    path_colors = [color for _r, _c, color in starts]
    grey = _infer_grey(g, bg, path_colors)

    paths: list[list[tuple[int, int]]] = []
    all_path_cells: set[tuple[int, int]] = set()
    grey_membership: Counter[tuple[int, int]] = Counter()
    for (_r, c, color) in starts:
        start = (0, c)
        path = _trace_path(g, start, color, grey)
        paths.append(path)

        path_color_cells = {pos for pos in path if int(g[pos]) == color}
        expected_color_cells = set(map(tuple, np.argwhere(g == color)))
        assert path_color_cells == expected_color_cells, f"failed to recover full path for color {color}"

        for pos in path:
            all_path_cells.add(pos)
            if grey is not None and int(g[pos]) == grey:
                grey_membership[pos] += 1

    non_bg_cells = set(map(tuple, np.argwhere(g != bg)))
    assert all_path_cells == non_bg_cells, "every non-background cell must belong to a traced path"

    if grey is not None:
        expected_grey = set(map(tuple, np.argwhere(g == grey)))
        assert set(grey_membership) == expected_grey, "every grey cell must be traversed by traced paths"
        assert all(count == 2 for count in grey_membership.values()), "every grey cell must belong to exactly two paths"

    return bg, grey, path_colors, start_cols, paths


def _resolve_crossings(
    paths: list[list[tuple[int, int]]],
    path_colors: list[int],
) -> tuple[dict[tuple[int, int], int], list[dict[str, object]]]:
    cross_to_paths: dict[tuple[int, int], list[int]] = {}
    for path_idx, path in enumerate(paths):
        for pos in path:
            cross_to_paths.setdefault(pos, []).append(path_idx)
    cross_to_paths = {pos: pids for pos, pids in cross_to_paths.items() if len(pids) > 1}

    encounter_orders: list[dict[tuple[int, int], int]] = []
    for path in paths:
        order: dict[tuple[int, int], int] = {}
        seen = 0
        for pos in path:
            if pos in cross_to_paths:
                seen += 1
                order[pos] = seen
        encounter_orders.append(order)

    winners: dict[tuple[int, int], int] = {}
    metadata: list[dict[str, object]] = []
    for pos, pids in cross_to_paths.items():
        assert len(pids) == 2, "each crossing must involve exactly two paths"
        up_flags = [encounter_orders[pid][pos] % 2 == 0 for pid in pids]
        if up_flags[0] != up_flags[1]:
            winner_pid = pids[0] if up_flags[0] else pids[1]
        else:
            winner_pid = min(pids)
        winners[pos] = path_colors[winner_pid]
        metadata.append(
            {
                "coord": pos,
                "tie": up_flags[0] == up_flags[1],
                "winner_pid": winner_pid,
            }
        )
    return winners, metadata


def main(grid: np.ndarray) -> np.ndarray:
    """Resolve grey path crossings into a woven over-under pattern.

    Input structure:
    - bg is the most common color.
    - Several different-colored 1-pixel-wide orthogonal paths start on the top edge and continue to another edge, possibly with 90-degree turns.
    - When two paths cross, their shared pixel is grey instead of either path color.

    Rule:
    - Trace each path from its top-edge start to its endpoint and record the grey crossings it encounters in path order.
    - For a given path, its 1st, 3rd, 5th, ... crossings are "under" and its 2nd, 4th, 6th, ... crossings are "over".
    - At each grey crossing, repaint the pixel with the color of the path that is "over"; if both paths get the same over/under status there, the path whose start is farther left on the top edge wins.
    - All non-grey pixels stay unchanged.

    Key insight:
    - A crossing is not decided by a global path priority: each path alternates under, over, under, over along its own traversal order, so solving the puzzle requires reconstructing every path and counting its crossings.
    """
    g = np.asarray(grid, dtype=int)
    _bg, grey, path_colors, _start_cols, paths = _trace_all_paths(g)
    if grey is None:
        return g.copy()

    winners, _metadata = _resolve_crossings(paths, path_colors)
    out = g.copy()
    for (r, c), color in winners.items():
        out[r, c] = color
    return out


def _build_path(start_col: int, turn_row: int, end_col: int, height: int) -> list[tuple[int, int]]:
    path = [(r, start_col) for r in range(turn_row + 1)]
    step = 1 if end_col > start_col else -1
    for col in range(start_col + step, end_col + step, step):
        path.append((turn_row, col))
    for row in range(turn_row + 1, height):
        path.append((row, end_col))
    return path


def _path_axis_at(path: list[tuple[int, int]], index: int) -> str:
    axes = set()
    r, c = path[index]
    if index > 0:
        pr, pc = path[index - 1]
        axes.add("h" if pr == r else "v")
    if index + 1 < len(path):
        nr, nc = path[index + 1]
        axes.add("h" if nr == r else "v")
    assert len(axes) == 1, "shared cells cannot be turns or endpoints"
    return next(iter(axes))


def _layout_is_valid(paths: list[list[tuple[int, int]]]) -> bool:
    pos_to_paths: dict[tuple[int, int], list[int]] = {}
    index_maps = [{pos: idx for idx, pos in enumerate(path)} for path in paths]
    for path_idx, path in enumerate(paths):
        for pos in path:
            pos_to_paths.setdefault(pos, []).append(path_idx)

    shared = [pos for pos, pids in pos_to_paths.items() if len(pids) > 1]
    if not shared:
        return False

    for pos in shared:
        pids = pos_to_paths[pos]
        if len(pids) != 2:
            return False
        axis0 = _path_axis_at(paths[pids[0]], index_maps[pids[0]][pos])
        axis1 = _path_axis_at(paths[pids[1]], index_maps[pids[1]][pos])
        if axis0 == axis1:
            return False

    for a_idx, a in enumerate(shared):
        for b in shared[a_idx + 1 :]:
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                return False
    return True


def _render_input(
    shape: tuple[int, int],
    paths: list[list[tuple[int, int]]],
    path_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, GEN_BG, dtype=int)
    pos_to_paths: dict[tuple[int, int], list[int]] = {}
    for path_idx, path in enumerate(paths):
        for pos in path:
            pos_to_paths.setdefault(pos, []).append(path_idx)

    for pos, pids in pos_to_paths.items():
        grid[pos] = GEN_GREY if len(pids) > 1 else path_colors[pids[0]]
    return grid


def _sample_columns(width: int, count: int) -> list[int] | None:
    offset = random.randint(0, 1)
    candidates = list(range(1 + offset, width - 1, 2))
    if len(candidates) < count:
        return None
    return random.sample(candidates, count)


def _sample_rows(height: int, count: int) -> list[int] | None:
    offset = random.randint(0, 1)
    candidates = list(range(2 + offset, height - 2, 2))
    if len(candidates) < count:
        return None
    return random.sample(candidates, count)


def _gen_attempt(mode: str) -> np.ndarray | None:
    path_count = 2 if mode == "tie" else random.randint(3, 5)
    min_h = max(12, 2 * path_count + 6)
    min_w = max(13, 4 * path_count + 1)
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)

    all_cols = _sample_columns(width, 2 * path_count)
    turn_rows = _sample_rows(height, path_count)
    if all_cols is None or turn_rows is None:
        return None

    all_cols = list(all_cols)
    start_cols = sorted(random.sample(all_cols, path_count))
    end_cols = [col for col in all_cols if col not in set(start_cols)]
    random.shuffle(end_cols)
    random.shuffle(turn_rows)

    path_colors = random.sample(GEN_PATH_COLORS, path_count)
    paths = [_build_path(start_cols[i], turn_rows[i], end_cols[i], height) for i in range(path_count)]
    if not _layout_is_valid(paths):
        return None

    _winners, metadata = _resolve_crossings(paths, path_colors)
    tie_count = sum(1 for item in metadata if bool(item["tie"]))
    non_tie_count = len(metadata) - tie_count

    if mode == "tie":
        if len(metadata) != 1 or tie_count != 1:
            return None
    else:
        if non_tie_count < 1:
            return None

    inp = _render_input((height, width), paths, path_colors)
    if int(np.count_nonzero(inp != GEN_BG)) >= inp.size // 2:
        return None

    if not np.array_equal(main(inp), _expected_output(inp, paths, path_colors)):
        return None
    return inp


def _expected_output(
    inp: np.ndarray,
    paths: list[list[tuple[int, int]]],
    path_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    winners, _metadata = _resolve_crossings(paths, path_colors)
    for pos, color in winners.items():
        out[pos] = color
    return out


def generate_input() -> np.ndarray:
    mode = random.choice(["tie", "mixed"])
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid {mode} weaving-path input")
