# Variant of 7b5033c1
# Dimension: D5 — 输出网格形状 (grid_shape)
# Change: tight crop 为 Nx1 单列（N = 路径像素数）→ 按颜色段分行输出

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the D5 variant
# - Concretization:
#   The input is still one multicolor 4-connected snake. The original solver reads
#   it from the top-left endpoint and emits a single vertical strip like
#   [1,1,1,3,3,2]. This variant keeps the same reading order, but repacks each
#   maximal same-color run into its own row. Example:
#   original sequence [1,1,1,3,3,2,2,2,4] -> original output 9x1;
#   variant output becomes
#       [[1,1,1],
#        [3,3,bg],
#        [2,2,2],
#        [4,bg,bg]]
#   so the changed step is only the final layout stage after path tracing.
# - Complexity calibration:
#   If every run had the same length, the new layout would still be multi-row but
#   would lose the visibly "ragged packed table" effect. The generator therefore
#   forces at least four runs and at least two distinct run lengths.
# - Implicit-assumption audit:
#   The path-tracing code still assumes a single simple chain with exactly two
#   degree-1 endpoints and no branches. That old assumption remains valid. The
#   changed part is the old final reshape((-1, 1)): that line depended on the old
#   invariant "output is always one column". In the variant it must be replaced by
#   run extraction + 2D packing. Another subtle assumption is that background
#   padding is safe; this remains valid because the generator never uses bg on the
#   snake itself, so padded cells are distinguishable from true segment cells.
# - Placement legality:
#   The generator keeps the original self-avoiding growth logic. `_try_extend`
#   rejects out-of-bounds moves, revisits, and side-touching cells, so the result
#   stays a legal 4-connected path with no overlaps and no accidental branches.
# - Visual observability:
#   Across examples, humans can still see the snake being read in endpoint order,
#   but now the output is a compact rectangular table whose row lengths match the
#   snake's color runs. The enforced length variation makes the D5 change visible.
# - Source-task baseline:
#   I read `/home/lixinyu/snap/new_code/solvers/arc2/58f5dbd5.py` for the
#   packed-grid inspiration. That solver preserves object order while repacking
#   items into a compact 2D arrangement; this variant applies the same idea to
#   ordered color runs instead of carved blocks.


def _trace_path_cells(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Return snake cells in traversal order from the top-left endpoint."""
    mask = g != int(bg)
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not pts:
        return []

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        d = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in pset:
                d += 1
        deg[(r, c)] = int(d)

    ends = [p for p in pts if int(deg[p]) == 1]
    assert len(ends) == 2
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)
        r, c = cur
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts)
    return order


def _extract_color_sequence(g: np.ndarray) -> tuple[int, list[int]]:
    bg = int(get_bg(g))
    order = _trace_path_cells(g, bg)
    cols = [int(g[r, c]) for r, c in order]
    return bg, cols


def _split_color_runs(cols: list[int]) -> list[tuple[int, int]]:
    """Split traversal colors into maximal contiguous runs."""
    if not cols:
        return []

    runs: list[tuple[int, int]] = []
    cur_col = int(cols[0])
    cur_len = 1
    for col in cols[1:]:
        col = int(col)
        if col == cur_col:
            cur_len += 1
        else:
            runs.append((int(cur_col), int(cur_len)))
            cur_col = int(col)
            cur_len = 1
    runs.append((int(cur_col), int(cur_len)))
    return runs


def _pack_runs(runs: list[tuple[int, int]], bg: int) -> np.ndarray:
    """
    Pack color runs into a compact table.

    New solver assumptions:
    - Each row corresponds to one maximal run in traversal order.
    - Rows are left-aligned; the unused suffix is bg padding.
    - Generator guarantees adjacent runs have different colors, so the solver's
      recovered run boundaries match the intended segment boundaries exactly.
    """
    if not runs:
        return np.zeros((0, 0), dtype=int)

    out_h = int(len(runs))
    out_w = int(max(length for _color, length in runs))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for r, (color, length) in enumerate(runs):
        out[int(r), : int(length)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a colored snake into a run-packed 2D table.

    Input structure:
    - bg = most common color.
    - Non-bg pixels form one simple 4-connected path with two endpoints.
    - Traversing from the top-left endpoint yields a 1D color sequence.

    Variant rule:
    - Keep the same traversal order as the original task.
    - Split the 1D sequence into maximal contiguous same-color runs.
    - Output a KxM grid where K = number of runs and M = longest run length.
    - Row i contains the i-th run left-aligned; remaining cells are bg.
    """
    g = np.array(grid, dtype=int)
    bg, cols = _extract_color_sequence(g)
    runs = _split_color_runs(cols)
    return _pack_runs(runs, bg)


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Try to extend by `count` steps; reject revisits and side-touching cells."""
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = int(r + dr * i), int(c + dc * i)
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in path_set:
            break

        pred = (int(r + dr * (i - 1)), int(c + dc * (i - 1)))
        bad = False
        for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (int(nr + ddr), int(nc + ddc))
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    """Partition total into bounded positive integers with visible length variety."""
    if num_parts * min_len > total or num_parts * max_len < total:
        return None

    for _ in range(80):
        parts = [int(min_len)] * int(num_parts)
        remaining = int(total - sum(parts))
        for _ in range(remaining):
            eligible = [i for i in range(num_parts) if parts[i] < max_len]
            if not eligible:
                break
            parts[random.choice(eligible)] += 1
        if sum(parts) != total:
            continue
        if len(set(parts)) < 2:
            continue
        random.shuffle(parts)
        return [int(p) for p in parts]
    return None


def _build_expected_output(colors: list[int], bg: int) -> np.ndarray:
    return _pack_runs(_split_color_runs(colors), bg)


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], non_bg_count)

    target_len = random.randint(22, 30)

    start_r = random.randint(1, 4)
    start_c = random.randint(1, W - 2)

    path = [(int(start_r), int(start_c))]
    path_set = {(int(start_r), int(start_c))}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

    while len(path) < target_len:
        run_len = random.randint(1, 5)
        run_len = min(int(run_len), int(target_len - len(path)))
        if run_len <= 0:
            break

        last = path[-1]
        segment: list[tuple[int, int]] = []

        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, H, W)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, H, W)
                if segment:
                    h_dir = -h_dir
            h_dir = -h_dir

        if not segment:
            break

        path.extend(segment)
        turns += 1
        is_vertical = not is_vertical

    total = len(path)
    if total < 22 or total > 30:
        return None
    if turns < 7 or turns > 11:
        return None

    num_segments = random.randint(max(4, non_bg_count), min(7, total // 2))
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=8)
    if seg_lengths is None:
        return None

    shuffled_palette = list(palette)
    random.shuffle(shuffled_palette)
    seg_colors = list(shuffled_palette[: min(len(shuffled_palette), num_segments)])
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([int(col)] * int(seg_len))

    grid = np.full((H, W), int(bg), dtype=int)
    for (r, c), col in zip(path, colors):
        grid[int(r), int(c)] = int(col)

    expected_out = _build_expected_output(colors, bg)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 300 attempts")
