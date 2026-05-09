# Variant of 7b5033c1
# Dimension: D10 — 符号编码参数 (meta)
# Change: 无符号编码——路径结构和颜色直接可见，无需查表或解码... → 图例指定遍历方向
#
# Design analysis
# a) Concrete behavior:
#    The original input only contains one multicolor snake. This variant adds a
#    disconnected 2-cell legend in a corner. If the legend is horizontal, the
#    output must be a 1xN row; if the legend is vertical, the output must be an
#    Nx1 column. Example: with snake colors [3, 3, 2, 4], the original task
#    always outputs [[3], [3], [2], [4]]. Here a top-corner legend [7, 6]
#    means the same traversal order becomes [[3, 3, 2, 4]], while a vertical
#    corner legend [[7], [6]] means the output stays a column. The new work is
#    therefore a front-loaded symbol-decoding step before the old tracing step.
# b) Complexity calibration:
#    The legend uses colors excluded from the snake palette and is forced into a
#    corner, so it is visually salient and cannot be confused with a short path
#    fragment. The path itself stays long (20-30 cells) with several turns, so
#    the base task remains visible instead of collapsing into a tiny toy case.
# c) Hidden assumptions from the original main():
#    - `mask = g != bg` used to mean "every non-bg cell belongs to the snake".
#      That is false now because the legend is also non-bg.
#    - Degree counting and endpoint finding assumed one connected chain. If the
#      legend is not removed first, the solver sees extra components and picks
#      the wrong endpoints.
#    - `reshape((-1, 1))` hard-coded the original column output. The legend now
#      controls whether the same color order should be laid out as a row or a
#      column.
# d) Placement legality:
#    The generator keeps the snake inside a 2-cell margin, reserving all corner
#    placements for the legend. The legend is exactly two straight cells rooted
#    at a corner, disconnected from the snake, and uses colors not used on the
#    path. This guarantees no overlap, no border ambiguity, and one valid path
#    component after the legend is removed.
# e) Visual observability:
#    A human can spot the only disconnected non-background object immediately.
#    Across samples, the snake logic stays the same but the output flips between
#    row and column according to the legend orientation, making the D10 change
#    visible rather than hidden inside an implementation detail.
# Solver assumption audit after modification:
#    - Exactly one size-2 straight component is anchored at a corner; that is
#      the legend. The generator enforces this.
#    - All remaining non-background cells form one simple 4-connected chain with
#      two endpoints. The generator verifies this by calling `main()` itself.
#    - The legend colors are irrelevant except for being two distinct colors;
#      the generator chooses them outside the snake palette to keep decoding
#      unambiguous for both solver and human observers.

from __future__ import annotations

import random
import sys
import os

import numpy as np

sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "..",
            "arc2_solver_select",
        )
    ),
)
from arc_utils import get_bg


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []

    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[Coord] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _is_corner_legend(comp: list[Coord], h: int, w: int) -> bool:
    if len(comp) != 2:
        return False

    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})

    if len(rows) == 1 and len(cols) == 2 and cols[1] == cols[0] + 1:
        return rows[0] in (0, h - 1) and cols in ([0, 1], [w - 2, w - 1])

    if len(cols) == 1 and len(rows) == 2 and rows[1] == rows[0] + 1:
        return cols[0] in (0, w - 1) and rows in ([0, 1], [h - 2, h - 1])

    return False


def _decode_legend_and_path(grid: np.ndarray, bg: int) -> tuple[bool, list[Coord]]:
    mask = grid != bg
    comps = _find_components(mask)
    h, w = grid.shape

    legends = [comp for comp in comps if _is_corner_legend(comp, h, w)]
    if len(legends) != 1:
        raise ValueError(f"expected exactly one corner legend, found {len(legends)}")

    legend = set(legends[0])
    output_is_row = len({r for r, _ in legend}) == 1

    path_mask = mask.copy()
    for r, c in legend:
        path_mask[r, c] = False

    path_comps = _find_components(path_mask)
    if len(path_comps) != 1:
        raise ValueError(f"expected exactly one snake component, found {len(path_comps)}")

    return output_is_row, path_comps[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a colored snake path preserving traversal order.

    Variant rule:
    - A disconnected 2-cell corner legend encodes the output orientation.
    - Horizontal legend => output a single row.
    - Vertical legend => output a single column.
    - After removing the legend, trace the snake exactly as in the original
      task, starting from the top-left-most endpoint.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    output_is_row, pts = _decode_legend_and_path(g, bg)
    if not pts:
        return np.zeros((1, 0), dtype=int) if output_is_row else np.zeros((0, 1), dtype=int)

    pset = set(pts)
    deg: dict[Coord, int] = {}
    for r, c in pts:
        d = 0
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pset:
                d += 1
        deg[(r, c)] = d

    ends = [p for p in pts if deg[p] == 1]
    if len(ends) != 2:
        raise ValueError(f"expected two snake endpoints, found {len(ends)}")
    start = min(ends)

    order: list[Coord] = []
    prev: Coord | None = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)

        r, c = cur
        nbrs = []
        for q in _neighbors4(r, c):
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break
        if len(nbrs) > 1:
            raise ValueError("snake branches after legend removal")

        prev, cur = cur, nbrs[0]

    if len(order) != len(pts):
        raise ValueError("failed to traverse the full snake")

    colors = np.array([int(g[r, c]) for r, c in order], dtype=int)
    return colors.reshape((1, -1)) if output_is_row else colors.reshape((-1, 1))


def _try_extend(
    path_set: set[Coord],
    last: Coord,
    dr: int,
    dc: int,
    count: int,
    h: int,
    w: int,
    margin: int,
) -> list[Coord]:
    """Try to extend the path by `count` steps in direction (dr,dc)."""
    segment: list[Coord] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < margin or nr >= h - margin or nc < margin or nc >= w - margin:
            break
        if (nr, nc) in path_set:
            break

        pred = (r + dr * (i - 1), c + dc * (i - 1))
        bad = False
        for q in _neighbors4(nr, nc):
            if q != pred and q in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    """Partition total into num_parts values each in [min_len, max_len]."""
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i in range(num_parts) if parts[i] < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def _legend_cells(h: int, w: int, output_is_row: bool, corner: str) -> list[Coord]:
    if output_is_row:
        if corner == "tl":
            return [(0, 0), (0, 1)]
        if corner == "tr":
            return [(0, w - 2), (0, w - 1)]
        if corner == "bl":
            return [(h - 1, 0), (h - 1, 1)]
        return [(h - 1, w - 2), (h - 1, w - 1)]

    if corner == "tl":
        return [(0, 0), (1, 0)]
    if corner == "tr":
        return [(0, w - 1), (1, w - 1)]
    if corner == "bl":
        return [(h - 2, 0), (h - 1, 0)]
    return [(h - 2, w - 1), (h - 1, w - 1)]


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    margin = 2

    bg = random.randint(0, 9)
    path_color_count = random.randint(3, 5)
    usable = [c for c in range(10) if c != bg]
    path_palette = random.sample(usable, path_color_count)
    remaining = [c for c in usable if c not in path_palette]
    legend_colors = random.sample(remaining, 2)

    target_len = random.randint(20, 30)
    start_r = random.randint(margin, margin + 2)
    start_c = random.randint(margin, w - margin - 1)

    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

    while len(path) < target_len:
        run_len = min(random.randint(1, 5), target_len - len(path))
        if run_len <= 0:
            break

        last = path[-1]
        segment: list[Coord] = []

        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, run_len, h, w, margin)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, h, w, margin)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, h, w, margin)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, h, w, margin)
                if segment:
                    h_dir = -h_dir
            h_dir = -h_dir

        if not segment:
            break

        path.extend(segment)
        turns += 1
        is_vertical = not is_vertical

    total = len(path)
    if total < 20 or total > 30:
        return None
    if turns < 7 or turns > 11:
        return None

    num_segments = random.randint(max(3, path_color_count), 6)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=9)
    if seg_lengths is None:
        return None

    shuffled_palette = list(path_palette)
    random.shuffle(shuffled_palette)
    seg_colors = list(shuffled_palette[: min(len(shuffled_palette), num_segments)])
    while len(seg_colors) < num_segments:
        choices = [c for c in path_palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in path_palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)

    output_is_row = random.random() < 0.5
    corner = random.choice(["tl", "tr", "bl", "br"])

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    for (r, c), col in zip(_legend_cells(h, w, output_is_row, corner), legend_colors):
        grid[r, c] = col

    expected = np.array(colors, dtype=int).reshape((1, -1) if output_is_row else (-1, 1))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 300 attempts")
