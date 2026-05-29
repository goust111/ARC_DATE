# Combo variant of 7b5033c1 (index 2)
# combo_spec: D1/0+D10/0
# Name: 图例方向+折叠矩阵
#
# Design analysis
# a) Concrete fused behavior:
#    Source task: trace the single 4-connected multicolor snake from the
#    top-left-most endpoint and output the colors as an N x 1 column.
#    Fused task: keep the same tracing order, but first decode a disconnected
#    2-cell corner legend. If that legend is horizontal, fold the color stream
#    across rows; if it is vertical, fold the same stream down columns. The fold
#    span K is the number of distinct snake colors, inherited from D1.
#    Example with traced colors [1,1,2,3,3,2,4,4,1,2,3,4] and K = 4:
#      original output:
#        [[1],[1],[2],[3],[3],[2],[4],[4],[1],[2],[3],[4]]
#      fused with horizontal legend:
#        [[1,1,2,3],
#         [4,2,3,3],
#         [4,4,1,2]]
#      fused with vertical legend:
#        [[1,2,4],
#         [1,3,4],
#         [2,3,1],
#         [3,2,4]]
#    The difference is not only "extra metadata exists"; the legend changes how
#    the same traced 1-D sequence is grouped into a 2-D snake-like matrix.
# b) Complexity calibration:
#    K is limited to 3..5 so the matrix is small enough to infer from 2-4 ARC
#    examples, but still nontrivial. Total snake length is chosen from 20..30
#    and must be divisible by K, giving the complementary dimension at least 4.
#    That avoids degenerate 1-row/1-column or tiny 2xN cases where row-vs-column
#    folding would be visually weak.
# c) Implicit-assumption audit of the original main():
#    - `mask = g != bg` no longer means "all non-bg cells belong to the snake";
#      the corner legend must be separated first.
#    - Endpoint detection and path traversal still assume one simple chain; that
#      remains valid only after removing the legend component.
#    - `start = min(ends)` still works because the snake reading direction itself
#      is unchanged by the fusion.
#    - The old `reshape((-1, 1))` hard-coded a column. D1 breaks that by folding
#      into a rectangle, and D10 adds a prior choice between row-wise and
#      column-wise folding.
#    - New interaction check: the fold span must be derived from snake colors
#      only, not legend colors, otherwise the extra symbol would corrupt K.
# d) Placement legality:
#    The generator reserves a 2-cell border margin for the path and places the
#    legend as a straight 2-cell component anchored in a corner. This prevents
#    overlap, keeps the legend disconnected, and leaves exactly one non-legend
#    component for the solver to trace.
# e) Visual observability:
#    A human sees one large serpentine object plus one tiny corner marker. Across
#    examples, the marker flips between horizontal and vertical, and the output
#    correspondingly alternates between row-folded and column-folded matrices.
#    Because K also varies with the number of snake colors, the solver must infer
#    both "which object is metadata" and "how the matrix dimensions are chosen."
# f) Visual cleanliness:
#    The legend stays in a corner using colors excluded from the snake palette.
#    The path stays at least one background cell away from every border, so the
#    two structures do not stack or visually compete in the same region.
# g) Color-change inferability:
#    No hidden recoloring is introduced. Output colors are copied directly from
#    the snake trace; only spatial arrangement changes. K depends on the visible
#    count of distinct snake colors, and the legend colors never participate in
#    that count.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: exactly one size-2 straight corner component is present. The
#      generator creates exactly one such legend.
#    - Assumption: after legend removal there is exactly one simple snake with two
#      endpoints and no branches. `_try_extend` enforces non-self-touching growth.
#    - Assumption: traced snake length is divisible by K. The generator chooses
#      total length from multiples of K only.
#    - Assumption: the legend orientation matters. The generator rejects samples
#      where row-folded and column-folded outputs would accidentally be identical.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

    legend = legends[0]
    fold_by_rows = len({r for r, _ in legend}) == 1

    path_mask = mask.copy()
    for r, c in legend:
        path_mask[r, c] = False

    path_comps = _find_components(path_mask)
    if len(path_comps) != 1:
        raise ValueError(f"expected exactly one snake component, found {len(path_comps)}")

    return fold_by_rows, path_comps[0]


def _trace_path(pts: list[Coord]) -> list[Coord]:
    if not pts:
        return []

    pset = set(pts)
    deg: dict[Coord, int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    if any(d > 2 for d in deg.values()):
        raise ValueError("snake component branches")

    ends = [p for p in pts if deg[p] == 1]
    if len(ends) != 2:
        raise ValueError(f"expected two snake endpoints, found {len(ends)}")

    order: list[Coord] = []
    prev: Coord | None = None
    cur = min(ends)
    for _ in range(len(pts)):
        order.append(cur)
        nbrs = [q for q in _neighbors4(*cur) if q in pset and q != prev]
        if not nbrs:
            break
        if len(nbrs) != 1:
            raise ValueError("snake traversal became ambiguous")
        prev, cur = cur, nbrs[0]

    if len(order) != len(pts):
        raise ValueError("failed to traverse full snake")
    return order


def _fold_colors(colors: list[int], span: int, fold_by_rows: bool) -> np.ndarray:
    if span <= 0:
        raise ValueError("fold span must be positive")
    if len(colors) % span != 0:
        raise ValueError("color stream length must be divisible by fold span")

    other = len(colors) // span
    if fold_by_rows:
        out = np.zeros((other, span), dtype=int)
        idx = 0
        for r in range(other):
            chunk = colors[idx : idx + span]
            idx += span
            out[r] = chunk if r % 2 == 0 else chunk[::-1]
        return out

    out = np.zeros((span, other), dtype=int)
    idx = 0
    for c in range(other):
        chunk = colors[idx : idx + span]
        idx += span
        out[:, c] = chunk if c % 2 == 0 else chunk[::-1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the corner legend, trace the snake from the top-left endpoint, then
    fold the color stream into a matrix. Horizontal legend => fold across rows.
    Vertical legend => fold down columns. The fold span is the count of distinct
    snake colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    fold_by_rows, pts = _decode_legend_and_path(g, bg)
    order = _trace_path(pts)
    if not order:
        return np.zeros((0, 0), dtype=int)

    colors = [int(g[r, c]) for r, c in order]
    span = len(set(colors))
    return _fold_colors(colors, span, fold_by_rows)


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


def _build_color_stream(total: int, palette: list[int]) -> list[int] | None:
    color_count = len(palette)
    min_segments = max(color_count + 1, 6)
    max_segments = min(9, total // 2)
    if min_segments > max_segments:
        return None

    num_segments = random.randint(min_segments, max_segments)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=7)
    if seg_lengths is None:
        return None

    seg_colors = palette[:]
    random.shuffle(seg_colors)
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)
    if len(set(colors)) != color_count:
        return None
    return colors


def _legend_cells(h: int, w: int, fold_by_rows: bool, corner: str) -> list[Coord]:
    if fold_by_rows:
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
    color_count = random.randint(3, 5)
    usable = [c for c in range(10) if c != bg]
    path_palette = random.sample(usable, color_count)
    legend_palette = [c for c in usable if c not in path_palette]
    legend_colors = random.sample(legend_palette, 2)

    target_lengths = [n for n in range(20, 31) if n % color_count == 0 and n // color_count >= 4]
    target_len = random.choice(target_lengths)

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
    if total != target_len:
        return None
    if turns < 6 or turns > 12:
        return None

    colors = _build_color_stream(total, path_palette)
    if colors is None:
        return None

    fold_by_rows = random.random() < 0.5
    corner = random.choice(["tl", "tr", "bl", "br"])
    expected = _fold_colors(colors, color_count, fold_by_rows)
    other = _fold_colors(colors, color_count, not fold_by_rows)
    if np.array_equal(expected, other):
        return None

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col
    for (r, c), col in zip(_legend_cells(h, w, fold_by_rows, corner), legend_colors):
        grid[r, c] = col

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
