# Combo variant of 7b5033c1 (index 17)
# combo_spec: D2/2+D10/0
# Name: 标记起点+图例方向

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import get_bg


# Design analysis for combo D2/2 + D10/0
# a) Concrete behavior and trace:
#    - The original puzzle has one multicolor 4-connected snake. The solver finds the
#      two degree-1 endpoints, chooses the top-left endpoint, traces the chain, and
#      always returns an N x 1 column.
#    - The combo input adds two disconnected symbols at once:
#      1) one isolated marker cell near exactly one endpoint, which chooses the start;
#      2) one 2-cell straight legend anchored at a corner, whose orientation chooses
#         the output layout (horizontal legend -> 1 x N row, vertical legend -> N x 1 column).
#    - Small example: if the snake colors along one endpoint-to-endpoint traversal are
#      [4, 4, 6, 8], the original solver outputs [[4], [4], [6], [8]]. If the marker
#      sits nearer the opposite endpoint and the corner legend is horizontal, the combo
#      solver must first reverse the traversal to [8, 6, 4, 4], then reshape to
#      [[8, 6, 4, 4]]. The differences are therefore independent but fused: one symbol
#      controls order, the other controls layout.
# b) Complexity calibration:
#    - The snake remains long (20-30 cells) with many turns, so the base visual pattern
#      is still the main object.
#    - The legend is restricted to an exact 2-cell corner pattern. The marker is a lone
#      off-path pixel near one endpoint. These are easy to spot and visually distinct,
#      but neither symbol carries too many degrees of freedom.
#    - The generator rejects cases where both controls accidentally match the original
#      task (marker selects the default top-left endpoint and legend asks for a column),
#      so accepted samples always exhibit at least one observable combo effect.
# c) Hidden assumption audit against the original solver:
#    - `mask = g != bg` no longer means "all foreground is the snake"; it now includes
#      a corner legend and a start marker.
#    - Degree counting cannot run on all foreground cells, because the legend and marker
#      would create extra components and break endpoint detection.
#    - `start = min(ends)` is no longer valid; the marker decides which endpoint to use.
#    - `reshape((-1, 1))` is no longer valid; the legend orientation decides row vs column.
#    - After fusion, solver order matters: first remove the corner legend, then identify
#      the largest remaining component as the snake, then use the leftover cell(s) as the
#      marker pool. Doing this in another order risks confusing the legend with the marker.
# d) Placement legality:
#    - The generator keeps the snake inside a 3-cell margin, reserving border corners for
#      the legend and leaving room for off-path marker placement near endpoints.
#    - The legend is exactly two orthogonally adjacent cells touching one corner.
#    - The marker is a single cell that is in bounds, not on the snake, not on the legend,
#      not 4-neighbor to either symbolized object, and uniquely closer to the chosen
#      endpoint than to the other endpoint under the solver's distance key.
# e) Visual observability:
#    - Across examples, a human sees the same snake-reading task plus two salient cues:
#      the lone nearby marker explains which endpoint color appears first, and the corner
#      legend explains whether the output is laid out as a row or column.
#    - Because accepted samples always differ from the original in at least order or shape,
#      the fusion behavior is visible in demonstrations rather than hidden in metadata.
# f) Visual neatness:
#    - Objects occupy separate regions: the snake stays interior, the legend hugs a corner,
#      and the marker sits near an endpoint but off the path. This avoids stacked symbols
#      or crowded overlaps in one local area.
# g) Color inferability:
#    - Colors themselves are not decoded semantically. The marker and legend colors are
#      only required to be outside the snake palette so the disconnected symbols remain
#      visually separable. The rule depends on geometry (near endpoint, corner orientation),
#      which is inferable from a few ARC examples.
# h) Fusion-behavior metric used after implementation:
#    - For each generated sample we can record:
#      `start_flips = (marked_start != min(endpoints))`
#      `layout_flips = output_is_row`
#      Degenerate original-behavior samples would have `(False, False)`, and the generator
#      rejects them. Sample statistics should therefore show zero such cases plus nonzero
#      counts for the remaining combinations.
# i) New solver assumptions and generator guarantees:
#    - Exactly one corner legend component exists and has size 2.
#    - After removing that legend, the largest remaining component is one simple snake.
#    - At least one extra non-legend foreground cell remains; the nearest such cell to the
#      endpoints is the start marker, and it uniquely indicates one endpoint.
#    - `generate_input()` checks these assumptions by construction and by validating the
#      produced grid with `main()`.


Coord = tuple[int, int]
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARK = 2
MARK_OFFSETS: list[Coord] = [
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-2, -1),
    (-2, 1),
    (2, -1),
    (2, 1),
    (-1, -2),
    (1, -2),
    (-1, 2),
    (1, 2),
    (-2, -2),
    (-2, 2),
    (2, -2),
    (2, 2),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _dist_key(a: Coord, b: Coord) -> tuple[int, int]:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    return max(dr, dc), dr + dc


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        comp: list[Coord] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
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


def _endpoint_degrees(path_pts: list[Coord]) -> tuple[dict[Coord, int], list[Coord]]:
    pset = set(path_pts)
    deg: dict[Coord, int] = {}
    ends: list[Coord] = []
    for r, c in path_pts:
        d = 0
        for q in _neighbors4(r, c):
            if q in pset:
                d += 1
        deg[(r, c)] = d
        if d == 1:
            ends.append((r, c))
    return deg, ends


def _trace_path(path_pts: list[Coord], start: Coord) -> list[Coord]:
    pset = set(path_pts)
    order: list[Coord] = []
    prev: Coord | None = None
    cur = start
    for _ in range(len(path_pts)):
        order.append(cur)
        r, c = cur
        nbrs = [q for q in _neighbors4(r, c) if q in pset and q != prev]
        if not nbrs:
            break
        if len(nbrs) > 1:
            raise ValueError("snake branches during tracing")
        prev, cur = cur, nbrs[0]
    if len(order) != len(path_pts):
        raise ValueError("failed to traverse full snake")
    return order


def _decode_scene(grid: np.ndarray, bg: int) -> tuple[bool, list[Coord], Coord]:
    mask = grid != bg
    comps = _components4(mask)
    h, w = grid.shape

    legends = [comp for comp in comps if _is_corner_legend(comp, h, w)]
    if len(legends) != 1:
        raise ValueError(f"expected exactly one corner legend, found {len(legends)}")
    legend = legends[0]
    legend_set = set(legend)
    output_is_row = len({r for r, _ in legend}) == 1

    non_legend_mask = mask.copy()
    for r, c in legend_set:
        non_legend_mask[r, c] = False

    rest = _components4(non_legend_mask)
    if len(rest) < 2:
        raise ValueError("expected snake plus marker after removing legend")

    path_pts = max(rest, key=len)
    extra_pts = [pt for comp in rest if comp is not path_pts for pt in comp]
    if not extra_pts:
        raise ValueError("expected at least one marker cell")

    _, ends = _endpoint_degrees(path_pts)
    if len(ends) != 2:
        raise ValueError(f"expected two snake endpoints, found {len(ends)}")

    marker = min(extra_pts, key=lambda pt: min(_dist_key(pt, end) for end in ends))
    return output_is_row, path_pts, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a colored snake path preserving traversal order.

    Combo rule:
    - A disconnected corner legend chooses the output layout.
      Horizontal legend => 1 x N row, vertical legend => N x 1 column.
    - A disconnected marker near one endpoint chooses the traversal start.
    - After decoding both symbols, output the snake colors in that order and layout.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    if not np.any(g != bg):
        return np.zeros((0, 1), dtype=int)

    output_is_row, path_pts, marker = _decode_scene(g, bg)
    _, ends = _endpoint_degrees(path_pts)
    start = min(ends, key=lambda end: (_dist_key(marker, end), end))
    order = _trace_path(path_pts, start)
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
    segment: list[Coord] = []
    r, c = last
    for i in range(1, count + 1):
        nr = int(r + dr * i)
        nc = int(c + dc * i)
        if nr < margin or nr >= h - margin or nc < margin or nc >= w - margin:
            break
        if (nr, nc) in path_set:
            break
        pred = (int(r + dr * (i - 1)), int(c + dc * (i - 1)))
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


def _choose_marker_cell(
    path_set: set[Coord],
    chosen_end: Coord,
    other_end: Coord,
    legend_cells: set[Coord],
    h: int,
    w: int,
) -> Coord | None:
    offsets = list(MARK_OFFSETS)
    random.shuffle(offsets)
    offsets.sort(key=lambda d: (max(abs(d[0]), abs(d[1])), abs(d[0]) + abs(d[1])))
    for dr, dc in offsets:
        nr = int(chosen_end[0] + dr)
        nc = int(chosen_end[1] + dc)
        cell = (nr, nc)
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if cell in path_set or cell in legend_cells:
            continue
        if any(nb in path_set or nb in legend_cells for nb in _neighbors4(nr, nc)):
            continue
        if _dist_key(cell, chosen_end) >= _dist_key(cell, other_end):
            continue
        return cell
    return None


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    margin = 3

    bg = random.choice([c for c in range(10) if c != MARK])
    path_color_count = random.randint(3, 5)
    usable = [c for c in range(10) if c not in (bg, MARK)]
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
        colors.extend([int(col)] * int(seg_len))

    _, ends = _endpoint_degrees(path)
    if len(ends) != 2:
        return None
    default_start = min(ends)
    marked_start = random.choice(ends)
    other_end = ends[0] if marked_start == ends[1] else ends[1]

    output_is_row = random.random() < 0.5
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend = _legend_cells(h, w, output_is_row, corner)
    legend_set = set(legend)

    marker = _choose_marker_cell(path_set, marked_start, other_end, legend_set, h, w)
    if marker is None:
        return None

    # Reject samples that behave exactly like the original task despite carrying symbols.
    if marked_start == default_start and not output_is_row:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = int(col)
    for (r, c), col in zip(legend, legend_colors):
        grid[r, c] = int(col)
    grid[marker] = int(MARK)

    expected_order = _trace_path(path, marked_start)
    expected_colors = np.array([int(grid[r, c]) for r, c in expected_order], dtype=int)
    expected = expected_colors.reshape((1, -1)) if output_is_row else expected_colors.reshape((-1, 1))
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
