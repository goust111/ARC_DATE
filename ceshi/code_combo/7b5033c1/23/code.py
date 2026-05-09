# Combo variant of 7b5033c1 (index 23)
# combo_spec: D1/0+D2/2+D10/0
# Name: 折叠+标记起点+图例方向
#
# Design analysis
# a) Concrete behavior and trace:
#    - Base task: one multicolor 4-connected snake is traced from an endpoint and
#      emitted as a straight column.
#    - Combined variant input adds two disconnected symbols:
#        1. one isolated marker pixel near exactly one snake endpoint;
#        2. one 2-cell corner legend whose geometry is horizontal or vertical.
#    - Solver pipeline is now:
#        step 1: decode the corner legend orientation;
#        step 2: decode which endpoint is marked and traverse from there;
#        step 3: fold the color stream into a matrix whose side K equals the
#                number of distinct snake colors.
#    - Small example:
#        traced colors = [1, 1, 3, 3, 2, 2], K = 3.
#        Original task -> [[1], [1], [3], [3], [2], [2]].
#        This combo with horizontal legend -> row-wise snake fold:
#            [[1, 1, 3],
#             [2, 2, 3]]
#        Same traced colors with vertical legend -> column-wise snake fold:
#            [[1, 2],
#             [1, 2],
#             [3, 3]]
#      If the marker points to the opposite endpoint, the traced sequence reverses
#      before folding, so both the read direction and the fold direction matter.
# b) Complexity calibration:
#    - K is restricted to 3..5, and path length is chosen from multiples of K with
#      at least 4 chunks. This keeps the output visibly rectangular in both fold
#      modes and avoids weak 1-row / 1-column edge cases.
#    - The legend has only two states (horizontal vs vertical), and the marker has
#      only two states (which endpoint). This is enough visible freedom without
#      turning the task into an over-parameterized symbol puzzle.
# c) Implicit-assumption audit of the original `main()`:
#    - Original `mask = g != bg` meant all foreground belonged to the snake. False
#      now: the legend and marker are also foreground, so we must split components.
#    - Original endpoint search assumed the whole foreground graph was one chain.
#      False now until the legend and marker are excluded.
#    - Original `start = min(ends)` assumed top-left endpoint always defines order.
#      False now: the marker chooses the start endpoint.
#    - Original `reshape((-1, 1))` assumed output shape was always a column. False
#      now: output is a folded matrix, and legend orientation chooses row-wise vs
#      column-wise boustrophedon filling.
#    - New interaction check: D10 cannot keep its single-row/single-column meaning
#      from the standalone variant, because D1 already changes the output to a
#      matrix. The combined interpretation is therefore "legend selects fold axis",
#      which preserves both dimensions instead of letting one overwrite the other.
# d) Placement legality:
#    - The generator keeps the snake inside a 3-cell margin, reserving corners for
#      the legend and nearby empty space for the marker.
#    - The marker must be in bounds, off the snake, not 4-neighbor to snake cells,
#      not on/adjacent to the legend, and strictly closer to the chosen endpoint
#      than to the other endpoint under the solver's distance rule.
#    - The legend is exactly one size-2 straight component anchored at a corner.
#      Because the snake stays away from the border, the three objects never merge.
# e) Visual observability:
#    - A human sees one large winding object plus two tiny symbols with different
#      jobs: the nearby isolated pixel indicates which end to start from, while the
#      corner 2-cell legend indicates how the output matrix snakes.
#    - Across examples, the output sometimes becomes tall-wide and sometimes
#      short-wide or transposed-looking; this makes the fold-axis cue visible.
# f) Visual cleanliness:
#    - Symbol roles are spatially separated: the marker lives beside an endpoint,
#      the legend lives in a corner, and the snake stays in the interior. This
#      avoids layered clutter and prevents ambiguity about which object controls
#      which part of the rule.
# g) Color-change inferability:
#    - There is no hidden color remapping. Snake colors are preserved verbatim in
#      traversal order; the only color-specific convention is that the marker color
#      is fixed and excluded from the snake/legend palettes, making the symbol easy
#      to spot without requiring any learned color table.
# h) Fusion anti-degeneration check:
#    - The generator rejects samples where using the opposite endpoint or the
#      opposite fold axis would accidentally produce the same output. This ensures
#      both the marker and the legend are behaviorally active rather than cosmetic.
# i) New solver assumptions and generator guarantees:
#    - Assumption: exactly one corner size-2 straight component exists. Guaranteed
#      by `_legend_cells()` and the interior path margin.
#    - Assumption: after removing that legend, the largest remaining component is a
#      simple snake with exactly two endpoints. Guaranteed by `_try_extend()`.
#    - Assumption: at least one extra non-background cell remains as a start marker.
#      Guaranteed by `_choose_marker_cell()`.
#    - Assumption: snake length is divisible by K = distinct snake colors. Guaranteed
#      by choosing target lengths from multiples of `color_count`.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import get_bg


Coord = tuple[int, int]
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARK = 2
GRID_SHAPE = (16, 16)
PATH_MARGIN = 3
MARK_OFFSETS = [
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
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


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
            cr, cc = stack.pop()
            comp.append((int(cr), int(cc)))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
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


def _dist_key(a: Coord, b: Coord) -> tuple[int, int]:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    return max(dr, dc), dr + dc


def _endpoint_degrees(path_pts: list[Coord]) -> tuple[dict[Coord, int], list[Coord]]:
    pset = set(path_pts)
    deg: dict[Coord, int] = {}
    ends: list[Coord] = []
    for r, c in path_pts:
        d = 0
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pset:
                d += 1
        deg[(int(r), int(c))] = int(d)
        if d == 1:
            ends.append((int(r), int(c)))
    return deg, ends


def _trace_path(path_pts: list[Coord], start: Coord) -> list[Coord]:
    pset = set(path_pts)
    order: list[Coord] = []
    prev: Coord | None = None
    cur = (int(start[0]), int(start[1]))
    for _ in range(len(path_pts)):
        order.append(cur)
        nbrs = [q for q in _neighbors4(cur[0], cur[1]) if q in pset and q != prev]
        if not nbrs:
            break
        if len(nbrs) > 1:
            raise ValueError("snake branches during traversal")
        prev, cur = cur, nbrs[0]
    if len(order) != len(path_pts):
        raise ValueError("failed to traverse the full snake")
    return order


def _fold_colors(colors: list[int], width: int, fold_by_rows: bool) -> np.ndarray:
    if width <= 0 or len(colors) % width != 0:
        raise ValueError("invalid fold dimensions")

    chunk_count = len(colors) // width
    if fold_by_rows:
        out = np.zeros((chunk_count, width), dtype=int)
        idx = 0
        for r in range(chunk_count):
            chunk = colors[idx : idx + width]
            idx += width
            out[r, :] = chunk if r % 2 == 0 else chunk[::-1]
        return out

    out = np.zeros((width, chunk_count), dtype=int)
    idx = 0
    for c in range(chunk_count):
        chunk = colors[idx : idx + width]
        idx += width
        out[:, c] = chunk if c % 2 == 0 else chunk[::-1]
    return out


def _decode_scene(grid: np.ndarray, bg: int) -> tuple[bool, list[Coord], Coord | None]:
    mask = grid != bg
    comps = _components4(mask)
    h, w = grid.shape

    legends = [comp for comp in comps if _is_corner_legend(comp, h, w)]
    if len(legends) != 1:
        raise ValueError(f"expected exactly one corner legend, found {len(legends)}")

    legend = legends[0]
    fold_by_rows = len({r for r, _ in legend}) == 1

    path_mask = mask.copy()
    for r, c in legend:
        path_mask[r, c] = False

    comps_wo_legend = _components4(path_mask)
    if not comps_wo_legend:
        return fold_by_rows, [], None

    snake = max(comps_wo_legend, key=len)
    extras = [pt for comp in comps_wo_legend if comp is not snake for pt in comp]
    marker = min(extras) if extras else None
    return fold_by_rows, snake, marker


def _marker_start(ends: list[Coord], marker: Coord | None) -> Coord:
    if marker is None:
        return min(ends)
    scored = [(_dist_key(marker, end), end) for end in ends]
    scored.sort()
    return scored[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combined rule:
    - decode the corner legend orientation;
    - remove legend, find the snake, and use the isolated marker to pick the start;
    - trace the snake and fold the resulting color stream into a boustrophedon
      matrix, using K = number of distinct snake colors.

    Legend meaning in the combo:
    - horizontal corner legend => fold by rows;
    - vertical corner legend => fold by columns.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    fold_by_rows, path_pts, marker = _decode_scene(g, bg)
    if not path_pts:
        return np.zeros((0, 0), dtype=int)

    deg, ends = _endpoint_degrees(path_pts)
    if any(d > 2 for d in deg.values()):
        raise ValueError("snake is not a simple chain")
    if len(ends) != 2:
        raise ValueError(f"expected two snake endpoints, found {len(ends)}")

    start = _marker_start(ends, marker)
    order = _trace_path(path_pts, start)
    colors = [int(g[r, c]) for r, c in order]
    width = len(set(colors))
    return _fold_colors(colors, width, fold_by_rows)


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
        nr, nc = int(r + dr * i), int(c + dc * i)
        if nr < margin or nr >= h - margin or nc < margin or nc >= w - margin:
            break
        if (nr, nc) in path_set:
            break

        pred = (int(r + dr * (i - 1)), int(c + dc * (i - 1)))
        if any(q != pred and q in path_set for q in _neighbors4(nr, nc)):
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
        prev = seg_colors[-1]
        choices = [c for c in palette if c != prev]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([int(col)] * int(seg_len))
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


def _touches_reserved(cell: Coord, reserved: set[Coord]) -> bool:
    if cell in reserved:
        return True
    return any(nb in reserved for nb in _neighbors4(cell[0], cell[1]))


def _choose_marker_cell(
    path_set: set[Coord],
    chosen_end: Coord,
    other_end: Coord,
    h: int,
    w: int,
    reserved: set[Coord],
) -> Coord | None:
    offsets = list(MARK_OFFSETS)
    random.shuffle(offsets)
    offsets.sort(key=lambda d: (max(abs(d[0]), abs(d[1])), abs(d[0]) + abs(d[1])))
    for dr, dc in offsets:
        nr, nc = int(chosen_end[0] + dr), int(chosen_end[1] + dc)
        cell = (nr, nc)
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if cell in path_set or _touches_reserved(cell, reserved):
            continue
        if any(nb in path_set for nb in _neighbors4(nr, nc)):
            continue
        if _dist_key(cell, chosen_end) >= _dist_key(cell, other_end):
            continue
        return cell
    return None


def _gen_attempt() -> np.ndarray | None:
    h, w = GRID_SHAPE
    bg = random.choice([c for c in range(10) if c != MARK])
    color_count = random.randint(3, 5)

    usable = [c for c in range(10) if c not in (bg, MARK)]
    path_palette = random.sample(usable, color_count)
    legend_pool = [c for c in usable if c not in path_palette]
    legend_colors = random.sample(legend_pool, 2)

    target_len = random.choice([n for n in range(18, 31) if n % color_count == 0 and n // color_count >= 4])
    fold_by_rows = random.random() < 0.5
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend = _legend_cells(h, w, fold_by_rows, corner)
    reserved = set(legend)

    start_r = random.randint(PATH_MARGIN, PATH_MARGIN + 2)
    start_c = random.randint(PATH_MARGIN, w - PATH_MARGIN - 1)

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
            segment = _try_extend(path_set, last, v_dir, 0, run_len, h, w, PATH_MARGIN)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, h, w, PATH_MARGIN)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, h, w, PATH_MARGIN)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, h, w, PATH_MARGIN)
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

    _, ends = _endpoint_degrees(path)
    if len(ends) != 2:
        return None

    marked_start = random.choice(ends)
    other_end = ends[0] if marked_start == ends[1] else ends[1]
    marker = _choose_marker_cell(path_set, marked_start, other_end, h, w, reserved)
    if marker is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = int(col)
    grid[marker] = int(MARK)
    for (r, c), col in zip(legend, legend_colors):
        grid[r, c] = int(col)

    traced_order = _trace_path(path, marked_start)
    traced_colors = [int(grid[r, c]) for r, c in traced_order]
    expected = _fold_colors(traced_colors, color_count, fold_by_rows)

    alt_order = _trace_path(path, other_end)
    alt_colors = [int(grid[r, c]) for r, c in alt_order]
    if np.array_equal(expected, _fold_colors(alt_colors, color_count, fold_by_rows)):
        return None
    if np.array_equal(expected, _fold_colors(traced_colors, color_count, not fold_by_rows)):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
