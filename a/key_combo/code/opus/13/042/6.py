from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4

GEN_RETRY_LIMIT = 500
GEN_SOURCE_LAYOUT_RETRIES = 120
GEN_COLORS = (1, 2, 6, 8, 9)


Coord = tuple[int, int]


def _border_cells(shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    cells = [(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
    cells += [(r, 0) for r in range(1, h - 1)] + [(r, w - 1) for r in range(1, h - 1)]
    return cells


def _source_priority(shape: tuple[int, int], coords: tuple[Coord, ...]) -> tuple[int, int, int]:
    h, w = shape
    ranks: list[tuple[int, int, int]] = []
    for r, c in coords:
        if r == 0:
            ranks.append((0, c, 0))
        elif c == 0:
            ranks.append((1, r, 0))
        elif r == h - 1:
            ranks.append((2, c, 0))
        elif c == w - 1:
            ranks.append((3, r, 0))
    assert ranks, "source must touch the border"
    return min(ranks)


def _find_sources(grid: np.ndarray) -> dict[int, tuple[Coord, ...]]:
    g = np.asarray(grid, dtype=int)
    source_by_color: dict[int, list[Coord]] = defaultdict(list)
    for r, c in _border_cells(g.shape):
        color = int(g[r, c])
        if color not in (BLACK, GREEN):
            source_by_color[color].append((r, c))
    assert len(source_by_color) == 2, "expected exactly two border source colors"
    return {color: tuple(sorted(coords)) for color, coords in source_by_color.items()}


def _marker_positions(grid: np.ndarray) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(np.asarray(grid) == GREEN)]


def _nearest_source_color(pos: Coord, sources: dict[int, tuple[Coord, ...]], shape: tuple[int, int]) -> int:
    r, c = pos
    best: tuple[int, tuple[int, int, int], int] | None = None
    for color, coords in sources.items():
        dist = min(abs(r - rr) + abs(c - cc) for rr, cc in coords)
        cand = (dist, _source_priority(shape, coords), int(color))
        if best is None or cand < best:
            best = cand
    assert best is not None
    return int(best[2])


def _recolor_markers(grid: np.ndarray, sources: dict[int, tuple[Coord, ...]]) -> dict[Coord, int]:
    return {
        pos: _nearest_source_color(pos, sources, grid.shape)
        for pos in _marker_positions(grid)
    }


def _pair_candidates(markers: dict[Coord, int], color: int) -> list[tuple[tuple[int, int, int, int, int], Coord, Coord]]:
    pts = sorted(pos for pos, col in markers.items() if col == color)
    out: list[tuple[tuple[int, int, int, int, int], Coord, Coord]] = []
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            if a[0] != b[0] and a[1] != b[1]:
                continue
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            lo = min(a, b)
            hi = max(a, b)
            out.append(((dist, lo[0], lo[1], hi[0], hi[1]), a, b))
    return sorted(out)


def _best_partner_map(markers: dict[Coord, int], color: int) -> dict[Coord, Coord]:
    choices: dict[Coord, tuple[int, int, int, Coord]] = {}
    pts = sorted(pos for pos, col in markers.items() if col == color)
    for a in pts:
        for b in pts:
            if a == b or (a[0] != b[0] and a[1] != b[1]):
                continue
            cand = (abs(a[0] - b[0]) + abs(a[1] - b[1]), b[0], b[1], b)
            if a not in choices or cand < choices[a]:
                choices[a] = cand
    return {a: cand[3] for a, cand in choices.items()}


def _pair_markers(markers: dict[Coord, int]) -> list[dict[str, object]]:
    used: set[Coord] = set()
    pairs: list[dict[str, object]] = []
    colors = sorted(set(markers.values()))

    for color in colors:
        best = _best_partner_map(markers, color)
        for a in sorted(pos for pos, col in markers.items() if col == color):
            if a in used:
                continue
            b = best.get(a)
            if b is None or b in used:
                continue
            if best.get(b) != a:
                continue
            used.add(a)
            used.add(b)
            pairs.append(
                {
                    "color": int(color),
                    "a": a,
                    "b": b,
                    "orient": "horizontal" if a[0] == b[0] else "vertical",
                }
            )

    for color in colors:
        for _key, a, b in _pair_candidates(markers, color):
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            pairs.append(
                {
                    "color": int(color),
                    "a": a,
                    "b": b,
                    "orient": "horizontal" if a[0] == b[0] else "vertical",
                }
            )
    return pairs


def _draw_segment(out: np.ndarray, a: Coord, b: Coord, color: int) -> None:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        out[r, c0 : c1 + 1] = int(color)
    else:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        out[r0 : r1 + 1, c] = int(color)


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sources = _find_sources(g)
    markers = _recolor_markers(g, sources)
    pairs = _pair_markers(markers)

    out = g.copy()
    for pos, color in markers.items():
        out[pos] = int(color)

    horizontals: list[dict[str, object]] = []
    verticals: list[dict[str, object]] = []
    for pair in pairs:
        a = pair["a"]
        b = pair["b"]
        color = int(pair["color"])
        _draw_segment(out, a, b, color)
        if pair["orient"] == "horizontal":
            r = a[0]
            c0, c1 = sorted((a[1], b[1]))
            horizontals.append({"color": color, "r": r, "c0": c0, "c1": c1, "a": a, "b": b})
        else:
            c = a[1]
            r0, r1 = sorted((a[0], b[0]))
            verticals.append({"color": color, "c": c, "r0": r0, "r1": r1, "a": a, "b": b})

    marker_positions = set(markers)
    yellow_points: set[Coord] = set()
    for h in horizontals:
        for v in verticals:
            rr = int(h["r"])
            cc = int(v["c"])
            if not (int(h["c0"]) <= cc <= int(h["c1"]) and int(v["r0"]) <= rr <= int(v["r1"])):
                continue
            if int(h["color"]) != int(v["color"]) and (rr, cc) not in marker_positions:
                out[rr, cc] = YELLOW
                yellow_points.add((rr, cc))

    for pos, color in markers.items():
        out[pos] = int(color)

    return {
        "sources": sources,
        "markers": markers,
        "pairs": pairs,
        "output": out,
        "yellow_points": yellow_points,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each green marker from the nearest border source, then connect aligned same-color pairs and mark different-color crossings yellow.

    Input structure:
    - A black background contains exactly two differently colored source lines on the border.
    - Inside the border, several green single-pixel markers are scattered on empty cells.
    - The source lines remain fixed and act only as visible color origins.

    Rule:
    - Keep the border source lines unchanged.
    - For each green marker, find the nearest source-line pixel by Manhattan distance and recolor the marker to that source color; ties prefer the source that is higher, then further left on the border.
    - Among recolored non-border markers, pair same-color markers that share a row or column, giving each marker at most one partner; nearest aligned partners win, with row/column lexicographic tie-breaks.
    - Draw each pair as an inclusive horizontal or vertical segment in its shared color.
    - When a horizontal segment and a vertical segment of different colors cross, recolor the crossing pixel yellow, but never overwrite a marker pixel with yellow.

    Key insight:
    - The border sources first partition the marker set by nearest-distance inheritance, and those inherited colors then create the only valid pairing graph that determines which line segments and crossings can appear.
    """
    return np.asarray(_solve_details(grid)["output"], dtype=int)


def _paint_source_segment(grid: np.ndarray, side: str, start: int, end: int, color: int) -> list[Coord]:
    h, w = grid.shape
    coords: list[Coord] = []
    if side == "top":
        for c in range(start, end + 1):
            grid[0, c] = int(color)
            coords.append((0, c))
    elif side == "right":
        for r in range(start, end + 1):
            grid[r, w - 1] = int(color)
            coords.append((r, w - 1))
    elif side == "left":
        for r in range(start, end + 1):
            grid[r, 0] = int(color)
            coords.append((r, 0))
    elif side == "bottom":
        for c in range(start, end + 1):
            grid[h - 1, c] = int(color)
            coords.append((h - 1, c))
    else:
        raise AssertionError("unknown side")
    return coords


def _sample_sources(h: int, w: int, colors: tuple[int, int]) -> tuple[np.ndarray, dict[int, tuple[Coord, ...]], str]:
    grid = np.zeros((h, w), dtype=int)
    mode = random.choice(("top-bottom", "left-right"))
    if mode == "top-bottom":
        long_top = random.choice((True, False))
        long_lo = max(5, w // 2)
        short_hi = max(4, min(w - 3, w // 3 + 1))
        if long_top:
            top_len = random.randint(long_lo, w - 2)
            bottom_len = random.randint(3, short_hi)
        else:
            top_len = random.randint(3, short_hi)
            bottom_len = random.randint(long_lo, w - 2)
        top_start = random.randint(0, w - top_len)
        bottom_start = random.randint(0, w - bottom_len)
        a = _paint_source_segment(grid, "top", top_start, top_start + top_len - 1, colors[0])
        b = _paint_source_segment(grid, "bottom", bottom_start, bottom_start + bottom_len - 1, colors[1])
    else:
        long_left = random.choice((True, False))
        long_lo = max(5, h // 2)
        short_hi = max(4, min(h - 3, h // 3 + 1))
        if long_left:
            left_len = random.randint(long_lo, h - 2)
            right_len = random.randint(3, short_hi)
        else:
            left_len = random.randint(3, short_hi)
            right_len = random.randint(long_lo, h - 2)
        left_start = random.randint(0, h - left_len)
        right_start = random.randint(0, h - right_len)
        a = _paint_source_segment(grid, "left", left_start, left_start + left_len - 1, colors[0])
        b = _paint_source_segment(grid, "right", right_start, right_start + right_len - 1, colors[1])
    return grid, {colors[0]: tuple(a), colors[1]: tuple(b)}, mode


def _classified_cells(shape: tuple[int, int], sources: dict[int, tuple[Coord, ...]]) -> dict[int, list[Coord]]:
    h, w = shape
    out: dict[int, list[Coord]] = defaultdict(list)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = _nearest_source_color((r, c), sources, shape)
            out[color].append((r, c))
    return {color: sorted(cells) for color, cells in out.items()}


def _distance_map(shape: tuple[int, int], sources: dict[int, tuple[Coord, ...]]) -> dict[Coord, tuple[int, int]]:
    color_order = sorted(sources)
    out: dict[Coord, tuple[int, int]] = {}
    h, w = shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            ds = []
            for color in color_order:
                ds.append(min(abs(r - rr) + abs(c - cc) for rr, cc in sources[color]))
            out[(r, c)] = tuple(ds)
    return out


def _find_cross_pair(
    color_h: int,
    color_v: int,
    cells: dict[int, list[Coord]],
) -> tuple[tuple[Coord, Coord], tuple[Coord, Coord]] | None:
    rows_h: dict[int, list[int]] = defaultdict(list)
    cols_v: dict[int, list[int]] = defaultdict(list)
    for r, c in cells[color_h]:
        rows_h[r].append(c)
    for r, c in cells[color_v]:
        cols_v[c].append(r)

    row_items = list(rows_h.items())
    col_items = list(cols_v.items())
    random.shuffle(row_items)
    random.shuffle(col_items)
    for r_h, cols in row_items:
        if len(cols) < 2:
            continue
        cols = sorted(cols)
        for c_v, rows in col_items:
            if len(rows) < 2:
                continue
            rows = sorted(rows)
            left = [c for c in cols if c < c_v]
            right = [c for c in cols if c > c_v]
            up = [r for r in rows if r < r_h]
            down = [r for r in rows if r > r_h]
            if left and right and up and down:
                h_pair = ((r_h, left[0]), (r_h, right[-1]))
                v_pair = ((up[0], c_v), (down[-1], c_v))
                return h_pair, v_pair
    return None


def _pair_conflicts(
    pair: tuple[Coord, Coord],
    color: int,
    row_counts: dict[int, dict[int, int]],
    col_counts: dict[int, dict[int, int]],
) -> bool:
    a, b = pair
    if a[0] == b[0]:
        row = a[0]
        if row_counts[color].get(row, 0):
            return True
        if col_counts[color].get(a[1], 0) or col_counts[color].get(b[1], 0):
            return True
    else:
        col = a[1]
        if col_counts[color].get(col, 0):
            return True
        if row_counts[color].get(a[0], 0) or row_counts[color].get(b[0], 0):
            return True
    return False


def _register_pair(
    pair: tuple[Coord, Coord],
    color: int,
    row_counts: dict[int, dict[int, int]],
    col_counts: dict[int, dict[int, int]],
    used: set[Coord],
) -> None:
    a, b = pair
    used.add(a)
    used.add(b)
    row_counts[color][a[0]] += 1
    row_counts[color][b[0]] += 1
    col_counts[color][a[1]] += 1
    col_counts[color][b[1]] += 1


def _find_additional_pair(
    color: int,
    orient: str,
    cells: dict[int, list[Coord]],
    used: set[Coord],
    row_counts: dict[int, dict[int, int]],
    col_counts: dict[int, dict[int, int]],
) -> tuple[Coord, Coord] | None:
    if orient == "horizontal":
        rows: dict[int, list[int]] = defaultdict(list)
        for r, c in cells[color]:
            if (r, c) in used:
                continue
            rows[r].append(c)
        items = list(rows.items())
        random.shuffle(items)
        for r, cols in items:
            if row_counts[color].get(r, 0):
                continue
            cols = sorted(c for c in cols if not col_counts[color].get(c, 0))
            if len(cols) >= 2:
                return (r, cols[0]), (r, cols[-1])
    else:
        cols: dict[int, list[int]] = defaultdict(list)
        for r, c in cells[color]:
            if (r, c) in used:
                continue
            cols[c].append(r)
        items = list(cols.items())
        random.shuffle(items)
        for c, rows in items:
            if col_counts[color].get(c, 0):
                continue
            rows = sorted(r for r in rows if not row_counts[color].get(r, 0))
            if len(rows) >= 2:
                return (rows[0], c), (rows[-1], c)
    return None


def _same_color_three_collinear(markers: dict[Coord, int]) -> bool:
    rows: dict[tuple[int, int], int] = defaultdict(int)
    cols: dict[tuple[int, int], int] = defaultdict(int)
    for (r, c), color in markers.items():
        rows[(color, r)] += 1
        cols[(color, c)] += 1
    return any(v >= 3 for v in rows.values()) or any(v >= 3 for v in cols.values())


def _pair_signature(pairs: list[dict[str, object]]) -> set[tuple[Coord, Coord, int]]:
    out: set[tuple[Coord, Coord, int]] = set()
    for pair in pairs:
        a = tuple(pair["a"])
        b = tuple(pair["b"])
        lo, hi = min(a, b), max(a, b)
        out.add((lo, hi, int(pair["color"])))
    return out


def _tie_cells(shape: tuple[int, int], sources: dict[int, tuple[Coord, ...]]) -> list[Coord]:
    colors = sorted(sources)
    out: list[Coord] = []
    h, w = shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            d0 = min(abs(r - rr) + abs(c - cc) for rr, cc in sources[colors[0]])
            d1 = min(abs(r - rr) + abs(c - cc) for rr, cc in sources[colors[1]])
            if d0 == d1:
                out.append((r, c))
    return out


def _build_candidate() -> tuple[np.ndarray, set[tuple[Coord, Coord, int]]] | None:
    h = random.randint(10, 15)
    w = random.randint(10, 15)
    colors = tuple(random.sample(GEN_COLORS, 2))
    grid = None
    sources = None
    cells = None
    cross = None
    for _ in range(GEN_SOURCE_LAYOUT_RETRIES):
        grid_try, sources_try, _mode = _sample_sources(h, w, colors)
        cells_try = _classified_cells(grid_try.shape, sources_try)
        if any(len(cells_try.get(color, ())) < 3 for color in colors):
            continue
        cross_try = _find_cross_pair(colors[0], colors[1], cells_try)
        if cross_try is None:
            continue
        grid, sources, cells, cross = grid_try, sources_try, cells_try, cross_try
        break
    if grid is None or sources is None or cells is None or cross is None:
        return None

    used: set[Coord] = set()
    row_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    col_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    expected_pairs: list[tuple[Coord, Coord, int]] = []

    h_pair, v_pair = cross
    if _pair_conflicts(h_pair, colors[0], row_counts, col_counts):
        return None
    if _pair_conflicts(v_pair, colors[1], row_counts, col_counts):
        return None
    _register_pair(h_pair, colors[0], row_counts, col_counts, used)
    _register_pair(v_pair, colors[1], row_counts, col_counts, used)
    expected_pairs.append((min(h_pair), max(h_pair), colors[0]))
    expected_pairs.append((min(v_pair), max(v_pair), colors[1]))

    third_options = [
        (colors[0], "vertical"),
        (colors[1], "horizontal"),
        (colors[0], "horizontal"),
        (colors[1], "vertical"),
    ]
    random.shuffle(third_options)
    extra_pair = None
    for color, orient in third_options:
        pair = _find_additional_pair(color, orient, cells, used, row_counts, col_counts)
        if pair is None:
            continue
        if _pair_conflicts(pair, color, row_counts, col_counts):
            continue
        extra_pair = (pair, color)
        break
    if extra_pair is None:
        return None

    pair, color = extra_pair
    _register_pair(pair, color, row_counts, col_counts, used)
    expected_pairs.append((min(pair), max(pair), color))

    markers = set(used)

    tie_bonus = random.random() < 0.45
    if tie_bonus:
        tie_candidates = [
            pos
            for pos in _tie_cells(grid.shape, sources)
            if pos not in markers
            and row_counts[_nearest_source_color(pos, sources, grid.shape)].get(pos[0], 0) == 0
            and col_counts[_nearest_source_color(pos, sources, grid.shape)].get(pos[1], 0) == 0
        ]
        random.shuffle(tie_candidates)
        if tie_candidates:
            markers.add(tie_candidates[0])

    target_count = random.randint(max(6, len(markers)), min(10, len(markers) + 3))
    filler_candidates = [
        pos
        for color in colors
        for pos in cells[color]
        if pos not in markers
        and row_counts[color].get(pos[0], 0) == 0
        and col_counts[color].get(pos[1], 0) == 0
    ]
    random.shuffle(filler_candidates)
    for pos in filler_candidates:
        if len(markers) >= target_count:
            break
        markers.add(pos)

    if not 6 <= len(markers) <= 10:
        return None

    for pos in markers:
        grid[pos] = GREEN

    details = _solve_details(grid)
    actual_pairs = _pair_signature(details["pairs"])
    expected_set = set(expected_pairs)
    if actual_pairs != expected_set:
        return None
    if not details["yellow_points"]:
        return None
    marker_colors = details["markers"]
    if _same_color_three_collinear(marker_colors):
        return None
    pair_colors = {int(pair["color"]) for pair in details["pairs"]}
    if pair_colors != set(colors) and not (pair_colors >= set(colors)):
        return None
    if sum(1 for v in marker_colors.values() if v == colors[0]) < 2:
        return None
    if sum(1 for v in marker_colors.values() if v == colors[1]) < 2:
        return None
    return grid, expected_set


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _build_candidate()
        if built is None:
            continue
        grid, expected_pairs = built
        details = _solve_details(grid)
        if _pair_signature(details["pairs"]) != expected_pairs:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
