import random

import numpy as np


BG = 0
MAGENTA = 6
PATH_COLOR_POOL = (1, 2, 3, 4, 5, 7, 8, 9)
COUNT_CHOICES = (2, 3)
GENERATION_ATTEMPTS = 400


def _legend_colors(row: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in row:
        color = int(value)
        if color == BG:
            continue
        assert color != MAGENTA, "legend must contain only path colors"
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _extract_lines(canvas: np.ndarray, path_colors: list[int]) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []

    for color in path_colors:
        coords = np.argwhere(canvas == int(color))
        assert coords.size > 0, f"path color {color} must appear in the canvas"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1 and len(cols) > 1:
            horizontals.append(
                {
                    "axis": 0,
                    "color": int(color),
                    "fixed": int(rows[0]),
                    "start": int(cols.min()),
                    "end": int(cols.max()),
                }
            )
        elif len(cols) == 1 and len(rows) > 1:
            verticals.append(
                {
                    "axis": 1,
                    "color": int(color),
                    "fixed": int(cols[0]),
                    "start": int(rows.min()),
                    "end": int(rows.max()),
                }
            )
        else:
            raise AssertionError(f"could not infer a straight line for color {color}")

    horizontals.sort(key=lambda line: int(line["fixed"]))
    verticals.sort(key=lambda line: int(line["fixed"]))
    return horizontals, verticals


def _paint_full_lines(shape: tuple[int, int], horizontals: list[dict[str, int]], verticals: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for line in horizontals:
        r = int(line["fixed"])
        out[r, int(line["start"]) : int(line["end"]) + 1] = int(line["color"])
    for line in verticals:
        c = int(line["fixed"])
        out[int(line["start"]) : int(line["end"]) + 1, c] = int(line["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve colored line crossings by applying the legend priority order and optional magenta inversion markers.

    Input structure:
    - The background is black, and the bottom row is a left-to-right legend of 1x1 path colors.
    - Above the legend are several straight 1-pixel-wide horizontal or vertical colored path segments.
    - Different-colored horizontal and vertical segments may geometrically cross, and some crossings have an orthogonally adjacent magenta marker.

    Rule:
    - Read the bottom legend row from left to right; earlier colors have higher priority.
    - Reconstruct each colored path as one straight segment, then consider every horizontal/vertical geometric crossing.
    - Without an adjacent magenta marker, the higher-priority path keeps the crossing cell; with an adjacent magenta marker, the lower-priority path keeps it instead.
    - The losing path is cut at that crossing by clearing the two cells immediately adjacent to the crossing along the losing path, and the output removes both the legend row and all magenta markers.

    Key insight:
    - The visible crossing result depends on two layers: legend-derived color priority and a local magenta exception that flips which path has right of way.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    legend = _legend_colors(g[-1])
    assert legend, "expected at least one legend color"
    priority = {color: rank for rank, color in enumerate(legend)}

    canvas = np.array(g[:-1], copy=False)
    markers = {tuple(map(int, rc)) for rc in np.argwhere(canvas == MAGENTA)}

    horizontals, verticals = _extract_lines(canvas, legend)
    out = _paint_full_lines(canvas.shape, horizontals, verticals)

    for hline in horizontals:
        hr = int(hline["fixed"])
        hc0 = int(hline["start"])
        hc1 = int(hline["end"])
        for vline in verticals:
            vc = int(vline["fixed"])
            vr0 = int(vline["start"])
            vr1 = int(vline["end"])
            if not (hc0 <= vc <= hc1 and vr0 <= hr <= vr1):
                continue

            inverted = any(
                (rr, cc) in markers
                for rr, cc in ((hr - 1, vc), (hr + 1, vc), (hr, vc - 1), (hr, vc + 1))
            )

            h_rank = int(priority[int(hline["color"])])
            v_rank = int(priority[int(vline["color"])])
            if h_rank < v_rank:
                higher, lower = hline, vline
            else:
                higher, lower = vline, hline

            winner = lower if inverted else higher
            loser = higher if inverted else lower
            out[hr, vc] = int(winner["color"])

            if int(loser["axis"]) == 0:
                if int(loser["start"]) <= vc - 1 <= int(loser["end"]):
                    out[hr, vc - 1] = BG
                if int(loser["start"]) <= vc + 1 <= int(loser["end"]):
                    out[hr, vc + 1] = BG
            else:
                if int(loser["start"]) <= hr - 1 <= int(loser["end"]):
                    out[hr - 1, vc] = BG
                if int(loser["start"]) <= hr + 1 <= int(loser["end"]):
                    out[hr + 1, vc] = BG

    return out


def _orientation_split(priority_order: list[int], n_horizontal: int) -> tuple[list[int], list[int]] | None:
    for _ in range(100):
        horizontal_colors = random.sample(priority_order, n_horizontal)
        vertical_colors = [color for color in priority_order if color not in horizontal_colors]

        hranks = [priority_order.index(color) for color in horizontal_colors]
        vranks = [priority_order.index(color) for color in vertical_colors]
        if min(hranks) < max(vranks) and min(vranks) < max(hranks):
            random.shuffle(horizontal_colors)
            random.shuffle(vertical_colors)
            return horizontal_colors, vertical_colors
    return None


def _line_positions(count: int, slack: int) -> tuple[list[int], int]:
    slots = count + slack
    positions = [3 + 4 * i for i in range(slots)]
    return sorted(random.sample(positions, count)), 4 * slots + 3


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    n_horizontal = random.choice(COUNT_CHOICES)
    n_vertical = random.choice(COUNT_CHOICES)
    total = n_horizontal + n_vertical

    all_colors = random.sample(PATH_COLOR_POOL, total)
    priority_order = random.sample(all_colors, total)
    split = _orientation_split(priority_order, n_horizontal)
    if split is None:
        return None
    horizontal_colors, vertical_colors = split

    rows, out_h = _line_positions(n_horizontal, random.randint(1, 3))
    cols, out_w_base = _line_positions(n_vertical, random.randint(1, 3))
    out_w = max(out_w_base, 2 * total + 3)

    min_col = min(cols)
    max_col = max(cols)
    min_row = min(rows)
    max_row = max(rows)

    horizontals: list[dict[str, int]] = []
    for row, color in zip(rows, horizontal_colors):
        start = random.randint(1, min_col - 2)
        end = random.randint(max_col + 2, out_w - 2)
        horizontals.append({"axis": 0, "color": int(color), "fixed": int(row), "start": int(start), "end": int(end)})

    verticals: list[dict[str, int]] = []
    for col, color in zip(cols, vertical_colors):
        start = random.randint(1, min_row - 2)
        end = random.randint(max_row + 2, out_h - 2)
        verticals.append({"axis": 1, "color": int(color), "fixed": int(col), "start": int(start), "end": int(end)})

    crossings: list[dict[str, object]] = []
    for hline in horizontals:
        for vline in verticals:
            crossings.append({"h": hline, "v": vline})

    cross_count = len(crossings)
    marked_count = cross_count // 2 if random.random() < 0.5 else (cross_count + 1) // 2
    marked_indices = set(random.sample(range(cross_count), marked_count))
    rank = {color: idx for idx, color in enumerate(priority_order)}

    input_canvas = _paint_full_lines((out_h, out_w), horizontals, verticals)
    expected = _paint_full_lines((out_h, out_w), horizontals, verticals)
    used_markers: set[tuple[int, int]] = set()

    for idx, cross in enumerate(crossings):
        hline = cross["h"]
        vline = cross["v"]
        hr = int(hline["fixed"])
        vc = int(vline["fixed"])
        inverted = idx in marked_indices

        if rank[int(hline["color"])] < rank[int(vline["color"])]:
            higher, lower = hline, vline
        else:
            higher, lower = vline, hline
        winner = lower if inverted else higher
        loser = higher if inverted else lower

        input_canvas[hr, vc] = int(winner["color"])
        expected[hr, vc] = int(winner["color"])

        if int(loser["axis"]) == 0:
            expected[hr, vc - 1] = BG
            expected[hr, vc + 1] = BG
            marker_options = [(hr, vc - 1), (hr, vc + 1)]
        else:
            expected[hr - 1, vc] = BG
            expected[hr + 1, vc] = BG
            marker_options = [(hr - 1, vc), (hr + 1, vc)]

        if inverted:
            random.shuffle(marker_options)
            marker = None
            for candidate in marker_options:
                if candidate not in used_markers:
                    marker = candidate
                    break
            assert marker is not None
            used_markers.add(marker)
            input_canvas[marker] = MAGENTA

    legend_row = np.full(out_w, BG, dtype=int)
    legend_start = random.randint(1, out_w - 2 * total)
    for idx, color in enumerate(priority_order):
        legend_row[legend_start + 2 * idx] = int(color)

    inp = np.vstack([input_canvas, legend_row])
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        inp, expected = built
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid crossing-priority scene")
