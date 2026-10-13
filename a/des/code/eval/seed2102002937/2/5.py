from __future__ import annotations

import random

import numpy as np


PLUS_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
CORNER_OPTIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
COUNT_OPTIONS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2))
GRID_MIN = 14
GRID_MAX = 24
MIN_SPAN = 7
PATH_GAP = 3
GENERATE_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_legend_candidate(
    grid: np.ndarray,
    bg: int,
    pair_row: int,
    result_row: int,
    align: str,
) -> dict[str, object] | None:
    h, w = grid.shape
    if not (0 <= pair_row < h and 0 <= result_row < h):
        return None

    row = grid[pair_row]
    if align == "left":
        run = 0
        while run < w and int(row[run]) != bg:
            run += 1
        start = 0
        if run == 0 or np.any(row[run:] != bg):
            return None
    else:
        start = w
        while start > 0 and int(row[start - 1]) != bg:
            start -= 1
        run = w - start
        if run == 0 or np.any(row[:start] != bg):
            return None

    if run < 2 or run % 2 != 0:
        return None

    result_slice = grid[result_row, start : start + run]
    if align == "left":
        outside_ok = bool(np.all(grid[result_row, run:] == bg))
    else:
        outside_ok = bool(np.all(grid[result_row, :start] == bg))
    if not outside_ok:
        return None

    for phase in (0, 1):
        mapping: dict[tuple[int, int], int] = {}
        ok = True
        for offset in range(run):
            value = int(result_slice[offset])
            expects_color = offset % 2 == phase
            if expects_color and value == bg:
                ok = False
                break
            if (not expects_color) and value != bg:
                ok = False
                break
        if not ok:
            continue

        for entry in range(run // 2):
            c0 = start + 2 * entry
            c1 = c0 + 1
            a = int(grid[pair_row, c0])
            b = int(grid[pair_row, c1])
            result = int(grid[result_row, c0 + phase])
            if a == bg or b == bg or result == bg or a == b:
                ok = False
                break
            key = tuple(sorted((a, b)))
            if key in mapping and mapping[key] != result:
                ok = False
                break
            mapping[key] = result
        if not ok:
            continue

        mask = np.zeros_like(grid, dtype=bool)
        r0, r1 = sorted((pair_row, result_row))
        mask[r0 : r1 + 1, start : start + run] = True
        return {
            "mask": mask,
            "mapping": mapping,
            "phase": phase,
            "pair_row": pair_row,
            "result_row": result_row,
            "start": start,
            "width": run,
        }

    return None


def _find_legend(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    h, _w = grid.shape
    candidates: list[dict[str, object]] = []
    bands = []
    if h >= 2:
        bands.append((0, 1))
        bands.append((h - 2, h - 1))
    for pair_row, result_row in bands:
        for align in ("left", "right"):
            candidate = _parse_legend_candidate(grid, bg, pair_row, result_row, align)
            if candidate is not None:
                candidates.append(candidate)

    assert len(candidates) == 1, "expected exactly one corner legend"
    candidate = candidates[0]
    return np.asarray(candidate["mask"], dtype=bool), dict(candidate["mapping"])


def _extract_paths(
    grid: np.ndarray,
    bg: int,
    legend_mask: np.ndarray,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    colors = sorted(int(color) for color in np.unique(grid[~legend_mask]) if int(color) != bg)
    assert colors, "expected non-legend paths"

    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []
    for color in colors:
        coords = np.argwhere((grid == color) & (~legend_mask))
        assert len(coords) >= 2, "each path color must remain visible outside the legend"
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if len(rows) == 1 and len(cols) >= 2:
            horizontals.append(
                {
                    "color": int(color),
                    "row": int(rows[0]),
                    "start": int(cols.min()),
                    "end": int(cols.max()),
                }
            )
        elif len(cols) == 1 and len(rows) >= 2:
            verticals.append(
                {
                    "color": int(color),
                    "col": int(cols[0]),
                    "start": int(rows.min()),
                    "end": int(rows.max()),
                }
            )
        else:
            raise AssertionError("each path must be a single straight horizontal or vertical segment")

    assert horizontals and verticals, "expected both horizontal and vertical paths"
    assert len({path["row"] for path in horizontals}) == len(horizontals)
    assert len({path["col"] for path in verticals}) == len(verticals)

    by_row = {path["row"]: path for path in horizontals}
    by_col = {path["col"]: path for path in verticals}

    # Any hidden cells inside a path span must be explained by an orthogonal crossing.
    for path in horizontals:
        row = path["row"]
        for col in range(path["start"], path["end"] + 1):
            value = int(grid[row, col])
            if value == path["color"]:
                continue
            other = by_col.get(col)
            assert other is not None and other["start"] <= row <= other["end"]

    for path in verticals:
        col = path["col"]
        for row in range(path["start"], path["end"] + 1):
            value = int(grid[row, col])
            if value == path["color"]:
                continue
            other = by_row.get(row)
            assert other is not None and other["start"] <= col <= other["end"]

    return horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover orthogonal line crossings and mark each crossing with a legend-defined plus.

    Input structure:
    - Background color fills most of the canvas.
    - A corner 2xN legend encodes unordered path-color pairs -> marker colors.
    - Outside the legend, each foreground color is one long horizontal or vertical path.
    - Crossings may be hidden because the later-drawn path overwrote the earlier one.

    Rule:
    - Infer each path's full span from its visible row/column extent.
    - Reconstruct every horizontal/vertical crossing between different colors.
    - Look up the crossing's unordered color pair in the legend.
    - Paint a 3x3 "+" centered on the crossing with the mapped color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    legend_mask, mapping = _find_legend(g, bg)
    horizontals, verticals = _extract_paths(g, bg, legend_mask)

    out = g.copy()
    h, w = out.shape
    for h_path in horizontals:
        row = h_path["row"]
        for v_path in verticals:
            col = v_path["col"]
            if not (h_path["start"] <= col <= h_path["end"]):
                continue
            if not (v_path["start"] <= row <= v_path["end"]):
                continue
            key = tuple(sorted((h_path["color"], v_path["color"])))
            assert key in mapping, "every crossing color pair must appear in the legend"
            mark = int(mapping[key])
            for dr, dc in PLUS_DELTAS:
                rr = row + dr
                cc = col + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = mark

    return out


def _sample_spaced_positions(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    compressed_hi = hi - (gap - 1) * (count - 1)
    if compressed_hi - lo + 1 < count:
        return None
    base = sorted(random.sample(range(lo, compressed_hi + 1), count))
    return [value + (gap - 1) * i for i, value in enumerate(base)]


def _legend_layout(corner: str, h: int, w: int, legend_width: int) -> tuple[int, int, int]:
    is_top = corner.startswith("top")
    is_left = corner.endswith("left")
    pair_row = 0 if is_top else h - 2
    result_row = pair_row + 1
    start_col = 0 if is_left else w - legend_width
    return pair_row, result_row, start_col


def _path_margins(corner: str, legend_width: int) -> tuple[int, int, int, int]:
    top_margin = 3 if corner.startswith("top") else 1
    bottom_margin = 3 if corner.startswith("bottom") else 1
    left_margin = legend_width + 1 if corner.endswith("left") else 1
    right_margin = legend_width + 1 if corner.endswith("right") else 1
    return top_margin, bottom_margin, left_margin, right_margin


def _random_span(lo: int, hi: int, min_span: int) -> tuple[int, int] | None:
    total = hi - lo + 1
    if total < min_span:
        return None
    extra = total - min_span
    cut = random.randint(0, extra)
    left_pad = random.randint(0, cut)
    right_pad = cut - left_pad
    return lo + left_pad, hi - right_pad


def _draw_legend(
    grid: np.ndarray,
    corner: str,
    entries: list[tuple[tuple[int, int], int]],
    phase: int,
) -> np.ndarray:
    legend_width = 2 * len(entries)
    pair_row, result_row, start_col = _legend_layout(corner, grid.shape[0], grid.shape[1], legend_width)
    mask = np.zeros_like(grid, dtype=bool)
    for idx, (pair, result) in enumerate(entries):
        c0 = start_col + 2 * idx
        a, b = pair
        if random.random() < 0.5:
            a, b = b, a
        grid[pair_row, c0] = a
        grid[pair_row, c0 + 1] = b
        grid[result_row, c0 + phase] = result
        mask[pair_row : result_row + 1, c0 : c0 + 2] = True
    return mask


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_h, n_v = random.choice(COUNT_OPTIONS)
    n_paths = n_h + n_v
    n_pairs = n_h * n_v
    corner = random.choice(CORNER_OPTIONS)
    legend_width = 2 * n_pairs

    row_span_min = max(MIN_SPAN, 3 * n_h)
    col_span_min = max(MIN_SPAN, 3 * n_v)
    top_margin, bottom_margin, left_margin, right_margin = _path_margins(corner, legend_width)
    min_h = max(GRID_MIN, top_margin + bottom_margin + row_span_min)
    min_w = max(GRID_MIN, left_margin + right_margin + col_span_min, legend_width + 1)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)
    palette = list(range(10))
    bg = random.choice(palette)
    available = [color for color in palette if color != bg]
    random.shuffle(available)
    path_colors = available[:n_paths]
    mark_palette = available[n_paths:]
    if not mark_palette:
        return None

    row_span = _random_span(top_margin, h - bottom_margin - 1, row_span_min)
    col_span = _random_span(left_margin, w - right_margin - 1, col_span_min)
    if row_span is None or col_span is None:
        return None
    top, bottom = row_span
    left, right = col_span

    rows = _sample_spaced_positions(top + 1, bottom - 1, n_h, PATH_GAP)
    cols = _sample_spaced_positions(left + 1, right - 1, n_v, PATH_GAP)
    if rows is None or cols is None:
        return None

    h_specs = [
        {"color": int(path_colors[i]), "row": int(rows[i]), "start": int(left), "end": int(right)}
        for i in range(n_h)
    ]
    v_specs = [
        {"color": int(path_colors[n_h + i]), "col": int(cols[i]), "start": int(top), "end": int(bottom)}
        for i in range(n_v)
    ]

    pair_to_mark: dict[tuple[int, int], int] = {}
    for h_spec in h_specs:
        for v_spec in v_specs:
            key = tuple(sorted((h_spec["color"], v_spec["color"])))
            pair_to_mark[key] = int(random.choice(mark_palette))

    legend_entries = list(pair_to_mark.items())
    random.shuffle(legend_entries)
    phase = random.randint(0, 1)

    inp = np.full((h, w), bg, dtype=int)
    legend_mask = _draw_legend(inp, corner, legend_entries, phase)

    draw_order = h_specs + v_specs
    random.shuffle(draw_order)
    for spec in draw_order:
        color = int(spec["color"])
        if "row" in spec:
            inp[spec["row"], spec["start"] : spec["end"] + 1] = color
        else:
            inp[spec["start"] : spec["end"] + 1, spec["col"]] = color

    for spec in h_specs:
        visible = int(np.sum((inp == spec["color"]) & (~legend_mask)))
        if visible < 2:
            return None
    for spec in v_specs:
        visible = int(np.sum((inp == spec["color"]) & (~legend_mask)))
        if visible < 2:
            return None

    out = inp.copy()
    for h_spec in h_specs:
        for v_spec in v_specs:
            mark = pair_to_mark[tuple(sorted((h_spec["color"], v_spec["color"])))]
            center_r = int(h_spec["row"])
            center_c = int(v_spec["col"])
            for dr, dc in PLUS_DELTAS:
                out[center_r + dr, center_c + dc] = mark

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        attempt = _generate_attempt()
        if attempt is None:
            continue
        inp, expected = attempt
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise AssertionError("failed to generate a valid puzzle instance")
