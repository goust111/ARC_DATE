from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400
HEIGHT_RANGE = (10, 18)
WIDTH_RANGE = (12, 22)
LINE_COUNT_RANGE = (2, 4)

_RNG = random.Random()


def _priority_from_legend(grid: np.ndarray) -> tuple[list[int], dict[int, int]]:
    legend_cols = np.flatnonzero(grid[0] != BG)
    assert legend_cols.size >= 2, "legend row must define at least two priorities"
    legend = [int(grid[0, c]) for c in legend_cols]
    assert len(legend) == len(set(legend)), "legend colors must be unique"
    return legend, {color: idx for idx, color in enumerate(legend)}


def _extract_lines(grid: np.ndarray) -> list[dict[str, int | str]]:
    lines: list[dict[str, int | str]] = []
    for color in (int(v) for v in np.unique(grid[1:, :]) if int(v) != BG):
        coords = np.argwhere(grid[1:, :] == color)
        rows = coords[:, 0] + 1
        cols = coords[:, 1]
        row_vals = np.unique(rows)
        col_vals = np.unique(cols)

        if row_vals.size == 1 and col_vals.size >= 2:
            lines.append(
                {
                    "orientation": "h",
                    "color": color,
                    "row": int(row_vals[0]),
                    "c0": int(cols.min()),
                    "c1": int(cols.max()),
                }
            )
        elif col_vals.size == 1 and row_vals.size >= 2:
            lines.append(
                {
                    "orientation": "v",
                    "color": color,
                    "col": int(col_vals[0]),
                    "r0": int(rows.min()),
                    "r1": int(rows.max()),
                }
            )
        else:
            raise AssertionError("each visible color must encode one axis-aligned segment")

    assert lines, "grid must contain colored segments below the legend"
    return lines


def _render_priority_output(
    shape: tuple[int, int],
    legend_row: np.ndarray,
    lines: list[dict[str, int | str]],
    priority: dict[int, int],
) -> np.ndarray:
    h, w = shape
    horiz = np.zeros((h, w), dtype=int)
    vert = np.zeros((h, w), dtype=int)

    for line in lines:
        color = int(line["color"])
        assert color in priority, "every line color must appear in the legend"
        if line["orientation"] == "h":
            r = int(line["row"])
            c0 = int(line["c0"])
            c1 = int(line["c1"])
            assert np.all(horiz[r, c0 : c1 + 1] == BG), "horizontal segments cannot overlap"
            horiz[r, c0 : c1 + 1] = color
        else:
            c = int(line["col"])
            r0 = int(line["r0"])
            r1 = int(line["r1"])
            assert np.all(vert[r0 : r1 + 1, c] == BG), "vertical segments cannot overlap"
            vert[r0 : r1 + 1, c] = color

    out = np.zeros((h, w), dtype=int)
    out[0, :] = legend_row
    for r in range(1, h):
        for c in range(w):
            h_color = int(horiz[r, c])
            v_color = int(vert[r, c])
            if h_color and v_color:
                out[r, c] = h_color if priority[h_color] < priority[v_color] else v_color
            else:
                out[r, c] = h_color or v_color
    return out


def _intersection_records(
    h_lines: list[dict[str, int | str]],
    v_lines: list[dict[str, int | str]],
    priority: dict[int, int],
) -> list[dict[str, int | bool]]:
    records: list[dict[str, int | bool]] = []
    for hi, h_line in enumerate(h_lines):
        hr = int(h_line["row"])
        hc0 = int(h_line["c0"])
        hc1 = int(h_line["c1"])
        h_color = int(h_line["color"])
        for vi, v_line in enumerate(v_lines):
            vc = int(v_line["col"])
            vr0 = int(v_line["r0"])
            vr1 = int(v_line["r1"])
            v_color = int(v_line["color"])
            if hc0 <= vc <= hc1 and vr0 <= hr <= vr1:
                h_wins = priority[h_color] < priority[v_color]
                low_at_endpoint = (hr == vr0 or hr == vr1) if h_wins else (vc == hc0 or vc == hc1)
                records.append(
                    {
                        "h_idx": hi,
                        "v_idx": vi,
                        "row": hr,
                        "col": vc,
                        "h_color": h_color,
                        "v_color": v_color,
                        "winner": h_color if h_wins else v_color,
                        "loser": v_color if h_wins else h_color,
                        "h_interior": hc0 < vc < hc1,
                        "v_interior": vr0 < hr < vr1,
                        "low_at_endpoint": low_at_endpoint,
                    }
                )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve horizontal/vertical line crossings using the top-row priority legend.

    The top row lists line colors from highest to lowest priority. Each body color
    belongs to one unique horizontal or vertical segment. At every crossing, the
    higher-priority color owns the shared cell and the lower-priority segment is cut.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 4

    _legend, priority = _priority_from_legend(g)
    lines = _extract_lines(g)
    return _render_priority_output(g.shape, g[0].copy(), lines, priority)


def _sample_interval(anchors: list[int], lo: int, hi: int, rng: random.Random) -> tuple[int, int] | None:
    assert len(anchors) >= 2
    left_idx = rng.randrange(len(anchors) - 1)
    right_idx = rng.randrange(left_idx + 1, len(anchors))

    if rng.random() < 0.4:
        start = anchors[left_idx]
    else:
        left_lo = lo if left_idx == 0 else anchors[left_idx - 1] + 1
        left_hi = anchors[left_idx] - 1
        start = rng.randint(left_lo, left_hi) if left_lo <= left_hi else anchors[left_idx]

    if rng.random() < 0.4:
        end = anchors[right_idx]
    else:
        right_lo = anchors[right_idx] + 1
        right_hi = hi if right_idx == len(anchors) - 1 else anchors[right_idx + 1] - 1
        end = rng.randint(right_lo, right_hi) if right_lo <= right_hi else anchors[right_idx]

    if end - start + 1 < 3:
        return None
    return int(start), int(end)


def _legend_row(priority_colors: list[int], width: int, rng: random.Random) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    cols = sorted(rng.sample(range(width), len(priority_colors)))
    for col, color in zip(cols, priority_colors):
        row[col] = int(color)
    return row


def _body_colors(grid: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(grid[1:, :]) if int(v) != BG}


def _visible_as_line(grid: np.ndarray, color: int) -> bool:
    coords = np.argwhere(grid[1:, :] == color)
    if coords.shape[0] < 2:
        return False
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    return (rows.size == 1 and cols.size >= 2) or (cols.size == 1 and rows.size >= 2)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_h = rng.randint(*LINE_COUNT_RANGE)
    n_v = rng.randint(*LINE_COUNT_RANGE)
    total = n_h + n_v
    if total > len(PALETTE):
        return None

    height = rng.randint(max(HEIGHT_RANGE[0], n_h + 6), HEIGHT_RANGE[1])
    width = rng.randint(max(WIDTH_RANGE[0], total + 4), WIDTH_RANGE[1])

    row_pool = list(range(2, height - 1))
    col_pool = list(range(1, width - 1))
    if len(row_pool) < n_h or len(col_pool) < n_v:
        return None

    h_rows = sorted(rng.sample(row_pool, n_h))
    v_cols = sorted(rng.sample(col_pool, n_v))
    colors = rng.sample(list(PALETTE), total)
    priority_colors = colors[:]
    rng.shuffle(priority_colors)
    priority = {color: idx for idx, color in enumerate(priority_colors)}

    h_lines: list[dict[str, int | str]] = []
    for row, color in zip(h_rows, colors[:n_h]):
        interval = _sample_interval(v_cols, 1, width - 2, rng)
        if interval is None:
            return None
        c0, c1 = interval
        h_lines.append({"orientation": "h", "color": int(color), "row": int(row), "c0": c0, "c1": c1})

    v_lines: list[dict[str, int | str]] = []
    for col, color in zip(v_cols, colors[n_h:]):
        interval = _sample_interval(h_rows, 2, height - 2, rng)
        if interval is None:
            return None
        r0, r1 = interval
        v_lines.append({"orientation": "v", "color": int(color), "col": int(col), "r0": r0, "r1": r1})

    records = _intersection_records(h_lines, v_lines, priority)
    if not (2 <= len(records) <= 10):
        return None
    if any(not any(rec["h_idx"] == i for rec in records) for i in range(n_h)):
        return None
    if any(not any(rec["v_idx"] == i for rec in records) for i in range(n_v)):
        return None
    if not any(int(rec["winner"]) == int(rec["h_color"]) for rec in records):
        return None
    if not any(int(rec["winner"]) == int(rec["v_color"]) for rec in records):
        return None
    if not any(bool(rec["h_interior"]) and bool(rec["v_interior"]) for rec in records):
        return None
    if rng.random() < 0.5 and not any(bool(rec["low_at_endpoint"]) for rec in records):
        return None

    legend = _legend_row(priority_colors, width, rng)
    lines = h_lines + v_lines
    out = _render_priority_output((height, width), legend, lines, priority)

    flippable = [rec for rec in records if bool(rec["h_interior"]) and bool(rec["v_interior"])]
    if not flippable:
        return None

    inp = out.copy()
    flip_count = rng.randint(1, len(flippable))
    for rec in rng.sample(flippable, flip_count):
        inp[int(rec["row"]), int(rec["col"])] = int(rec["loser"])

    palette = set(priority_colors)
    if _body_colors(inp) != palette or _body_colors(out) != palette:
        return None
    if np.array_equal(inp, out):
        return None
    if any(not _visible_as_line(inp, color) for color in palette):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
