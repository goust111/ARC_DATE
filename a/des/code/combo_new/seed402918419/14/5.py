from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
BODY_H_RANGE = (8, 18)
BODY_W_RANGE = (8, 22)
MAX_PATHS_PER_AXIS = 3
LINE_MIN_STEP = 2
GEN_ATTEMPTS = 400
POSITION_ATTEMPTS = 200


def _read_legend(grid: np.ndarray) -> list[int]:
    legend = grid[-1]
    colors = [int(legend[c]) for c in np.flatnonzero(legend != BG)]
    assert colors, "expected a non-empty bottom legend"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return colors


def _extract_paths(body: np.ndarray, legend_colors: list[int]) -> dict[int, tuple[str, int, int, int]]:
    seen_colors = set(map(int, np.unique(body))) - {BG}
    assert seen_colors == set(legend_colors), "body colors must match legend colors"

    paths: dict[int, tuple[str, int, int, int]] = {}
    for color in legend_colors:
        coords = np.argwhere(body == color)
        assert coords.size > 0, f"missing path for color {color}"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1 and len(cols) >= 2:
            r = int(rows[0])
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            paths[color] = ("h", r, c0, c1)
            continue
        if len(cols) == 1 and len(rows) >= 2:
            c = int(cols[0])
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            paths[color] = ("v", c, r0, r1)
            continue
        raise AssertionError(f"color {color} does not form a recoverable straight path")

    return paths


def _paint_paths(shape: tuple[int, int], paths: dict[int, tuple[str, int, int, int]], order: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order:
        axis, fixed, start, end = paths[color]
        if axis == "h":
            out[fixed, start : end + 1] = color
        else:
            out[start : end + 1, fixed] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint crossed straight paths using the bottom legend as the true layer order.

    Input structure:
    - The background is black.
    - Above the last row, each non-black color forms one single-pixel-wide straight path that is either horizontal or vertical.
    - The bottom row is a legend of colored single pixels ordered from lower layer on the left to higher layer on the right.

    Rule:
    - Read the bottom legend to obtain the global layer priority.
    - Recover each path from its visible same-color cells, extending it across the full span between its visible endpoints.
    - Redraw all recovered paths from low layer to high layer so each crossing takes the higher-priority color.
    - Remove the legend row from the output.

    Key insight:
    - The crossing colors cannot be trusted directly; the reliable signal is the legend, and the visible fragments of each color are enough to reconstruct the hidden straight path underneath.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a bottom legend row"
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9, "expected ARC color ids"

    legend_colors = _read_legend(g)
    body = g[:-1]
    paths = _extract_paths(body, legend_colors)
    return _paint_paths(body.shape, paths, legend_colors)


def _sample_spaced_positions(n: int, lo: int, hi: int, min_step: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if n > len(values):
        return None
    for _ in range(POSITION_ATTEMPTS):
        picks = sorted(random.sample(values, n))
        if all(b - a >= min_step for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _feasible_axis_count(limit: int) -> int:
    return min(MAX_PATHS_PER_AXIS, len(list(range(1, limit - 1, LINE_MIN_STEP))))


def _crossings(
    h_paths: list[dict[str, int | str]], v_paths: list[dict[str, int | str]]
) -> list[tuple[int, int, int, int]]:
    hits: list[tuple[int, int, int, int]] = []
    for hp in h_paths:
        hr = int(hp["fixed"])
        hc0 = int(hp["start"])
        hc1 = int(hp["end"])
        for vp in v_paths:
            vc = int(vp["fixed"])
            vr0 = int(vp["start"])
            vr1 = int(vp["end"])
            if hc0 <= vc <= hc1 and vr0 <= hr <= vr1:
                hits.append((hr, vc, int(hp["color"]), int(vp["color"])))
    return hits


def _render_input_body(
    shape: tuple[int, int],
    h_paths: list[dict[str, int | str]],
    v_paths: list[dict[str, int | str]],
    order: list[int],
) -> np.ndarray:
    body = np.full(shape, BG, dtype=int)

    for path in h_paths + v_paths:
        color = int(path["color"])
        fixed = int(path["fixed"])
        start = int(path["start"])
        end = int(path["end"])
        if path["axis"] == "h":
            body[fixed, start : end + 1] = color
        else:
            body[start : end + 1, fixed] = color

    priority = {color: idx for idx, color in enumerate(order)}
    crossings = _crossings(h_paths, v_paths)
    keep_high = [bool(random.getrandbits(1)) for _ in crossings]
    if len(keep_high) >= 2 and len(set(keep_high)) == 1:
        flip_idx = random.randrange(len(keep_high))
        keep_high[flip_idx] = not keep_high[flip_idx]

    for (r, c, h_color, v_color), high_flag in zip(crossings, keep_high):
        if priority[h_color] > priority[v_color]:
            high_color, low_color = h_color, v_color
        else:
            high_color, low_color = v_color, h_color
        body[r, c] = high_color if high_flag else low_color

    return body


def _gen_attempt() -> np.ndarray | None:
    body_h = random.randint(*BODY_H_RANGE)
    body_w = random.randint(*BODY_W_RANGE)

    max_h = _feasible_axis_count(body_h)
    max_v = _feasible_axis_count(body_w)
    count_pairs = [
        (n_h, n_v)
        for n_h in range(1, max_h + 1)
        for n_v in range(1, max_v + 1)
        if n_h * n_v >= 2
    ]
    if not count_pairs:
        return None
    n_h, n_v = random.choice(count_pairs)

    rows = _sample_spaced_positions(n_h, 1, body_h - 2, LINE_MIN_STEP)
    cols = _sample_spaced_positions(n_v, 1, body_w - 2, LINE_MIN_STEP)
    if rows is None or cols is None:
        return None

    colors = random.sample(COLOR_POOL, n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]

    min_col = min(cols)
    max_col = max(cols)
    min_row = min(rows)
    max_row = max(rows)

    h_paths: list[dict[str, int | str]] = []
    for row, color in zip(rows, h_colors):
        start = random.randint(0, min_col - 1)
        end = random.randint(max_col + 1, body_w - 1)
        h_paths.append({"color": color, "axis": "h", "fixed": row, "start": start, "end": end})

    v_paths: list[dict[str, int | str]] = []
    for col, color in zip(cols, v_colors):
        start = random.randint(0, min_row - 1)
        end = random.randint(max_row + 1, body_h - 1)
        v_paths.append({"color": color, "axis": "v", "fixed": col, "start": start, "end": end})

    order = random.sample(colors, len(colors))
    body = _render_input_body((body_h, body_w), h_paths, v_paths, order)

    if any(np.count_nonzero(body == color) < 2 for color in colors):
        return None

    legend = np.full(body_w, BG, dtype=int)
    legend_cols = sorted(random.sample(range(body_w), len(order)))
    for col, color in zip(legend_cols, order):
        legend[col] = color

    inp = np.vstack([body, legend[None, :]])
    path_lookup = {
        int(path["color"]): (
            str(path["axis"]),
            int(path["fixed"]),
            int(path["start"]),
            int(path["end"]),
        )
        for path in h_paths + v_paths
    }
    expected = _paint_paths((body_h, body_w), path_lookup, order)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-path input")
