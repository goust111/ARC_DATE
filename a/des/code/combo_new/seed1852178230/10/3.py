from __future__ import annotations

import numpy as np


def _extract_segments(grid: np.ndarray) -> list[dict[str, int | str]]:
    segments: list[dict[str, int | str]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != 0):
        coords = np.argwhere(grid == color)
        assert len(coords) >= 2, "each line must leave at least two visible cells"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            segments.append(
                {
                    "color": color,
                    "orientation": "h",
                    "row": int(rows[0]),
                    "start": int(cols.min()),
                    "end": int(cols.max()),
                }
            )
        elif len(cols) == 1:
            segments.append(
                {
                    "color": color,
                    "orientation": "v",
                    "col": int(cols[0]),
                    "start": int(rows.min()),
                    "end": int(rows.max()),
                }
            )
        else:
            raise AssertionError(f"color {color} is not a single horizontal or vertical segment")
    assert segments, "expected at least one colored segment"
    return segments


def _crosses(horizontal: dict[str, int | str], vertical: dict[str, int | str]) -> bool:
    return (
        int(horizontal["start"]) <= int(vertical["col"]) <= int(horizontal["end"])
        and int(vertical["start"]) <= int(horizontal["row"]) <= int(vertical["end"])
    )


def _reachability(edges: dict[int, set[int]]) -> dict[int, set[int]]:
    reach: dict[int, set[int]] = {}
    for color in edges:
        seen: set[int] = set()
        stack = list(edges[color])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(edges[node] - seen)
        reach[color] = seen
    return reach


def _bottom_to_top_order(segments: list[dict[str, int | str]], grid: np.ndarray) -> list[dict[str, int | str]]:
    horizontals = [segment for segment in segments if segment["orientation"] == "h"]
    verticals = [segment for segment in segments if segment["orientation"] == "v"]

    colors = [int(segment["color"]) for segment in segments]
    edges: dict[int, set[int]] = {color: set() for color in colors}

    for horizontal in horizontals:
        for vertical in verticals:
            if not _crosses(horizontal, vertical):
                continue

            r = int(horizontal["row"])
            c = int(vertical["col"])
            visible = int(grid[r, c])
            h_color = int(horizontal["color"])
            v_color = int(vertical["color"])
            if visible == h_color:
                edges[v_color].add(h_color)
            elif visible == v_color:
                edges[h_color].add(v_color)
            else:
                raise AssertionError("a crossing must show one of the two crossing segment colors")

    reach = _reachability(edges)
    for color in colors:
        assert color not in reach[color], "occlusion order must be acyclic"

    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            a_before_b = b in reach[a]
            b_before_a = a in reach[b]
            assert a_before_b ^ b_before_a, "crossing relations must determine a total order"

    rank = {color: len(reach[color]) for color in colors}
    assert len(set(rank.values())) == len(colors), "a total order must give each segment a unique rank"

    by_color = {int(segment["color"]): segment for segment in segments}
    ordered_colors = sorted(colors, key=lambda color: -rank[color])
    return [by_color[color] for color in ordered_colors]


def _render(shape: tuple[int, int], segments: list[dict[str, int | str]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for segment in segments:
        color = int(segment["color"])
        if segment["orientation"] == "h":
            row = int(segment["row"])
            out[row, int(segment["start"]) : int(segment["end"]) + 1] = color
        else:
            col = int(segment["col"])
            out[int(segment["start"]) : int(segment["end"]) + 1, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the full crossed line segments and redraw them in their bottom-to-top stacking order.

    Input structure:
    - The background is black.
    - Each foreground color belongs to exactly one one-cell-wide horizontal or vertical line segment.
    - Horizontal and vertical segments may cross; at each crossing only the upper segment's color is visible.
    - Every segment still has enough visible pixels to reveal its row/column and its two endpoints.

    Rule:
    - For each color, infer the complete straight segment from the visible cells on its single row or single column.
    - At every reconstructed horizontal/vertical crossing, read the visible color to decide which segment is above the other.
    - Combine those pairwise crossing relations into the unique global order from bottom to top.
    - Return a same-size black canvas and redraw every full segment in that order, painting lower segments first and upper segments last.

    Key insight:
    - A hidden crossing cell removes only one pixel of the lower segment, so the segment's full extent is still recoverable from its visible endpoints; the visible crossing colors then encode the stack order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments = _extract_segments(g)
    ordered = _bottom_to_top_order(segments, g)
    return _render(g.shape, ordered)


# --------------- generator (o5x) ---------------

_RNG = np.random.default_rng()

BACKGROUND = 0
COLOR_POOL = np.arange(1, 10, dtype=int)
LINE_COUNT_RANGE = (3, 8)
MIN_GAP = 2
MAX_POSITION_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _sample_spaced_positions(count: int, size: int) -> list[int] | None:
    candidates = np.arange(1, size - 1, dtype=int)
    if len(candidates) < count:
        return None

    for _ in range(MAX_POSITION_ATTEMPTS):
        picks = sorted(int(x) for x in _RNG.choice(candidates, size=count, replace=False))
        if all(b - a >= MIN_GAP for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _order_orientations(line_count: int) -> list[str]:
    start = "h" if int(_RNG.integers(2)) == 0 else "v"
    other = "v" if start == "h" else "h"
    return [start if i % 2 == 0 else other for i in range(line_count)]


def _build_segments() -> tuple[tuple[int, int], list[dict[str, int | str]]] | None:
    line_count = int(_RNG.integers(LINE_COUNT_RANGE[0], LINE_COUNT_RANGE[1] + 1))
    orientations = _order_orientations(line_count)
    h_count = orientations.count("h")
    v_count = orientations.count("v")

    height_min = max(7, 2 * h_count + 3)
    width_min = max(7, 2 * v_count + 3)
    height = int(_RNG.integers(height_min, 31))
    width = int(_RNG.integers(width_min, 31))

    rows = _sample_spaced_positions(h_count, height)
    cols = _sample_spaced_positions(v_count, width)
    if rows is None or cols is None:
        return None

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    if min_row < 1 or max_row > height - 2 or min_col < 1 or max_col > width - 2:
        return None

    row_pool = rows[:]
    col_pool = cols[:]
    _RNG.shuffle(row_pool)
    _RNG.shuffle(col_pool)
    colors = [int(c) for c in _RNG.choice(COLOR_POOL, size=line_count, replace=False)]

    segments: list[dict[str, int | str]] = []
    h_idx = 0
    v_idx = 0
    for color, orientation in zip(colors, orientations):
        if orientation == "h":
            row = row_pool[h_idx]
            h_idx += 1
            segments.append(
                {
                    "color": color,
                    "orientation": "h",
                    "row": int(row),
                    "start": int(_RNG.integers(0, min_col)),
                    "end": int(_RNG.integers(max_col + 1, width)),
                }
            )
        else:
            col = col_pool[v_idx]
            v_idx += 1
            segments.append(
                {
                    "color": color,
                    "orientation": "v",
                    "col": int(col),
                    "start": int(_RNG.integers(0, min_row)),
                    "end": int(_RNG.integers(max_row + 1, height)),
                }
            )

    return (height, width), segments


def _is_valid_visible_line(grid: np.ndarray, segment: dict[str, int | str]) -> bool:
    color = int(segment["color"])
    coords = np.argwhere(grid == color)
    if len(coords) < 2:
        return False
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if segment["orientation"] == "h":
        return len(rows) == 1 and int(rows[0]) == int(segment["row"]) and int(cols.min()) == int(segment["start"]) and int(cols.max()) == int(segment["end"])
    return len(cols) == 1 and int(cols[0]) == int(segment["col"]) and int(rows.min()) == int(segment["start"]) and int(rows.max()) == int(segment["end"])


def _gen_attempt() -> np.ndarray | None:
    built = _build_segments()
    if built is None:
        return None

    shape, ordered_segments = built
    inp = np.full(shape, BACKGROUND, dtype=int)
    for segment in ordered_segments:
        color = int(segment["color"])
        if segment["orientation"] == "h":
            inp[int(segment["row"]), int(segment["start"]) : int(segment["end"]) + 1] = color
        else:
            inp[int(segment["start"]) : int(segment["end"]) + 1, int(segment["col"])] = color

    for segment in ordered_segments:
        if not _is_valid_visible_line(inp, segment):
            return None

    try:
        out = main(inp)
    except AssertionError:
        return None

    if np.array_equal(out, _render(shape, ordered_segments)):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossed-line scene")
