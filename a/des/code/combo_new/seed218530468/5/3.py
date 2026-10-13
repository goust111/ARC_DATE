from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()

_MIN_SIZE = 12
_MAX_SIZE = 30
_OUTER_MARGIN = 2
_MIN_GAP = 1
_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _is_consecutive(values: np.ndarray) -> bool:
    if len(values) == 0:
        return False
    start = int(values[0])
    return bool(np.array_equal(values, np.arange(start, start + len(values))))


def _extract_stripes(grid: np.ndarray, bg: int) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])

        row_count = int(len(rows))
        col_count = int(len(cols))
        if row_count in (1, 2) and _is_consecutive(rows) and (c1 - c0 + 1) > row_count:
            horizontals.append(
                {
                    "color": color,
                    "width": row_count,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                }
            )
            continue
        if col_count in (1, 2) and _is_consecutive(cols) and (r1 - r0 + 1) > col_count:
            verticals.append(
                {
                    "color": color,
                    "width": col_count,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                }
            )
            continue
        raise ValueError(f"cannot classify stripe color {color}")

    return horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """Reweave hollow stripe crossings by letting the wider band cover each overlap.

    Input structure:
    - The dominant background surrounds several monochrome horizontal and vertical
      straight bands, each with its own color.
    - Every band is one or two pixels wide and extends as a single axis-aligned
      segment, but its crossing cells have been removed.
    - Each horizontal/vertical overlap rectangle is therefore visible as a
      background-colored hole.

    Rule:
    - Identify each band from its color, orientation, span, and width.
    - For every horizontal/vertical overlap rectangle, fill the whole missing
      crossing with the color of the wider band.
    - If both bands have the same width, use the horizontal band color.
    - Leave every non-crossing pixel unchanged.

    Key insight:
    - The background holes are intentional weave crossings, and the visible top
      band is determined purely by geometric width rather than by color or
      position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    horizontals, verticals = _extract_stripes(g, bg)

    out = g.copy()
    for h_band in horizontals:
        for v_band in verticals:
            rr0 = max(int(h_band["r0"]), int(v_band["r0"]))
            rr1 = min(int(h_band["r1"]), int(v_band["r1"]))
            cc0 = max(int(h_band["c0"]), int(v_band["c0"]))
            cc1 = min(int(h_band["c1"]), int(v_band["c1"]))
            if rr0 > rr1 or cc0 > cc1:
                continue

            if int(h_band["width"]) >= int(v_band["width"]):
                color = int(h_band["color"])
            else:
                color = int(v_band["color"])
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color

    return out


def _sample_widths(count: int) -> list[int]:
    widths = [1, 2]
    while len(widths) < count:
        widths.append(int(_RNG.integers(1, 3)))
    _RNG.shuffle(widths)
    return widths


def _place_blocks(axis_len: int, widths: list[int]) -> list[tuple[int, int]] | None:
    needed = 2 * _OUTER_MARGIN + sum(widths) + _MIN_GAP * (len(widths) - 1)
    extra = axis_len - needed
    if extra < 0:
        return None

    slack = _RNG.multinomial(extra, np.full(len(widths) + 1, 1.0 / (len(widths) + 1)))
    blocks: list[tuple[int, int]] = []
    pos = _OUTER_MARGIN + int(slack[0])
    for i, width in enumerate(widths):
        blocks.append((int(pos), int(width)))
        pos += int(width)
        if i < len(widths) - 1:
            pos += _MIN_GAP + int(slack[i + 1])
    return blocks


def _build_stripes(
    shape: tuple[int, int],
    h_widths: list[int],
    v_widths: list[int],
    colors: list[int],
) -> tuple[list[dict[str, int]], list[dict[str, int]]] | None:
    rows, cols = shape
    row_blocks = _place_blocks(rows, h_widths)
    col_blocks = _place_blocks(cols, v_widths)
    if row_blocks is None or col_blocks is None:
        return None

    top = min(start for start, _width in row_blocks)
    bottom = max(start + width - 1 for start, width in row_blocks)
    left = min(start for start, _width in col_blocks)
    right = max(start + width - 1 for start, width in col_blocks)

    if not (1 <= top - 1 and bottom + 1 <= rows - 2 and 1 <= left - 1 and right + 1 <= cols - 2):
        return None

    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []
    color_idx = 0

    for r0, width in row_blocks:
        c0 = int(_RNG.integers(1, left))
        c1 = int(_RNG.integers(right + 1, cols - 1))
        horizontals.append(
            {
                "color": int(colors[color_idx]),
                "width": int(width),
                "r0": int(r0),
                "r1": int(r0 + width - 1),
                "c0": int(c0),
                "c1": int(c1),
            }
        )
        color_idx += 1

    for c0, width in col_blocks:
        r0 = int(_RNG.integers(1, top))
        r1 = int(_RNG.integers(bottom + 1, rows - 1))
        verticals.append(
            {
                "color": int(colors[color_idx]),
                "width": int(width),
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c0 + width - 1),
            }
        )
        color_idx += 1

    return horizontals, verticals


def _render_input(shape: tuple[int, int], bg: int, horizontals: list[dict[str, int]], verticals: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for stripe in horizontals:
        grid[int(stripe["r0"]) : int(stripe["r1"]) + 1, int(stripe["c0"]) : int(stripe["c1"]) + 1] = int(stripe["color"])
    for stripe in verticals:
        grid[int(stripe["r0"]) : int(stripe["r1"]) + 1, int(stripe["c0"]) : int(stripe["c1"]) + 1] = int(stripe["color"])

    for h_band in horizontals:
        for v_band in verticals:
            rr0 = max(int(h_band["r0"]), int(v_band["r0"]))
            rr1 = min(int(h_band["r1"]), int(v_band["r1"]))
            cc0 = max(int(h_band["c0"]), int(v_band["c0"]))
            cc1 = min(int(h_band["c1"]), int(v_band["c1"]))
            if rr0 <= rr1 and cc0 <= cc1:
                grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(bg)
    return grid


def _render_expected(shape: tuple[int, int], bg: int, horizontals: list[dict[str, int]], verticals: list[dict[str, int]]) -> np.ndarray:
    out = _render_input(shape, bg, horizontals, verticals)
    for h_band in horizontals:
        for v_band in verticals:
            rr0 = max(int(h_band["r0"]), int(v_band["r0"]))
            rr1 = min(int(h_band["r1"]), int(v_band["r1"]))
            cc0 = max(int(h_band["c0"]), int(v_band["c0"]))
            cc1 = min(int(h_band["c1"]), int(v_band["c1"]))
            if rr0 > rr1 or cc0 > cc1:
                continue
            if int(h_band["width"]) >= int(v_band["width"]):
                color = int(h_band["color"])
            else:
                color = int(v_band["color"])
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
    return out


def _branch_signature(horizontals: list[dict[str, int]], verticals: list[dict[str, int]]) -> set[str]:
    kinds: set[str] = set()
    for h_band in horizontals:
        for v_band in verticals:
            if int(h_band["width"]) > int(v_band["width"]):
                kinds.add("h")
            elif int(h_band["width"]) < int(v_band["width"]):
                kinds.add("v")
            else:
                kinds.add("eq")
    return kinds


def _gen_attempt() -> np.ndarray | None:
    h_count = int(_RNG.integers(2, 5))
    v_count = int(_RNG.integers(2, 5))
    h_widths = _sample_widths(h_count)
    v_widths = _sample_widths(v_count)

    min_rows = max(_MIN_SIZE, 2 * _OUTER_MARGIN + sum(h_widths) + _MIN_GAP * (h_count - 1))
    min_cols = max(_MIN_SIZE, 2 * _OUTER_MARGIN + sum(v_widths) + _MIN_GAP * (v_count - 1))
    if min_rows > _MAX_SIZE or min_cols > _MAX_SIZE:
        return None

    rows = int(_RNG.integers(min_rows, _MAX_SIZE + 1))
    cols = int(_RNG.integers(min_cols, _MAX_SIZE + 1))

    bg = int(_RNG.integers(0, 10))
    palette = [color for color in range(10) if color != bg]
    total = h_count + v_count
    colors = [int(c) for c in _RNG.choice(palette, size=total, replace=False)]

    built = _build_stripes((rows, cols), h_widths, v_widths, colors)
    if built is None:
        return None
    horizontals, verticals = built

    if _branch_signature(horizontals, verticals) != {"h", "v", "eq"}:
        return None

    inp = _render_input((rows, cols), bg, horizontals, verticals)
    expected = _render_expected((rows, cols), bg, horizontals, verticals)

    if np.array_equal(inp, expected):
        return None
    values, counts = np.unique(inp, return_counts=True)
    bg_count = int(counts[np.where(values == bg)[0][0]])
    if _get_bg(inp) != bg or any(int(count) >= bg_count for value, count in zip(values, counts, strict=False) if int(value) != bg):
        return None
    if len(np.unique(inp)) != total + 1:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving puzzle")
