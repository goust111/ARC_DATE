from __future__ import annotations

from collections import Counter, deque
import hashlib
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

LEGEND_ENTRY_COUNT = 3
SQUARE_SIZE = 2
MAX_GENERATION_ATTEMPTS = 600

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(cells),
                }
            )
    return comps


def _separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] == int(grid[r, 0])) and int(grid[r, 0]) != bg
    ]
    assert len(rows) == 1, "expected exactly one solid separator row"
    return int(rows[0])


def _parse_legend(grid: np.ndarray, sep_row: int, bg: int):
    top = grid[:sep_row]
    comps = _find_components(top, bg)
    squares = []
    markers = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(comp["area"])
        if area == 1:
            markers.append(comp)
        else:
            assert h == SQUARE_SIZE and w == SQUARE_SIZE and area == SQUARE_SIZE * SQUARE_SIZE
            squares.append(comp)

    assert len(squares) == LEGEND_ENTRY_COUNT
    assert len(markers) == LEGEND_ENTRY_COUNT
    assert len({int(comp["color"]) for comp in squares}) == LEGEND_ENTRY_COUNT
    assert len({int(comp["color"]) for comp in markers}) == LEGEND_ENTRY_COUNT

    squares.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    markers.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))

    marker_order: list[int] = []
    mapping: dict[int, int] = {}
    offsets = set()
    for square, marker in zip(squares, markers, strict=True):
        r0, c0, r1, c1 = square["bbox"]
        mr, mc = map(int, marker["cells"][0])
        assert mr > int(r1), "legend marker must be below its square"
        assert mc in {int(c0), int(c1)}, "marker must align with one of the square's middle columns"
        marker_order.append(int(marker["color"]))
        mapping[int(marker["color"])] = int(square["color"])
        offsets.add((int(r0) - mr, int(c0) - mc))

    assert len(offsets) == 1, "all legend entries must share one anchor convention"
    stamp_offset = next(iter(offsets))
    return mapping, marker_order, stamp_offset, (SQUARE_SIZE, SQUARE_SIZE)


def _panel_boundaries(lower: np.ndarray, sep_color: int) -> list[int]:
    cols = [c for c in range(lower.shape[1]) if np.all(lower[:, c] == sep_color)]
    assert len(cols) >= 4, "need 3+ panels separated by shared border columns"
    assert cols[0] == 0 and cols[-1] == lower.shape[1] - 1
    strides = [b - a for a, b in zip(cols, cols[1:])]
    assert len(set(strides)) == 1 and strides[0] >= 3
    return cols


def _parse_frame_sequence(lower: np.ndarray, sep_color: int, bg: int, allowed_colors: set[int]):
    boundaries = _panel_boundaries(lower, sep_color)
    panel_h = lower.shape[0]
    panel_w = boundaries[1] - boundaries[0] - 1
    assert panel_w >= 3

    rows: list[int] = []
    cols: list[int] = []
    colors: list[int] = []
    for c0, c1 in zip(boundaries, boundaries[1:]):
        panel = lower[:, c0 + 1 : c1]
        coords = np.argwhere(panel != bg)
        assert len(coords) == 1, "each frame must contain exactly one marker"
        r, c = map(int, coords[0])
        color = int(panel[r, c])
        assert color in allowed_colors
        rows.append(r)
        cols.append(c)
        colors.append(color)

    assert len(set(cols)) == 1, "marker column must stay fixed in panel coordinates"
    deltas = [b - a for a, b in zip(rows, rows[1:])]
    assert deltas and len(set(deltas)) == 1 and deltas[0] != 0
    return panel_h, panel_w, rows, cols[0], colors


def _build_output(
    panel_h: int,
    panel_w: int,
    marker_row: int,
    marker_col: int,
    square_color: int,
    stamp_offset: tuple[int, int],
    square_shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros((panel_h, panel_w), dtype=int)
    r0 = marker_row + int(stamp_offset[0])
    c0 = marker_col + int(stamp_offset[1])
    h, w = square_shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= panel_h and c0 + w <= panel_w
    out[r0 : r0 + h, c0 : c0 + w] = int(square_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the next panel marker, use the legend to reinterpret its color as a 2x2 sprite, and stamp that sprite in the new panel.

    Input structure:
    - The grid has a black background, a top legend, one solid grey separator row, and a lower strip of equal-width panels split by shared grey boundary columns.
    - The legend contains three filled 2x2 squares of distinct colors, each paired with one distinct singleton marker color directly below it.
    - Each lower panel contains exactly one singleton marker using one of those legend marker colors; its panel-local column stays fixed while its row changes by a constant amount from panel to panel.

    Rule:
    - Read the legend left-to-right to obtain the cyclic marker-color order and the mapping from each marker color to the color of the square above it.
    - From the lower panel strip, recover the constant vertical delta and continue the marker one more panel forward at the same local column; continue the marker color one step forward in the legend cycle.
    - Output only one clean panel of the same interior size, with black background, and stamp the mapped 2x2 square at the extrapolated marker position using the same marker-to-square anchor relation shown in the legend.

    Key insight:
    - The observed marker color is only an index into the legend, so solving requires combining three independent clues: panel-motion extrapolation for position, legend order for the next query color, and legend mapping for the square that must actually be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK, "background should be black"

    sep_row = _separator_row(g, bg)
    sep_color = int(g[sep_row, 0])
    mapping, marker_order, stamp_offset, square_shape = _parse_legend(g, sep_row, bg)

    lower = g[sep_row + 1 :]
    panel_h, panel_w, rows, col, colors = _parse_frame_sequence(lower, sep_color, bg, set(marker_order))

    start_idx = marker_order.index(colors[0])
    for i, color in enumerate(colors):
        assert color == marker_order[(start_idx + i) % len(marker_order)], "panel colors must follow legend order"

    delta = rows[1] - rows[0]
    next_row = rows[-1] + delta
    next_color = marker_order[(start_idx + len(colors)) % len(marker_order)]
    assert 0 <= next_row < panel_h

    return _build_output(
        panel_h,
        panel_w,
        next_row,
        col,
        mapping[next_color],
        stamp_offset,
        square_shape,
    )


def _choose_distinct_colors() -> tuple[list[int], list[int]]:
    picked = random.sample(PALETTE, 6)
    square_colors = picked[:LEGEND_ENTRY_COUNT]
    marker_colors = picked[LEGEND_ENTRY_COUNT:]
    random.shuffle(square_colors)
    while True:
        random.shuffle(marker_colors)
        if marker_colors != sorted(marker_colors):
            return square_colors, marker_colors


def _motion_choices(panel_h: int, n_panels: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for delta in (-2, -1, 1, 2):
        for start_row in range(3, panel_h):
            rows = [start_row + i * delta for i in range(n_panels + 1)]
            if all(3 <= row < panel_h for row in rows):
                choices.append((start_row, delta))
    return choices


def _oracle_output(spec: dict[str, object]) -> np.ndarray:
    marker_order = spec["marker_order"]
    square_colors = spec["square_colors"]
    marker_colors = spec["marker_colors"]
    color_map = {int(m): int(s) for m, s in zip(marker_colors, square_colors, strict=True)}
    next_color = marker_order[(int(spec["start_idx"]) + int(spec["n_panels"])) % LEGEND_ENTRY_COUNT]
    next_row = int(spec["rows"][-1]) + int(spec["delta"])
    return _build_output(
        int(spec["panel_h"]),
        int(spec["panel_w"]),
        next_row,
        int(spec["marker_col"]),
        color_map[int(next_color)],
        spec["stamp_offset"],
        (SQUARE_SIZE, SQUARE_SIZE),
    )


def _build_input(spec: dict[str, object]) -> np.ndarray:
    panel_h = int(spec["panel_h"])
    panel_w = int(spec["panel_w"])
    n_panels = int(spec["n_panels"])
    top_pad = int(spec["top_pad"])
    legend_left = int(spec["legend_left"])
    legend_stride = int(spec["legend_stride"])
    marker_anchor_col = int(spec["marker_anchor_col"])
    square_colors = [int(x) for x in spec["square_colors"]]
    marker_colors = [int(x) for x in spec["marker_colors"]]
    rows = [int(x) for x in spec["rows"]]
    marker_col = int(spec["marker_col"])
    start_idx = int(spec["start_idx"])

    lower_w = n_panels * panel_w + (n_panels + 1)
    sep_row = top_pad + 5
    height = sep_row + 1 + panel_h
    grid = np.zeros((height, lower_w), dtype=int)

    for i, color in enumerate(square_colors):
        c0 = legend_left + i * legend_stride
        grid[top_pad : top_pad + SQUARE_SIZE, c0 : c0 + SQUARE_SIZE] = color
        grid[top_pad + 3, c0 + marker_anchor_col] = marker_colors[i]

    grid[sep_row, :] = GREY

    boundaries = [i * (panel_w + 1) for i in range(n_panels + 1)]
    lower = grid[sep_row + 1 :]
    for bc in boundaries:
        lower[:, bc] = GREY
    for i, row in enumerate(rows):
        color = marker_colors[(start_idx + i) % LEGEND_ENTRY_COUNT]
        lower[row, boundaries[i] + 1 + marker_col] = color
    return grid


def _validate_generated(inp: np.ndarray, expected: np.ndarray, spec: dict[str, object]) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not isinstance(expected, np.ndarray) or expected.ndim != 2 or max(expected.shape) > 30:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if len(set(map(int, spec["marker_order"]))) != LEGEND_ENTRY_COUNT:
        return False
    marker_colors = [int(x) for x in spec["marker_colors"]]
    square_colors = [int(x) for x in spec["square_colors"]]
    if set(marker_colors) & set(square_colors):
        return False

    observed_colors = [
        marker_colors[(int(spec["start_idx"]) + i) % LEGEND_ENTRY_COUNT]
        for i in range(int(spec["n_panels"]))
    ]
    if set(observed_colors) != set(marker_colors):
        return False

    next_row = int(spec["rows"][-1]) + int(spec["delta"])
    next_color = marker_colors[(int(spec["start_idx"]) + int(spec["n_panels"])) % LEGEND_ENTRY_COUNT]
    mapped_color = square_colors[marker_colors.index(next_color)]
    if mapped_color == next_color:
        return False

    wrong_direct = _build_output(
        int(spec["panel_h"]),
        int(spec["panel_w"]),
        next_row,
        int(spec["marker_col"]),
        next_color,
        spec["stamp_offset"],
        (SQUARE_SIZE, SQUARE_SIZE),
    )
    if np.array_equal(wrong_direct, expected):
        return False

    wrong_last_position = _build_output(
        int(spec["panel_h"]),
        int(spec["panel_w"]),
        int(spec["rows"][-1]),
        int(spec["marker_col"]),
        mapped_color,
        spec["stamp_offset"],
        (SQUARE_SIZE, SQUARE_SIZE),
    )
    if np.array_equal(wrong_last_position, expected):
        return False

    last_color = observed_colors[-1]
    wrong_last_color = _build_output(
        int(spec["panel_h"]),
        int(spec["panel_w"]),
        next_row,
        int(spec["marker_col"]),
        square_colors[marker_colors.index(last_color)],
        spec["stamp_offset"],
        (SQUARE_SIZE, SQUARE_SIZE),
    )
    if np.array_equal(wrong_last_color, expected):
        return False

    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(7, 9)
    panel_w = random.randint(5, 7)
    n_panels = random.choice((3, 4))
    lower_w = n_panels * panel_w + (n_panels + 1)

    legend_stride = random.choice((4, 5, 6))
    legend_width = 2 + 2 * legend_stride
    if legend_width > lower_w:
        return None

    motion = _motion_choices(panel_h, n_panels)
    if not motion:
        return None
    start_row, delta = random.choice(motion)

    square_colors, marker_colors = _choose_distinct_colors()
    marker_anchor_col = random.choice((0, 1))
    marker_col_low = 1
    marker_col_high = panel_w - 2
    if marker_col_low > marker_col_high:
        return None
    marker_col = random.randint(marker_col_low, marker_col_high)

    top_pad = random.choice((0, 1))
    legend_left = random.randint(0, lower_w - legend_width)
    start_idx = random.randrange(LEGEND_ENTRY_COUNT)
    rows = [start_row + i * delta for i in range(n_panels)]

    spec: dict[str, object] = {
        "panel_h": panel_h,
        "panel_w": panel_w,
        "n_panels": n_panels,
        "top_pad": top_pad,
        "legend_left": legend_left,
        "legend_stride": legend_stride,
        "marker_anchor_col": marker_anchor_col,
        "marker_col": marker_col,
        "square_colors": square_colors,
        "marker_colors": marker_colors,
        "marker_order": marker_colors[:],
        "start_idx": start_idx,
        "rows": rows,
        "delta": delta,
        "stamp_offset": (-3, -marker_anchor_col),
    }

    inp = _build_input(spec)
    out = _oracle_output(spec)
    if not _validate_generated(inp, out, spec):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _generate_attempt()
        if attempt is None:
            continue
        inp, out = attempt
        in_sig = hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest()
        out_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(in_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return inp
    raise RuntimeError("failed to generate a valid non-degenerate puzzle instance")
