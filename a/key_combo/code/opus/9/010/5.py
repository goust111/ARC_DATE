from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
LENGTH_CHOICES = tuple(range(2, 8))
BAR_COUNT_RANGE = (3, 5)
GRID_H_RANGE = (10, 16)
GRID_W_RANGE = (8, 16)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_bars(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert 3 <= len(colors) <= 5

    bars = []
    for color in colors:
        mask = g == int(color)
        assert _is_single_component(mask), "each bar color must form one 4-connected component"
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = int(coords.shape[0])
        assert min(height, width) == 1, "each object must be one pixel thick"
        assert area == max(height, width), "each object must be a solid thin bar"
        length = max(height, width)
        assert 2 <= length <= 7
        bars.append(
            {
                "color": int(color),
                "length": int(length),
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "orientation": "h" if height == 1 else "v",
            }
        )

    lengths = [bar["length"] for bar in bars]
    assert len(set(lengths)) == len(lengths), "bar lengths must be distinct"

    all_coords = []
    owners = []
    for idx, bar in enumerate(bars):
        for r, c in bar["coords"]:
            all_coords.append((int(r), int(c)))
            owners.append(idx)

    for i, (r1, c1) in enumerate(all_coords):
        for j in range(i + 1, len(all_coords)):
            r2, c2 = all_coords[j]
            if owners[i] != owners[j]:
                assert max(abs(r1 - r2), abs(c1 - c2)) > 1, "bars must be separated by background"

    return bg, bars


def _sequence_from_bars(bars: list[dict], bg: int) -> list[int]:
    ordered = sorted(bars, key=lambda bar: bar["length"])
    seq: list[int] = []
    for i, bar in enumerate(ordered):
        if i:
            seq.append(int(bg))
        seq.extend([int(bar["color"])] * int(bar["length"]))
    return seq


def _wrap_sequence(seq: list[int], width: int, bg: int) -> np.ndarray:
    height = (len(seq) + width - 1) // width
    out = np.full((height, width), int(bg), dtype=int)
    for idx, color in enumerate(seq):
        out[idx // width, idx % width] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Turn sorted thin bars into a wrapped multicolor run-length sequence.

    Input structure:
    - The grid has a dominant background color and 3 to 5 one-pixel-thick bars.
    - Each bar is horizontal or vertical, uses a distinct non-background color, and has a distinct length.
    - Bars are separated from one another by at least one background cell.

    Rule:
    - Measure every bar's length and keep its color.
    - Sort the bars by length in ascending order.
    - Build a 1D sequence by concatenating one run of that bar's color repeated by its length, inserting one background separator pixel between consecutive runs.
    - Wrap the sequence row-major using the input width, and pad the final row with background cells if needed.

    Key insight:
    - Each thin bar jointly encodes a run length and a run color, while sorting by bar length determines the run order before wrapping.
    """
    g = np.asarray(grid, dtype=int)
    bg, bars = _extract_bars(g)
    seq = _sequence_from_bars(bars, bg)
    return _wrap_sequence(seq, g.shape[1], bg)


def _bar_cells(start_r: int, start_c: int, length: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(start_r, start_c + dc) for dc in range(length)]
    return [(start_r + dr, start_c) for dr in range(length)]


def _expanded_cells(shape: tuple[int, int], cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    expanded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _place_bars(
    height: int, width: int, specs: list[dict]
) -> list[dict] | None:
    occupied = set()
    blocked = set()
    placed = []

    for spec in sorted(specs, key=lambda item: item["length"], reverse=True):
        length = int(spec["length"])
        orientation = spec["orientation"]
        candidates = []
        if orientation == "h":
            for r in range(height):
                for c in range(width - length + 1):
                    cells = _bar_cells(r, c, length, orientation)
                    if any(cell in blocked for cell in cells):
                        continue
                    expanded = _expanded_cells((height, width), cells)
                    if occupied & expanded:
                        continue
                    candidates.append((cells, expanded))
        else:
            for r in range(height - length + 1):
                for c in range(width):
                    cells = _bar_cells(r, c, length, orientation)
                    if any(cell in blocked for cell in cells):
                        continue
                    expanded = _expanded_cells((height, width), cells)
                    if occupied & expanded:
                        continue
                    candidates.append((cells, expanded))

        if not candidates:
            return None

        cells, expanded = random.choice(candidates)
        occupied.update(cells)
        blocked.update(expanded)
        arr = np.array(cells, dtype=int)
        placed.append(
            {
                "color": int(spec["color"]),
                "length": int(spec["length"]),
                "orientation": orientation,
                "coords": arr,
                "bbox": _bbox_coords(arr),
            }
        )

    return placed


def _row_major_order(bars: list[dict]) -> list[int]:
    return [
        idx
        for idx, _bar in sorted(
            enumerate(bars),
            key=lambda item: (item[1]["bbox"][0], item[1]["bbox"][1], item[1]["bbox"][2], item[1]["bbox"][3]),
        )
    ]


def _color_order(bars: list[dict]) -> list[int]:
    return [idx for idx, _bar in sorted(enumerate(bars), key=lambda item: item[1]["color"])]


def _length_order(bars: list[dict]) -> list[int]:
    return [idx for idx, _bar in sorted(enumerate(bars), key=lambda item: item[1]["length"])]


def _render_input(height: int, width: int, bars: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for bar in bars:
        coords = bar["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(bar["color"])
    return grid


def _valid_widths(seq_len: int, exact_fill: bool) -> list[int]:
    widths = []
    for width in range(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1):
        if seq_len <= width:
            continue
        if exact_fill and seq_len % width != 0:
            continue
        if (not exact_fill) and seq_len % width == 0:
            continue
        heights = (seq_len + width - 1) // width
        if heights <= 6:
            widths.append(width)
    return widths


def _choose_orientations(count: int) -> list[str]:
    if count >= 2 and random.random() < 0.8:
        orientations = ["h", "v"]
        orientations.extend(random.choice(("h", "v")) for _ in range(count - 2))
        random.shuffle(orientations)
        return orientations
    return [random.choice(("h", "v")) for _ in range(count)]


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        bg, bars = _extract_bars(grid)
    except AssertionError:
        return False

    seq_len = len(_sequence_from_bars(bars, bg))
    width = int(grid.shape[1])
    if seq_len <= width:
        return False

    length_order = _length_order(bars)
    row_major = _row_major_order(bars)
    if row_major == length_order or row_major == list(reversed(length_order)):
        return False

    color_order = _color_order(bars)
    if color_order == length_order or color_order == list(reversed(length_order)):
        return False

    orientations = {bar["orientation"] for bar in bars}
    if len(bars) >= 4 and len(orientations) == 1:
        return False

    out = main(grid)
    if out.shape[1] != width:
        return False
    if out.shape[0] != (seq_len + width - 1) // width:
        return False
    if seq_len % width != 0:
        if not np.all(out[-1, seq_len % width :] == bg):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*BAR_COUNT_RANGE)
    lengths = random.sample(LENGTH_CHOICES, count)
    colors = random.sample(PALETTE, count)
    specs = [
        {"length": int(length), "color": int(color), "orientation": orientation}
        for length, color, orientation in zip(lengths, colors, _choose_orientations(count))
    ]

    seq_len = sum(lengths) + count - 1
    exact_fill = random.random() < 0.4
    width_options = _valid_widths(seq_len, exact_fill)
    if not width_options:
        width_options = _valid_widths(seq_len, not exact_fill)
    if not width_options:
        return None

    random.shuffle(width_options)
    height_options = list(range(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    random.shuffle(height_options)

    for width in width_options:
        for height in height_options:
            if max(lengths) > max(height, width):
                continue
            placed = _place_bars(height, width, specs)
            if placed is None:
                continue
            grid = _render_input(height, width, placed)
            if not _semantic_ok(grid):
                continue
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
