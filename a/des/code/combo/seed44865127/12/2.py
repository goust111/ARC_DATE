from __future__ import annotations

import random

import numpy as np


MIN_HEIGHT = 6
MAX_HEIGHT = 18
MIN_WIDTH = 12
MAX_WIDTH = 30
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _non_bg_rows(grid: np.ndarray, bg: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool(np.any(grid[r] != bg))]


def _row_cells(row: np.ndarray, bg: int) -> dict[int, int]:
    return {int(c): int(row[c]) for c in np.flatnonzero(row != bg)}


def _decode_mapping(rule_row: np.ndarray, bg: int) -> dict[int, int]:
    colors = [int(value) for value in rule_row.tolist() if int(value) != bg]
    assert colors and len(colors) % 2 == 0, "rule row must contain color pairs"

    mapping: dict[int, int] = {}
    for i in range(0, len(colors), 2):
        mapping[int(colors[i])] = int(colors[i + 1])
    return mapping


def _simulate_rows(
    mapping: dict[int, int],
    seed_cells: dict[int, int],
    available_rows: int,
    width: int,
) -> tuple[list[dict[int, int]], str, bool, bool]:
    current = dict(seed_cells)
    rows: list[dict[int, int]] = []
    conflict = False
    touched_unmapped = False

    for _ in range(int(available_rows)):
        next_cells: dict[int, int] = {}
        has_mapped_pixel = False
        overflow = False

        for col in sorted(current):
            color = int(current[col])
            out_color = mapping.get(color)
            if out_color is None:
                touched_unmapped = True
                continue

            has_mapped_pixel = True
            for next_col in (col - 1, col, col + 1):
                if not (0 <= next_col < width):
                    overflow = True
                    break
                previous = next_cells.get(int(next_col))
                if previous is not None and int(previous) != int(out_color):
                    conflict = True
                next_cells[int(next_col)] = int(out_color)
            if overflow:
                break

        if overflow:
            return rows, "width", conflict, touched_unmapped
        if not has_mapped_pixel:
            return rows, "extinct", conflict, touched_unmapped

        rows.append(next_cells)
        current = next_cells

    return rows, "bottom", conflict, touched_unmapped


def _expected_output(
    inp: np.ndarray,
    bg: int,
    seed_row: int,
    rows: list[dict[int, int]],
) -> np.ndarray:
    out = np.array(inp, copy=True)
    for step, row_cells in enumerate(rows, start=1):
        rr = int(seed_row + step)
        out[rr, :] = bg
        for col, color in row_cells.items():
            out[rr, int(col)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a seed row downward using replacement pairs encoded on the top rule row.

    Input structure:
    - One rule row near the top. Reading its non-background cells from left to right,
      each consecutive pair encodes input color -> output color.
    - After a blank separator row, one seed row contains the initial foreground pattern.
    - Lower rows are blank background.

    Rule:
    - The seed row is step 0.
    - For each later row, every mapped pixel produces the output color at the same
      column and at the immediate left/right columns in the next row.
    - Background pixels do nothing.
    - Stop before writing a row that would extend past the left/right edge, or when
      there is no room below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    rows = _non_bg_rows(g, bg)
    assert len(rows) >= 2, "expected a rule row and a seed row"

    rule_row = int(rows[0])
    seed_row = int(rows[1])
    mapping = _decode_mapping(g[rule_row], bg)
    seed_cells = _row_cells(g[seed_row], bg)

    grown_rows, _reason, _conflict, _touched_unmapped = _simulate_rows(
        mapping,
        seed_cells,
        g.shape[0] - seed_row - 1,
        g.shape[1],
    )
    return _expected_output(g, bg, seed_row, grown_rows)


def _build_live_mapping(
    live_colors: list[int],
    live_mode: str,
    persistent_sources: list[int],
    sink_sources: list[int],
    sink_output: int | None,
) -> dict[int, int]:
    mapping: dict[int, int] = {}

    for color in persistent_sources:
        mapping[int(color)] = int(live_colors[0])

    if len(live_colors) == 1:
        mapping[int(live_colors[0])] = int(live_colors[0])
    elif live_mode == "cycle":
        for i, color in enumerate(live_colors):
            mapping[int(color)] = int(live_colors[(i + 1) % len(live_colors)])
    else:
        for i, color in enumerate(live_colors[:-1]):
            mapping[int(color)] = int(live_colors[i + 1])
        mapping[int(live_colors[-1])] = int(live_colors[-1])

    if sink_output is not None:
        for color in sink_sources:
            mapping[int(color)] = int(sink_output)

    return mapping


def _build_persistent_band(source_colors: list[int]) -> tuple[list[tuple[int, int]], int]:
    segment_colors = list(source_colors)
    for _ in range(random.randint(0, 2)):
        segment_colors.append(int(random.choice(source_colors)))
    random.shuffle(segment_colors)

    pixels: list[tuple[int, int]] = []
    cursor = 0
    for i, color in enumerate(segment_colors):
        length = random.randint(1, 3)
        for offset in range(length):
            pixels.append((int(cursor + offset), int(color)))
        cursor += length
        if i + 1 < len(segment_colors):
            cursor += random.randint(0, 2)

    return pixels, int(cursor)


def _build_sink_segments(source_colors: list[int]) -> list[dict[str, int]]:
    if not source_colors:
        return []

    segment_colors = list(source_colors)
    for _ in range(random.randint(0, 1)):
        segment_colors.append(int(random.choice(source_colors)))
    random.shuffle(segment_colors)

    return [
        {"color": int(color), "length": int(random.randint(1, 2))}
        for color in segment_colors
    ]


def _candidate_starts(
    occupied: np.ndarray,
    lo: int,
    hi: int,
    length: int,
) -> list[int]:
    starts: list[int] = []
    for start in range(int(lo), int(hi) - int(length) + 2):
        end = int(start + length - 1)
        if bool(np.any(occupied[start : end + 1])):
            continue
        if start - 1 >= 0 and bool(occupied[start - 1]):
            continue
        if end + 1 < occupied.shape[0] and bool(occupied[end + 1]):
            continue
        starts.append(int(start))
    return starts


def _place_sink_segments(
    width: int,
    band_start: int,
    band_end: int,
    segments: list[dict[str, int]],
) -> list[tuple[int, int]] | None:
    if not segments:
        return []

    occupied = np.zeros(width, dtype=bool)
    placed: list[tuple[int, int]] = []

    intervals = [
        (1, band_start - 3),
        (band_end + 3, width - 2),
    ]

    for segment in segments:
        choices: list[list[int]] = []
        for lo, hi in intervals:
            if lo > hi:
                continue
            starts = _candidate_starts(occupied, int(lo), int(hi), int(segment["length"]))
            if starts:
                choices.append(starts)
        if not choices:
            return None

        starts = random.choice(choices)
        start = int(random.choice(starts))
        for col in range(start, start + int(segment["length"])):
            occupied[col] = True
            placed.append((int(col), int(segment["color"])))

    return placed


def _render_rule_row(width: int, bg: int, mapping: dict[int, int]) -> np.ndarray:
    items = list(mapping.items())
    random.shuffle(items)

    row = np.full(width, bg, dtype=int)
    n = len(items)
    min_width = 2 * n + 2
    assert width >= min_width

    gaps = [1] + [0] * (n - 1) + [1]
    spare = width - min_width
    for _ in range(spare):
        gaps[random.randrange(len(gaps))] += 1

    cursor = gaps[0]
    for i, (src, dst) in enumerate(items):
        row[cursor] = int(src)
        row[cursor + 1] = int(dst)
        cursor += 2
        cursor += gaps[i + 1]

    return row


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    stop_mode = random.choice(["bottom", "width"])
    include_sink = random.choice([False, True])
    live_len = random.randint(1, 3)
    live_mode = random.choice(["cycle", "tail"])
    persistent_count = random.randint(1, 3)
    sink_count = random.randint(1, 2) if include_sink else 0

    needed = live_len + persistent_count + sink_count + (1 if include_sink else 0)
    if needed > 9:
        return None

    palette = random.sample([color for color in range(10) if color != bg], needed)
    idx = 0
    live_colors = [int(color) for color in palette[idx : idx + live_len]]
    idx += live_len
    persistent_sources = [int(color) for color in palette[idx : idx + persistent_count]]
    idx += persistent_count
    sink_sources = [int(color) for color in palette[idx : idx + sink_count]]
    idx += sink_count
    sink_output = int(palette[idx]) if include_sink else None

    mapping = _build_live_mapping(
        live_colors,
        live_mode,
        persistent_sources,
        sink_sources,
        sink_output,
    )

    band_pixels, band_width = _build_persistent_band(persistent_sources)
    sink_segments = _build_sink_segments(sink_sources)
    rule_min_width = 2 * len(mapping) + 2
    target_steps = random.randint(3, 7)

    if stop_mode == "bottom":
        height = target_steps + 3
        if not (MIN_HEIGHT <= height <= MAX_HEIGHT):
            return None

        left_margin = target_steps + 1 + random.randint(0, 3)
        right_margin = target_steps + 1 + random.randint(0, 4)
        width = max(MIN_WIDTH, rule_min_width, band_width + left_margin + right_margin)
        if width > MAX_WIDTH:
            return None
        band_start = int(left_margin)
    else:
        height = target_steps + 4 + random.randint(0, 3)
        if not (MIN_HEIGHT <= height <= MAX_HEIGHT):
            return None

        far_margin = target_steps + random.randint(0, 4)
        width = max(MIN_WIDTH, rule_min_width, band_width + target_steps + far_margin)
        if width > MAX_WIDTH:
            return None
        if random.choice([False, True]):
            band_start = int(target_steps)
        else:
            band_start = int(width - band_width - target_steps)

    band_end = int(band_start + band_width - 1)
    if band_start < 0 or band_end >= width:
        return None

    sink_pixels = _place_sink_segments(width, band_start, band_end, sink_segments)
    if sink_pixels is None:
        return None

    inp = np.full((height, width), bg, dtype=int)
    inp[0] = _render_rule_row(width, bg, mapping)
    for offset, color in band_pixels:
        inp[2, band_start + int(offset)] = int(color)
    for col, color in sink_pixels:
        inp[2, int(col)] = int(color)

    seed_cells = _row_cells(inp[2], bg)
    rows, reason, conflict, touched_unmapped = _simulate_rows(
        mapping,
        seed_cells,
        height - 3,
        width,
    )
    if reason != stop_mode:
        return None
    if len(rows) != target_steps:
        return None
    if conflict:
        return None
    if include_sink and not touched_unmapped:
        return None
    if len(rows) < 3:
        return None

    expected = _expected_output(inp, bg, 2, rows)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    non_bg_input_rows = _non_bg_rows(inp, bg)
    if non_bg_input_rows != [0, 2]:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
