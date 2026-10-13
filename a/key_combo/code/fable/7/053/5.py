import random
from typing import Dict, List, Tuple

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
COUNT_PAIRS = ((4, 5), (5, 4))
GRID_H_RANGE = (14, 22)
LEFT_BAND_RANGE = (10, 13)
RIGHT_EXTRA_RANGE = (2, 4)
GAP_RANGE = (2, 4)
GEN_RETRIES = 400


Line = Dict[str, int | np.ndarray | str]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_line(grid: np.ndarray, color: int) -> Line:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "each non-background color must appear"

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if rows.size == 1:
        row = int(rows[0])
        ordered = np.sort(coords[:, 1])
        expected = np.arange(int(ordered[0]), int(ordered[-1]) + 1)
        assert np.array_equal(ordered, expected), "horizontal segment must be contiguous"
        return {
            "color": int(color),
            "orientation": "H",
            "fixed": row,
            "start": int(ordered[0]),
            "end": int(ordered[-1]),
            "length": int(len(ordered)),
            "coords": coords,
        }

    if cols.size == 1:
        col = int(cols[0])
        ordered = np.sort(coords[:, 0])
        expected = np.arange(int(ordered[0]), int(ordered[-1]) + 1)
        assert np.array_equal(ordered, expected), "vertical segment must be contiguous"
        return {
            "color": int(color),
            "orientation": "V",
            "fixed": col,
            "start": int(ordered[0]),
            "end": int(ordered[-1]),
            "length": int(len(ordered)),
            "coords": coords,
        }

    raise AssertionError("each foreground object must be a straight horizontal or vertical segment")


def _extract_families(grid: np.ndarray) -> Tuple[List[Line], List[Line]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected foreground line segments"

    lines = [_parse_line(g, color) for color in colors]
    horizontals = [line for line in lines if line["orientation"] == "H"]
    verticals = [line for line in lines if line["orientation"] == "V"]

    assert horizontals and verticals, "expected both horizontal and vertical families"
    assert len({int(line["fixed"]) for line in horizontals}) == len(horizontals), "horizontal rows must be unique"
    assert len({int(line["fixed"]) for line in verticals}) == len(verticals), "vertical columns must be unique"

    horizontals.sort(key=lambda line: int(line["fixed"]))
    verticals.sort(key=lambda line: int(line["fixed"]))
    return horizontals, verticals


def _reference_line(lines: List[Line], target_length: int) -> Line:
    matches = [line for line in lines if int(line["length"]) == int(target_length)]
    assert len(matches) == 1, "reference line must be uniquely selected by the opposite family count"
    return matches[0]


def _aligned_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    horizontals, verticals = _extract_families(g)
    h_ref = _reference_line(horizontals, len(verticals))
    v_ref = _reference_line(verticals, len(horizontals))

    out = g.copy()
    for line in horizontals + verticals:
        if int(line["color"]) in (int(h_ref["color"]), int(v_ref["color"])):
            continue
        coords = line["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    h_start, h_end = int(h_ref["start"]), int(h_ref["end"])
    for line in horizontals:
        if int(line["color"]) == int(h_ref["color"]):
            continue
        out[int(line["fixed"]), h_start : h_end + 1] = int(line["color"])

    v_start, v_end = int(v_ref["start"]), int(v_ref["end"])
    for line in verticals:
        if int(line["color"]) == int(v_ref["color"]):
            continue
        out[v_start : v_end + 1, int(line["fixed"])] = int(line["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each orthogonal line family to its own reference line, where each reference is selected by the other family's count.

    Input structure:
    - Black cells are the background.
    - The foreground contains one-cell-thick horizontal segments on distinct rows and vertical segments on distinct columns.
    - Exactly one horizontal segment has length equal to the number of vertical segments, and exactly one vertical segment has length equal to the number of horizontal segments.

    Rule:
    - Keep the two reference segments unchanged.
    - For every other horizontal segment, preserve its row and color but rewrite its start and end columns to match the horizontal reference.
    - For every other vertical segment, preserve its column and color but rewrite its start and end rows to match the vertical reference.

    Key insight:
    - The reference in each family is not chosen by extremal length, position, or color; it is the unique segment whose length self-describes the opposite family's cardinality.
    """
    return _aligned_output(np.asarray(grid, dtype=int))


def _admissible_ref_indices(count: int) -> List[int]:
    indices = [i for i in range(count) if i not in (0, count - 1)]
    if count % 2 == 1:
        middle = count // 2
        indices = [i for i in indices if i != middle]
    return indices


def _sample_sorted_positions(count: int, low: int, high: int) -> List[int]:
    return sorted(random.sample(range(low, high + 1), count))


def _sample_reference_interval(space: int, ref_length: int) -> Tuple[int, int]:
    start = random.randint(1, space - ref_length - 2)
    end = start + ref_length - 1
    return start, end


def _sample_other_lengths(count: int, ref_length: int, own_count: int, max_length: int) -> List[int]:
    fixed = [2, ref_length + 2]
    pool = [
        length
        for length in range(2, max_length + 1)
        if length not in {ref_length, own_count} and length not in fixed
    ]
    assert len(pool) >= count - 3, "not enough alternative lengths"
    return fixed + random.sample(pool, count - 3)


def _sample_random_interval(space: int, ref_start: int, ref_end: int, length: int) -> Tuple[int, int]:
    starts = []
    for start in range(0, space - length + 1):
        end = start + length - 1
        if start == ref_start and end == ref_end:
            continue
        starts.append(start)
    assert starts, "expected at least one non-reference placement"
    start = random.choice(starts)
    return start, start + length - 1


def _sample_family_specs(count: int, space: int, ref_length: int, own_count: int) -> Tuple[List[Dict[str, int]], int]:
    ref_index = random.choice(_admissible_ref_indices(count))
    ref_start, ref_end = _sample_reference_interval(space, ref_length)

    other_lengths = _sample_other_lengths(count, ref_length, own_count, space - 1)
    used_short_witness = False
    used_long_witness = False
    other_specs = []

    for length in other_lengths:
        if not used_short_witness and length == 2:
            start = random.randint(ref_start + 1, ref_end - 2)
            end = start + length - 1
            used_short_witness = True
        elif not used_long_witness and length == ref_length + 2:
            start = ref_start - 1
            end = ref_end + 1
            used_long_witness = True
        else:
            start, end = _sample_random_interval(space, ref_start, ref_end, length)
        other_specs.append({"start": start, "end": end, "length": length})

    random.shuffle(other_specs)
    specs = []
    other_iter = iter(other_specs)
    for index in range(count):
        if index == ref_index:
            specs.append({"start": ref_start, "end": ref_end, "length": ref_length})
        else:
            specs.append(next(other_iter))
    return specs, ref_index


def _render_grid(
    height: int,
    width: int,
    rows: List[int],
    row_specs: List[Dict[str, int]],
    row_colors: List[int],
    cols: List[int],
    col_specs: List[Dict[str, int]],
    col_colors: List[int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for row, spec, color in zip(rows, row_specs, row_colors):
        grid[row, spec["start"] : spec["end"] + 1] = color
    for col, spec, color in zip(cols, col_specs, col_colors):
        grid[spec["start"] : spec["end"] + 1, col] = color
    return grid


def _family_map(lines: List[Line]) -> Dict[int, Line]:
    return {int(line["color"]): line for line in lines}


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        horizontals, verticals = _extract_families(grid)
        h_count = len(horizontals)
        v_count = len(verticals)
        if (h_count, v_count) not in COUNT_PAIRS:
            return False
        if len(np.unique(grid[grid != BG])) != h_count + v_count:
            return False

        h_ref = _reference_line(horizontals, v_count)
        v_ref = _reference_line(verticals, h_count)

        h_lengths = [int(line["length"]) for line in horizontals]
        v_lengths = [int(line["length"]) for line in verticals]
        if any(length == h_count for length in h_lengths):
            return False
        if any(length == v_count for length in v_lengths):
            return False
        if int(h_ref["length"]) in (min(h_lengths), max(h_lengths)):
            return False
        if int(v_ref["length"]) in (min(v_lengths), max(v_lengths)):
            return False

        h_ref_index = next(i for i, line in enumerate(horizontals) if int(line["color"]) == int(h_ref["color"]))
        v_ref_index = next(i for i, line in enumerate(verticals) if int(line["color"]) == int(v_ref["color"]))
        if h_ref_index not in _admissible_ref_indices(h_count):
            return False
        if v_ref_index not in _admissible_ref_indices(v_count):
            return False

        if not any(int(line["length"]) < int(h_ref["length"]) for line in horizontals if int(line["color"]) != int(h_ref["color"])):
            return False
        if not any(int(line["length"]) > int(h_ref["length"]) for line in horizontals if int(line["color"]) != int(h_ref["color"])):
            return False
        if not any(int(line["length"]) < int(v_ref["length"]) for line in verticals if int(line["color"]) != int(v_ref["color"])):
            return False
        if not any(int(line["length"]) > int(v_ref["length"]) for line in verticals if int(line["color"]) != int(v_ref["color"])):
            return False

        if not any(
            int(line["start"]) > int(h_ref["start"]) and int(line["end"]) < int(h_ref["end"])
            for line in horizontals
            if int(line["color"]) != int(h_ref["color"])
        ):
            return False
        if not any(
            int(line["start"]) < int(h_ref["start"]) and int(line["end"]) > int(h_ref["end"])
            for line in horizontals
            if int(line["color"]) != int(h_ref["color"])
        ):
            return False
        if not any(
            int(line["start"]) > int(v_ref["start"]) and int(line["end"]) < int(v_ref["end"])
            for line in verticals
            if int(line["color"]) != int(v_ref["color"])
        ):
            return False
        if not any(
            int(line["start"]) < int(v_ref["start"]) and int(line["end"]) > int(v_ref["end"])
            for line in verticals
            if int(line["color"]) != int(v_ref["color"])
        ):
            return False

        out = main(grid)
        h_out, v_out = _extract_families(out)
        h_by_color = _family_map(horizontals)
        v_by_color = _family_map(verticals)
        h_out_by_color = _family_map(h_out)
        v_out_by_color = _family_map(v_out)

        for color, line in h_by_color.items():
            new_line = h_out_by_color[color]
            if color == int(h_ref["color"]):
                if (int(new_line["start"]), int(new_line["end"])) != (int(line["start"]), int(line["end"])):
                    return False
            elif (int(new_line["start"]), int(new_line["end"])) != (int(h_ref["start"]), int(h_ref["end"])):
                return False

        for color, line in v_by_color.items():
            new_line = v_out_by_color[color]
            if color == int(v_ref["color"]):
                if (int(new_line["start"]), int(new_line["end"])) != (int(line["start"]), int(line["end"])):
                    return False
            elif (int(new_line["start"]), int(new_line["end"])) != (int(v_ref["start"]), int(v_ref["end"])):
                return False

        return True
    except Exception:
        return False


def _attempt() -> np.ndarray | None:
    h_count, v_count = random.choice(COUNT_PAIRS)
    height = random.randint(*GRID_H_RANGE)
    left_band = random.randint(*LEFT_BAND_RANGE)
    gap = random.randint(*GAP_RANGE)
    right_band = v_count + random.randint(*RIGHT_EXTRA_RANGE)
    width = left_band + gap + right_band

    rows = _sample_sorted_positions(h_count, 1, height - 2)
    cols = _sample_sorted_positions(v_count, left_band + gap, width - 1)
    row_specs, _ = _sample_family_specs(h_count, left_band, v_count, h_count)
    col_specs, _ = _sample_family_specs(v_count, height, h_count, v_count)

    colors = random.sample(PALETTE, h_count + v_count)
    row_colors = colors[:h_count]
    col_colors = colors[h_count:]
    grid = _render_grid(height, width, rows, row_specs, row_colors, cols, col_specs, col_colors)

    if width > 30 or height > 30:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 053/5")
