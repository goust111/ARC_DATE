import numpy as np


BG = 0
GREY = 5
LEGEND_HEIGHT = 3
PALETTE = tuple(color for color in range(10) if color not in (BG, GREY))
GEN_ATTEMPTS = 400

DIR_VECS = {
    "H": (0, 1),
    "V": (1, 0),
    "D1": (1, 1),
    "D2": (1, -1),
}

PATTERN_MASKS = {
    "plus": np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    "x": np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "mixed": np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
}


def _edge_distance(cell: tuple[int, int], height: int, width: int) -> int:
    r, c = cell
    return min(r, c, height - 1 - r, width - 1 - c)


def _depth_key(endpoints: tuple[tuple[int, int], tuple[int, int]], height: int, width: int) -> tuple[int, int]:
    distances = sorted(_edge_distance(cell, height, width) for cell in endpoints)
    return distances[0], distances[1]


def _nonempty_spans(block: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    width = block.shape[1]
    c = 0
    while c < width:
        if np.any(block[:, c] != BG):
            c0 = c
            while c < width and np.any(block[:, c] != BG):
                c += 1
            spans.append((c0, c - 1))
        else:
            c += 1
    return spans


def _flags_from_3x3(block: np.ndarray) -> dict[str, bool]:
    return {
        "H": int(block[1, 0]) != BG and int(block[1, 2]) != BG,
        "V": int(block[0, 1]) != BG and int(block[2, 1]) != BG,
        "D1": int(block[0, 0]) != BG and int(block[2, 2]) != BG,
        "D2": int(block[0, 2]) != BG and int(block[2, 0]) != BG,
    }


def _cross_type_from_flags(flags: dict[str, bool]) -> str:
    active = {name for name, present in flags.items() if present}
    assert len(active) == 2, f"expected exactly two line directions, got {active}"
    if active == {"H", "V"}:
        return "plus"
    if active == {"D1", "D2"}:
        return "x"
    return "mixed"


def _parse_legend(grid: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    assert grid.shape[0] > LEGEND_HEIGHT
    legend = grid[-LEGEND_HEIGHT:, :]
    grey_coords = np.argwhere(legend == GREY)
    assert grey_coords.shape == (3, 2), f"expected 3 legend grey centers, got {grey_coords}"
    assert set(int(r) for r, _ in grey_coords) == {1}, "legend grey centers must sit on the middle row"

    mapping: dict[str, np.ndarray] = {}
    for _, c_center in sorted((int(r), int(c)) for r, c in grey_coords):
        example = legend[:, c_center - 1 : c_center + 2]
        pattern = legend[:, c_center + 3 : c_center + 6]
        assert example.shape == (3, 3), "legend example must be 3x3"
        assert pattern.shape == (3, 3), "legend pattern must be 3x3"
        cross_type = _cross_type_from_flags(_flags_from_3x3(example))
        mask = pattern != BG
        assert np.any(mask), "legend pattern must be non-empty"
        mapping[cross_type] = mask

    assert set(mapping) == {"plus", "x", "mixed"}
    return mapping, grid.shape[0] - LEGEND_HEIGHT


def _real_cross_flags(grid: np.ndarray, r: int, c: int) -> dict[str, bool]:
    height, width = grid.shape
    assert 0 < r < height - 1 and 0 < c < width - 1
    return {
        "H": int(grid[r, c - 1]) != BG and int(grid[r, c + 1]) != BG,
        "V": int(grid[r - 1, c]) != BG and int(grid[r + 1, c]) != BG,
        "D1": int(grid[r - 1, c - 1]) != BG and int(grid[r + 1, c + 1]) != BG,
        "D2": int(grid[r - 1, c + 1]) != BG and int(grid[r + 1, c - 1]) != BG,
    }


def _trace_line(grid: np.ndarray, center: tuple[int, int], direction: str, play_height: int) -> dict[str, object]:
    r0, c0 = center
    height, width = grid.shape
    dr, dc = DIR_VECS[direction]
    colors: set[int] = set()
    endpoints: list[tuple[int, int]] = []

    for sign in (-1, 1):
        step_r = dr * sign
        step_c = dc * sign
        rr = r0 + step_r
        cc = c0 + step_c
        last = (r0, c0)
        saw_segment = False
        while 0 <= rr < play_height and 0 <= cc < width and int(grid[rr, cc]) != BG:
            saw_segment = True
            if int(grid[rr, cc]) != GREY:
                colors.add(int(grid[rr, cc]))
            last = (rr, cc)
            rr += step_r
            cc += step_c
        assert saw_segment, f"direction {direction} does not continue away from {center}"
        endpoints.append(last)

    assert len(colors) == 1, f"line through {center} in {direction} is not monochrome: {colors}"
    return {
        "direction": direction,
        "color": colors.pop(),
        "endpoints": (endpoints[0], endpoints[1]),
        "depth_key": _depth_key((endpoints[0], endpoints[1]), height, width),
    }


def _stamp_pattern(out: np.ndarray, center: tuple[int, int], mask: np.ndarray, color: int) -> None:
    height, width = out.shape
    r0, c0 = center
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if not mask[dr + 1, dc + 1]:
                continue
            rr = r0 + dr
            cc = c0 + dc
            if 0 <= rr < height and 0 <= cc < width and int(out[rr, cc]) == BG:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each grey line crossing and stamp the legend-selected ornament using the upper line's color.

    Input structure:
    - The background is black.
    - The main area contains single-pixel straight line segments in horizontal,
      vertical, or 45-degree diagonal directions.
    - Wherever exactly two segments cross, the crossing pixel is grey.
    - The bottom three rows form a legend row of three entries; each entry has a
      3x3 crossing example and a 3x3 ornament pattern.

    Rule:
    - Remove the legend from the output.
    - For each grey crossing in the main area, identify whether the two lines are
      orthogonal, diagonal-vs-diagonal, or orthogonal-vs-diagonal by inspecting
      the occupied neighbor pairs around the grey pixel.
    - Trace the two lines through the crossing, compute each line's nearest-endpoint
      distance to the grid edge, restore the crossing pixel with the lower line's
      color, and use the other line's color for the ornament.
    - Stamp the corresponding 3x3 ornament centered on the crossing, but only
      onto black background cells.

    Key insight:
    - The grey pixel is an occluded crossing, while the bottom legend maps
      crossing geometry to a shape mask; the line whose endpoint reaches closer
      to the border stays underneath and the farther one supplies the decoration color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pattern_map, play_height = _parse_legend(g)
    out = g[:play_height, :].copy()

    for r, c in map(tuple, np.argwhere(out == GREY)):
        flags = _real_cross_flags(g, int(r), int(c))
        cross_type = _cross_type_from_flags(flags)
        directions = [name for name, present in flags.items() if present]
        lines = [_trace_line(g, (int(r), int(c)), direction, play_height) for direction in directions]
        lines.sort(key=lambda item: item["depth_key"])
        lower, upper = lines
        assert lower["depth_key"] != upper["depth_key"], f"ambiguous layer ordering at {(int(r), int(c))}"

        out[int(r), int(c)] = int(lower["color"])
        _stamp_pattern(out, (int(r), int(c)), pattern_map[cross_type], int(upper["color"]))

    return out


def _line_cells(center: tuple[int, int], direction: str, neg_len: int, pos_len: int) -> list[tuple[int, int]]:
    r0, c0 = center
    dr, dc = DIR_VECS[direction]
    cells = [(r0 - dr * k, c0 - dc * k) for k in range(neg_len, 0, -1)]
    cells.append(center)
    cells.extend((r0 + dr * k, c0 + dc * k) for k in range(1, pos_len + 1))
    return cells


def _max_lengths(center: tuple[int, int], direction: str, play_height: int, width: int) -> tuple[int, int]:
    r, c = center
    if direction == "H":
        return c, width - 1 - c
    if direction == "V":
        return r, play_height - 1 - r
    if direction == "D1":
        return min(r, c), min(play_height - 1 - r, width - 1 - c)
    assert direction == "D2"
    return min(r, width - 1 - c), min(play_height - 1 - r, c)


def _zone3(center: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = center
    return {(r0 + dr, c0 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _paint_line(grid: np.ndarray, cells: list[tuple[int, int]], color: int, center: tuple[int, int]) -> None:
    for cell in cells:
        if cell == center:
            continue
        grid[cell] = color


def _build_example_block(cross_type: str, colors: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    color_a, color_b = colors
    block = np.full((3, 3), BG, dtype=int)
    block[1, 1] = GREY
    if cross_type == "plus":
        block[1, 0] = color_a
        block[1, 2] = color_a
        block[0, 1] = color_b
        block[2, 1] = color_b
    elif cross_type == "x":
        block[0, 0] = color_a
        block[2, 2] = color_a
        block[0, 2] = color_b
        block[2, 0] = color_b
    else:
        orth = "H" if int(rng.integers(0, 2)) == 0 else "V"
        diag = "D1" if int(rng.integers(0, 2)) == 0 else "D2"
        for rr, cc in _line_cells((1, 1), orth, 1, 1):
            if (rr, cc) != (1, 1):
                block[rr, cc] = color_a
        for rr, cc in _line_cells((1, 1), diag, 1, 1):
            if (rr, cc) != (1, 1):
                block[rr, cc] = color_b
    return block


def _build_pattern_block(mask: np.ndarray, color: int) -> np.ndarray:
    block = np.full((3, 3), BG, dtype=int)
    block[mask] = color
    return block


def _place_cross(
    grid: np.ndarray,
    cross_type: str,
    colors: tuple[int, int],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    total_height, width = grid.shape
    play_height = total_height - LEGEND_HEIGHT

    for _ in range(250):
        center = (
            int(rng.integers(1, play_height - 1)),
            int(rng.integers(1, width - 1)),
        )

        if cross_type == "plus":
            directions = ["H", "V"]
        elif cross_type == "x":
            directions = ["D1", "D2"]
        else:
            directions = [
                "H" if int(rng.integers(0, 2)) == 0 else "V",
                "D1" if int(rng.integers(0, 2)) == 0 else "D2",
            ]

        line_specs = []
        for direction, color in zip(directions, colors):
            max_neg, max_pos = _max_lengths(center, direction, play_height, width)
            if min(max_neg, max_pos) < 2:
                line_specs = []
                break
            neg_len = int(rng.integers(2, max_neg + 1))
            pos_len = int(rng.integers(2, max_pos + 1))
            cells = _line_cells(center, direction, neg_len, pos_len)
            endpoints = (cells[0], cells[-1])
            line_specs.append(
                {
                    "direction": direction,
                    "color": color,
                    "cells": cells,
                    "depth_key": _depth_key(endpoints, total_height, width),
                }
            )
        if not line_specs:
            continue
        if line_specs[0]["depth_key"] == line_specs[1]["depth_key"]:
            continue

        cell_sets = [set(spec["cells"]) for spec in line_specs]
        if cell_sets[0] & cell_sets[1] != {center}:
            continue

        union = cell_sets[0] | cell_sets[1]
        zone = _zone3(center)
        if zone - union:
            if any(cell in occupied for cell in zone - union):
                continue
        if union & occupied:
            continue
        if union & protected:
            continue
        if any(
            (nr, nc) in occupied
            for r, c in union
            for nr in range(r - 1, r + 2)
            for nc in range(c - 1, c + 2)
        ):
            continue
        if any(not (0 <= r < play_height and 0 <= c < width) for r, c in zone):
            continue

        for spec in line_specs:
            _paint_line(grid, spec["cells"], int(spec["color"]), center)
        grid[center] = GREY
        occupied.update(union)
        protected.update(zone)
        return True

    return False


def _add_legend(grid: np.ndarray, rng: np.random.Generator) -> None:
    total_height, width = grid.shape
    legend_top = total_height - LEGEND_HEIGHT
    order = ["plus", "x", "mixed"]
    rng.shuffle(order)

    used_width = 23
    x0 = int(rng.integers(0, width - used_width + 1))
    for idx, cross_type in enumerate(order):
        example_x = x0 + idx * 8
        pattern_x = example_x + 4
        colors = list(PALETTE)
        rng.shuffle(colors)
        example = _build_example_block(cross_type, (colors[0], colors[1]), rng)
        pattern = _build_pattern_block(PATTERN_MASKS[cross_type], colors[2])
        grid[legend_top : legend_top + 3, example_x : example_x + 3] = example
        grid[legend_top : legend_top + 3, pattern_x : pattern_x + 3] = pattern


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    play_height = int(rng.integers(12, 21))
    width = int(rng.integers(25, 31))
    grid = np.full((play_height + LEGEND_HEIGHT, width), BG, dtype=int)

    cross_types = ["plus", "x", "mixed"]
    if int(rng.integers(0, 2)) == 1:
        cross_types.append(cross_types[int(rng.integers(0, 3))])
    rng.shuffle(cross_types)

    line_colors = list(PALETTE)
    rng.shuffle(line_colors)
    needed_colors = 2 * len(cross_types)
    chosen = line_colors[:needed_colors]

    occupied: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    for idx, cross_type in enumerate(cross_types):
        colors = (chosen[2 * idx], chosen[2 * idx + 1])
        if not _place_cross(grid, cross_type, colors, occupied, protected, rng):
            return None

    _add_legend(grid, rng)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
