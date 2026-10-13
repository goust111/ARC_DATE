from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
PALETTE = list(range(10))
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[str, int, int]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for r in range(h):
            if np.all(grid[r, :] == color):
                candidates.append(("h", r, int(color)))
        for c in range(w):
            if np.all(grid[:, c] == color):
                candidates.append(("v", c, int(color)))
    assert len(candidates) == 1, "expected exactly one full separator line"
    return candidates[0]


def _parse_strip(
    coords: list[tuple[int, int]],
    values: dict[tuple[int, int], int],
    start_color: int,
) -> list[int]:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    assert len(rows) == 1 or len(cols) == 1, "legend strip must be straight"

    if len(rows) == 1:
        row = next(iter(rows))
        ordered_coords = [(row, c) for c in range(min(cols), max(cols) + 1)]
    else:
        col = next(iter(cols))
        ordered_coords = [(r, col) for r in range(min(rows), max(rows) + 1)]

    assert len(ordered_coords) == len(coords), "legend strip must be contiguous"
    ordered_colors = [values[coord] for coord in ordered_coords]
    endpoint_matches = [
        ordered_colors[0] == start_color,
        ordered_colors[-1] == start_color,
    ]
    assert sum(endpoint_matches) == 1, "start color must match exactly one legend endpoint"
    return ordered_colors if endpoint_matches[0] else list(reversed(ordered_colors))


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)
    sep_orient, sep_idx, sep_color = _find_separator(g, bg)

    if sep_orient == "v":
        left_w = sep_idx
        right_w = w - sep_idx - 1
        assert left_w != right_w, "legend side must be uniquely narrower"
        if left_w < right_w:
            legend_r0, legend_c0, legend_r1, legend_c1 = 0, 0, h - 1, sep_idx - 1
            arena_r0, arena_c0, arena_r1, arena_c1 = 0, sep_idx + 1, h - 1, w - 1
            separator_wall = ("c", arena_c0)
        else:
            legend_r0, legend_c0, legend_r1, legend_c1 = 0, sep_idx + 1, h - 1, w - 1
            arena_r0, arena_c0, arena_r1, arena_c1 = 0, 0, h - 1, sep_idx - 1
            separator_wall = ("c", arena_c1)
    else:
        top_h = sep_idx
        bottom_h = h - sep_idx - 1
        assert top_h != bottom_h, "legend side must be uniquely narrower"
        if top_h < bottom_h:
            legend_r0, legend_c0, legend_r1, legend_c1 = 0, 0, sep_idx - 1, w - 1
            arena_r0, arena_c0, arena_r1, arena_c1 = sep_idx + 1, 0, h - 1, w - 1
            separator_wall = ("r", arena_r0)
        else:
            legend_r0, legend_c0, legend_r1, legend_c1 = sep_idx + 1, 0, h - 1, w - 1
            arena_r0, arena_c0, arena_r1, arena_c1 = 0, 0, sep_idx - 1, w - 1
            separator_wall = ("r", arena_r1)

    legend = g[legend_r0 : legend_r1 + 1, legend_c0 : legend_c1 + 1]
    arena = g[arena_r0 : arena_r1 + 1, arena_c0 : arena_c1 + 1]

    legend_coords_local = np.argwhere((legend != bg) & (legend != sep_color))
    assert legend_coords_local.size > 0, "missing legend strip"
    legend_coords = [
        (int(r + legend_r0), int(c + legend_c0))
        for r, c in legend_coords_local
    ]
    legend_values = {coord: int(g[coord]) for coord in legend_coords}

    arena_coords_local = np.argwhere((arena != bg) & (arena != sep_color))
    arena_coords = [
        (int(r + arena_r0), int(c + arena_c0))
        for r, c in arena_coords_local
    ]
    arena_values = {coord: int(g[coord]) for coord in arena_coords}
    grey_coords = [coord for coord, color in arena_values.items() if color == GREY]
    assert len(grey_coords) == 1, "expected exactly one grey direction marker"
    grey = grey_coords[0]

    start_candidates = [
        coord
        for coord, color in arena_values.items()
        if color != GREY and abs(coord[0] - grey[0]) == 1 and abs(coord[1] - grey[1]) == 1
    ]
    assert len(start_candidates) == 1, "expected one start pixel diagonally adjacent to grey"
    start = start_candidates[0]
    start_color = int(g[start])
    direction = (grey[0] - start[0], grey[1] - start[1])

    legend_colors = _parse_strip(legend_coords, legend_values, start_color)
    assert legend_colors[0] == start_color, "legend order must start with the start color"

    return {
        "bg": bg,
        "separator": (sep_orient, sep_idx, sep_color),
        "legend_colors": legend_colors,
        "start": start,
        "direction": direction,
        "arena_bounds": (arena_r0, arena_c0, arena_r1, arena_c1),
        "separator_wall": separator_wall,
    }


def _simulate_segments(
    start: tuple[int, int],
    direction: tuple[int, int],
    arena_bounds: tuple[int, int, int, int],
    segment_count: int,
    separator_wall: tuple[str, int],
) -> tuple[list[list[tuple[int, int]]], list[str]]:
    r0, c0, r1, c1 = map(int, arena_bounds)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    axis, wall_value = separator_wall

    segments: list[list[tuple[int, int]]] = []
    hits: list[str] = []
    for seg_idx in range(segment_count):
        segment = [(r, c)]
        while True:
            nr = r + dr
            nc = c + dc
            hit_row = nr < r0 or nr > r1
            hit_col = nc < c0 or nc > c1
            if hit_row or hit_col:
                hit_kind = "corner"
                if hit_row and not hit_col:
                    hit_kind = "separator" if axis == "r" and (nr < r0 or nr > r1) and (r == wall_value) else "outer"
                elif hit_col and not hit_row:
                    hit_kind = "separator" if axis == "c" and (nc < c0 or nc > c1) and (c == wall_value) else "outer"
                else:
                    touched_separator = (axis == "r" and r == wall_value) or (axis == "c" and c == wall_value)
                    hit_kind = "separator+outer" if touched_separator else "corner"
                segments.append(segment)
                hits.append(hit_kind)
                if seg_idx + 1 < segment_count:
                    if hit_row:
                        dr *= -1
                    if hit_col:
                        dc *= -1
                break
            r, c = nr, nc
            segment.append((r, c))
    return segments, hits


def _paint_solution(grid: np.ndarray, parsed: dict, legend_colors: list[int] | None = None) -> np.ndarray:
    colors = parsed["legend_colors"] if legend_colors is None else legend_colors
    segments, _hits = _simulate_segments(
        parsed["start"],
        parsed["direction"],
        parsed["arena_bounds"],
        len(colors),
        parsed["separator_wall"],
    )
    out = np.array(grid, dtype=int, copy=True)
    for color, segment in zip(colors, segments):
        for r, c in segment:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a diagonally moving light beam that reflects inside the arena and recolor each reflected segment by the legend strip order.

    Input structure:
    - A full separator line splits the grid into a narrow legend zone and a larger arena.
    - The legend zone contains one straight contiguous color strip.
    - The arena contains one start pixel and one grey pixel that is diagonally adjacent to it.

    Rule:
    - Identify the narrow side as the legend zone, keep the separator and the legend strip unchanged, and read the strip from the endpoint whose color matches the start pixel.
    - Launch a diagonal ray from the start pixel toward the grey pixel; whenever the next step would leave the arena or cross the separator line, reflect that direction component as a mirror bounce.
    - Paint the 1st, 2nd, ... reflected trajectory segments with the 1st, 2nd, ... legend colors, stopping after exactly as many segments as the strip length.

    Key insight:
    - The legend strip encodes both the color order and the required number of bounces, while the start-color match disambiguates which end of the strip is first.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    return _paint_solution(np.asarray(grid, dtype=int), parsed)


def _choose_layout() -> dict:
    sep_color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    if random.random() < 0.5:
        legend_w = random.randint(2, 4)
        arena_w = random.randint(8, min(20, 29 - legend_w))
        h = random.randint(10, 24)
        w = legend_w + 1 + arena_w
        side = random.choice(["left", "right"])
        if side == "left":
            sep_idx = legend_w
            legend_bounds = (0, 0, h - 1, legend_w - 1)
            arena_bounds = (0, sep_idx + 1, h - 1, w - 1)
            separator_wall = ("c", arena_bounds[1])
        else:
            sep_idx = arena_w
            legend_bounds = (0, sep_idx + 1, h - 1, w - 1)
            arena_bounds = (0, 0, h - 1, sep_idx - 1)
            separator_wall = ("c", arena_bounds[3])
        strip_axis = "v"
        separator = ("v", sep_idx, sep_color)
    else:
        legend_h = random.randint(2, 4)
        arena_h = random.randint(8, min(20, 29 - legend_h))
        w = random.randint(10, 24)
        h = legend_h + 1 + arena_h
        side = random.choice(["top", "bottom"])
        if side == "top":
            sep_idx = legend_h
            legend_bounds = (0, 0, legend_h - 1, w - 1)
            arena_bounds = (sep_idx + 1, 0, h - 1, w - 1)
            separator_wall = ("r", arena_bounds[0])
        else:
            sep_idx = arena_h
            legend_bounds = (sep_idx + 1, 0, h - 1, w - 1)
            arena_bounds = (0, 0, sep_idx - 1, w - 1)
            separator_wall = ("r", arena_bounds[2])
        strip_axis = "h"
        separator = ("h", sep_idx, sep_color)

    bg_choices = [c for c in PALETTE if c not in {GREY, sep_color}]
    bg = random.choice(bg_choices)
    return {
        "shape": (h, w),
        "bg": bg,
        "separator": separator,
        "legend_bounds": legend_bounds,
        "arena_bounds": arena_bounds,
        "separator_wall": separator_wall,
        "strip_axis": strip_axis,
        "sep_color": sep_color,
    }


def _place_separator(grid: np.ndarray, separator: tuple[str, int, int]) -> None:
    orient, idx, color = separator
    if orient == "v":
        grid[:, idx] = int(color)
    else:
        grid[idx, :] = int(color)


def _place_legend(grid: np.ndarray, layout: dict, colors: list[int]) -> None:
    r0, c0, r1, c1 = layout["legend_bounds"]
    if layout["strip_axis"] == "v":
        col = random.randint(c0, c1)
        start_r = random.randint(0, (r1 - r0 + 1) - len(colors))
        rows = [r0 + start_r + i for i in range(len(colors))]
        if random.random() < 0.5:
            ordered_colors = colors
        else:
            ordered_colors = list(reversed(colors))
        for r, color in zip(rows, ordered_colors):
            grid[r, col] = int(color)
    else:
        row = random.randint(r0, r1)
        start_c = random.randint(0, (c1 - c0 + 1) - len(colors))
        cols = [c0 + start_c + i for i in range(len(colors))]
        if random.random() < 0.5:
            ordered_colors = colors
        else:
            ordered_colors = list(reversed(colors))
        for c, color in zip(cols, ordered_colors):
            grid[row, c] = int(color)


def _simulate_metadata(layout: dict, start: tuple[int, int], direction: tuple[int, int], colors: list[int]) -> dict:
    segments, hits = _simulate_segments(
        start,
        direction,
        layout["arena_bounds"],
        len(colors),
        layout["separator_wall"],
    )
    return {"segments": segments, "hits": hits}


def _hit_requirements(hits: list[str]) -> bool:
    return any("separator" in hit for hit in hits) and any(hit in {"outer", "separator+outer", "corner"} for hit in hits)


def _alt_direction_output(grid: np.ndarray, parsed: dict, direction: tuple[int, int]) -> np.ndarray:
    alt = dict(parsed)
    alt["direction"] = direction
    return _paint_solution(grid, alt)


def _validate_generated(grid: np.ndarray, spec: dict) -> bool:
    try:
        parsed = _parse_grid(grid)
    except AssertionError:
        return False

    if parsed["start"] != spec["start"] or parsed["direction"] != spec["direction"]:
        return False
    if parsed["legend_colors"] != spec["legend_colors"]:
        return False
    if parsed["arena_bounds"] != spec["arena_bounds"]:
        return False

    expected = _paint_solution(grid, parsed)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(actual, grid):
        return False

    if any(len(segment) < 2 for segment in spec["segments"]):
        return False
    if len({segment[-1] for segment in spec["segments"][:-1]}) < 2:
        return False
    if not _hit_requirements(spec["hits"]):
        return False

    reversed_out = _paint_solution(grid, parsed, list(reversed(parsed["legend_colors"])))
    if np.array_equal(reversed_out, actual):
        return False

    shorter_out = _paint_solution(grid, parsed, parsed["legend_colors"][:-1])
    if np.array_equal(shorter_out, actual):
        return False

    for alt_direction in DIAGONALS:
        if alt_direction == parsed["direction"]:
            continue
        alt_out = _alt_direction_output(grid, parsed, alt_direction)
        if np.array_equal(alt_out, actual):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    h, w = layout["shape"]
    bg = layout["bg"]
    grid = np.full((h, w), bg, dtype=int)
    _place_separator(grid, layout["separator"])

    usable_colors = [c for c in PALETTE if c not in {bg, GREY, layout["sep_color"]}]
    segment_count = random.randint(3, min(6, len(usable_colors)))
    legend_colors = random.sample(usable_colors, segment_count)
    _place_legend(grid, layout, legend_colors)

    ar0, ac0, ar1, ac1 = layout["arena_bounds"]
    start = (
        random.randint(ar0 + 1, ar1 - 1),
        random.randint(ac0 + 1, ac1 - 1),
    )
    direction = random.choice(DIAGONALS)
    grey = (start[0] + direction[0], start[1] + direction[1])
    if not (ar0 <= grey[0] <= ar1 and ac0 <= grey[1] <= ac1):
        return None

    grid[start] = legend_colors[0]
    grid[grey] = GREY

    metadata = _simulate_metadata(layout, start, direction, legend_colors)
    spec = {
        "start": start,
        "direction": direction,
        "legend_colors": legend_colors,
        "arena_bounds": layout["arena_bounds"],
        "segments": metadata["segments"],
        "hits": metadata["hits"],
    }
    return grid if _validate_generated(grid, spec) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-beam ARC input")
