from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_RETRIES = 400
MIN_SIZE = 14
MAX_SIZE = 24
COUNT_OPTIONS = (2, 3)


def _consecutive_groups(indices: np.ndarray) -> list[list[int]]:
    if indices.size == 0:
        return []
    groups: list[list[int]] = []
    current = [int(indices[0])]
    for value in map(int, indices[1:]):
        if value == current[-1] + 1:
            current.append(value)
        else:
            groups.append(current)
            current = [value]
    groups.append(current)
    return groups


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.flat if int(v) != BG)
    assert counts, "path footprint must contain a visible color"
    return counts.most_common(1)[0][0]


def _detect_paths(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    row_counts = np.count_nonzero(grid != BG, axis=1)
    col_counts = np.count_nonzero(grid != BG, axis=0)

    horizontal_rows = np.where(row_counts == w)[0]
    vertical_cols = np.where(col_counts == h)[0]

    row_groups = _consecutive_groups(horizontal_rows)
    col_groups = _consecutive_groups(vertical_cols)

    assert row_groups and col_groups, "expected both horizontal and vertical paths"

    paths: list[dict[str, object]] = []
    for rows in row_groups:
        assert len(rows) in (1, 2), "horizontal path width must be 1 or 2"
        color = _mode_color(grid[np.array(rows), :])
        paths.append({"orient": "H", "rows": rows, "width": len(rows), "color": color})

    for cols in col_groups:
        assert len(cols) in (1, 2), "vertical path width must be 1 or 2"
        color = _mode_color(grid[:, np.array(cols)])
        paths.append({"orient": "V", "cols": cols, "width": len(cols), "color": color})

    return paths


def _attach_profiles(paths: list[dict[str, object]], height: int, width: int, recolor_after_wide: bool) -> None:
    horizontal_wide = sorted(
        (path for path in paths if path["orient"] == "H" and path["width"] == 2),
        key=lambda path: int(path["rows"][0]),
    )
    vertical_wide = sorted(
        (path for path in paths if path["orient"] == "V" and path["width"] == 2),
        key=lambda path: int(path["cols"][0]),
    )

    for path in paths:
        if path["orient"] == "H":
            profile = np.full(width, int(path["color"]), dtype=int)
            if recolor_after_wide and int(path["width"]) == 1:
                for vertical in vertical_wide:
                    start = int(vertical["cols"][-1]) + 1
                    if start < width:
                        profile[start:] = int(vertical["color"])
            path["profile"] = profile
        else:
            profile = np.full(height, int(path["color"]), dtype=int)
            if recolor_after_wide and int(path["width"]) == 1:
                for horizontal in horizontal_wide:
                    start = int(horizontal["rows"][-1]) + 1
                    if start < height:
                        profile[start:] = int(horizontal["color"])
            path["profile"] = profile


def _render_paths(paths: list[dict[str, object]], height: int, width: int, recolor_after_wide: bool) -> np.ndarray:
    _attach_profiles(paths, height, width, recolor_after_wide)

    order = {"V1": 0, "H1": 1, "V2": 2, "H2": 3}
    out = np.zeros((height, width), dtype=int)

    for path in sorted(paths, key=lambda item: order[f"{item['orient']}{item['width']}"]):
        if path["orient"] == "H":
            profile = np.asarray(path["profile"], dtype=int)
            for row in path["rows"]:
                out[int(row), :] = profile
        else:
            profile = np.asarray(path["profile"], dtype=int)
            for col in path["cols"]:
                out[:, int(col)] = profile

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each thin path after it passes underneath a wider crossing path.

    Input structure:
    - The background is black.
    - Non-black pixels form straight horizontal or vertical full-length paths, each
      with constant width 1 or 2 and one visible color away from crossings.
    - At an overlap, only the top path's color is visible in the input grid.

    Rule:
    - Recover every horizontal and vertical path from the occupied row-bands and
      column-bands, along with its original color.
    - Width-2 paths sit above width-1 paths; when widths match, the horizontal
      path sits above the vertical path.
    - Treat horizontal paths as progressing left-to-right and vertical paths as
      progressing top-to-bottom.
    - After a width-1 path passes under a width-2 crossing band, recolor the
      remainder of that thin path to the wider path's color; same-width crossings
      never recolor the lower path.
    - Re-render the scene with the same visibility order at all crossings.

    Key insight:
    The full occupied rows and columns reveal the hidden straight paths even when
    colors are occluded, and only thin paths change state after moving beneath a
    wider path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    paths = _detect_paths(g)
    return _render_paths(paths, g.shape[0], g.shape[1], recolor_after_wide=True)


def _place_bands(size: int, widths: list[int]) -> list[list[int]] | None:
    for _ in range(200):
        occupied: set[int] = set()
        starts = [None] * len(widths)
        order = list(range(len(widths)))
        random.shuffle(order)

        for idx in order:
            width = widths[idx]
            candidates: list[int] = []
            for start in range(1, size - width):
                halo = set(range(start - 1, start + width + 1))
                if occupied & halo:
                    continue
                candidates.append(start)
            if not candidates:
                break
            start = random.choice(candidates)
            starts[idx] = start
            occupied.update(range(start - 1, start + width + 1))
        else:
            return [list(range(int(start), int(start) + width)) for start, width in zip(starts, widths)]
    return None


def _gen_attempt() -> np.ndarray | None:
    h_count = random.choice(COUNT_OPTIONS)
    v_count = random.choice(COUNT_OPTIONS)

    h_widths = [1, 2] + [random.randint(1, 2) for _ in range(h_count - 2)]
    v_widths = [1, 2] + [random.randint(1, 2) for _ in range(v_count - 2)]
    random.shuffle(h_widths)
    random.shuffle(v_widths)

    min_h = max(MIN_SIZE, sum(h_widths) + 2 * len(h_widths) + 1)
    min_w = max(MIN_SIZE, sum(v_widths) + 2 * len(v_widths) + 1)
    if min_h > MAX_SIZE or min_w > MAX_SIZE:
        return None

    height = random.randint(min_h, MAX_SIZE)
    width = random.randint(min_w, MAX_SIZE)

    occupied_area = height * sum(v_widths) + width * sum(h_widths) - sum(h_widths) * sum(v_widths)
    if occupied_area * 2 >= height * width:
        return None

    row_bands = _place_bands(height, h_widths)
    col_bands = _place_bands(width, v_widths)
    if row_bands is None or col_bands is None:
        return None

    colors = random.sample(COLOR_POOL, h_count + v_count)
    paths: list[dict[str, object]] = []
    for rows, width_value, color in zip(row_bands, h_widths, colors[:h_count]):
        paths.append({"orient": "H", "rows": rows, "width": width_value, "color": color})
    for cols, width_value, color in zip(col_bands, v_widths, colors[h_count:]):
        paths.append({"orient": "V", "cols": cols, "width": width_value, "color": color})

    input_grid = _render_paths([path.copy() for path in paths], height, width, recolor_after_wide=False)
    output_grid = _render_paths([path.copy() for path in paths], height, width, recolor_after_wide=True)
    if np.array_equal(input_grid, output_grid):
        return None
    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
