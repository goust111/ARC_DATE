from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
PATH_COUNT_RANGE = (3, 5)
MIN_PATH_LEN = 5
MAX_PATH_LEN = 9
MAX_GENERATION_ATTEMPTS = 400


def _extract_priority(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in grid[:, 0]:
        color = int(value)
        if color == BG:
            continue
        assert color not in seen, "legend colors must be unique"
        seen.add(color)
        colors.append(color)
    assert len(colors) >= 2, "legend must contain at least two colors"
    return colors


def _path(color: int, orient: str, line: int, lo: int, hi: int) -> dict[str, int | str]:
    assert orient in {"h", "v"}
    assert lo <= hi
    return {"color": int(color), "orient": orient, "line": int(line), "lo": int(lo), "hi": int(hi)}


def _transpose_path(path: dict[str, int | str]) -> dict[str, int | str]:
    orient = str(path["orient"])
    return _path(
        int(path["color"]),
        "v" if orient == "h" else "h",
        int(path["line"]),
        int(path["lo"]),
        int(path["hi"]),
    )


def _paint_full_path(grid: np.ndarray, path: dict[str, int | str]) -> None:
    color = int(path["color"])
    line = int(path["line"])
    lo = int(path["lo"])
    hi = int(path["hi"])
    if str(path["orient"]) == "h":
        grid[line, lo : hi + 1] = color
    else:
        grid[lo : hi + 1, line] = color


def _crossing(
    a: dict[str, int | str], b: dict[str, int | str]
) -> tuple[int, int] | None:
    ao = str(a["orient"])
    bo = str(b["orient"])
    if ao == bo:
        if int(a["line"]) == int(b["line"]):
            overlap_lo = max(int(a["lo"]), int(b["lo"]))
            overlap_hi = min(int(a["hi"]), int(b["hi"]))
            assert overlap_lo > overlap_hi, "parallel paths may not overlap"
        return None

    if ao == "h":
        h_path = a
        v_path = b
    else:
        h_path = b
        v_path = a

    row = int(h_path["line"])
    col = int(v_path["line"])
    if int(h_path["lo"]) <= col <= int(h_path["hi"]) and int(v_path["lo"]) <= row <= int(v_path["hi"]):
        return (row, col)
    return None


def _infer_paths(grid: np.ndarray, legend: list[int]) -> list[dict[str, int | str]]:
    h, w = grid.shape
    assert w >= 3

    outside_colors = set(map(int, np.unique(grid[:, 1:]))) - {BG}
    assert outside_colors == set(legend), "every legend color must have exactly one visible path"

    paths: list[dict[str, int | str]] = []
    for color in legend:
        coords = np.argwhere(grid[:, 1:] == int(color))
        assert coords.size > 0, f"missing path for color {color}"
        coords = coords + np.array([0, 1], dtype=int)

        rows = sorted({int(r) for r in coords[:, 0]})
        cols = sorted({int(c) for c in coords[:, 1]})
        if len(rows) == 1 and len(cols) >= 2:
            row = rows[0]
            lo = min(cols)
            hi = max(cols)
            paths.append(_path(color, "h", row, lo, hi))
            continue
        if len(cols) == 1 and len(rows) >= 2:
            col = cols[0]
            lo = min(rows)
            hi = max(rows)
            paths.append(_path(color, "v", col, lo, hi))
            continue
        raise AssertionError(f"color {color} is not a single straight path")

    return paths


def _crossing_records(
    paths: list[dict[str, int | str]], priority: dict[int, int]
) -> list[dict[str, int | tuple[int, int]]]:
    records: list[dict[str, int | tuple[int, int]]] = []
    for i, first in enumerate(paths):
        for second in paths[i + 1 :]:
            coord = _crossing(first, second)
            if coord is None:
                continue
            first_color = int(first["color"])
            second_color = int(second["color"])
            if priority[first_color] < priority[second_color]:
                high = first
                low = second
            else:
                high = second
                low = first
            records.append(
                {
                    "coord": coord,
                    "high": int(high["color"]),
                    "low": int(low["color"]),
                }
            )
    return records


def _segment_infos(path: dict[str, int | str], cuts: list[int]) -> list[dict[str, int | bool]]:
    lo = int(path["lo"])
    hi = int(path["hi"])
    unique_cuts = sorted({int(cut) for cut in cuts if lo <= int(cut) <= hi})

    infos: list[dict[str, int | bool]] = []
    prev = lo - 1
    for next_cut in unique_cuts + [hi + 1]:
        start = prev + 1
        end = next_cut - 1
        if start <= end:
            broken_left = prev != lo - 1
            broken_right = next_cut != hi + 1
            shrunk_lo = start + (2 if broken_left else 0)
            shrunk_hi = end - (2 if broken_right else 0)
            infos.append(
                {
                    "start": start,
                    "end": end,
                    "broken_left": broken_left,
                    "broken_right": broken_right,
                    "shrunk_lo": shrunk_lo,
                    "shrunk_hi": shrunk_hi,
                    "survives": shrunk_lo <= shrunk_hi,
                }
            )
        prev = next_cut
    return infos


def _render_output(
    shape: tuple[int, int],
    paths: list[dict[str, int | str]],
    priority: dict[int, int],
) -> tuple[np.ndarray, dict[int, dict[str, object]], list[dict[str, int | tuple[int, int]]]]:
    out = np.zeros(shape, dtype=int)
    cut_map = {int(path["color"]): [] for path in paths}
    records = _crossing_records(paths, priority)
    for record in records:
        row, col = record["coord"]
        low_color = int(record["low"])
        low_path = next(path for path in paths if int(path["color"]) == low_color)
        cut_map[low_color].append(col if str(low_path["orient"]) == "h" else row)

    stats: dict[int, dict[str, object]] = {}
    for path in paths:
        color = int(path["color"])
        infos = _segment_infos(path, cut_map[color])
        stats[color] = {"cuts": sorted({int(c) for c in cut_map[color]}), "segments": infos}
        for info in infos:
            if not bool(info["survives"]):
                continue
            line = int(path["line"])
            lo = int(info["shrunk_lo"])
            hi = int(info["shrunk_hi"])
            if str(path["orient"]) == "h":
                existing = out[line, lo : hi + 1]
                assert bool(np.all((existing == BG) | (existing == color)))
                out[line, lo : hi + 1] = color
            else:
                existing = out[lo : hi + 1, line]
                assert bool(np.all((existing == BG) | (existing == color)))
                out[lo : hi + 1, line] = color

    return out, stats, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Break lower-priority crossing paths and retract the broken segments.

    Input structure:
    - The background is black.
    - Column 0 is a vertical legend: each non-black pixel appears once, and higher
      pixels have higher priority.
    - Each legend color forms exactly one straight monochrome path elsewhere in the
      grid. Paths are horizontal or vertical and may cross.

    Rule:
    - At each crossing, the higher-priority path keeps the crossing pixel.
    - The lower-priority path is cut at that crossing.
    - After all cuts, each resulting segment retracts by two pixels from every cut end.
      Segments that disappear after this retraction are erased.
    - The legend column is cleared to black in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend = _extract_priority(g)
    priority = {color: idx for idx, color in enumerate(legend)}
    paths = _infer_paths(g, legend)
    out, _stats, _records = _render_output(g.shape, paths, priority)
    return out


def _construct_canonical(
    height: int,
    width: int,
    colors: list[int],
    base_idx: int,
    mode: str,
    crosser_count: int,
) -> list[dict[str, int | str]] | None:
    if height < 7 or width < 11:
        return None
    if crosser_count > base_idx:
        return None

    base_row = random.randint(2, height - 3)
    if mode == "survive":
        cross_col = random.randint(4, width - 5)
        left_max = min(6, cross_col - 1)
        right_max = min(6, (width - 1) - cross_col)
        if left_max < 3 or right_max < 3:
            return None
        left_len = random.randint(3, left_max)
        right_len = random.randint(3, right_max)
        cross_cols = [cross_col]
        start = cross_col - left_len
        end = cross_col + right_len
    elif crosser_count == 1:
        short_side = random.choice(["left", "right"])
        short_len = random.randint(1, 2)
        if short_side == "left":
            col_min = 1 + short_len
            col_max = (width - 1) - 4
            if col_min > col_max:
                return None
            cross_col = random.randint(col_min, col_max)
            right_max = min(7, (width - 1) - cross_col)
            if right_max < 4:
                return None
            right_len = random.randint(4, right_max)
            start = cross_col - short_len
            end = cross_col + right_len
        else:
            col_min = 1 + 4
            col_max = (width - 1) - short_len
            if col_min > col_max:
                return None
            cross_col = random.randint(col_min, col_max)
            left_max = min(7, cross_col - 1)
            if left_max < 4:
                return None
            left_len = random.randint(4, left_max)
            start = cross_col - left_len
            end = cross_col + short_len
        cross_cols = [cross_col]
    else:
        mid_len = random.randint(1, 2)
        sep = mid_len + 1
        left_len = random.randint(4, 6)
        right_len = random.randint(4, 6)
        c1_min = 1 + left_len
        c1_max = (width - 1) - right_len - sep
        if c1_min > c1_max:
            return None
        c1 = random.randint(c1_min, c1_max)
        c2 = c1 + sep
        cross_cols = [c1, c2]
        start = c1 - left_len
        end = c2 + right_len

    if end - start + 1 < MIN_PATH_LEN:
        return None

    base_color = colors[base_idx]
    paths: list[dict[str, int | str] | None] = [None] * len(colors)
    paths[base_idx] = _path(base_color, "h", base_row, start, end)

    higher_slots = random.sample(list(range(base_idx)), crosser_count)
    for slot, col in zip(higher_slots, cross_cols):
        up_max = min(6, base_row)
        down_max = min(6, height - 1 - base_row)
        if up_max < 2 or down_max < 2:
            return None
        up = random.randint(2, up_max)
        down = random.randint(2, down_max)
        paths[slot] = _path(colors[slot], "v", col, base_row - up, base_row + down)

    min_cross = min(cross_cols)
    max_cross = max(cross_cols)
    side_ranges: list[tuple[int, int]] = []
    if min_cross - 1 >= MIN_PATH_LEN:
        side_ranges.append((1, min_cross - 1))
    if (width - 1) - max_cross >= MIN_PATH_LEN:
        side_ranges.append((max_cross + 1, width - 1))
    if not side_ranges and any(path is None for path in paths):
        return None

    free_rows = [row for row in range(height) if row != base_row]
    random.shuffle(free_rows)
    for idx, current in enumerate(paths):
        if current is not None:
            continue
        if not free_rows:
            return None
        row = free_rows.pop()
        lo_bound, hi_bound = random.choice(side_ranges)
        span = hi_bound - lo_bound + 1
        max_len = min(MAX_PATH_LEN, span)
        if max_len < MIN_PATH_LEN:
            return None
        length = random.randint(MIN_PATH_LEN, max_len)
        start = random.randint(lo_bound, hi_bound - length + 1)
        paths[idx] = _path(colors[idx], "h", row, start, start + length - 1)

    return [path for path in paths if path is not None]


def _render_input(
    shape: tuple[int, int],
    legend_rows: list[int],
    legend: list[int],
    paths: list[dict[str, int | str]],
    priority: dict[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for row, color in zip(legend_rows, legend):
        grid[row, 0] = int(color)
    for path in paths:
        _paint_full_path(grid, path)

    records = _crossing_records(paths, priority)
    if len(records) == 1:
        record = records[0]
        row, col = record["coord"]
        grid[row, col] = random.choice([int(record["high"]), int(record["low"])])
        return grid

    records = records.copy()
    random.shuffle(records)
    for idx, record in enumerate(records):
        row, col = record["coord"]
        if idx == 0:
            shown = int(record["high"])
        elif idx == 1:
            shown = int(record["low"])
        else:
            shown = random.choice([int(record["high"]), int(record["low"])])
        grid[row, col] = shown
    return grid


def _canonicalize_paths(
    paths: list[dict[str, int | str]]
) -> dict[int, tuple[str, int, int, int]]:
    return {
        int(path["color"]): (
            str(path["orient"]),
            int(path["line"]),
            int(path["lo"]),
            int(path["hi"]),
        )
        for path in paths
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        path_count = random.randint(*PATH_COUNT_RANGE)
        colors = random.sample(COLOR_POOL, path_count)

        mode = random.choice(["survive", "erase"])
        crosser_count = 1
        if mode == "erase" and path_count >= 4 and random.random() < 0.5:
            crosser_count = 2

        base_idx = random.randint(crosser_count, path_count - 1)
        base_orient = random.choice(["h", "v"])

        if base_orient == "h":
            paths = _construct_canonical(height, width, colors, base_idx, mode, crosser_count)
        else:
            canonical = _construct_canonical(width, height, colors, base_idx, mode, crosser_count)
            paths = None if canonical is None else [_transpose_path(path) for path in canonical]
        if paths is None:
            continue

        legend_rows = sorted(random.sample(range(height), path_count))
        priority = {color: idx for idx, color in enumerate(colors)}
        expected, stats, records = _render_output((height, width), paths, priority)
        if len(records) != crosser_count:
            continue

        base_color = colors[base_idx]
        base_stats = stats[base_color]
        base_segments = list(base_stats["segments"])
        base_cuts = list(base_stats["cuts"])
        if len(base_cuts) != crosser_count:
            continue
        if mode == "survive":
            if not all(bool(segment["survives"]) for segment in base_segments):
                continue
            if sum(bool(segment["survives"]) for segment in base_segments) != 2:
                continue
        else:
            if not any(not bool(segment["survives"]) for segment in base_segments):
                continue

        inp = _render_input((height, width), legend_rows, colors, paths, priority)
        try:
            inferred = _infer_paths(inp, colors)
        except AssertionError:
            continue
        if _canonicalize_paths(inferred) != _canonicalize_paths(paths):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp

    raise RuntimeError("failed to generate a valid input")
