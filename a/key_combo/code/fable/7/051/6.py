from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, crop_bbox, get_bg, paste
except ModuleNotFoundError:
    utils_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(utils_dir) not in sys.path:
        sys.path.append(str(utils_dir))
    from arc_utils import Color, crop_bbox, get_bg, paste


PALETTE = tuple(Color.ALL_COLORS)
CONTENT_WIDTH_RANGE = (3, 6)
EXTRA_COLS_RANGE = (4, 8)
EXTRA_ROWS_RANGE = (2, 6)
MAX_GEN_ATTEMPTS = 600
NOISE_MODES = ("wrong_stop", "wrong_width", "split")


def _contiguous_non_bg_run(row: np.ndarray, bg: int) -> tuple[int, int] | None:
    cols = np.flatnonzero(row != int(bg))
    if len(cols) < 3:
        return None
    expected = np.arange(cols[0], cols[-1] + 1)
    if len(cols) != len(expected) or not np.array_equal(cols, expected):
        return None
    return int(cols[0]), int(cols[-1])


def _row_record(grid: np.ndarray, r: int, bg: int) -> dict | None:
    run = _contiguous_non_bg_run(grid[r], bg)
    if run is None:
        return None
    c0, c1 = run
    segment = crop_bbox(grid, (r, c0, r, c1)).reshape(-1)
    return {
        "row": int(r),
        "start": int(c0),
        "left": int(segment[0]),
        "content": segment[1:-1].astype(int).copy(),
        "right": int(segment[-1]),
        "width": int(segment.shape[0] - 2),
    }


def _find_directory_group(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    records = []
    for r in range(g.shape[0]):
        rec = _row_record(g, r, bg)
        if rec is not None:
            records.append(rec)

    by_signature: dict[tuple[int, int], list[dict]] = {}
    for rec in records:
        by_signature.setdefault((rec["right"], rec["width"]), []).append(rec)

    candidates: list[tuple[list[dict], int]] = []
    for (_right, width), group in by_signature.items():
        if len(group) != width + 1:
            continue
        left_colors = [int(rec["left"]) for rec in group]
        if len(set(left_colors)) != len(left_colors):
            continue

        directory_indices = []
        for idx, rec in enumerate(group):
            others = left_colors[:idx] + left_colors[idx + 1 :]
            content = rec["content"].tolist()
            if len(content) != len(others):
                continue
            if len(set(content)) != len(others):
                continue
            if sorted(map(int, content)) == sorted(map(int, others)):
                directory_indices.append(idx)

        if len(directory_indices) == 1:
            candidates.append((sorted(group, key=lambda item: item["row"]), directory_indices[0]))

    assert len(candidates) == 1, "expected exactly one directory-governed row group"
    group, unsorted_dir_idx = candidates[0]
    directory_row = by_signature[(group[0]["right"], group[0]["width"])]
    dir_row_value = directory_row[unsorted_dir_idx]["row"]
    for idx, rec in enumerate(group):
        if rec["row"] == dir_row_value:
            return group, idx
    raise AssertionError("failed to recover directory row after sorting")


def main(grid: np.ndarray) -> np.ndarray:
    """Extract bracketed equal-width row segments, use the unique permutation segment as a directory, and stack the remaining segments in directory order.

    Input structure:
    - Several rows contain one contiguous non-background run whose first cell is that row's unique left marker and whose last cell is a shared right terminator color.
    - The cells strictly between those two markers form equal-width content segments; other rows are irrelevant and must be ignored.
    - Exactly one content segment is a permutation of the left-marker colors of all the other bracketed rows.

    Rule:
    - Extract every bracketed row segment and discard the two marker cells as well as all non-bracketed rows.
    - Identify the unique extracted segment whose colors are exactly the left-marker colors of the other extracted rows in some order; this segment is the directory.
    - Read the directory left-to-right and, for each color, select the content segment whose row has that left marker.
    - Stack those selected content segments from top to bottom in that directory order, omitting the directory segment itself.

    Key insight:
    - One extracted row is metadata rather than payload: its visible colors point to the other rows by their left markers and simultaneously encode the output stacking order.
    """
    group, directory_idx = _find_directory_group(np.asarray(grid, dtype=int))
    directory = group[directory_idx]
    directory_order = [int(color) for color in directory["content"].tolist()]

    content_by_left = {
        int(rec["left"]): rec["content"].astype(int).copy()
        for idx, rec in enumerate(group)
        if idx != directory_idx
    }
    assert len(content_by_left) == len(directory_order), "directory must refer to every non-directory row exactly once"

    return np.vstack([content_by_left[color] for color in directory_order]).astype(int)


def _content_segment(width: int, color: int, accent: int, primary_pos: int, secondary_pos: int | None) -> np.ndarray:
    segment = np.full(width, int(accent), dtype=int)
    segment[int(primary_pos)] = int(color)
    if secondary_pos is not None:
        segment[int(secondary_pos)] = int(color)
    return segment


def _noise_run(length: int, left: int, right: int, fill: int) -> np.ndarray:
    row = np.full((1, int(length)), int(fill), dtype=int)
    row[0, 0] = int(left)
    row[0, -1] = int(right)
    return row


def _build_instance() -> tuple[np.ndarray, dict]:
    width = random.randint(*CONTENT_WIDTH_RANGE)
    row_count = width + 1

    bg = random.choice(PALETTE)
    available = [color for color in PALETTE if color != bg]
    random.shuffle(available)
    stop = available.pop()
    accent = available.pop()
    left_colors = random.sample(available, row_count)

    directory_idx = random.randrange(row_count)
    content_colors = [color for idx, color in enumerate(left_colors) if idx != directory_idx]
    primary_positions = random.sample(range(width), width)
    secondary_positions = random.sample(range(width), width) if width >= 4 else [None] * width

    segments_by_left: dict[int, np.ndarray] = {}
    for idx, color in enumerate(content_colors):
        second = None
        if width >= 4 and random.random() < 0.5:
            cand = secondary_positions[idx]
            if cand != primary_positions[idx]:
                second = int(cand)
        segments_by_left[int(color)] = _content_segment(width, color, accent, primary_positions[idx], second)

    relevant_order = list(range(row_count))
    random.shuffle(relevant_order)
    natural_order = [left_colors[idx] for idx in relevant_order if idx != directory_idx]
    directory_order = natural_order[:]
    while directory_order == natural_order:
        directory_order = natural_order[:]
        random.shuffle(directory_order)

    directory_segment = np.array(directory_order, dtype=int)
    segments: dict[int, np.ndarray] = {}
    for idx, left_color in enumerate(left_colors):
        if idx == directory_idx:
            segments[int(left_color)] = directory_segment.copy()
        else:
            segments[int(left_color)] = segments_by_left[int(left_color)].copy()

    extra_cols = random.randint(*EXTRA_COLS_RANGE)
    run_length = width + 2
    grid_w = run_length + extra_cols
    starts = [random.randint(0, extra_cols) for _ in range(row_count)]
    while len(set(starts)) == 1:
        starts = [random.randint(0, extra_cols) for _ in range(row_count)]

    extra_rows = random.randint(*EXTRA_ROWS_RANGE)
    grid_h = row_count + extra_rows
    relevant_rows = sorted(random.sample(range(grid_h), row_count))

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    row_specs = []
    for place_idx, row_idx in enumerate(relevant_rows):
        source_idx = relevant_order[place_idx]
        left_color = int(left_colors[source_idx])
        start = int(starts[source_idx])
        patch = np.concatenate(([left_color], segments[left_color], [int(stop)])).reshape(1, -1)
        paste(grid, patch, row_idx, start, clip=False)
        row_specs.append(
            {
                "row": int(row_idx),
                "start": int(start),
                "left": int(left_color),
                "content": segments[left_color].copy(),
                "is_directory": bool(source_idx == directory_idx),
            }
        )

    noise_rows = [r for r in range(grid_h) if r not in set(relevant_rows)]
    filler_choices = [color for color in PALETTE if color not in {bg, stop}]
    for idx, r in enumerate(noise_rows):
        mode = NOISE_MODES[idx % len(NOISE_MODES)]
        fill = random.choice(filler_choices)
        left = random.choice(filler_choices)
        if mode == "wrong_stop":
            other_stop = random.choice([color for color in filler_choices if color != left])
            patch = _noise_run(run_length, left, other_stop, fill)
            start = random.randint(0, grid_w - run_length)
            paste(grid, patch, r, start, clip=False)
        elif mode == "wrong_width":
            noise_len = random.choice([run_length - 1, run_length + 1])
            noise_len = max(3, min(grid_w, noise_len))
            patch = _noise_run(noise_len, left, stop, fill)
            start = random.randint(0, grid_w - noise_len)
            paste(grid, patch, r, start, clip=False)
        else:
            first_len = random.randint(1, max(1, width - 1))
            second_len = random.randint(1, max(1, width - 1))
            total = first_len + second_len + 1
            if total > grid_w:
                total = grid_w
                first_len = max(1, total // 2 - 1)
                second_len = max(1, total - first_len - 1)
            start = random.randint(0, grid_w - total)
            grid[r, start : start + first_len] = left
            grid[r, start + first_len + 1 : start + total] = fill

    expected = np.vstack([segments_by_left[int(color)] for color in directory_order]).astype(int)
    metadata = {
        "bg": int(bg),
        "stop": int(stop),
        "accent": int(accent),
        "width": int(width),
        "directory_order": list(map(int, directory_order)),
        "natural_order": list(map(int, natural_order)),
        "row_specs": row_specs,
        "expected": expected,
    }
    return grid, metadata


def _validate_instance(grid: np.ndarray, meta: dict) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert int(np.count_nonzero(grid == meta["bg"])) > grid.size // 2, "background must remain dominant"
    assert meta["directory_order"] != meta["natural_order"], "directory order must differ from visual row order"

    group, directory_idx = _find_directory_group(grid)
    recovered_directory = group[directory_idx]
    recovered_output = main(grid)
    assert np.array_equal(recovered_output, meta["expected"]), "solver output must match generated target"

    directory_content = recovered_directory["content"].tolist()
    assert directory_content == meta["directory_order"], "directory content must encode the intended order"

    row_map = {int(rec["left"]): rec["content"] for idx, rec in enumerate(group) if idx != directory_idx}
    assert len(row_map) == meta["width"], "expected one content row per directory color"
    assert len({tuple(map(int, seg.tolist())) for seg in row_map.values()}) == meta["width"], "content rows must stay distinct"

    natural_stack = np.vstack(
        [rec["content"] for rec in sorted(group, key=lambda item: item["row"]) if rec is not recovered_directory]
    ).astype(int)
    assert not np.array_equal(natural_stack, recovered_output), "directory ordering must matter"
    assert len({rec["start"] for rec in group}) >= 2, "row-local bracket extraction should be necessary"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid, meta = _build_instance()
        try:
            _validate_instance(grid, meta)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid directory-segment puzzle")
