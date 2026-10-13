from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_top_legend(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, ...]]:
    row = [int(v) for v in grid[0]]
    segments: list[tuple[int, tuple[int, ...]]] = []
    c = 0
    while c < len(row):
        if row[c] == bg:
            c += 1
            continue
        c0 = c
        vals: list[int] = []
        while c < len(row) and row[c] != bg:
            vals.append(int(row[c]))
            c += 1
        segments.append((c0, tuple(vals)))

    assert len(segments) == 1, "expected exactly one top-edge legend string"
    legend_start, legend = segments[0]
    assert len(legend) >= 2, "legend must contain at least two colors"
    assert len(set(legend)) == len(legend), "legend colors must be pairwise distinct"
    return int(legend_start), legend


def _find_frames(grid: np.ndarray, bg: int, legend_colors: Iterable[int]) -> list[dict]:
    legend_set = {int(v) for v in legend_colors}
    palette = {int(v) for v in np.unique(grid)}
    border_colors = palette - legend_set - {int(bg)}
    assert len(border_colors) == 1, "expected exactly one frame-border color"
    border = int(next(iter(border_colors)))

    frames: list[dict] = []
    for comp in _components_of_mask(grid == border):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "frames must have a non-empty strict interior"

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros((height, width), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        assert np.array_equal(sub == border, border_mask), "frame must be a 1-cell-thick hollow rectangle"

        interior = sub[1:-1, 1:-1]
        marker_coords = np.argwhere(interior != bg)
        assert marker_coords.shape[0] == 1, "each frame interior must contain exactly one marker"
        mr_local, mc_local = map(int, marker_coords[0])
        marker_color = int(interior[mr_local, mc_local])
        assert marker_color in legend_set, "marker color must occur in the top legend"

        legend_idx = list(legend_colors).index(marker_color)
        frames.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "inner_h": int(height - 2),
                "inner_w": int(width - 2),
                "marker_r": int(r0 + 1 + mr_local),
                "marker_c": int(c0 + 1 + mc_local),
                "marker_offset": int(mc_local),
                "marker_color": int(marker_color),
                "legend_idx": int(legend_idx),
            }
        )

    frames.sort(key=lambda item: (item["r0"], item["c0"]))
    assert frames, "expected at least one hollow frame"
    return frames


def _fill_frame_interior(out: np.ndarray, frame: dict, legend: tuple[int, ...]) -> None:
    left = int(frame["c0"]) + 1
    right = int(frame["c1"])
    top = int(frame["r0"]) + 1
    bottom = int(frame["r1"])
    marker_c = int(frame["marker_c"])
    anchor = int(frame["legend_idx"])
    period = len(legend)

    offsets = np.arange(left, right, dtype=int) - marker_c
    cols = np.asarray([legend[(anchor + int(delta)) % period] for delta in offsets], dtype=int)
    out[top:bottom, left:right] = cols[np.newaxis, :]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the top legend string as a cyclic wallpaper and use each frame's marker pixel to phase-align a full interior fill.

    Input structure:
    - The top edge contains exactly one horizontal legend string whose colors are all distinct.
    - The scene also contains one or more hollow rectangular frames whose border color is not part of the legend.
    - Each frame interior contains exactly one isolated marker pixel, and that marker color appears in the legend.

    Rule:
    - View the legend as an infinite periodic sequence extending left and right.
    - For a frame whose marker sits at column m and whose marker color is the legend's p-th entry, fill every interior column c with the legend color at index (p + c - m) mod L, where L is the legend length.
    - Preserve the frame borders and the top legend unchanged; only replace the strict interiors.

    Key insight:
    - The marker provides both the matched legend color and the horizontal phase, so the rectangle interior is just a window onto one cyclic wallpaper.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    _legend_start, legend = _find_top_legend(g, bg)
    frames = _find_frames(g, bg, legend)

    out = g.copy()
    for frame in frames:
        _fill_frame_interior(out, frame, legend)
    return out


MAX_GENERATION_ATTEMPTS = 400
COLOR_POOL = tuple(range(1, 10))


def _choose_row_split(frame_count: int) -> tuple[int, int]:
    options = [(k, frame_count - k) for k in range(1, frame_count) if max(k, frame_count - k) <= 3]
    return random.choice(options)


def _boxes_have_gap(boxes: list[tuple[int, int, int, int]], new_box: tuple[int, int, int, int]) -> bool:
    nr0, nc0, nr1, nc1 = new_box
    for r0, c0, r1, c1 in boxes:
        if not (nr1 < r0 - 1 or nr0 > r1 + 1 or nc1 < c0 - 1 or nc0 > c1 + 1):
            return False
    return True


def _build_expected(grid: np.ndarray, legend: tuple[int, ...], frames: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        _fill_frame_interior(out, frame, legend)
    return out


def _validate_case(grid: np.ndarray, expected: np.ndarray) -> None:
    bg = _dominant_color(grid)
    _legend_start, legend = _find_top_legend(grid, bg)
    frames = _find_frames(grid, bg, legend)
    period = len(legend)

    assert len(frames) >= 3
    assert any(frame["inner_w"] > period for frame in frames), "need a frame wider than the legend"
    assert np.any(grid != expected), "output must differ from input"

    phases = [int((frame["legend_idx"] - frame["marker_offset"]) % period) for frame in frames]
    assert len(set(phases)) >= 2, "need multiple wallpaper phases across frames"

    wide_rows = []
    for frame in frames:
        if frame["inner_w"] <= period:
            continue
        row = expected[frame["r0"] + 1, frame["c0"] + 1 : frame["c1"]]
        assert np.array_equal(row[:-period], row[period:]), "wide frame must visibly enforce periodic extension"
        wide_rows.append(tuple(int(v) for v in row))
    assert wide_rows

    saw_same_color_diff_offset = False
    saw_same_offset_diff_color = False
    for i in range(len(frames)):
        left_i = int(expected[frames[i]["r0"] + 1, frames[i]["c0"] + 1])
        for j in range(i + 1, len(frames)):
            left_j = int(expected[frames[j]["r0"] + 1, frames[j]["c0"] + 1])
            if (
                frames[i]["legend_idx"] == frames[j]["legend_idx"]
                and frames[i]["marker_offset"] != frames[j]["marker_offset"]
                and (frames[i]["marker_offset"] - frames[j]["marker_offset"]) % period != 0
                and left_i != left_j
            ):
                saw_same_color_diff_offset = True
            if (
                frames[i]["marker_offset"] == frames[j]["marker_offset"]
                and frames[i]["legend_idx"] != frames[j]["legend_idx"]
                and left_i != left_j
            ):
                saw_same_offset_diff_color = True
    assert saw_same_color_diff_offset, "need a pair proving marker column matters"
    assert saw_same_offset_diff_color, "need a pair proving marker color matters"


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.randint(3, 6)
    frame_count = random.randint(3, 5)

    inner_ws = [random.randint(2, 7) for _ in range(frame_count)]
    wide_index = random.randrange(frame_count)
    inner_ws[wide_index] = random.randint(legend_len + 1, min(12, legend_len + 5))
    inner_hs = [random.randint(2, 5) for _ in range(frame_count)]
    outer_ws = [w + 2 for w in inner_ws]
    outer_hs = [h + 2 for h in inner_hs]

    order = list(range(frame_count))
    random.shuffle(order)
    row1_count, row2_count = _choose_row_split(frame_count)
    row1_ids = order[:row1_count]
    row2_ids = order[row1_count:]

    gap_choices_1 = [random.randint(1, 3) for _ in range(max(0, row1_count - 1))]
    gap_choices_2 = [random.randint(1, 3) for _ in range(max(0, row2_count - 1))]
    span1 = sum(outer_ws[idx] for idx in row1_ids) + sum(gap_choices_1)
    span2 = sum(outer_ws[idx] for idx in row2_ids) + sum(gap_choices_2)
    min_width = max(legend_len, span1, span2)
    if min_width > 30:
        return None
    width = random.randint(min_width, min(30, min_width + 8))

    band1_h = max(outer_hs[idx] for idx in row1_ids)
    band2_h = max(outer_hs[idx] for idx in row2_ids)
    between_gap = random.randint(1, 3)
    min_height = 2 + band1_h + between_gap + band2_h
    if min_height > 30:
        return None
    height = random.randint(min_height, min(30, min_height + 6))

    legend_colors = tuple(random.sample(COLOR_POOL, legend_len))
    border_choices = [color for color in COLOR_POOL if color not in legend_colors]
    border = random.choice(border_choices)

    grid = np.zeros((height, width), dtype=int)
    legend_start = random.randint(0, width - legend_len)
    grid[0, legend_start : legend_start + legend_len] = np.asarray(legend_colors, dtype=int)

    boxes: list[tuple[int, int, int, int]] = []
    frame_specs: list[dict | None] = [None] * frame_count

    def place_row(indices: list[int], gaps: list[int], band_top: int, band_h: int) -> bool:
        row_span = sum(outer_ws[idx] for idx in indices) + sum(gaps)
        start_c = random.randint(0, width - row_span)
        cursor = start_c
        for pos, idx in enumerate(indices):
            frame_h = outer_hs[idx]
            frame_w = outer_ws[idx]
            r0 = band_top + random.randint(0, band_h - frame_h)
            c0 = cursor
            box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            if not _boxes_have_gap(boxes, box):
                return False
            boxes.append(box)
            frame_specs[idx] = {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + frame_h - 1),
                "c1": int(c0 + frame_w - 1),
                "inner_h": int(inner_hs[idx]),
                "inner_w": int(inner_ws[idx]),
            }
            cursor += frame_w
            if pos < len(gaps):
                cursor += gaps[pos]
        return True

    band1_top = 2
    band2_top = band1_top + band1_h + between_gap
    if not place_row(row1_ids, gap_choices_1, band1_top, band1_h):
        return None
    if not place_row(row2_ids, gap_choices_2, band2_top, band2_h):
        return None

    assert all(spec is not None for spec in frame_specs)
    frames = [dict(spec) for spec in frame_specs if spec is not None]

    control_ids = random.sample(range(frame_count), 3)
    a_idx, b_idx, c_idx = control_ids
    a_w = frames[a_idx]["inner_w"]
    b_w = frames[b_idx]["inner_w"]
    c_w = frames[c_idx]["inner_w"]

    k0 = random.randint(0, min(a_w, c_w) - 1)
    k1_choices = [k for k in range(b_w) if k != k0 and (k - k0) % legend_len != 0]
    if not k1_choices:
        return None
    k1 = random.choice(k1_choices)

    p_shared = random.randrange(legend_len)
    p_other_choices = [p for p in range(legend_len) if p != p_shared]
    p_other = random.choice(p_other_choices)

    assignments: dict[int, tuple[int, int]] = {
        a_idx: (p_shared, k0),
        b_idx: (p_shared, k1),
        c_idx: (p_other, k0),
    }

    for idx in range(frame_count):
        if idx in assignments:
            legend_idx, marker_offset = assignments[idx]
        else:
            legend_idx = random.randrange(legend_len)
            marker_offset = random.randrange(frames[idx]["inner_w"])
        marker_row_offset = random.randrange(frames[idx]["inner_h"])
        marker_r = frames[idx]["r0"] + 1 + marker_row_offset
        marker_c = frames[idx]["c0"] + 1 + marker_offset
        frames[idx]["legend_idx"] = int(legend_idx)
        frames[idx]["marker_offset"] = int(marker_offset)
        frames[idx]["marker_r"] = int(marker_r)
        frames[idx]["marker_c"] = int(marker_c)
        frames[idx]["marker_color"] = int(legend_colors[legend_idx])

    for frame in frames:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        grid[r0, c0 : c1 + 1] = border
        grid[r1, c0 : c1 + 1] = border
        grid[r0 : r1 + 1, c0] = border
        grid[r0 : r1 + 1, c1] = border
        grid[frame["marker_r"], frame["marker_c"]] = frame["marker_color"]

    expected = _build_expected(grid, legend_colors, frames)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    try:
        _validate_case(grid, expected)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cyclic-wallpaper frame input")
