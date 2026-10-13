from __future__ import annotations

import random

import numpy as np


BG = 7
PATH = 1
FILL = 4
MARKER_COLORS = (2, 3, 5, 6, 8, 9)
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _trace_ray(shape: tuple[int, int], start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    r, c = start
    out: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        out.append((int(r), int(c)))
        r += dr
        c += dc
    return out


def _extract_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != bg)]
    assert 2 <= len(markers) <= 3, "expected 2-3 markers"
    cols = [c for _r, c, _color in markers]
    assert len(set(cols)) == len(cols), "marker columns must be distinct"
    return sorted(markers, key=lambda item: (item[1], item[0], item[2]))


def _segments_for_column(h: int, w: int, c: int) -> list[dict]:
    center = w // 2
    is_center = (w % 2 == 1) and (c == center)
    if is_center:
        top_dir = (1, 1)
        bottom_dir = (-1, -1)
        slope = 1
    else:
        top_dir = (1, -1)
        bottom_dir = (-1, 1)
        slope = -1

    top_cells = _trace_ray((h, w), (0, c), *top_dir)
    bot_cells = _trace_ray((h, w), (h - 1, c), *bottom_dir)
    return [
        {
            "role": "top",
            "slope": slope,
            "cells": top_cells,
            "const_k": len({r + cc for r, cc in top_cells}) == 1,
            "const": int(top_cells[0][0] + top_cells[0][1]) if slope == -1 else int(top_cells[0][0] - top_cells[0][1]),
            "axis_values": {int(r - cc) for r, cc in top_cells} if slope == -1 else {int(r + cc) for r, cc in top_cells},
        },
        {
            "role": "bottom",
            "slope": slope,
            "cells": bot_cells,
            "const_k": len({r + cc for r, cc in bot_cells}) == 1,
            "const": int(bot_cells[0][0] + bot_cells[0][1]) if slope == -1 else int(bot_cells[0][0] - bot_cells[0][1]),
            "axis_values": {int(r - cc) for r, cc in bot_cells} if slope == -1 else {int(r + cc) for r, cc in bot_cells},
        },
    ]


def _build_path_data(markers: list[tuple[int, int, int]], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    data: list[dict] = []
    for marker_id, (_r, c, _color) in enumerate(markers):
        for segment in _segments_for_column(h, w, c):
            item = dict(segment)
            item["marker_id"] = marker_id
            item["column"] = int(c)
            data.append(item)
    return data


def _fill_between_pair(
    out: np.ndarray,
    seg_a: dict,
    seg_b: dict,
    bg: int,
    fill_stats: dict[str, int] | None = None,
) -> None:
    h, w = out.shape
    role = str(seg_a["role"])
    if seg_a["slope"] == -1:
        k0, k1 = sorted((int(seg_a["const"]), int(seg_b["const"])))
        shared_b = sorted(set(seg_a["axis_values"]) & set(seg_b["axis_values"]))
        for b in shared_b:
            r_min = max(0, b)
            r_max = min(h - 1, b + w - 1)
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k0 < k < k1 and out[r, c] == bg:
                    out[r, c] = FILL
                    if fill_stats is not None:
                        fill_stats[role] += 1
    else:
        b0, b1 = sorted((int(seg_a["const"]), int(seg_b["const"])))
        shared_k = sorted(set(seg_a["axis_values"]) & set(seg_b["axis_values"]))
        for k in shared_k:
            r_min = max(0, k - (w - 1))
            r_max = min(h - 1, k)
            for r in range(r_min, r_max + 1):
                c = k - r
                b = r - c
                if b0 < b < b1 and out[r, c] == bg:
                    out[r, c] = FILL
                    if fill_stats is not None:
                        fill_stats[role] += 1


def _render_from_markers(
    markers: list[tuple[int, int, int]],
    shape: tuple[int, int],
    bg: int = BG,
    *,
    restore_markers: bool = True,
) -> tuple[np.ndarray, dict[str, int], list[dict]]:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    segments = _build_path_data(markers, shape)

    for _r, c, _color in markers:
        out[:, c] = PATH
    for segment in segments:
        for r, c in segment["cells"]:
            if out[r, c] == bg:
                out[r, c] = PATH

    fill_stats = {"top": 0, "bottom": 0}
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a = segments[i]
            b = segments[j]
            if a["marker_id"] == b["marker_id"]:
                continue
            if a["role"] != b["role"]:
                continue
            if a["slope"] != b["slope"]:
                continue
            _fill_between_pair(out, a, b, bg, fill_stats)

    if restore_markers:
        for r, c, color in markers:
            out[r, c] = int(color)
    return out, fill_stats, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each marker's blue folded path and fill the emergent region between parallel diagonal arms.

    Input structure:
    - The grid is mostly orange background with 2-3 isolated non-background marker pixels; blue is absent.
    - Marker columns are distinct, and the grid width is odd so there is a unique center column.
    - Each marker defines one full blue vertical column plus two diagonal arms attached to the column's top and bottom ends.

    Rule:
    - Paint the marker's whole column blue while preserving the marker pixel itself.
    - If a marker is in the center column, extend its top endpoint along a down-right diagonal and its bottom endpoint along an up-left diagonal.
    - Otherwise, extend its top endpoint along a down-left diagonal and its bottom endpoint along an up-right diagonal.
    - After all paths are drawn, for every pair of different markers whose corresponding top arms are parallel, and likewise whose corresponding bottom arms are parallel, fill the orange background cells strictly between those two diagonal arms with yellow.
    - Never overwrite marker pixels or existing blue path pixels.

    Key insight:
    - Path construction is independent per marker, but the yellow regions are emergent: they appear only where two different paths produce matching diagonal orientations and therefore bound a diagonal strip.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 5
    assert w % 2 == 1, "width must be odd"
    assert not bool(np.any(g == PATH)), "blue is introduced by the output"

    bg = _get_bg(g)
    markers = _extract_markers(g, bg)
    out, _stats, _segments = _render_from_markers(markers, (h, w), bg)
    return out


def _candidate_column_pairs(w: int, use_center_marker: bool) -> list[tuple[int, int]]:
    center = w // 2
    interior = [c for c in range(2, w - 2) if c != center]
    pairs: list[tuple[int, int]] = []
    modes = ["left", "right"] if use_center_marker else ["left", "right", "split"]
    for c0 in interior:
        for c1 in interior:
            if c0 >= c1:
                continue
            if (c0 - c1) % 2 != 0:
                continue
            if c1 - c0 < 4:
                continue
            side = (
                "left"
                if c1 < center
                else "right"
                if c0 > center
                else "split"
                if c0 < center < c1
                else "mixed"
            )
            if side in modes:
                pairs.append((int(c0), int(c1)))
    return pairs


def _path_cells_for_column(h: int, w: int, c: int) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(h)}
    for segment in _segments_for_column(h, w, c):
        cells.update(segment["cells"])
    return cells


def _choose_marker_rows(
    h: int,
    w: int,
    cols: list[int],
    fill_cells: set[tuple[int, int]],
) -> list[int] | None:
    path_cells = {c: _path_cells_for_column(h, w, c) for c in cols}
    rows: list[int] = []
    for c in cols:
        blocked = set(fill_cells)
        for other_c, other_cells in path_cells.items():
            if other_c != c:
                blocked |= other_cells
        candidates = [r for r in range(1, h - 1) if (r, c) not in blocked]
        if not candidates:
            return None
        rows.append(int(random.choice(candidates)))
    return rows


def _semantic_check_input(inp: np.ndarray, out: np.ndarray, fill_stats: dict[str, int]) -> None:
    h, w = inp.shape
    assert inp.shape == out.shape
    assert max(inp.shape) <= 30
    assert _get_bg(inp) == BG
    assert not bool(np.any(inp == PATH))
    markers = _extract_markers(inp, BG)
    cols = [c for _r, c, _color in markers]
    center = w // 2
    off_center = [c for c in cols if c != center]
    center_count = sum(c == center for c in cols)

    assert len(off_center) == 2, "generator should produce exactly one fill-producing off-center pair"
    assert center_count in (0, 1)
    assert abs(off_center[0] - off_center[1]) >= 4
    assert off_center[0] % 2 == off_center[1] % 2
    assert fill_stats["top"] > 0 and fill_stats["bottom"] > 0

    total_fill = int(np.sum(out == FILL))
    assert total_fill == fill_stats["top"] + fill_stats["bottom"]
    assert 4 <= total_fill <= (h * w) // 4

    for r, c, color in markers:
        assert 0 < r < h - 1
        assert out[r, c] == color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 21)
    w = random.choice(list(range(15, 30, 2)))
    use_center_marker = random.choice([False, True])

    off_pair_options = _candidate_column_pairs(w, use_center_marker)
    if not off_pair_options:
        return None
    c0, c1 = random.choice(off_pair_options)
    cols = [c0, c1]
    if use_center_marker:
        cols.append(w // 2)
    cols.sort()

    dummy_markers = [(1, c, MARKER_COLORS[i]) for i, c in enumerate(cols)]
    template_out, fill_stats, _segments = _render_from_markers(
        dummy_markers, (h, w), BG, restore_markers=False
    )
    fill_cells = {tuple(map(int, p)) for p in np.argwhere(template_out == FILL)}
    if fill_stats["top"] == 0 or fill_stats["bottom"] == 0:
        return None

    rows = _choose_marker_rows(h, w, cols, fill_cells)
    if rows is None:
        return None

    colors = random.sample(MARKER_COLORS, len(cols))
    markers = sorted([(rows[i], cols[i], colors[i]) for i in range(len(cols))], key=lambda item: item[1])
    raw_out, _raw_fill_stats, _segments = _render_from_markers(markers, (h, w), BG, restore_markers=False)
    out, fill_stats, _segments = _render_from_markers(markers, (h, w), BG)

    inp = np.full((h, w), BG, dtype=int)
    for r, c, color in markers:
        inp[r, c] = int(color)

    # Independent semantic checks: structure, branch coverage, fill existence, and marker preservation.
    _semantic_check_input(inp, out, fill_stats)
    if any(raw_out[r, c] == FILL for r, c, _color in markers):
        return None
    if np.any((inp != BG) & (out == FILL)):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
