from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


BG_COLOR = Color.BLACK
MAX_GEN_ATTEMPTS = 500
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
CHAIN_LEN_CHOICES = (2, 2, 3)
EXTRA_RECT_COUNT_CHOICES = (0, 1, 1, 2)
NOISE_LEN_RANGE = (1, 5)


def _area(rect: dict) -> int:
    return int((rect["r1"] - rect["r0"] + 1) * (rect["c1"] - rect["c0"] + 1))


def _overlap_cells(rect_a: dict, rect_b: dict) -> int:
    rows = max(0, min(rect_a["r1"], rect_b["r1"]) - max(rect_a["r0"], rect_b["r0"]) + 1)
    cols = max(0, min(rect_a["c1"], rect_b["c1"]) - max(rect_a["c0"], rect_b["c0"]) + 1)
    return int(rows * cols)


def _same_or_gap_separated(box_a: dict, box_b: dict, gap: int = 1) -> bool:
    return (
        box_a["r1"] + gap < box_b["r0"]
        or box_b["r1"] + gap < box_a["r0"]
        or box_a["c1"] + gap < box_b["c0"]
        or box_b["c1"] + gap < box_a["c0"]
    )


def _color_spans_1d(line: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    for idx, value in enumerate(line.tolist()):
        color = int(value)
        if color == int(bg):
            continue
        if color not in spans:
            spans[color] = (idx, idx)
        else:
            lo, _hi = spans[color]
            spans[color] = (lo, idx)
    return spans


def _render_from_spans(
    shape: tuple[int, int],
    bg: int,
    row_spans: dict[int, tuple[int, int]],
    col_spans: dict[int, tuple[int, int]],
) -> np.ndarray:
    shared = set(row_spans) & set(col_spans)
    rects = []
    for color in shared:
        r0, r1 = row_spans[color]
        c0, c1 = col_spans[color]
        rects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )

    rects.sort(key=lambda rect: rect["area"], reverse=True)
    out = np.full(shape, int(bg), dtype=int)
    for rect in rects:
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    out[0, :] = int(bg)
    out[:, 0] = int(bg)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden filled rectangles from matching top-edge and left-edge color projections, then stack them by descending area.

    Input structure:
    - The grid is mostly background, with colored projection marks only on the top row and left column.
    - Each participating color occupies one visible run or several split runs on an edge; all visible cells of that color belong to one hidden rectangle span on that edge.
    - Some colors appear on only one edge and therefore have no complete row/column pairing.

    Rule:
    - For every color that appears on both the top row and the left column, take the full row extent of that color on the left edge and the full column extent of that color on the top edge.
    - Reconstruct the filled rectangle defined by those row and column ranges.
    - Draw all reconstructed rectangles from largest area to smallest area so smaller rectangles overwrite larger ones where they overlap.
    - Clear the projection marks themselves, so the output contains only the reconstructed interior rectangles on the original canvas.

    Key insight:
    - The input never shows the rectangles directly; it only shows one-dimensional edge projections, and split runs of the same color must be merged into one full span before pairing row and column projections.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    assert max(grid.shape) <= 30, f"grid exceeds ARC bounds: {grid.shape}"

    bg = int(get_bg(grid))
    top_spans = _color_spans_1d(grid[0], bg)
    left_spans = _color_spans_1d(grid[:, 0], bg)
    return _render_from_spans(grid.shape, bg, left_spans, top_spans)


def _sample_outer_rect(height: int, width: int, chain_len: int) -> dict | None:
    min_side = 2 * chain_len + 2
    if height - 2 < min_side or width - 2 < min_side:
        return None

    rect_h = random.randint(min_side, min(height - 2, 12))
    rect_w = random.randint(min_side, min(width - 2, 12))
    r0 = random.randint(1, height - rect_h - 1)
    c0 = random.randint(1, width - rect_w - 1)
    return {"r0": r0, "r1": r0 + rect_h - 1, "c0": c0, "c1": c0 + rect_w - 1}


def _sample_child_rect(parent: dict, levels_left: int) -> dict | None:
    parent_h = parent["r1"] - parent["r0"] + 1
    parent_w = parent["c1"] - parent["c0"] + 1
    min_side = 2 * levels_left
    if parent_h - 2 < min_side or parent_w - 2 < min_side:
        return None

    rect_h = random.randint(min_side, parent_h - 2)
    rect_w = random.randint(min_side, parent_w - 2)
    r0_min = parent["r0"] + 1
    r0_max = parent["r1"] - rect_h
    c0_min = parent["c0"] + 1
    c0_max = parent["c1"] - rect_w
    if r0_min > r0_max or c0_min > c0_max:
        return None

    r0 = random.randint(r0_min, r0_max)
    c0 = random.randint(c0_min, c0_max)
    return {"r0": r0, "r1": r0 + rect_h - 1, "c0": c0, "c1": c0 + rect_w - 1}


def _sample_extra_rect(height: int, width: int, placed: list[dict]) -> dict | None:
    for _ in range(80):
        rect_h = random.randint(1, min(6, height - 2))
        rect_w = random.randint(1, min(6, width - 2))
        r0 = random.randint(1, height - rect_h - 1)
        c0 = random.randint(1, width - rect_w - 1)
        rect = {"r0": r0, "r1": r0 + rect_h - 1, "c0": c0, "c1": c0 + rect_w - 1}
        if all(_same_or_gap_separated(rect, other, gap=1) for other in placed):
            return rect
    return None


def _draw_edge_projections(length: int, rects: list[dict], bg: int, axis: str) -> np.ndarray:
    edge = np.full(length, int(bg), dtype=int)
    ordered = sorted(rects, key=_area, reverse=True)
    for rect in ordered:
        start = rect["c0"] if axis == "top" else rect["r0"]
        end = rect["c1"] if axis == "top" else rect["r1"]
        edge[start : end + 1] = rect["color"]
    edge[0] = int(bg)
    return edge


def _background_runs(edge: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx in range(1, len(edge)):
        if int(edge[idx]) == int(bg):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(edge) - 1))
    return runs


def _add_noise_intervals(edge: np.ndarray, colors: list[int], bg: int) -> list[dict]:
    intervals = []
    runs = _background_runs(edge, bg)
    random.shuffle(runs)
    for color in colors:
        placed = False
        for run_i, (start, end) in enumerate(runs):
            run_len = end - start + 1
            if run_len <= 0:
                continue
            seg_len = random.randint(NOISE_LEN_RANGE[0], min(run_len, NOISE_LEN_RANGE[1]))
            seg_start = random.randint(start, end - seg_len + 1)
            seg_end = seg_start + seg_len - 1
            edge[seg_start : seg_end + 1] = int(color)
            intervals.append({"color": int(color), "start": int(seg_start), "end": int(seg_end)})
            new_runs = []
            if start <= seg_start - 1:
                new_runs.append((start, seg_start - 1))
            if seg_end + 1 <= end:
                new_runs.append((seg_end + 1, end))
            runs = runs[:run_i] + new_runs + runs[run_i + 1 :]
            random.shuffle(runs)
            placed = True
            break
        if not placed:
            return []
    return intervals


def _boxes_to_expected(shape: tuple[int, int], rects: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for rect in sorted(rects, key=_area, reverse=True):
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    out[0, :] = int(bg)
    out[:, 0] = int(bg)
    return out


def _build_case(return_meta: bool = False):
    for _ in range(MAX_GEN_ATTEMPTS):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        chain_len = random.choice(CHAIN_LEN_CHOICES)
        outer = _sample_outer_rect(height, width, chain_len)
        if outer is None:
            continue

        rects = [outer]
        parent = outer
        ok = True
        for depth in range(1, chain_len):
            child = _sample_child_rect(parent, chain_len - depth)
            if child is None:
                ok = False
                break
            rects.append(child)
            parent = child
        if not ok:
            continue

        extra_count = random.choice(EXTRA_RECT_COUNT_CHOICES)
        for _unused in range(extra_count):
            extra = _sample_extra_rect(height, width, rects)
            if extra is not None:
                rects.append(extra)

        areas = [_area(rect) for rect in rects]
        if len(set(areas)) != len(areas):
            continue
        if all(_overlap_cells(rects[i], rects[j]) == 0 for i in range(len(rects)) for j in range(i + 1, len(rects))):
            continue

        top_noise_count = random.randint(0, 2)
        left_noise_count = random.randint(0, 2)
        if top_noise_count + left_noise_count == 0:
            if random.random() < 0.5:
                top_noise_count = 1
            else:
                left_noise_count = 1

        total_colors = len(rects) + top_noise_count + left_noise_count
        if total_colors > len(Color.NOT_BLACK):
            continue
        palette = random.sample(Color.NOT_BLACK, total_colors)
        for rect, color in zip(rects, palette[: len(rects)]):
            rect["color"] = int(color)
            rect["area"] = _area(rect)

        top_edge = _draw_edge_projections(width, rects, BG_COLOR, axis="top")
        left_edge = _draw_edge_projections(height, rects, BG_COLOR, axis="left")

        noise_palette = palette[len(rects) :]
        top_noise_colors = noise_palette[:top_noise_count]
        left_noise_colors = noise_palette[top_noise_count : top_noise_count + left_noise_count]
        top_noise = _add_noise_intervals(top_edge, top_noise_colors, BG_COLOR)
        left_noise = _add_noise_intervals(left_edge, left_noise_colors, BG_COLOR)
        if len(top_noise) != top_noise_count or len(left_noise) != left_noise_count:
            continue

        input_grid = np.full((height, width), BG_COLOR, dtype=int)
        input_grid[0, :] = top_edge
        input_grid[:, 0] = left_edge
        input_grid[0, 0] = BG_COLOR

        top_spans = _color_spans_1d(input_grid[0], BG_COLOR)
        left_spans = _color_spans_1d(input_grid[:, 0], BG_COLOR)
        visible_colors = set(top_spans) & set(left_spans)
        expected_rects = [rect for rect in rects if rect["color"] in visible_colors]
        if len(expected_rects) < 2:
            continue
        if len(expected_rects) != len(rects):
            # Keep paired colors complete; incomplete colors come only from explicit one-edge noise.
            continue

        expected = _boxes_to_expected((height, width), expected_rects, BG_COLOR)
        actual = main(input_grid)
        if not np.array_equal(actual, expected):
            continue

        top_only_colors = set(top_spans) - set(left_spans)
        left_only_colors = set(left_spans) - set(top_spans)
        if not top_only_colors and not left_only_colors:
            continue

        # Edge extents of visible paired colors must still match the original hidden rectangles.
        span_ok = True
        for rect in expected_rects:
            row_span = left_spans[rect["color"]]
            col_span = top_spans[rect["color"]]
            if row_span != (rect["r0"], rect["r1"]) or col_span != (rect["c0"], rect["c1"]):
                span_ok = False
                break
        if not span_ok:
            continue

        meta = {
            "input": input_grid,
            "output": expected,
            "rects": [rect.copy() for rect in expected_rects],
            "top_spans": top_spans,
            "left_spans": left_spans,
            "top_only_colors": sorted(map(int, top_only_colors)),
            "left_only_colors": sorted(map(int, left_only_colors)),
            "overlap_pairs": [
                (expected_rects[i]["color"], expected_rects[j]["color"])
                for i in range(len(expected_rects))
                for j in range(i + 1, len(expected_rects))
                if _overlap_cells(expected_rects[i], expected_rects[j]) > 0
            ],
        }
        return meta if return_meta else input_grid

    raise RuntimeError("failed to generate a valid projection-reconstruction input")


def generate_input() -> np.ndarray:
    return _build_case(return_meta=False)


if __name__ == "__main__":
    case = _build_case(return_meta=True)
    counts = Counter(int(value) for value in case["input"].ravel())
    print("input_shape", case["input"].shape)
    print("output_shape", case["output"].shape)
    print("input_colors", dict(sorted(counts.items())))
