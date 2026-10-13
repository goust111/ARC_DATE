import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
LINE = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return int(Counter(corners).most_common(1)[0][0])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _segment_from_cells(cells: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    size = len(cells)
    if r0 == r1 and size == c1 - c0 + 1:
        return {
            "orientation": "h",
            "row": r0,
            "c0": c0,
            "c1": c1,
            "length": size,
        }
    if c0 == c1 and size == r1 - r0 + 1:
        return {
            "orientation": "v",
            "col": c0,
            "r0": r0,
            "r1": r1,
            "length": size,
        }
    return None


def _detect_line_color(components: list[dict]) -> int | None:
    scores: dict[int, tuple[int, int]] = {}
    for comp in components:
        seg = _segment_from_cells(comp["cells"])
        if seg is None or int(seg["length"]) < 2:
            continue
        color = int(comp["color"])
        total, count = scores.get(color, (0, 0))
        scores[color] = (total + int(seg["length"]), count + 1)
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1][0], item[1][1], -item[0]))[0]


def _pair_segments(segments: list[dict]) -> list[tuple[dict, dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for seg in segments:
        if seg["orientation"] == "h":
            key = ("h", int(seg["c0"]), int(seg["c1"]))
        else:
            key = ("v", int(seg["r0"]), int(seg["r1"]))
        groups[key].append(seg)

    pairs = []
    for key, group in groups.items():
        if key[0] == "h":
            group.sort(key=lambda seg: int(seg["row"]))
        else:
            group.sort(key=lambda seg: int(seg["col"]))
        for idx in range(0, len(group) - 1, 2):
            pairs.append((group[idx], group[idx + 1]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each strip enclosed by a matched pair of aligned grey line segments
    using the unique colored seed pixel that lies between them.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    components = _components(g, bg)
    line_color = _detect_line_color(components)
    if line_color is None:
        return g.copy()

    segments = []
    for comp in components:
        if int(comp["color"]) != int(line_color):
            continue
        seg = _segment_from_cells(comp["cells"])
        if seg is not None and int(seg["length"]) >= 2:
            segments.append(seg)

    out = g.copy()
    for first, second in _pair_segments(segments):
        if first["orientation"] == "h":
            top = min(int(first["row"]), int(second["row"]))
            bottom = max(int(first["row"]), int(second["row"]))
            if bottom - top <= 1:
                continue
            row_slice = slice(top + 1, bottom)
            col_slice = slice(int(first["c0"]), int(first["c1"]) + 1)
        else:
            left = min(int(first["col"]), int(second["col"]))
            right = max(int(first["col"]), int(second["col"]))
            if right - left <= 1:
                continue
            row_slice = slice(int(first["r0"]), int(first["r1"]) + 1)
            col_slice = slice(left + 1, right)

        band = g[row_slice, col_slice]
        seed_colors = {
            int(color)
            for color in np.unique(band)
            if int(color) not in (int(bg), int(line_color))
        }
        if len(seed_colors) != 1:
            continue
        fill_color = next(iter(seed_colors))
        patch = out[row_slice, col_slice]
        mask = (band == bg) | (band == fill_color)
        patch[mask] = fill_color

    return out


def _orientation_plan(pair_count: int) -> list[str]:
    mode = random.choice(("h_only", "v_only", "mixed"))
    if mode == "h_only":
        return ["h"] * pair_count
    if mode == "v_only":
        return ["v"] * pair_count
    h_count = pair_count // 2
    v_count = pair_count - h_count
    if pair_count % 2 == 1 and random.random() < 0.5:
        h_count, v_count = v_count, h_count
    plan = ["h"] * h_count + ["v"] * v_count
    random.shuffle(plan)
    return plan


def _bbox_with_margin(
    top: int, left: int, height: int, width: int, max_h: int, max_w: int, margin: int = 1
) -> tuple[int, int, int, int]:
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(max_h, top + height + margin)
    c1 = min(max_w, left + width + margin)
    return r0, c0, r1, c1


def _pair_output_region(pair: dict) -> tuple[slice, slice]:
    if pair["orientation"] == "h":
        return (
            slice(int(pair["top"]) + 1, int(pair["top"]) + int(pair["gap"]) + 1),
            slice(int(pair["left"]), int(pair["left"]) + int(pair["length"])),
        )
    return (
        slice(int(pair["top"]), int(pair["top"]) + int(pair["length"])),
        slice(int(pair["left"]) + 1, int(pair["left"]) + int(pair["gap"]) + 1),
    )


def _expected_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        rows, cols = _pair_output_region(pair)
        out[rows, cols] = int(pair["seed_color"])
    return out


def _try_place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    used_h_spans: set[tuple[int, int]],
    used_v_spans: set[tuple[int, int]],
    orientation: str,
) -> dict | None:
    h, w = grid.shape
    max_length = min(10, w - 2) if orientation == "h" else min(10, h - 2)
    max_gap = min(5, h - 4) if orientation == "h" else min(5, w - 4)
    if max_length < 4 or max_gap < 1:
        return None

    for _ in range(120):
        length = random.randint(4, max_length)
        gap = random.randint(1, max_gap)
        if orientation == "h":
            box_h = gap + 2
            box_w = length
            if h - box_h - 1 < 1 or w - box_w - 1 < 1:
                continue
            top = random.randint(1, h - box_h - 1)
            left = random.randint(1, w - box_w - 1)
            span = (left, left + length - 1)
            if span in used_h_spans:
                continue
        else:
            box_h = length
            box_w = gap + 2
            if h - box_h - 1 < 1 or w - box_w - 1 < 1:
                continue
            top = random.randint(1, h - box_h - 1)
            left = random.randint(1, w - box_w - 1)
            span = (top, top + length - 1)
            if span in used_v_spans:
                continue

        r0, c0, r1, c1 = _bbox_with_margin(top, left, box_h, box_w, h, w, margin=1)
        if bool(occupied[r0:r1, c0:c1].any()):
            continue

        seed_color = random.choice(SEED_COLORS)
        if orientation == "h":
            grid[top, left : left + length] = LINE
            grid[top + gap + 1, left : left + length] = LINE
            seed_r = random.randint(top + 1, top + gap)
            seed_c = random.randint(left, left + length - 1)
            used_h_spans.add(span)
        else:
            grid[top : top + length, left] = LINE
            grid[top : top + length, left + gap + 1] = LINE
            seed_r = random.randint(top, top + length - 1)
            seed_c = random.randint(left + 1, left + gap)
            used_v_spans.add(span)

        grid[seed_r, seed_c] = seed_color
        occupied[r0:r1, c0:c1] = True
        return {
            "orientation": orientation,
            "top": top,
            "left": left,
            "length": length,
            "gap": gap,
            "seed_color": seed_color,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 5)
    base_side = 12 + pair_count * 2
    height = random.randint(base_side, base_side + 6)
    width = random.randint(base_side, base_side + 6)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    pairs = []
    used_h_spans: set[tuple[int, int]] = set()
    used_v_spans: set[tuple[int, int]] = set()
    for orientation in _orientation_plan(pair_count):
        pair = _try_place_pair(grid, occupied, used_h_spans, used_v_spans, orientation)
        if pair is None:
            return None
        pairs.append(pair)

    expected = _expected_output(grid, pairs)
    actual = main(grid)
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid strip-fill input")
