from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            comps.append(
                {
                    "index": len(comps),
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
    return comps


def _is_full_rectangle(comp: dict[str, object]) -> bool:
    return int(comp["area"]) == int(comp["height"]) * int(comp["width"])


def _detect_legend(
    comps: list[dict[str, object]], height: int, width: int
) -> tuple[dict[tuple[int, ...], int], set[int]]:
    candidates: list[tuple[str, list[dict[str, object]]]] = []
    for side in ("left", "right"):
        if side == "left":
            outputs = [
                comp
                for comp in comps
                if int(comp["height"]) == 1 and int(comp["width"]) == 1 and int(comp["c0"]) == 0
            ]
        else:
            outputs = [
                comp
                for comp in comps
                if int(comp["height"]) == 1 and int(comp["width"]) == 1 and int(comp["c1"]) == width - 1
            ]
        if len(outputs) < 2:
            continue
        rows = sorted(int(comp["r0"]) for comp in outputs)
        if any(b - a != 2 for a, b in zip(rows, rows[1:])):
            continue
        if not (rows[0] == 0 or rows[-1] == height - 1):
            continue
        candidates.append((side, outputs))

    assert len(candidates) == 1, "legend outputs must form a unique border-aligned column"
    side, outputs = candidates[0]

    output_by_row = {int(comp["r0"]): int(comp["color"]) for comp in outputs}
    rows = sorted(output_by_row)
    legend_ids = {int(comp["index"]) for comp in outputs}
    legend_map: dict[tuple[int, ...], int] = {}

    for row in rows:
        row_parts = [
            comp
            for comp in comps
            if int(comp["height"]) == 1
            and int(comp["r0"]) == row
            and int(comp["index"]) not in legend_ids
        ]
        assert row_parts, "each legend row must contain at least one width bar"
        row_parts.sort(key=lambda comp: int(comp["c0"]))
        seq = tuple(int(comp["width"]) for comp in row_parts)
        assert seq not in legend_map, "legend sequences must be unique"
        legend_map[seq] = output_by_row[row]
        legend_ids.update(int(comp["index"]) for comp in row_parts)

    return legend_map, legend_ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the foreground objects in row-major visual order, turn their widths into a code,
    and use the corner legend to map that code to a single output color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "grid must contain foreground components"

    legend_map, legend_ids = _detect_legend(comps, h, w)

    objects = [comp for comp in comps if int(comp["index"]) not in legend_ids]
    assert objects, "grid must contain non-legend objects"
    for comp in objects:
        assert int(comp["height"]) >= 2, "data objects must be taller than legend bars"
        assert _is_full_rectangle(comp), "data objects must be filled rectangles"

    objects.sort(key=lambda comp: (int(comp["r0"]), int(comp["c0"])))
    code = tuple(int(comp["width"]) for comp in objects)
    assert code in legend_map, "object-width code must appear in the legend"
    return np.array([[legend_map[code]]], dtype=int)


def _rand_seq(length: int) -> tuple[int, ...]:
    return tuple(random.randint(1, 5) for _ in range(length))


def _split_positive(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    out: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _build_object_patch(bg: int, widths: tuple[int, ...], colors: list[int]) -> np.ndarray:
    count = len(widths)
    lane_count = random.randint(1, min(3, count))
    lane_sizes = _split_positive(count, lane_count)
    heights = [random.randint(2, 3) for _ in widths]

    rects: list[tuple[int, int, int, int, int]] = []
    top = 0
    patch_w = 0
    idx = 0
    for lane_size in lane_sizes:
        lane_h = max(heights[idx : idx + lane_size])
        col = random.randint(0, 2)
        lane_right = 0
        for j in range(lane_size):
            rect_h = heights[idx + j]
            rect_w = int(widths[idx + j])
            color = int(colors[idx + j])
            rects.append((top, col, rect_h, rect_w, color))
            lane_right = col + rect_w
            if j < lane_size - 1:
                col = lane_right + random.randint(2, 4)
        patch_w = max(patch_w, lane_right + random.randint(0, 2))
        top += lane_h + 2
        idx += lane_size

    patch_h = top - 2
    patch = np.full((patch_h, patch_w), bg, dtype=int)
    for r0, c0, rect_h, rect_w, color in rects:
        patch[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
    return patch


def _legend_row_width(seq: tuple[int, ...]) -> int:
    return sum(seq) + max(len(seq) - 1, 0) + 3


def _paint_legend(
    grid: np.ndarray,
    entries: list[tuple[tuple[int, ...], int]],
    digit_color: int,
    side: str,
    vertical: str,
) -> None:
    h, w = grid.shape
    legend_h = 2 * len(entries) - 1
    base_r = 0 if vertical == "top" else h - legend_h

    for i, (seq, out_color) in enumerate(entries):
        r = base_r + 2 * i
        if side == "left":
            grid[r, 0] = int(out_color)
            c = 2
        else:
            grid[r, w - 1] = int(out_color)
            bar_len = sum(seq) + len(seq) - 1
            c = w - 2 - bar_len
        for width in seq:
            grid[r, c : c + width] = int(digit_color)
            c += width + 1


def _sample_entries(target_seq: tuple[int, ...], count: int) -> list[tuple[tuple[int, ...], int]]:
    rows = {target_seq}
    while len(rows) < count:
        rows.add(_rand_seq(random.randint(2, 5)))
    seqs = list(rows)
    random.shuffle(seqs)
    return [(seq, 0) for seq in seqs]


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    non_bg = [color for color in PALETTE if color != bg]

    object_count = random.randint(2, 5)
    target_seq = _rand_seq(object_count)
    legend_rows = random.randint(3, 5)

    entries = _sample_entries(target_seq, legend_rows)
    output_colors = random.sample(non_bg, legend_rows)
    for i, color in enumerate(output_colors):
        seq, _ = entries[i]
        entries[i] = (seq, int(color))

    target_color = next(color for seq, color in entries if seq == target_seq)
    remaining_for_digits = [color for color in non_bg if color not in output_colors]
    digit_color = random.choice(remaining_for_digits) if remaining_for_digits else random.choice(non_bg)

    object_colors = random.sample(non_bg, object_count)
    object_patch = _build_object_patch(bg, target_seq, object_colors)

    legend_w = max(_legend_row_width(seq) for seq, _ in entries)
    legend_h = 2 * legend_rows - 1
    grid_h = legend_h + int(object_patch.shape[0]) + 6 + random.randint(0, 5)
    grid_w = max(legend_w, int(object_patch.shape[1]) + 4) + random.randint(0, 6)

    side = random.choice(("left", "right"))
    vertical = random.choice(("top", "bottom"))
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    _paint_legend(grid, entries, digit_color, side, vertical)

    patch_h, patch_w = map(int, object_patch.shape)
    obj_left = random.randint(2, grid_w - patch_w - 2)
    if vertical == "top":
        obj_top = random.randint(legend_h + 2, grid_h - patch_h - 2)
    else:
        obj_top = random.randint(2, grid_h - legend_h - patch_h - 2)
    mask = object_patch != bg
    grid[obj_top : obj_top + patch_h, obj_left : obj_left + patch_w][mask] = object_patch[mask]

    expected = np.array([[target_color]], dtype=int)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
