from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_from_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict[str, int] | None:
    pts = np.array(comp, dtype=int)
    rows = pts[:, 0]
    cols = pts[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    colors = np.unique(grid[rows, cols])
    if colors.size != 1:
        return None
    color = int(colors[0])
    if color in (BLACK, ORANGE):
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True

    if not np.array_equal(sub == color, ring):
        return None

    interior = sub[1:-1, 1:-1]
    if np.any((interior != BLACK) & (interior != ORANGE)):
        return None

    count = int(np.count_nonzero(interior == ORANGE))
    if count <= 0:
        return None

    return {
        "color": color,
        "count": count,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "center_r2": r0 + r1,
        "center_c2": c0 + c1,
    }


def _extract_frames(grid: np.ndarray) -> list[dict[str, int]]:
    mask = (grid != BLACK) & (grid != ORANGE)
    frames: list[dict[str, int]] = []
    for comp in _components(mask):
        frame = _frame_from_component(grid, comp)
        assert frame is not None, "expected every colored component to be a 1-pixel rectangular frame"
        frames.append(frame)
    assert frames, "expected at least one rectangular frame"
    frames.sort(key=lambda item: (item["center_r2"], item["center_c2"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each colored rectangular frame into a solid square whose side length equals
    the number of orange pixels inside that frame, then pack the squares left-to-right
    in input-position order with bottom alignment.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)

    max_side = max(frame["count"] for frame in frames)
    total_width = sum(frame["count"] for frame in frames)
    out = np.full((max_side, total_width), BLACK, dtype=int)

    x = 0
    for frame in frames:
        side = frame["count"]
        out[max_side - side :, x : x + side] = frame["color"]
        x += side
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out: list[int] = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _make_frame_spec(color: int, inner_h: int) -> dict[str, object] | None:
    count = random.randint(1, 5)
    min_inner_w = max(2, (count + 1 + inner_h - 1) // inner_h)
    choices = [w for w in range(min_inner_w, 8) if inner_h * w > count]
    if not choices:
        return None
    inner_w = random.choice(choices)
    orange_cells = random.sample([(r, c) for r in range(inner_h) for c in range(inner_w)], count)
    return {
        "color": color,
        "count": count,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
        "orange_cells": orange_cells,
    }


def _draw_frame(grid: np.ndarray, r0: int, c0: int, spec: dict[str, object]) -> None:
    color = int(spec["color"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = color
    grid[r0 + 1 : r0 + outer_h - 1, c0 + 1 : c0 + outer_w - 1] = BLACK
    for rr, cc in spec["orange_cells"]:
        grid[r0 + 1 + rr, c0 + 1 + cc] = ORANGE


def _expected_output(placed: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(placed, key=lambda item: (item["center_r2"], item["center_c2"]))
    max_side = max(item["count"] for item in ordered)
    total_width = sum(item["count"] for item in ordered)
    out = np.full((max_side, total_width), BLACK, dtype=int)
    x = 0
    for item in ordered:
        side = int(item["count"])
        out[max_side - side :, x : x + side] = int(item["color"])
        x += side
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(1, 6)
    row_count = random.randint(1 if frame_count < 5 else 2, min(3, frame_count))
    row_sizes = _random_partition(frame_count, row_count)
    colors = random.sample(FRAME_COLORS, frame_count)

    rows: list[list[dict[str, object]]] = []
    color_idx = 0
    for row_size in row_sizes:
        inner_h = random.randint(2, 5)
        row_specs: list[dict[str, object]] = []
        for _ in range(row_size):
            spec = _make_frame_spec(colors[color_idx], inner_h)
            color_idx += 1
            if spec is None:
                return None
            row_specs.append(spec)
        rows.append(row_specs)

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    row_offsets = [random.randint(0, 4) for _ in range(row_count)]
    row_gaps = [[random.randint(2, 4) for _ in range(max(0, len(row) - 1))] for row in rows]
    vertical_gaps = [random.randint(2, 4) for _ in range(max(0, row_count - 1))]

    row_heights = [int(row[0]["outer_h"]) for row in rows]
    row_widths = []
    for row, gaps in zip(rows, row_gaps):
        width = sum(int(spec["outer_w"]) for spec in row) + sum(gaps)
        row_widths.append(width)

    height = top_margin + sum(row_heights) + sum(vertical_gaps) + bottom_margin
    width = max(left_margin + offset + row_width + right_margin for offset, row_width in zip(row_offsets, row_widths))
    if not (12 <= height <= 30 and 12 <= width <= 30):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    placed: list[dict[str, int]] = []

    y = top_margin
    for row_idx, row in enumerate(rows):
        x = left_margin + row_offsets[row_idx]
        gaps = row_gaps[row_idx]
        for col_idx, spec in enumerate(row):
            _draw_frame(grid, y, x, spec)
            outer_h = int(spec["outer_h"])
            outer_w = int(spec["outer_w"])
            placed.append(
                {
                    "color": int(spec["color"]),
                    "count": int(spec["count"]),
                    "center_r2": 2 * y + outer_h - 1,
                    "center_c2": 2 * x + outer_w - 1,
                }
            )
            x += outer_w
            if col_idx < len(gaps):
                x += gaps[col_idx]
        y += row_heights[row_idx]
        if row_idx < len(vertical_gaps):
            y += vertical_gaps[row_idx]

    expected = _expected_output(placed)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
