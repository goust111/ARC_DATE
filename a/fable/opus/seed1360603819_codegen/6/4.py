from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
AZURE = 8
MAROON = 9

SEED_COLORS = (RED, GREEN, MAROON)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _is_solid_rect(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return int(comp["area"]) == area and bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _is_eight_isolated(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    for dr, dc in NEIGHBORS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != bg:
            return False
    return True


def _find_structure(grid: np.ndarray) -> dict:
    bg = _dominant_color(grid)
    comps = _components(grid, bg)
    rects = [comp for comp in comps if _is_solid_rect(grid, comp)]
    candidates = []

    for i, first in enumerate(rects):
        for second in rects[i + 1 :]:
            if int(first["color"]) != int(second["color"]):
                continue

            fr0, fc0, fr1, fc1 = first["bbox"]
            sr0, sc0, sr1, sc1 = second["bbox"]
            fh = fr1 - fr0 + 1
            fw = fc1 - fc0 + 1
            sh = sr1 - sr0 + 1
            sw = sc1 - sc0 + 1
            if fh != sh or fw != sw:
                continue

            if fr0 < sr0 and fc0 < sc0:
                source, dest = first, second
            elif sr0 < fr0 and sc0 < fc0:
                source, dest = second, first
            else:
                continue

            sr0, sc0, sr1, sc1 = source["bbox"]
            dr0, dc0, dr1, dc1 = dest["bbox"]
            anchor_color = int(source["color"])

            vertical = [
                comp
                for comp in rects
                if int(comp["color"]) != anchor_color
                and comp["bbox"][0] == sr0
                and comp["bbox"][2] == sr1
                and comp["bbox"][1] == comp["bbox"][3]
                and sc1 < comp["bbox"][1] < dc0
            ]
            horizontal = [
                comp
                for comp in rects
                if int(comp["color"]) != anchor_color
                and comp["bbox"][1] == sc0
                and comp["bbox"][3] == sc1
                and comp["bbox"][0] == comp["bbox"][2]
                and sr1 < comp["bbox"][0] < dr0
            ]
            if len(vertical) != 2 or len(horizontal) != 2:
                continue

            vertical.sort(key=lambda comp: int(comp["bbox"][1]))
            horizontal.sort(key=lambda comp: int(comp["bbox"][0]))
            control_colors = {int(comp["color"]) for comp in vertical + horizontal}
            if len(control_colors) != 1:
                continue
            control_color = int(vertical[0]["color"])

            first_tick_c = int(vertical[0]["bbox"][1])
            second_tick_c = int(vertical[1]["bbox"][1])
            first_tick_r = int(horizontal[0]["bbox"][0])
            second_tick_r = int(horizontal[1]["bbox"][0])
            period_x = second_tick_c - first_tick_c
            period_y = second_tick_r - first_tick_r
            if first_tick_c != sc1 + 1 or first_tick_r != sr1 + 1 or period_x <= 0 or period_x != period_y:
                continue

            seed_candidates = [
                comp
                for comp in comps
                if int(comp["area"]) == 1 and int(comp["color"]) not in {anchor_color, control_color}
            ]
            if len(seed_candidates) != 1:
                continue
            seed_r, seed_c = seed_candidates[0]["cells"][0]
            if not _is_eight_isolated(grid, seed_r, seed_c, bg):
                continue

            candidates.append(
                {
                    "bg": bg,
                    "anchor_color": anchor_color,
                    "control_color": control_color,
                    "seed_color": int(seed_candidates[0]["color"]),
                    "source_bbox": source["bbox"],
                    "dest_bbox": dest["bbox"],
                    "period": period_x,
                }
            )

    assert len(candidates) == 1, "expected one valid decorated diagonal-anchor structure"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a tick period from the paired control bars, complete the missing corner
    anchors, and recolor the full periodic outer tick frame with the remote seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    structure = _find_structure(g)
    anchor_color = int(structure["anchor_color"])
    seed_color = int(structure["seed_color"])
    period = int(structure["period"])
    sr0, sc0, sr1, sc1 = structure["source_bbox"]
    dr0, dc0, dr1, dc1 = structure["dest_bbox"]

    out = g.copy()

    out[sr0 : sr1 + 1, dc0 : dc1 + 1] = anchor_color
    out[dr0 : dr1 + 1, sc0 : sc1 + 1] = anchor_color

    for c in range(sc1 + 1, dc0, period):
        out[sr0 : sr1 + 1, c] = seed_color
        out[dr0 : dr1 + 1, c] = seed_color

    for r in range(sr1 + 1, dr0, period):
        out[r, sc0 : sc1 + 1] = seed_color
        out[r, dc0 : dc1 + 1] = seed_color

    return out


MIN_GRID = 14
MAX_GRID = 30
ANCHOR_SIZE_RANGE = (2, 5)
PERIOD_RANGE = (2, 5)
TICK_COUNT_RANGE = (2, 6)
BASE_MARGIN = 2
MAX_GENERATION_ATTEMPTS = 400


def _split_amount(total: int) -> tuple[int, int]:
    left = random.randint(0, total)
    return left, total - left


def _seed_position(corner: str, rows: int, cols: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, cols - 1
    if corner == "bl":
        return rows - 1, 0
    return rows - 1, cols - 1


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    anchor_h = random.randint(*ANCHOR_SIZE_RANGE)
    anchor_w = random.randint(*ANCHOR_SIZE_RANGE)
    period = random.randint(*PERIOD_RANGE)
    tail_x = random.randint(1, period)
    tail_y = random.randint(1, period)

    max_tick_cols = (MAX_GRID - 2 * BASE_MARGIN - 2 * anchor_w - tail_x) // period + 1
    max_tick_rows = (MAX_GRID - 2 * BASE_MARGIN - 2 * anchor_h - tail_y) // period + 1
    max_tick_cols = min(TICK_COUNT_RANGE[1], max_tick_cols)
    max_tick_rows = min(TICK_COUNT_RANGE[1], max_tick_rows)
    if max_tick_cols < TICK_COUNT_RANGE[0] or max_tick_rows < TICK_COUNT_RANGE[0]:
        raise ValueError("structure does not fit inside the maximum canvas")

    tick_cols_count = random.randint(TICK_COUNT_RANGE[0], max_tick_cols)
    tick_rows_count = random.randint(TICK_COUNT_RANGE[0], max_tick_rows)

    span_h = 2 * anchor_h + (tick_rows_count - 1) * period + tail_y
    span_w = 2 * anchor_w + (tick_cols_count - 1) * period + tail_x
    min_rows = span_h + 2 * BASE_MARGIN
    min_cols = span_w + 2 * BASE_MARGIN
    if min_rows > MAX_GRID or min_cols > MAX_GRID:
        raise ValueError("minimum canvas exceeds size limit")

    extra_rows_min = max(0, MIN_GRID - min_rows)
    extra_cols_min = max(0, MIN_GRID - min_cols)
    extra_rows = random.randint(extra_rows_min, MAX_GRID - min_rows)
    extra_cols = random.randint(extra_cols_min, MAX_GRID - min_cols)
    extra_top, extra_bottom = _split_amount(extra_rows)
    extra_left, extra_right = _split_amount(extra_cols)

    top_margin = BASE_MARGIN + extra_top
    bottom_margin = BASE_MARGIN + extra_bottom
    left_margin = BASE_MARGIN + extra_left
    right_margin = BASE_MARGIN + extra_right
    rows = span_h + top_margin + bottom_margin
    cols = span_w + left_margin + right_margin

    source_r = top_margin
    source_c = left_margin
    source_r1 = source_r + anchor_h - 1
    source_c1 = source_c + anchor_w - 1
    tick_cols = [source_c1 + 1 + period * i for i in range(tick_cols_count)]
    tick_rows = [source_r1 + 1 + period * i for i in range(tick_rows_count)]
    dest_r = tick_rows[-1] + tail_y
    dest_c = tick_cols[-1] + tail_x

    seed_color = random.choice(SEED_COLORS)
    seed_corner = random.choice(("tl", "tr", "bl", "br"))
    seed_r, seed_c = _seed_position(seed_corner, rows, cols)

    inp = np.full((rows, cols), BLACK, dtype=int)
    _paint_rect(inp, source_r, source_c, anchor_h, anchor_w, AZURE)
    _paint_rect(inp, dest_r, dest_c, anchor_h, anchor_w, AZURE)
    for c in tick_cols[:2]:
        inp[source_r : source_r + anchor_h, c] = YELLOW
    for r in tick_rows[:2]:
        inp[r, source_c : source_c + anchor_w] = YELLOW
    inp[seed_r, seed_c] = seed_color

    out = inp.copy()
    _paint_rect(out, source_r, dest_c, anchor_h, anchor_w, AZURE)
    _paint_rect(out, dest_r, source_c, anchor_h, anchor_w, AZURE)
    for c in tick_cols:
        out[source_r : source_r + anchor_h, c] = seed_color
        out[dest_r : dest_r + anchor_h, c] = seed_color
    for r in tick_rows:
        out[r, source_c : source_c + anchor_w] = seed_color
        out[r, dest_c : dest_c + anchor_w] = seed_color

    if _dominant_color(inp) != BLACK:
        raise ValueError("background must stay dominant")

    if not _is_eight_isolated(inp, seed_r, seed_c, BLACK):
        raise ValueError("seed must be isolated")

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected = _build_case()
        except ValueError:
            continue

        try:
            actual = main(inp)
        except Exception:
            continue

        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("failed to generate a valid input for puzzle 4")
