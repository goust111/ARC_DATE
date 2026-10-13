from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIVIDER = 5
STOP = 1
PASS = 9
BODY_COLORS = (1, 2, 3, 4, 7)
GEN_BODY_COLORS = (2, 3, 4, 7)

MIN_H = 2
MAX_H = 3
MIN_W = 2
MAX_W = 4

LEFT_SLOT_START = 1
LEFT_SLOT_WIDTH = 4
CENTER_SLOT_START = 6
CENTER_SLOT_WIDTH = 5
GEN_RETRIES = 600


def _separator_rows(grid: np.ndarray) -> list[int]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    assert 2 <= len(rows) <= 3, "expected 2-3 full-width divider rows"
    assert rows == sorted(rows), "divider rows must be ordered"
    return rows


def _band_ranges(height: int, sep_rows: Iterable[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for row in sep_rows:
        assert start <= row - 1, "divider rows must leave a non-empty band"
        bands.append((start, int(row) - 1))
        start = int(row) + 1
    assert start <= height - 1, "expected a trailing band"
    bands.append((start, height - 1))
    return bands


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    sep_rows = _separator_rows(g)
    bands = _band_ranges(h, sep_rows)
    used = np.zeros_like(g, dtype=bool)
    rects: list[dict] = []

    for band_index, (r0, r1) in enumerate(bands):
        band_rects = 0
        for body_top in range(r0 + 1, r1):
            c = 0
            while c < w:
                color = int(g[body_top, c])
                if color in (BG, DIVIDER, PASS) or used[body_top, c]:
                    c += 1
                    continue

                c1 = c
                while c1 + 1 < w and int(g[body_top, c1 + 1]) == color:
                    c1 += 1
                width = c1 - c + 1
                if not (MIN_W <= width <= MAX_W):
                    c = c1 + 1
                    continue

                above = g[body_top - 1, c : c1 + 1]
                control_offsets = [
                    idx for idx, value in enumerate(above.tolist()) if int(value) in (STOP, PASS)
                ]
                if len(control_offsets) != 1:
                    c = c1 + 1
                    continue
                if int(np.count_nonzero(above != BG)) != 1:
                    c = c1 + 1
                    continue

                body_height = 1
                while (
                    body_top + body_height <= r1
                    and body_height < MAX_H
                    and np.all(g[body_top + body_height, c : c1 + 1] == color)
                ):
                    body_height += 1
                if body_height < MIN_H:
                    c = c1 + 1
                    continue

                body_bottom = body_top + body_height - 1
                if body_bottom >= r1:
                    c = c1 + 1
                    continue

                block = g[body_top : body_bottom + 1, c : c1 + 1]
                if not np.all(block == color):
                    c = c1 + 1
                    continue
                if np.any(used[body_top - 1 : body_bottom + 1, c : c1 + 1]):
                    c = c1 + 1
                    continue

                if not np.all(g[body_bottom + 1, c : c1 + 1] == BG):
                    c = c1 + 1
                    continue

                if c > 0 and np.any(g[body_top - 1 : body_bottom + 1, c - 1] != BG):
                    c = c1 + 1
                    continue
                if c1 + 1 < w and np.any(g[body_top - 1 : body_bottom + 1, c1 + 1] != BG):
                    c = c1 + 1
                    continue

                ctrl_col = c + control_offsets[0]
                ctrl_color = int(g[body_top - 1, ctrl_col])
                used[body_top - 1, ctrl_col] = True
                used[body_top : body_bottom + 1, c : c1 + 1] = True
                rects.append(
                    {
                        "band": band_index,
                        "color": color,
                        "control": ctrl_color,
                        "top": body_top,
                        "left": c,
                        "height": body_height,
                        "width": width,
                        "area": body_height * width,
                        "control_col": ctrl_col,
                    }
                )
                band_rects += 1
                c = c1 + 1

        assert 1 <= band_rects <= 2, "expected 1-2 rectangles per band"

    relevant = (g != BG) & (g != DIVIDER)
    assert np.array_equal(used, relevant), "every non-divider foreground cell must belong to one rectangle+control"
    return rects


def _landing_top(rect: dict, sep_rows: list[int], height: int, mode: str = "normal") -> int:
    body_bottom = int(rect["top"]) + int(rect["height"]) - 1
    control = int(rect["control"])

    if mode == "all_pass":
        control = PASS
    elif mode == "all_stop":
        control = STOP

    if control == STOP:
        floors = [row for row in sep_rows if row > body_bottom]
        if floors:
            return int(floors[0] - rect["height"])
    return int(height - rect["height"])


def _priority_key(rect: dict) -> tuple[int, int, int]:
    return int(rect["area"]), -int(rect["band"]), int(rect["left"])


def _reverse_priority_key(rect: dict) -> tuple[int, int, int]:
    return -int(rect["area"]), int(rect["band"]), int(rect["left"])


def _render_from_rectangles(
    shape: tuple[int, int],
    sep_rows: list[int],
    rects: list[dict],
    *,
    mode: str = "normal",
    reverse_priority: bool = False,
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    for row in sep_rows:
        out[int(row), :] = DIVIDER

    key = _reverse_priority_key if reverse_priority else _priority_key
    for rect in sorted(rects, key=key):
        top = _landing_top(rect, sep_rows, h, mode=mode)
        left = int(rect["left"])
        rh = int(rect["height"])
        rw = int(rect["width"])
        out[top : top + rh, left : left + rw] = int(rect["color"])
    return out


def _overlap_area(a: dict, b: dict, shape: tuple[int, int], sep_rows: list[int]) -> int:
    top_a = _landing_top(a, sep_rows, shape[0])
    top_b = _landing_top(b, sep_rows, shape[0])
    bottom_a = top_a + int(a["height"]) - 1
    bottom_b = top_b + int(b["height"]) - 1
    left_a = int(a["left"])
    left_b = int(b["left"])
    right_a = left_a + int(a["width"]) - 1
    right_b = left_b + int(b["width"]) - 1
    rh = max(0, min(bottom_a, bottom_b) - max(top_a, top_b) + 1)
    rw = max(0, min(right_a, right_b) - max(left_a, left_b) + 1)
    return int(rh * rw)


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        sep_rows = _separator_rows(grid)
        bands = _band_ranges(grid.shape[0], sep_rows)
        heights = {r1 - r0 + 1 for r0, r1 in bands}
        if len(heights) != 1:
            return False

        rects = _find_rectangles(grid)
        per_band: dict[int, list[dict]] = {}
        for rect in rects:
            per_band.setdefault(int(rect["band"]), []).append(rect)

        for band_index in range(len(bands) - 1):
            band_rects = per_band.get(band_index, [])
            if len(band_rects) != 2:
                return False
            controls = {int(rect["control"]) for rect in band_rects}
            if controls != {STOP, PASS}:
                return False

        if len(per_band.get(len(bands) - 1, [])) not in (1, 2):
            return False

        overlaps: list[tuple[dict, dict]] = []
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                area = _overlap_area(rects[i], rects[j], grid.shape, sep_rows)
                if area > 0:
                    overlaps.append((rects[i], rects[j]))
                    if rects[i]["area"] == rects[j]["area"] and rects[i]["band"] == rects[j]["band"]:
                        return False
        if not overlaps:
            return False

        correct = _render_from_rectangles(grid.shape, sep_rows, rects)
        if np.array_equal(correct, _render_from_rectangles(grid.shape, sep_rows, rects, mode="all_pass")):
            return False
        if np.array_equal(correct, _render_from_rectangles(grid.shape, sep_rows, rects, mode="all_stop")):
            return False
        if np.array_equal(
            correct,
            _render_from_rectangles(grid.shape, sep_rows, rects, reverse_priority=True),
        ):
            return False

        if not np.array_equal(correct, main(grid)):
            return False
        if np.array_equal(correct, grid):
            return False
        return True
    except AssertionError:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each banded rectangle downward, using the control pixel to decide whether grey dividers block it or not.

    Input structure
    The background is black. Two or three full-width grey rows split the grid into
    equal horizontal bands. Each band contains one or two solid monochrome
    rectangles, each with exactly one adjacent control pixel directly above it:
    blue means "stop at the first divider below", maroon means "ignore dividers".

    Rule
    Erase every control pixel and move every rectangle straight downward while
    keeping its shape and columns. A blue-controlled rectangle stops with its
    bottom edge immediately above the first grey divider below it, or on the grid
    bottom if no divider lies below. A maroon-controlled rectangle ignores every
    grey divider and bottom-aligns. Keep the grey divider rows unless a landed
    rectangle covers them. When landed rectangles overlap, the larger-area
    rectangle is drawn on top; equal areas are broken by the rectangle that
    started in the higher band.

    Key insight
    The same grey rows serve both as visible band separators and as selective
    obstacles: the control color decides whether a rectangle treats those rows as
    floors, and only after that selective fall can the overlap-priority rule be
    applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_rows = _separator_rows(g)
    rects = _find_rectangles(g)
    return _render_from_rectangles(g.shape, sep_rows, rects)


def _rand_size(band_h: int) -> tuple[int, int]:
    height = random.randint(MIN_H, min(MAX_H, band_h - 2))
    width = random.randint(MIN_W, MAX_W)
    return height, width


def _top_for_height(band: tuple[int, int], band_h: int, rect_h: int) -> int:
    max_extra = band_h - rect_h - 2
    extra = random.randint(0, max_extra)
    return int(band[0] + 1 + extra)


def _slot_left(slot_start: int, slot_width: int, rect_w: int) -> int:
    return int(slot_start + random.randint(0, slot_width - rect_w))


def _sample_overlap_pair(width_a: int, width_b: int) -> tuple[int, int]:
    start = CENTER_SLOT_START
    choices: list[tuple[int, int]] = []
    for left_a in range(start, start + CENTER_SLOT_WIDTH - width_a + 1):
        for left_b in range(start, start + CENTER_SLOT_WIDTH - width_b + 1):
            overlap = max(
                0,
                min(left_a + width_a - 1, left_b + width_b - 1) - max(left_a, left_b) + 1,
            )
            if overlap > 0:
                choices.append((left_a, left_b))
    assert choices, "expected at least one overlapping placement"
    return random.choice(choices)


def _right_slot_start(width: int) -> int:
    return int(width - LEFT_SLOT_WIDTH)


def _slot_bounds(side: str, width: int) -> tuple[int, int]:
    if side == "left":
        return LEFT_SLOT_START, LEFT_SLOT_WIDTH
    if side == "right":
        return _right_slot_start(width), LEFT_SLOT_WIDTH
    raise ValueError("invalid side")


def _build_band_rects(
    bands: list[tuple[int, int]],
    band_h: int,
    width: int,
) -> list[dict] | None:
    rects: list[dict] = []
    overlap_bands = sorted(random.sample(range(len(bands) - 1), 2))
    equal_area_overlap = random.random() < 0.5

    h1, w1 = _rand_size(band_h)
    if equal_area_overlap:
        h2, w2 = h1, w1
    else:
        for _ in range(40):
            h2, w2 = _rand_size(band_h)
            if h2 * w2 != h1 * w1:
                break
        else:
            return None
    overlap_left_1, overlap_left_2 = _sample_overlap_pair(w1, w2)
    overlap_dims = {
        overlap_bands[0]: (h1, w1, overlap_left_1),
        overlap_bands[1]: (h2, w2, overlap_left_2),
    }

    overlap_colors = random.sample(GEN_BODY_COLORS, 2)
    overlap_ctrl_cols = {
        overlap_bands[0]: random.randint(0, w1 - 1),
        overlap_bands[1]: random.randint(0, w2 - 1),
    }

    for band_index, band in enumerate(bands[:-1]):
        if band_index in overlap_dims:
            pass_h, pass_w, pass_left = overlap_dims[band_index]
            pass_color = overlap_colors[overlap_bands.index(band_index)]
            stop_side = random.choice(["left", "right"])
        else:
            pass_h, pass_w = _rand_size(band_h)
            stop_side = random.choice(["left", "right"])
            pass_side = "right" if stop_side == "left" else "left"
            pass_slot_start, pass_slot_width = _slot_bounds(pass_side, width)
            pass_left = _slot_left(pass_slot_start, pass_slot_width, pass_w)
            pass_color = random.choice(GEN_BODY_COLORS)

        stop_h, stop_w = _rand_size(band_h)
        top_pass = _top_for_height(band, band_h, pass_h)
        top_stop = _top_for_height(band, band_h, stop_h)
        stop_slot_start, stop_slot_width = _slot_bounds(stop_side, width)
        stop_left = _slot_left(stop_slot_start, stop_slot_width, stop_w)

        stop_color_choices = [color for color in GEN_BODY_COLORS if color != pass_color]
        stop_color = random.choice(stop_color_choices)
        rects.append(
            {
                "band": band_index,
                "color": pass_color,
                "control": PASS,
                "top": top_pass,
                "left": pass_left,
                "height": pass_h,
                "width": pass_w,
                "area": pass_h * pass_w,
                "control_col": pass_left + (overlap_ctrl_cols[band_index] if band_index in overlap_ctrl_cols else random.randint(0, pass_w - 1)),
            }
        )
        rects.append(
            {
                "band": band_index,
                "color": stop_color,
                "control": STOP,
                "top": top_stop,
                "left": stop_left,
                "height": stop_h,
                "width": stop_w,
                "area": stop_h * stop_w,
                "control_col": stop_left + random.randint(0, stop_w - 1),
            }
        )

    bottom_band = bands[-1]
    bottom_count = random.randint(1, 2)
    controls = [STOP] if bottom_count == 1 else [STOP, PASS]
    if bottom_count == 1 and random.random() < 0.45:
        controls = [PASS]
    sides = ["left", "right"] if bottom_count == 2 else [random.choice(["left", "right"])]
    random.shuffle(sides)
    used_colors: list[int] = []
    for idx in range(bottom_count):
        rect_h, rect_w = _rand_size(band_h)
        slot_start, slot_width = _slot_bounds(sides[idx], width)
        rect_left = _slot_left(slot_start, slot_width, rect_w)
        rect_color_choices = [color for color in GEN_BODY_COLORS if color not in used_colors] or list(GEN_BODY_COLORS)
        rect_color = random.choice(rect_color_choices)
        used_colors.append(rect_color)
        rects.append(
            {
                "band": len(bands) - 1,
                "color": rect_color,
                "control": controls[idx],
                "top": _top_for_height(bottom_band, band_h, rect_h),
                "left": rect_left,
                "height": rect_h,
                "width": rect_w,
                "area": rect_h * rect_w,
                "control_col": rect_left + random.randint(0, rect_w - 1),
            }
        )

    return rects


def _place_input(shape: tuple[int, int], sep_rows: list[int], rects: list[dict]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    for row in sep_rows:
        out[int(row), :] = DIVIDER
    for rect in rects:
        top = int(rect["top"])
        left = int(rect["left"])
        rh = int(rect["height"])
        rw = int(rect["width"])
        out[top : top + rh, left : left + rw] = int(rect["color"])
        out[top - 1, int(rect["control_col"])] = int(rect["control"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_bands = random.choice([3, 4])
    band_h = random.choice([5, 6])
    height = n_bands * band_h + (n_bands - 1)
    width = random.randint(16, 18)

    sep_rows = [band_h + i * (band_h + 1) for i in range(n_bands - 1)]
    bands = _band_ranges(height, sep_rows)
    rects = _build_band_rects(bands, band_h, width)
    if rects is None:
        return None

    grid = _place_input((height, width), sep_rows, rects)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-divider ARC input")
