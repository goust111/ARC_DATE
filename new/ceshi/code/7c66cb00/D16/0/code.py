# Variant of 7c66cb00
# Dimension: D16 — 精灵的x位置与条的列约束 (meta)
# Change: 精灵在条中保持输入中的原始x位置，精灵需在cols 1..W-2范围内（避开边框列）... → 精灵水平滑动到条的边框旁

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

# Design analysis
# variant_analysis:
#   source_task: none
#   concretization:
#     - The bottom bars stay unchanged: full-width horizontal bands with a fill color in cols 1..W-2 and a border color at cols 0 and W-1.
#     - The top area still contains 2-3 small sprites made from bar-fill colors. The solver still detects them the same way as the original:
#       it scans each color separately, labels connected components, takes each component bbox, and treats the bbox content as one pattern.
#       This means a two-tone sprite still yields both its outer bbox-pattern and, when present, an inner monochrome sub-pattern.
#     - D16 changes only the horizontal docking rule. Each detected pattern decides a side from its visible bbox center:
#       center < grid midpoint => dock to the left edge of the matched bar; otherwise dock to the right edge.
#       Within the same bar and same side, patterns are ordered by their original c0 from left to right, then packed contiguously.
#   behavior_trace:
#     - Original rule: a sprite with bbox c0=8 that matches the green bar docks bottom-aligned at c0=8.
#     - Variant rule: the same sprite still matches the green bar and still bottom-aligns, but if its bbox center is in the left half,
#       it now docks starting at col 1. A second green-matching sprite from the same side docks immediately to its right.
#       If a matching pattern comes from the right half, the whole ordered block for that side is flushed against col W-2 instead.
#     - The visible difference is therefore not recoloring or matching, but horizontal sliding and edge packing.
#   complexity_calibration:
#     - The side choice is a binary, visible attribute derived only from geometry, not from hidden color ids.
#     - Ordering stays low-complexity: only original left-to-right order matters inside each side group.
#     - The generator keeps sprites wholly inside the left or right half of the top area, so the side signal is visually unambiguous.
#   implicit_assumptions_audited:
#     - In the original solver, `place_c = pat["c0"]` was safe because the generator reserved original x-ranges inside each bar.
#       Under D16 that assumption fails; placement must be recomputed from grouped widths.
#     - The original generator reserved bar occupancy using original x positions. That is invalid now because all matched patterns slide.
#       The new generator reserves total widths per `(bar, side)` instead.
#     - The original generator only guaranteed `sprite height <= outer bar height`. Because inner monochrome components dock independently,
#       D16 generation must also guarantee `inner height <= inner target bar height`.
#   placement_legality:
#     - Top sprites are placed with a 1-cell gap from each other and are kept entirely inside a left zone or a right zone.
#     - For every bar, the generator tracks the total packed width contributed on the left and right. It rejects samples that would make
#       the two side blocks overlap after docking.
#     - Output placement uses the same parsing logic as `main()`, so generated samples are solver-consistent by construction.
#   visual_observability:
#     - The generator biases left-side sprites away from col 1 and right-side sprites away from col W-2, so the horizontal slide is visible.
#     - Both edge choices can appear across generated samples, but each sprite itself stays clearly on one half in the input.
#   visual_cleanliness:
#     - Bars remain separated by one background row.
#     - Packed patterns never overlap inside a bar, and when both sides are used in one bar the generator keeps at least one fill cell gap.
#   color_reasoning:
#     - Colors keep the original meaning: outer color identifies the target bar, then swaps to the bar border color; inner color swaps to
#       the bar fill. No new color remapping is introduced, so D16 stays the only conceptual change.


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 8)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 400

LEFT_INPUT_MIN = 2
LEFT_INPUT_MAX_END = GRID_W // 2 - 2
RIGHT_INPUT_MIN = GRID_W // 2 + 1
RIGHT_INPUT_MAX_END = GRID_W - 3
INTERIOR_W = GRID_W - 2
MIN_MOVED_PATTERNS = 1
MIN_TOTAL_SHIFT = 2


def _pattern_side(c0: int, w: int, width: int) -> str:
    center = int(c0) + (int(w) - 1) / 2.0
    return "left" if center < (int(width) - 1) / 2.0 else "right"


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    current_bar: dict | None = None
    H, W = g.shape
    for r in range(H):
        row = g[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if not non_bg:
            if current_bar is not None:
                bars.append(current_bar)
                current_bar = None
            continue

        color_counts = Counter(non_bg)
        fill_color, count = color_counts.most_common(1)[0]
        if int(count) <= int(W * 0.7):
            if current_bar is not None:
                bars.append(current_bar)
                current_bar = None
            continue

        border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
        if current_bar is not None and int(current_bar["fill"]) == int(fill_color):
            current_bar["r_max"] = int(r)
        else:
            if current_bar is not None:
                bars.append(current_bar)
            current_bar = {
                "fill": int(fill_color),
                "border": int(border),
                "r_min": int(r),
                "r_max": int(r),
            }

    if current_bar is not None:
        bars.append(current_bar)
    return bars


def _extract_patterns(g: np.ndarray, bars: list[dict], bg: int) -> tuple[list[dict], int]:
    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    top = g[:pattern_area_end, :]
    patterns: list[dict] = []
    for col in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(col)
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = map(int, bbox_coords(coords))
            bbox = crop_bbox(g, (r0, c0, r1, c1))
            colors_in_bbox = Counter(map(int, bbox.flatten()))
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue

            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            inner_colors = [int(c) for c in colors_in_bbox if int(c) != int(outer_color)]
            inner_color = int(inner_colors[0]) if inner_colors else None
            patterns.append(
                {
                    "outer_color": int(outer_color),
                    "inner_color": inner_color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                    "side": _pattern_side(int(c0), int(bbox.shape[1]), g.shape[1]),
                }
            )
    return patterns, pattern_area_end


def _transform_pattern(pattern: dict, bar: dict) -> np.ndarray:
    transformed = np.array(pattern["content"], dtype=int).copy()
    transformed[transformed == int(pattern["outer_color"])] = int(bar["border"])
    if pattern["inner_color"] is not None:
        transformed[transformed == int(pattern["inner_color"])] = int(bar["fill"])
    return transformed


def _plan_pattern_placements(patterns: list[dict], bars: list[dict], width: int) -> list[dict]:
    placements: list[dict] = []
    for bar in bars:
        matching = [p for p in patterns if int(p["outer_color"]) == int(bar["fill"])]
        left = sorted(
            [p for p in matching if p["side"] == "left"],
            key=lambda p: (int(p["c0"]), int(p["r0"]), int(p["w"]), int(p["h"])),
        )
        right = sorted(
            [p for p in matching if p["side"] == "right"],
            key=lambda p: (int(p["c0"]), int(p["r0"]), int(p["w"]), int(p["h"])),
        )
        left_total = sum(int(p["w"]) for p in left)
        right_total = sum(int(p["w"]) for p in right)
        right_cursor = int(width - 1 - right_total)
        left_cursor = 1
        if left_total > width - 2 or right_total > width - 2 or left_cursor + left_total > right_cursor:
            raise ValueError("packed patterns do not fit inside target bar")

        for pattern in left:
            placements.append(
                {
                    "pattern": pattern,
                    "bar": bar,
                    "place_r": int(bar["r_max"] - pattern["h"] + 1),
                    "place_c": int(left_cursor),
                }
            )
            left_cursor += int(pattern["w"])

        for pattern in right:
            placements.append(
                {
                    "pattern": pattern,
                    "bar": bar,
                    "place_r": int(bar["r_max"] - pattern["h"] + 1),
                    "place_c": int(right_cursor),
                }
            )
            right_cursor += int(pattern["w"])

    return placements


def _solve_core(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    bars = _find_bars(g, bg)
    assert bars, "No bars found — grid must contain at least one horizontal bar"

    patterns, pattern_area_end = _extract_patterns(g, bars, bg)
    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for placement in _plan_pattern_placements(patterns, bars, g.shape[1]):
        transformed = _transform_pattern(placement["pattern"], placement["bar"])
        paste(out, transformed, placement["place_r"], placement["place_c"], transparent=int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid)


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {"fill": int(fills[i]), "border": int(borders[i]), "h": h, "r_min": r0, "r_max": r1}
        r = r0 - 1
    return grid, bars


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    outer_fill = int(bars[outer_idx]["fill"])
    max_h = min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"]))
    h = random.randint(SPRITE_H_RANGE[0], max_h)
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        random.shuffle(inner_choices)
        for candidate_idx in inner_choices:
            valid_inner = [
                (ih, iw)
                for ih in range(1, h - 1)
                for iw in range(1, w - 1)
                if ih <= int(bars[candidate_idx]["h"]) and (w - iw) % 2 == 0 and ih * iw * 2 < h * w
            ]
            if not valid_inner:
                continue
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[candidate_idx]["fill"])
            inner_idx = candidate_idx
            inner_bbox = (ir0, ic0, ih, iw)
            break

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": inner_idx,
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _empty_reservations(n_bars: int) -> list[dict]:
    return [{"left": 0, "right": 0} for _ in range(n_bars)]


def _sprite_contributions(sprite: dict) -> list[tuple[int, str, int]]:
    parts = [(int(sprite["outer_idx"]), str(sprite["side"]), int(sprite["w"]))]
    if sprite["inner_idx"] is not None:
        _, _ic0, _ih, iw = sprite["inner_bbox"]
        parts.append((int(sprite["inner_idx"]), str(sprite["side"]), int(iw)))
    return parts


def _reservations_allow(reserved: list[dict], sprite: dict) -> bool:
    trial = [{"left": int(item["left"]), "right": int(item["right"])} for item in reserved]
    for bar_idx, side, width in _sprite_contributions(sprite):
        trial[bar_idx][side] += int(width)
    for item in trial:
        left_w = int(item["left"])
        right_w = int(item["right"])
        if left_w > INTERIOR_W or right_w > INTERIOR_W:
            return False
        if left_w + right_w > INTERIOR_W:
            return False
        if left_w > 0 and right_w > 0 and left_w + right_w > INTERIOR_W - 1:
            return False
    return True


def _apply_reservations(reserved: list[dict], sprite: dict) -> None:
    for bar_idx, side, width in _sprite_contributions(sprite):
        reserved[bar_idx][side] += int(width)


def _candidate_cols(side: str, w: int) -> list[int]:
    if side == "left":
        lo = LEFT_INPUT_MIN
        hi = LEFT_INPUT_MAX_END - int(w) + 1
        cols = list(range(lo, hi + 1))
        cols.sort(reverse=True)
        return cols
    lo = RIGHT_INPUT_MIN
    hi = RIGHT_INPUT_MAX_END - int(w) + 1
    cols = list(range(lo, hi + 1))
    cols.sort()
    return cols


def _place_sprites(sprites: list[dict], top_h: int) -> list[dict] | None:
    top_rects: list[tuple[int, int, int, int]] = []
    used_c0s: set[int] = set()
    placed: list[dict] = []

    for sprite in sorted(sprites, key=lambda s: (-int(s["h"]) * int(s["w"]), int(s["outer_idx"]))):
        h, w = int(sprite["h"]), int(sprite["w"])
        cols = _candidate_cols(str(sprite["side"]), w)
        if not cols:
            return None

        candidates: list[tuple[int, int]] = []
        rows = list(range(0, top_h - h + 1))
        random.shuffle(rows)
        for c0 in cols:
            if c0 in used_c0s:
                continue
            for r0 in rows:
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))
            if candidates:
                break

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        placed.append({**sprite, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
        used_c0s.add(int(c0))

    return placed


def _render_input(base: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    inp = base.copy()
    for sprite in placed:
        paste(inp, sprite["patch"], int(sprite["r0"]), int(sprite["c0"]), transparent=int(bg))
    return inp


def _movement_metrics(inp: np.ndarray) -> dict:
    bg = int(get_bg(inp))
    bars = _find_bars(inp, bg)
    patterns, _ = _extract_patterns(inp, bars, bg)
    placements = _plan_pattern_placements(patterns, bars, inp.shape[1])
    shifts = [abs(int(item["place_c"]) - int(item["pattern"]["c0"])) for item in placements]
    return {
        "pattern_count": len(placements),
        "moved_count": sum(1 for shift in shifts if shift > 0),
        "total_shift": sum(shifts),
        "left_count": sum(1 for item in placements if item["pattern"]["side"] == "left"),
        "right_count": sum(1 for item in placements if item["pattern"]["side"] == "right"),
    }


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)
    reserved = _empty_reservations(n_bars)
    sprites: list[dict] = []

    for i in range(n_sprites):
        outer_idx = int(outer_slots[i])
        built = None
        for _ in range(80):
            sprite = _make_sprite(outer_idx, bars)
            side = random.choice(["left", "right"])
            candidate = {**sprite, "side": side}
            if _reservations_allow(reserved, candidate):
                _apply_reservations(reserved, candidate)
                built = candidate
                break
        if built is None:
            return None
        sprites.append(built)

    placed = _place_sprites(sprites, top_h)
    if placed is None:
        return None

    inp = _render_input(base, placed, bg)
    metrics = _movement_metrics(inp)
    if metrics["moved_count"] < MIN_MOVED_PATTERNS or metrics["total_shift"] < MIN_TOTAL_SHIFT:
        return None
    if metrics["left_count"] == 0 or metrics["right_count"] == 0:
        return None
    if get_bg(inp) != bg:
        return None

    out = _solve_core(inp)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
