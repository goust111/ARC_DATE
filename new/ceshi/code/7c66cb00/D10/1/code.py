# Variant of 7c66cb00
# Dimension: D10 — 空间布局：精灵在顶部区域的排列 (meta)
# Change: 精灵在顶部区域自由放置，各精灵互不接触（有背景色间隔），保持在条区域之上... → 精灵放置在条的左侧或右侧
#
# Design analysis
# ---------------
# spec:
#   concrete_behavior:
#     - Bars are still horizontal fill+border rectangles stacked near the bottom, but
#       they are inset horizontally so the grid has a 4-column lane on the left and
#       another 4-column lane on the right.
#     - Every sprite lives entirely in one side lane, not above the bars. Its rows lie
#       inside the height span of its outer-color bar, so the visual relation is now
#       lateral instead of vertical.
#     - Output keeps the original color rule: outer_color matches a bar fill, outer
#       recolors to that bar's border, inner recolors to that bar's fill. The sprite is
#       erased from the side lane and docked inside the matched bar.
#     - The new spatial rule is side-aware: a sprite from the left lane docks against
#       the left edge of its target bar; a sprite from the right lane docks against the
#       right edge. Vertical placement remains bottom-aligned so inner monochrome
#       sub-components can still dock into their own bars without depending on the
#       source bar's row span.
#   traced_example:
#     - Original logic: a sprite at top columns 3..5 with outer fill 2 is moved to the
#       bar whose fill is 2, keeping x=3 and bottom-aligning.
#     - Variant logic: a sprite in the left lane beside some bar rows, still with outer
#       fill 2, is moved to the left edge of the bar whose fill is 2 and bottom-aligned.
#     - Difference: the solver must no longer treat "everything above first bar" as the
#       sprite area, and x-position is replaced by a left/right side choice.
#   complexity_calibration:
#     - Bars keep unique fill colors and unique border colors.
#     - Side is binary (left/right), which is small enough to infer from 2-3 examples but
#       still visually obvious.
#     - Lane width is fixed at 4 columns, so sprites stay compact (2-4 columns wide) and
#       do not visually merge with bars.
#     - Slot usage is controlled per (bar, side). This prevents ambiguous output
#       overlaps when multiple detected components target the same bar edge.
#   implicit_assumption_audit:
#     - Old `pattern_area_end = min(bar.r_min)` assumed all sprites are above the first
#       bar. This is false now because sprites share rows with bars.
#     - Old `top = g[:pattern_area_end, :]` would miss every sprite in the variant.
#     - Old clearing `out[:pattern_area_end, :] = bg` would erase nothing useful.
#     - Old `place_c = pat["c0"]` depended on the input x-position being meaningful for
#       output docking. In the variant the meaningful spatial datum is side, not x.
#     - Bar detection also can no longer rely on the bar touching column 0 / W-1, so the
#       solver now detects wide non-bg runs and records their own left/right borders.
#   placement_legality:
#     - Bars occupy fixed middle columns 5..18, leaving column 4 and 19 as bg separators.
#     - Sprites stay fully inside cols 0..3 or 20..23 and fully inside the row span of
#       their outer bar, so they never touch a bar in the input.
#     - Output docking is safe because at most one detected component may target each
#       `(bar, side)` slot.
#   visual_observability:
#     - The top area is empty; all action sits in the side lanes aligned with the bars.
#     - Examples can contain both left and right placements, so the side rule is visible.
#     - Because bars all share the same horizontal extent, "left of the bars" and "right
#       of the bars" is stable across all examples and easy to read.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


GRID_H = 30
GRID_W = 24

BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_MARGIN = 2

SIDE_LANE_W = 4
SIDE_GAP = 1
BAR_C0 = SIDE_LANE_W + SIDE_GAP
BAR_C1 = GRID_W - SIDE_LANE_W - SIDE_GAP - 1
MIN_BAR_RUN = 10

SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, SIDE_LANE_W)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 400


def _row_runs(row: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    W = int(row.shape[0])
    while c < W:
        if int(row[c]) == int(bg):
            c += 1
            continue
        c0 = c
        while c < W and int(row[c]) != int(bg):
            c += 1
        runs.append((int(c0), int(c - 1)))
    return runs


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    bars: list[dict] = []
    current: dict | None = None
    min_run = max(int(MIN_BAR_RUN), int(W // 2))

    for r in range(H):
        wide_runs = [(c0, c1) for c0, c1 in _row_runs(g[r], bg) if c1 - c0 + 1 >= min_run]
        if len(wide_runs) != 1:
            if current is not None:
                bars.append(current)
                current = None
            continue

        c0, c1 = wide_runs[0]
        seg = g[r, c0:c1 + 1]
        colors = Counter(int(v) for v in seg if int(v) != int(bg))
        if not colors:
            if current is not None:
                bars.append(current)
                current = None
            continue

        fill, _ = colors.most_common(1)[0]
        border = int(seg[0])
        if border == int(fill):
            border = int(seg[-1])
        if border == int(fill):
            others = [int(col) for col in colors if int(col) != int(fill)]
            if not others:
                if current is not None:
                    bars.append(current)
                    current = None
                continue
            border = others[0]

        signature = (int(c0), int(c1), int(fill), int(border))
        if current is not None and current["signature"] == signature and int(current["r_max"]) + 1 == int(r):
            current["r_max"] = int(r)
        else:
            if current is not None:
                bars.append(current)
            current = {
                "fill": int(fill),
                "border": int(border),
                "c_min": int(c0),
                "c_max": int(c1),
                "r_min": int(r),
                "r_max": int(r),
                "signature": signature,
            }

    if current is not None:
        bars.append(current)

    for bar in bars:
        bar["h"] = int(bar["r_max"] - bar["r_min"] + 1)

    assert bars, "No bars found"
    return bars


def _extract_patterns(g: np.ndarray, bars: list[dict], bg: int) -> tuple[list[dict], np.ndarray]:
    sprite_grid = np.array(g, dtype=int).copy()
    for bar in bars:
        sprite_grid[int(bar["r_min"]): int(bar["r_max"]) + 1, int(bar["c_min"]): int(bar["c_max"]) + 1] = int(bg)

    left_edge = min(int(bar["c_min"]) for bar in bars)
    right_edge = max(int(bar["c_max"]) for bar in bars)
    patterns: list[dict] = []

    for col in sorted(set(map(int, sprite_grid.flatten())) - {int(bg)}):
        mask = sprite_grid == int(col)
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = map(int, bbox_coords(coords))
            side = None
            if c1 < left_edge:
                side = "left"
            elif c0 > right_edge:
                side = "right"
            if side is None:
                continue

            bbox = crop_bbox(sprite_grid, (r0, c0, r1, c1))
            colors_in_bbox = Counter(int(v) for v in bbox.flatten())
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue

            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            inner_candidates = [int(color) for color in colors_in_bbox if int(color) != int(outer_color)]
            inner_color = inner_candidates[0] if inner_candidates else None

            patterns.append(
                {
                    "outer_color": int(outer_color),
                    "inner_color": inner_color,
                    "side": side,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )

    return patterns, sprite_grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Detect inset horizontal bars in the middle columns.
    - Detect side-lane sprites by removing the bars and scanning the remaining non-bg
      components by color.
    - Match each detected pattern to the bar whose fill matches the pattern's outer color.
    - Recolor outer -> bar border, inner -> bar fill.
    - Dock left-lane patterns to the left edge of the target bar, right-lane patterns to
      the right edge, always bottom-aligned.
    - Erase the original side-lane sprites.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    patterns, sprite_grid = _extract_patterns(g, bars, bg)
    out[sprite_grid != int(bg)] = int(bg)

    bar_by_fill = {int(bar["fill"]): bar for bar in bars}
    for pat in patterns:
        bar = bar_by_fill.get(int(pat["outer_color"]))
        if bar is None:
            continue

        transformed = np.array(pat["content"], dtype=int).copy()
        transformed[transformed == int(pat["outer_color"])] = int(bar["border"])
        if pat["inner_color"] is not None:
            transformed[transformed == int(pat["inner_color"])] = int(bar["fill"])

        place_r = int(bar["r_max"] - pat["h"] + 1)
        if pat["side"] == "left":
            place_c = int(bar["c_min"])
        else:
            place_c = int(bar["c_max"] - pat["w"] + 1)
        paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_MARGIN:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars = [None] * len(heights)
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
        grid[r0:r1 + 1, BAR_C0 + 1:BAR_C1] = int(fills[i])
        grid[r0:r1 + 1, BAR_C0] = int(borders[i])
        grid[r0:r1 + 1, BAR_C1] = int(borders[i])
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
            "c_min": int(BAR_C0),
            "c_max": int(BAR_C1),
        }
        r = r0 - 1
    return grid, bars


def _make_sprite(outer_idx: int, side: str, bars: list[dict], used_slots: set[tuple[int, str]]) -> dict:
    outer_fill = int(bars[outer_idx]["fill"])
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"])))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    available_inner = [
        i
        for i in range(len(bars))
        if i != int(outer_idx) and (int(i), str(side)) not in used_slots
    ]
    if available_inner and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, w - 1)
            if ih * iw * 2 < h * w
        ]
        if valid_inner:
            inner_idx = random.choice(available_inner)
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = random.randint(1, w - iw - 1)
            patch[ir0:ir0 + ih, ic0:ic0 + iw] = int(bars[inner_idx]["fill"])
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": inner_idx,
        "side": str(side),
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _choose_sprites(n_sprites: int, bars: list[dict]) -> list[dict] | None:
    all_slots = [(i, side) for i in range(len(bars)) for side in ("left", "right")]
    random.shuffle(all_slots)
    used_slots: set[tuple[int, str]] = set()
    sprites: list[dict] = []

    for _ in range(n_sprites):
        candidates = [slot for slot in all_slots if slot not in used_slots]
        if not candidates:
            return None
        outer_idx, side = random.choice(candidates)
        sprite = _make_sprite(int(outer_idx), str(side), bars, used_slots)
        used_slots.add((int(outer_idx), str(side)))
        if sprite["inner_idx"] is not None:
            used_slots.add((int(sprite["inner_idx"]), str(side)))
        sprites.append(sprite)

    return sprites


def _place_sprites(sprites: list[dict], bars: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    for sp in sprites:
        bar = bars[int(sp["outer_idx"])]
        max_r0 = int(bar["r_max"] - sp["h"] + 1)
        min_r0 = int(bar["r_min"])
        if max_r0 < min_r0:
            return None

        r0 = random.randint(min_r0, max_r0)
        if sp["side"] == "left":
            c0 = random.randint(0, SIDE_LANE_W - sp["w"])
        else:
            c0 = random.randint(GRID_W - SIDE_LANE_W, GRID_W - sp["w"])

        placed.append({**sp, "r0": int(r0), "c0": int(c0)})

    return placed


def _dock_position(bar: dict, side: str, h: int, w: int) -> tuple[int, int]:
    r0 = int(bar["r_max"] - h + 1)
    if side == "left":
        c0 = int(bar["c_min"])
    else:
        c0 = int(bar["c_max"] - w + 1)
    return r0, c0


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()

    for sp in placed:
        patch = sp["patch"].copy()
        outer_bar = bars[int(sp["outer_idx"])]
        side = str(sp["side"])

        paste(inp, patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

        transformed = patch.copy()
        transformed[transformed == int(outer_bar["fill"])] = int(outer_bar["border"])
        if sp["inner_idx"] is not None:
            inner_bar = bars[int(sp["inner_idx"])]
            transformed[transformed == int(inner_bar["fill"])] = int(outer_bar["fill"])

        out_r, out_c = _dock_position(outer_bar, side, int(sp["h"]), int(sp["w"]))
        paste(out, transformed, out_r, out_c, transparent=int(bg))

        if sp["inner_idx"] is not None:
            _, _, ih, iw = sp["inner_bbox"]
            inner_bar = bars[int(sp["inner_idx"])]
            inner_patch = np.full((ih, iw), int(inner_bar["border"]), dtype=int)
            inner_r, inner_c = _dock_position(inner_bar, side, int(ih), int(iw))
            paste(out, inner_patch, inner_r, inner_c)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1:1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars:1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    sprites = _choose_sprites(n_sprites, bars)
    if sprites is None:
        return None

    placed = _place_sprites(sprites, bars)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)
    if get_bg(inp) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
