from __future__ import annotations

# Combo variant of 7c66cb00 (index 19)
# combo_spec: D5/0+D12/1+D15/0
# Name: 大网格紧密条排列紧凑裁剪输出

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the fused variant:
# a) Concrete behavior:
#    - Original: 30x24 grid, bars are separated by one blank row, solver can stop a bar
#      when it sees background, and the output keeps the whole canvas.
#    - Fused: 30x30 grid, bars are still full-width border+fill stripes, but they touch
#      vertically with no blank separator. A sprite still matches a bar by outer color and
#      docks bottom-aligned at its original x, but the final answer is cropped to the
#      minimal bbox of non-background pixels. Because the top sprites are erased first,
#      that bbox becomes exactly the stacked bar block.
#    - Example trace: if the top bar starts at row 11, the old task returns a 30-row
#      output with rows 0..10 blank; the fused task returns only rows 11..29.
# b) Complexity calibration:
#    - Bars still use only 2-3 fill colors and 2-3 border colors, so the color-matching
#      rule remains learnable. Tight stacking is visually obvious because the bar boundary
#      must be inferred from a color change rather than a blank spacer row.
#    - We keep 2-3 sprites and ensure at least one bicolor sprite, so the swap rule has
#      repeated support without exploding the number of cases.
# c) Broken original assumptions:
#    - Original bar finder assumed background rows delimit bars. That is invalid here, so
#      detection must instead recognize each row's exact "border, solid fill..., border"
#      signature and merge adjacent rows with the same (fill, border) pair.
#    - Original output-size assumption is invalid, so main() must crop after docking.
# d) Placement legality:
#    - Generator enforces sprite bbox separation in the top area, width bounds inside
#      cols 1..W-2, and per-bar horizontal occupancy so independently docked components do
#      not overlap within the same destination bar.
# e) Visual observability:
#    - Human-solvable cues are all visible: the bars are flush, the fill color still
#      identifies the destination, and the output loses the blank top margin.
# f) Visual cleanliness:
#    - Top sprites never overlap. Inside bars, objects may touch across a bar boundary
#      because there is no spacer row, but they remain locally explainable by docking.
# g) Color-change inference:
#    - Colors are limited and structured by region: outer sprite color identifies the
#      target bar; after docking that region becomes the bar border color, while any inner
#      color becomes the target fill. This is visible across multiple samples.


def _detect_bars(g: np.ndarray, bg: int) -> list[dict]:
    """Find tightly stacked full-width bars by row signature, not by blank gaps."""
    H, W = g.shape
    assert W >= 3

    row_info: list[tuple[int, int, int]] = []
    for r in range(H):
        row = g[r]
        fill = int(row[1])
        border = int(row[0])
        if fill == bg or border == bg or border == fill:
            continue
        if int(row[-1]) != border:
            continue
        if not np.all(row[1:-1] == fill):
            continue
        row_info.append((int(r), fill, border))

    assert row_info, "no bars found"

    bars: list[dict] = []
    start_r, cur_fill, cur_border = row_info[0]
    prev_r = start_r
    for r, fill, border in row_info[1:]:
        if int(r) == int(prev_r + 1) and int(fill) == int(cur_fill) and int(border) == int(cur_border):
            prev_r = int(r)
            continue
        bars.append(
            {
                "fill": int(cur_fill),
                "border": int(cur_border),
                "r_min": int(start_r),
                "r_max": int(prev_r),
                "h": int(prev_r - start_r + 1),
            }
        )
        start_r, cur_fill, cur_border, prev_r = int(r), int(fill), int(border), int(r)

    bars.append(
        {
            "fill": int(cur_fill),
            "border": int(cur_border),
            "r_min": int(start_r),
            "r_max": int(prev_r),
            "h": int(prev_r - start_r + 1),
        }
    )
    return bars


def _extract_patterns(g: np.ndarray, top_h: int, bg: int) -> list[dict]:
    """Recover both whole sprites and inner monochrome sub-components from the top area."""
    patterns: list[dict] = []
    top = g[:top_h, :]
    colors = sorted(set(map(int, top.flatten())) - {int(bg)})

    for col in colors:
        lab, n = label_mask(top == int(col), connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = map(int, bbox_coords(coords))
            bbox = crop_bbox(g, (r0, c0, r1, c1))
            color_counts = Counter(map(int, bbox.flatten()))
            color_counts.pop(int(bg), None)
            if not color_counts:
                continue
            outer_color = int(color_counts.most_common(1)[0][0])
            inner_candidates = [int(v) for v in color_counts if int(v) != int(outer_color)]
            patterns.append(
                {
                    "outer_color": int(outer_color),
                    "inner_color": inner_candidates[0] if inner_candidates else None,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )

    patterns.sort(key=lambda p: (p["c0"], p["r0"], p["h"], p["w"], p["outer_color"]))
    return patterns


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    return crop_bbox(grid, bbox_coords(nz))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock top sprites into tightly stacked color-matched bars, then crop away the blank margin.

    Input structure:
    - Background is the most common color in a 30x30 grid.
    - The lower portion contains 2-3 full-width horizontal bars whose rows have the pattern
      border / solid fill / border, and consecutive bars touch with no empty spacer row.
    - Above the first bar sit 2-3 small sprites made from bar fill colors, usually with an
      outer color and sometimes a smaller inner region.

    Rule:
    - Each sprite component targets the bar whose fill color matches that component's outer color.
    - Repaint the docked copy so the outer color becomes the target bar's border color and any
      inner color becomes the target bar's fill color, then place it bottom-aligned at the same x.
    - Erase the original sprites from the top area and return only the minimal non-background crop.

    Key insight:
    - Even without gap rows, each bar row is identifiable by its full-width border/fill signature;
      after docking, the remaining non-background area is exactly the bar block, so tight cropping
      is part of the transformation rather than a cosmetic post-process.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    bars = _detect_bars(g, bg)
    top_h = int(min(bar["r_min"] for bar in bars))
    patterns = _extract_patterns(g, top_h, bg)

    out[:top_h, :] = int(bg)

    for bar in bars:
        for pat in patterns:
            if int(pat["outer_color"]) != int(bar["fill"]):
                continue
            transformed = np.array(pat["content"], dtype=int).copy()
            transformed[transformed == int(pat["outer_color"])] = int(bar["border"])
            if pat["inner_color"] is not None:
                transformed[transformed == int(pat["inner_color"])] = int(bar["fill"])
            place_r = int(bar["r_max"] - pat["h"] + 1)
            paste(out, transformed, place_r, int(pat["c0"]), transparent=int(bg))

    return _crop_non_bg(out, bg)


GRID_H = 30
GRID_W = 30
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (5, 10)
MIN_TOP_AREA = 8
SPRITE_H_RANGE = (2, 6)
SPRITE_W_RANGE = (2, 10)
INNER_COLOR_PROB = 0.6
GEN_RETRIES = 400


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if GRID_H - sum(heights) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars = [None] * len(heights)
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0 = int(r - h)
        r1 = int(r - 1)
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = int(r0)
    return grid, bars


def _make_sprite(outer_idx: int, bars: list[dict], *, force_inner: bool) -> dict:
    outer_fill = int(bars[outer_idx]["fill"])
    max_h = min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"]))
    h = random.randint(SPRITE_H_RANGE[0], max_h)
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), int(outer_fill), dtype=int)

    inner_idx = None
    inner_bbox = None
    allow_inner = len(bars) >= 2 and h >= 3 and w >= 3
    use_inner = allow_inner and (force_inner or random.random() < INNER_COLOR_PROB)
    if use_inner:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        inner_idx = random.choice(inner_choices)
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, w - 1)
            if (w - iw) % 2 == 0 and ih * iw * 2 < h * w
        ]
        if valid_inner:
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[inner_idx]["fill"])
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))
        else:
            inner_idx = None

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": None if inner_idx is None else int(inner_idx),
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[int(start) : int(start + length)].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx = int(sp["outer_idx"])
        inner_idx = sp["inner_idx"]
        candidates: list[tuple[int, int]] = []

        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_idx], c0, w):
                continue
            if inner_idx is not None:
                _, ic0, _, iw = sp["inner_bbox"]
                if not _interval_free(bar_occ[int(inner_idx)], c0 + int(ic0), int(iw)):
                    continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((int(r0), int(c0)))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][c0 : c0 + w] = True
        if inner_idx is not None:
            _, ic0, _, iw = sp["inner_bbox"]
            bar_occ[int(inner_idx)][c0 + int(ic0) : c0 + int(ic0) + int(iw)] = True

        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render_expected(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[int(sp["outer_idx"])]
        h = int(sp["h"])
        c0 = int(sp["c0"])

        paste(inp, patch, int(sp["r0"]), c0, transparent=int(bg))

        transformed = patch.copy()
        transformed[transformed == int(outer_bar["fill"])] = int(outer_bar["border"])
        if sp["inner_idx"] is not None:
            inner_bar = bars[int(sp["inner_idx"])]
            transformed[transformed == int(inner_bar["fill"])] = int(outer_bar["fill"])
        paste(out, transformed, int(outer_bar["r_max"] - h + 1), c0, transparent=int(bg))

        if sp["inner_idx"] is not None:
            _, ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[int(sp["inner_idx"])]
            inner_patch = np.full((int(ih), int(iw)), int(inner_bar["border"]), dtype=int)
            paste(out, inner_patch, int(inner_bar["r_max"] - ih + 1), int(c0 + ic0))

    return inp, _crop_non_bg(out, int(bg))


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
    sprites = []
    for i in range(n_sprites):
        sprites.append(_make_sprite(int(outer_slots[i]), bars, force_inner=(i == 0)))
    if not any(sp["inner_idx"] is not None for sp in sprites):
        return None

    sprites.sort(key=lambda s: (s["inner_idx"] is None, -int(s["w"]), -int(s["h"])))
    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, expected = _render_expected(base, placed, bars, bg)
    actual = main(inp)

    if get_bg(inp) != int(bg):
        return None
    if inp.shape != (GRID_H, GRID_W):
        return None
    if expected.shape[0] >= GRID_H:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
