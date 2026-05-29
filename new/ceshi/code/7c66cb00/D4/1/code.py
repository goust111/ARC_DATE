# Variant of 7c66cb00
# Dimension: D4 — 目标对象：水平条结构 (dst)
# Change: 全宽水平条，底部堆叠2-3个，每个条4-8行高，由填充色（内部）和边框色（首尾两列）组成，条间间隔1行背景... → 非全宽条（部分宽度）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

"""
Design analysis
===============

a) Concrete variant + behavior trace
YAML-style spec for this D4 variant:
  grid: 30x24, bg = most common color
  bars:
    count: 2-3 bars total
    width: 10-15 for single bars, 10-11 when two bars share one row band
    height: 4-8
    structure:
      - each bar is a solid rectangle component
      - leftmost and rightmost columns are border_color
      - interior columns are fill_color
    layout:
      - bottom area consists of 1-2 row bands
      - consecutive bands have exactly 1 bg row gap
      - at least one band contains 2 bars placed side-by-side with a 1-col bg gap
      - every bar leaves bg visible on both outer sides of the canvas, so no bar is full-width
  sprites:
    count: 2-3
    location: above the topmost bar band
    colors: same as original, outer color is some bar fill, optional inner region is another bar fill
    x-rule:
      - a sprite keeps its original global x when docked
      - therefore the full sprite width must fit inside its target bar's horizontal span
      - if an inner-color subcomponent exists, that subcomponent keeps its own x too and must fit inside its own target bar span
  transform:
    - erase the top sprites
    - match by fill color as before
    - dock bottom-aligned into the matched bar at the same x
    - recolor outer -> target border, inner -> target fill
    - inner-color connected components still dock independently, exactly like the original solver behavior

Concrete trace:
  Original task:
    - a bar spans almost the whole row, so `place_c = sprite.c0` is always safe
    - bar detection can rely on "this row is mostly one fill color"
  Variant:
    - suppose a sprite of width 5 starts at x=14
    - its matching bar might be cols 12..21; docking at x=14 is legal
    - another bar on the same rows might be cols 1..10; matching by x would be illegal even if the color matched
    - therefore the solver must identify each bar's bbox and respect its c_min/c_max during placement

b) Complexity calibration
  The change is visually obvious because at least one band has two separate bars in one row range.
  Matching is still color-based, so the core concept remains learnable.
  Widths stay in a narrow range and fills remain unique, preventing ambiguity.

c) Hidden assumptions in the original main() that break here
  - `count > W * 0.7`: assumes each bar row covers most columns; false for width 10-15.
  - `border = row[0] ... row[-1]`: assumes borders touch canvas edges; false for interior bars.
  - extending a bar only by repeated fill color rows: fails when two different bars share the same rows.
  - unconditional `place_c = pat["c0"]` was harmless before because bars were full-width; now it needs an explicit fit assumption.
  The fix is to detect full rectangular non-bg components and read border/fill from each component bbox.

d) Placement legality
  Generator guarantees:
  - every sprite fits horizontally inside its outer target bar
  - every inner subcomponent interval fits inside its own target bar
  - sprites in the top area are separated by at least 1 bg cell
  - output dockings for the same bar do not overlap horizontally
  - all pastes are in-bounds

e) Visual observability
  A human sees that bars no longer span the canvas and that two bars can occupy the same rows.
  Because sprites keep x, the correct destination is constrained by both color and bar span.
  This is visible in 2-3 samples without relying on color ids as numbers.

f) Visual cleanliness
  The lower area uses 1-2 clean bands instead of arbitrary packed rectangles.
  Bars remain axis-aligned rectangles with single-column side borders, so they are easy to parse.
  Top sprites stay above the bars and do not touch each other.

g) Color-change inferability
  The recolor rule is unchanged from the original task.
  Colors are still tied structurally to bar fill/border roles, so the mapping remains inferable.

New solver assumptions to audit against the generator
  - Bars are the only non-bg rectangular components of width >= 10. Generator keeps sprites width <= 9.
  - Each bar bbox is fully filled with exactly 2 colors in border/fill arrangement. Generator enforces this exactly.
  - Every detected pattern's docking x interval fits its matched bar. Generator enforces this during placement.
"""


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 400

PAIR_BAR_WIDTHS = ((10, 10), (10, 11), (11, 10))
SINGLE_BAR_WIDTH_RANGE = (10, 15)
BAND_GAP = 1
PAIR_GAP = 1


def _extract_bars(g: np.ndarray, bg: int) -> list[dict]:
    """Detect partial-width bars as rectangular non-bg components."""
    lab, n = label_mask(g != int(bg), connectivity=4)
    bars: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        patch = crop_bbox(g, (r0, c0, r1, c1))
        h, w = map(int, patch.shape)
        if h < BAR_HEIGHT_RANGE[0] or w < 10:
            continue
        if np.any(patch == int(bg)) or w < 3:
            continue
        colors = sorted(set(map(int, patch.flatten())))
        if len(colors) != 2:
            continue
        if len(set(map(int, patch[:, 0]))) != 1 or len(set(map(int, patch[:, -1]))) != 1:
            continue
        border = int(patch[0, 0])
        if not np.all(patch[:, 0] == int(border)) or not np.all(patch[:, -1] == int(border)):
            continue
        inner = patch[:, 1:-1]
        if inner.size == 0 or len(set(map(int, inner.flatten()))) != 1:
            continue
        fill = int(inner[0, 0])
        if fill == border or not np.all(inner == int(fill)):
            continue
        bars.append(
            {
                "fill": fill,
                "border": border,
                "r_min": r0,
                "r_max": r1,
                "c_min": c0,
                "c_max": c1,
                "h": h,
                "w": w,
            }
        )
    bars.sort(key=lambda b: (int(b["r_min"]), int(b["c_min"])))
    assert bars, "No partial-width bars found"
    return bars


def _extract_patterns(g: np.ndarray, bg: int, pattern_area_end: int) -> list[dict]:
    """Mirror the original per-color component logic so inner regions dock independently."""
    patterns = []
    top = g[:pattern_area_end, :]
    for col in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(col)
        if not bool(mask.any()):
            continue
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
            patterns.append(
                {
                    "outer_color": outer_color,
                    "inner_color": int(inner_colors[0]) if inner_colors else None,
                    "r0": r0,
                    "c0": c0,
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )
    return patterns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched partial-width bars with the original color swap.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _extract_bars(g, bg)
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}
    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    patterns = _extract_patterns(g, bg, pattern_area_end)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for pat in patterns:
        bar = fill_to_bar.get(int(pat["outer_color"]))
        if bar is None:
            continue
        place_r = int(bar["r_max"] - pat["h"] + 1)
        place_c = int(pat["c0"])
        assert int(bar["c_min"]) <= place_c <= int(bar["c_max"])
        assert place_c + int(pat["w"]) - 1 <= int(bar["c_max"])

        src = np.array(pat["content"], dtype=int)
        transformed = src.copy()
        outer_mask = src == int(pat["outer_color"])
        transformed[outer_mask] = int(bar["border"])
        if pat["inner_color"] is not None:
            inner_mask = src == int(pat["inner_color"])
            transformed[inner_mask] = int(bar["fill"])
        paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


def _sample_band_sizes(n_bars: int) -> list[int]:
    if n_bars == 2:
        return [2]
    return random.choice(([2, 1], [1, 2]))


def _sample_band_heights(n_bands: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bands)]
        if GRID_H - sum(heights) - (n_bands - 1) * BAND_GAP >= MIN_TOP_AREA:
            return heights


def _bars_overlap(a: dict, b: dict) -> bool:
    return max(int(a["c_min"]), int(b["c_min"])) <= min(int(a["c_max"]), int(b["c_max"]))


def _build_bars(bg: int, fills: list[int], borders: list[int], band_sizes: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Render partial-width bars inside 1-2 stacked row bands."""
    band_heights = _sample_band_heights(len(band_sizes))
    band_rows: list[tuple[int, int]] = [(-1, -1)] * len(band_sizes)

    r = GRID_H
    for band_idx in range(len(band_sizes) - 1, -1, -1):
        h = int(band_heights[band_idx])
        r0, r1 = int(r - h), int(r - 1)
        band_rows[band_idx] = (r0, r1)
        r = r0 - BAND_GAP

    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = []
    color_idx = 0

    for band_idx, band_size in enumerate(band_sizes):
        r0, r1 = band_rows[band_idx]
        h = r1 - r0 + 1

        if band_size == 1:
            w = random.randint(*SINGLE_BAR_WIDTH_RANGE)
            c0 = random.randint(1, GRID_W - w - 1)
            specs = [(c0, w)]
        else:
            w1, w2 = random.choice(PAIR_BAR_WIDTHS)
            used = w1 + w2 + PAIR_GAP
            remaining = GRID_W - used
            left_margin = random.randint(1, remaining - 1)
            specs = [(left_margin, w1), (left_margin + w1 + PAIR_GAP, w2)]

        for c0, w in specs:
            c1 = c0 + w - 1
            fill = int(fills[color_idx])
            border = int(borders[color_idx])
            grid[r0 : r1 + 1, c0] = border
            grid[r0 : r1 + 1, c1] = border
            grid[r0 : r1 + 1, c0 + 1 : c1] = fill
            bars.append(
                {
                    "fill": fill,
                    "border": border,
                    "r_min": r0,
                    "r_max": r1,
                    "c_min": c0,
                    "c_max": c1,
                    "h": h,
                    "w": w,
                }
            )
            color_idx += 1

    if len(bars) == 3:
        top_band = [b for b in bars if int(b["r_min"]) == min(int(x["r_min"]) for x in bars)]
        bottom_band = [b for b in bars if int(b["r_min"]) == max(int(x["r_min"]) for x in bars)]
        assert len(top_band) in (1, 2) and len(bottom_band) in (1, 2)

    return grid, bars


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    """Create a sprite patch using bar fill colors, preserving the original 1-2 color structure."""
    outer_bar = bars[outer_idx]
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(outer_bar["h"])))
    w = random.randint(SPRITE_W_RANGE[0], min(SPRITE_W_RANGE[1], int(outer_bar["w"])))
    patch = np.full((h, w), int(outer_bar["fill"]), dtype=int)

    inner_idx = None
    inner_bbox = None
    overlap_choices = [i for i, bar in enumerate(bars) if i != outer_idx and _bars_overlap(outer_bar, bar)]
    if overlap_choices and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_idx = random.choice(overlap_choices)
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
            inner_bbox = (ir0, ic0, ih, iw)
        else:
            inner_idx = None

    return {
        "outer_idx": outer_idx,
        "inner_idx": inner_idx,
        "patch": patch,
        "h": h,
        "w": w,
        "inner_bbox": inner_bbox,
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[int(start) : int(start) + int(length)].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    """Find sprite placements that satisfy both the outer-bar and inner-bar x-span constraints."""
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed = []

    for sp in sprites:
        outer_bar = bars[sp["outer_idx"]]
        h, w = int(sp["h"]), int(sp["w"])

        c0_lo = int(outer_bar["c_min"])
        c0_hi = int(outer_bar["c_max"]) - w + 1
        if sp["inner_idx"] is not None:
            inner_bar = bars[sp["inner_idx"]]
            _, ic0, _, iw = sp["inner_bbox"]
            c0_lo = max(c0_lo, int(inner_bar["c_min"]) - int(ic0))
            c0_hi = min(c0_hi, int(inner_bar["c_max"]) - (int(ic0) + int(iw)) + 1)
        if c0_lo > c0_hi:
            return None

        candidates = []
        for c0 in range(c0_lo, c0_hi + 1):
            if not _interval_free(bar_occ[sp["outer_idx"]], c0, w):
                continue
            if sp["inner_idx"] is not None:
                _, ic0, _, iw = sp["inner_bbox"]
                inner_c0 = c0 + int(ic0)
                if not _interval_free(bar_occ[sp["inner_idx"]], inner_c0, int(iw)):
                    continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[sp["outer_idx"]][c0 : c0 + w] = True
        if sp["inner_idx"] is not None:
            _, ic0, _, iw = sp["inner_bbox"]
            inner_c0 = c0 + int(ic0)
            bar_occ[sp["inner_idx"]][inner_c0 : inner_c0 + int(iw)] = True
        placed.append({**sp, "r0": r0, "c0": c0})
        top_rects.append((r0, c0, r0 + h - 1, c0 + w - 1))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[sp["outer_idx"]]
        paste(inp, patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

        transformed = patch.copy()
        outer_mask = patch == int(outer_bar["fill"])
        transformed[outer_mask] = int(outer_bar["border"])
        if sp["inner_idx"] is not None:
            inner_bar = bars[sp["inner_idx"]]
            inner_mask = patch == int(inner_bar["fill"])
            transformed[inner_mask] = int(outer_bar["fill"])
        paste(out, transformed, int(outer_bar["r_max"] - sp["h"] + 1), int(sp["c0"]), transparent=int(bg))

        if sp["inner_idx"] is not None:
            _, ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[sp["inner_idx"]]
            inner_patch = np.full((int(ih), int(iw)), int(inner_bar["border"]), dtype=int)
            paste(out, inner_patch, int(inner_bar["r_max"] - ih + 1), int(sp["c0"] + ic0))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(x) for x in palette[1 : 1 + n_bars]]
    borders = [int(x) for x in palette[1 + n_bars : 1 + 2 * n_bars]]

    band_sizes = _sample_band_sizes(n_bars)
    base, bars = _build_bars(bg, fills, borders, band_sizes)
    top_h = min(int(bar["r_min"]) for bar in bars)

    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)
    sprites = [_make_sprite(outer_slots[i], bars) for i in range(n_sprites)]
    sprites.sort(key=lambda s: (s["inner_idx"] is None, -int(s["w"]), -int(s["h"])))

    placed = _place_sprites(sprites, bars, top_h)
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
