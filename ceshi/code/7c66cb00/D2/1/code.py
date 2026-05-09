# Variant of 7c66cb00
# Dimension: D2 — 匹配条件：颜色映射方向 (match_cond)
# Change: 精灵的外层色（多数色/主色调）作为匹配键，映射到条的填充色... → 匹配条的边框色而非填充色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior:
#    The bottom bars stay visually identical to the source task:
#    each bar spans the row, has a large fill color in cols 1..W-2,
#    and a thin border color only at cols 0 and W-1. The top sprites
#    are still 1-2 color rectangles with a majority "outer" color and
#    an optional centered minority "inner" block. The crucial change is
#    that the outer color is now copied from a bar BORDER, not from a
#    bar FILL. In the output, each sprite still docks into its matched
#    bar at the same x-position and bottom alignment; the inner color is
#    recolored to the destination bar fill, while the outer color stays
#    at the bar border color.
#    Small example:
#    - Original logic: sprite outer=3 matches bar fill=3, then outer->4
#      if that bar border is 4.
#    - Variant logic: sprite outer=4 matches the bar whose border is 4;
#      after docking the outer remains 4 and any inner color becomes the
#      matched bar fill (say 3).
#    The visible cue is therefore "tiny edge color on a bar picks the
#    destination", not the large fill area.
#
# b) Complexity calibration:
#    To keep the change observable, generator chooses bar fills and bar
#    borders from disjoint color sets. That prevents accidental overlap
#    where an outer border-color could also equal some fill-color and
#    make the old solver appear correct. We still keep 2-3 bars and 2-3
#    sprites so multiple examples support the rule.
#
# c) Implicit assumptions audit for the original main():
#    - Bar detection assumed the majority non-bg color in a bar row is
#      the fill. This still holds because bars are unchanged.
#    - Matching used p["outer_color"] == bar["fill"]. This must change
#      to border matching.
#    - Pattern detection iterated per non-bg color component. That was
#      harmless before because all sprite colors were fill-colors and the
#      generator explicitly rendered extra "inner sub-sprite" outputs.
#      Under D2 it would wrongly treat fill-colored inner regions as
#      independent sprites, even though only the outer border-colored
#      shell should control docking. We therefore switch to connected
#      components on the UNION of non-bg pixels, so one visual sprite
#      becomes one detected object.
#
# d) Placement legality:
#    Sprites are placed above the bars with a one-cell bg gap between
#    their bounding boxes. Within each target bar, x-intervals of docked
#    sprites may not overlap. Heights are clamped to fit inside the
#    destination bar, and widths are restricted to stay within cols
#    1..W-2 so bars' thin side borders remain visible.
#
# e) Visual observability:
#    Every sample uses border-only outer colors. Since border colors
#    appear only on the first/last columns of bars, the solver must
#    notice those narrow edge pixels. We also force at least one two-tone
#    sprite whenever possible so recoloring to the bar fill is visible.
#
# f) Visual cleanliness:
#    There is no bar overlay beyond the intended docked sprite, sprites
#    are separated in the top area, and all bars remain simple horizontal
#    bands. This avoids clutter despite the border-match cue being small.
#
# g) Color-change inference:
#    Colors remain structurally grouped: fills occupy wide bar interiors,
#    borders occupy the two edge columns, and sprite outers reuse only
#    border colors. Inner colors reuse fill colors from other bars, so
#    the output consistently shows "inner becomes destination fill".
#
# New solver assumptions introduced by this variant:
# - Top sprites are connected non-bg objects separated from one another
#   by background; the generator enforces that.
# - Each sprite uses 1-2 non-bg colors with a unique majority outer
#   color; the generator enforces an outer frame and minority inner box.
# - Each outer color matches exactly one bar border; the generator makes
#   all bar border colors distinct.


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    """Find horizontal full-width bars in the lower area."""
    h, w = g.shape
    bars: list[dict] = []
    current_bar: dict | None = None

    for r in range(h):
        row = g[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if not non_bg:
            if current_bar is not None:
                bars.append(current_bar)
                current_bar = None
            continue

        counts = Counter(non_bg)
        fill_color, count = counts.most_common(1)[0]
        if int(count) <= int(w * 0.7):
            if current_bar is not None:
                bars.append(current_bar)
                current_bar = None
            continue

        left = int(row[0])
        right = int(row[-1])
        border_color = left if left != fill_color else right
        if current_bar and current_bar["fill"] == int(fill_color) and current_bar["border"] == int(border_color):
            current_bar["r_max"] = int(r)
        else:
            if current_bar is not None:
                bars.append(current_bar)
            current_bar = {
                "fill": int(fill_color),
                "border": int(border_color),
                "r_min": int(r),
                "r_max": int(r),
            }

    if current_bar is not None:
        bars.append(current_bar)

    return bars


def _find_sprites(top: np.ndarray, bg: int) -> list[dict]:
    """Detect top sprites as connected non-bg visual objects."""
    lab, n = label_mask(top != int(bg), connectivity=4)
    sprites: list[dict] = []

    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        bbox = crop_bbox(top, (r0, c0, r1, c1))
        colors = Counter(int(v) for v in bbox.flatten() if int(v) != int(bg))
        if not colors:
            continue

        outer_color = int(colors.most_common(1)[0][0])
        inner_candidates = [int(c) for c in colors if int(c) != int(outer_color)]
        sprites.append(
            {
                "outer_color": outer_color,
                "inner_color": inner_candidates[0] if inner_candidates else None,
                "r0": int(r0),
                "c0": int(c0),
                "h": int(bbox.shape[0]),
                "w": int(bbox.shape[1]),
                "content": bbox,
            }
        )

    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock top sprites into horizontal bars using BORDER-color matching.

    Variant rule:
    - Match each sprite to the bar whose border color equals the sprite's
      majority outer color.
    - Keep x-position, bottom-align inside that bar.
    - Recolor any inner minority region to the destination bar fill.
    - Erase the original top sprites.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(g.flatten()).most_common(1)[0][0])

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    pattern_area_end = min(int(bar["r_min"]) for bar in bars)
    top = g[:pattern_area_end, :]
    sprites = _find_sprites(top, bg)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] == int(bg), out[:pattern_area_end, :], int(bg))

    for sp in sprites:
        bar = next((candidate for candidate in bars if int(candidate["border"]) == int(sp["outer_color"])), None)
        if bar is None:
            continue

        transformed = np.array(sp["content"], dtype=int).copy()
        if sp["inner_color"] is not None:
            transformed[transformed == int(sp["inner_color"])] = int(bar["fill"])

        place_r = int(bar["r_max"] - sp["h"] + 1)
        paste(out, transformed, place_r, int(sp["c0"]), transparent=int(bg))

    return out


# --------------- generator ---------------

GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.55
GEN_RETRIES = 300


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        bar_h = int(heights[i])
        r0, r1 = r - bar_h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(bar_h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = r0 - 1
    return grid, bars


def _make_sprite(target_idx: int, bars: list[dict], *, force_inner: bool) -> dict:
    target_bar = bars[target_idx]

    min_h = max(SPRITE_H_RANGE[0], 3 if force_inner else SPRITE_H_RANGE[0])
    min_w = max(SPRITE_W_RANGE[0], 3 if force_inner else SPRITE_W_RANGE[0])
    h = random.randint(min_h, min(SPRITE_H_RANGE[1], int(target_bar["h"])))
    w = random.randint(min_w, SPRITE_W_RANGE[1])

    patch = np.full((h, w), int(target_bar["border"]), dtype=int)
    inner_fill_idx = None
    inner_bbox = None

    allow_inner = len(bars) >= 2 and h >= 3 and w >= 3 and (force_inner or random.random() < INNER_COLOR_PROB)
    if allow_inner:
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, w - 1)
            if (w - iw) % 2 == 0 and ih * iw * 2 < h * w
        ]
        if valid_inner:
            inner_fill_idx = random.choice([idx for idx in range(len(bars)) if idx != target_idx])
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[inner_fill_idx]["fill"])
            inner_bbox = (ir0, ic0, ih, iw)

    return {
        "target_idx": int(target_idx),
        "inner_fill_idx": inner_fill_idx,
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[start : start + length].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
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
        target_idx = int(sp["target_idx"])
        candidates: list[tuple[int, int]] = []

        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[target_idx], c0, w):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[target_idx][c0 : c0 + w] = True
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        target_bar = bars[int(sp["target_idx"])]
        h = int(sp["h"])
        c0 = int(sp["c0"])

        paste(inp, patch, int(sp["r0"]), c0, transparent=int(bg))

        transformed = patch.copy()
        if sp["inner_fill_idx"] is not None:
            inner_fill = int(bars[int(sp["inner_fill_idx"])]["fill"])
            transformed[transformed == inner_fill] = int(target_bar["fill"])
        paste(out, transformed, int(target_bar["r_max"] - h + 1), c0, transparent=int(bg))

    return inp, out


def _choose_targets(n_bars: int, n_sprites: int) -> list[int]:
    targets = list(range(n_bars))
    random.shuffle(targets)
    targets = targets[: min(n_bars, n_sprites)]
    while len(targets) < n_sprites:
        targets.append(random.randrange(n_bars))
    random.shuffle(targets)
    return targets


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

    targets = _choose_targets(n_bars, n_sprites)
    forced_inner_idx = random.randrange(n_sprites)
    sprites = [_make_sprite(targets[i], bars, force_inner=(i == forced_inner_idx)) for i in range(n_sprites)]
    sprites.sort(key=lambda sprite: (-int(sprite["w"]), -int(sprite["h"])))

    if not any(sprite["inner_fill_idx"] is not None for sprite in sprites):
        return None

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
