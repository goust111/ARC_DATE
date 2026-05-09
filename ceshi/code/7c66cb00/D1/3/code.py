# Variant of 7c66cb00
# Dimension: D1 — 匹配条件：精灵与条的配对方式 (match_cond)
# Change: 精灵的外层颜色（多数色）等于目标条的填充色：s.outer_color == d.fill_color... → 精灵面积排序匹配条的位置排序

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior and pipeline impact
# - Visual objects remain the same broad family as the source task: a 30x24 grid,
#   2-3 full-width bars stacked in the bottom area, and 2-3 small two-tone sprites
#   floating in the top area.
# - The original task paired a sprite to a bar through color identity:
#     sprite.outer_color == bar.fill_color
#   Then the sprite was bottom-aligned into that bar and recolored:
#     outer -> bar.border, inner -> bar.fill.
# - This variant changes only the pairing rule. Bars are ordered by vertical
#   position (top to bottom). Sprites are ordered by total occupied area
#   (non-background cell count, largest to smallest). The k-th largest sprite
#   docks into the k-th bar.
# - Small trace:
#   Original: top sprite A is mostly red, so it must go into the red-filled bar.
#   Variant: sprite A may be mostly green, but if it is the largest sprite it must
#   go into the topmost bar, regardless of color.
#   The recoloring rule is unchanged once the pairing is known.
#
# b) Complexity calibration
# - To keep the new rule visible and non-ambiguous, each sample uses exactly one
#   sprite per bar and enforces distinct sprite areas.
# - If equal areas were allowed, the ordering would require extra tie-breakers that
#   are not visually obvious to a human solver. Distinct areas make the ordinal map
#   readable from 2-4 examples.
# - We also deliberately derange sprite outer colors relative to their target bars,
#   so the task does not collapse back into color matching.
#
# c) Implicit-assumption audit of the original solver
# - Original top-object extraction iterated by color and then by monochrome
#   component. That is only safe when color identity is the pairing key; for this
#   variant it is wrong because an inner patch would be detected as its own object
#   and enter the size ranking independently.
# - Therefore the solver must detect whole visual sprites as connected components
#   in the non-background top area.
# - Original matching line:
#     matching = [p for p in patterns if p["outer_color"] == bar["fill"]]
#   directly encodes the old D1 assumption and must be replaced by an ordinal zip
#   between sorted sprites and sorted bars.
#
# d) Placement legality
# - Generator keeps the original bar geometry: full-width bars, 1-row gaps, bottom
#   stack, sprite x inside cols 1..W-2.
# - Because each bar receives exactly one sprite, output placements cannot collide
#   within a bar. Top-area placements still enforce one-cell separation.
# - Each sprite height is generated with respect to the height of its target bar,
#   because the target is now size-order based rather than color based.
#
# e) Visual observability
# - Humans can see bar order directly from top to bottom.
# - Humans can compare sprite size directly because sprites are compact rectangles
#   (optionally with a smaller inner rectangle) and all areas are distinct.
# - Colors still matter for recoloring, but no hidden numeric color ordering is
#   used anywhere in the new matching rule.
#
# f) Visual cleanliness
# - Sprites are sparse and separated in the top area.
# - Bars remain unchanged and non-overlapping.
# - Inner regions are minority blocks fully enclosed by the outer color, so each
#   sprite remains a single coherent object instead of a noisy collage.
#
# g) Color-change inferability
# - The recolor rule stays structurally tied to the destination bar:
#   outer cells take the bar border color, inner cells take the bar fill color.
# - The number of colors remains small. Since bar order determines destination and
#   the target bar's own colors are visible in-place, the post-docking recolor is
#   still easy to infer from examples.


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    bars: list[dict] = []
    current_bar: dict | None = None
    for r in range(H):
        row = g[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            c = Counter(non_bg)
            fill_color, count = c.most_common(1)[0]
            if int(count) > int(W * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar:
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                    }
            else:
                if current_bar:
                    bars.append(current_bar)
                    current_bar = None
        else:
            if current_bar:
                bars.append(current_bar)
                current_bar = None
    if current_bar:
        bars.append(current_bar)
    return bars


def _find_sprites(g: np.ndarray, bg: int, pattern_area_end: int) -> list[dict]:
    top = g[:pattern_area_end, :]
    mask = top != int(bg)
    lab, n = label_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        bbox = crop_bbox(g, (r0, c0, r1, c1))

        colors_in_bbox = Counter(int(v) for v in bbox.flatten() if int(v) != int(bg))
        if not colors_in_bbox:
            continue
        outer_color = int(colors_in_bbox.most_common(1)[0][0])
        inner_colors = [int(c) for c in colors_in_bbox if int(c) != int(outer_color)]
        inner_color = int(inner_colors[0]) if inner_colors else None

        sprites.append(
            {
                "outer_color": int(outer_color),
                "inner_color": inner_color,
                "r0": int(r0),
                "c0": int(c0),
                "h": int(bbox.shape[0]),
                "w": int(bbox.shape[1]),
                "area": int(len(coords)),
                "content": bbox,
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into horizontal bars using size-order matching.

    Input structure:
    - bg = most common color.
    - Top area: 2-3 connected two-tone sprites.
    - Bottom area: 2-3 full-width horizontal bars with fill and border colors.

    Rule:
    - Order bars from top to bottom.
    - Order sprites by total area from large to small.
    - Pair them by rank.
    - Place each sprite at its original x-position, bottom-aligned in its paired bar.
    - Recolor: sprite outer -> paired bar border, sprite inner -> paired bar fill.
    - Erase the original sprites from the top area.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found — grid must contain at least one horizontal bar"
    pattern_area_end = int(min(int(b["r_min"]) for b in bars))

    sprites = _find_sprites(g, bg, pattern_area_end)
    assert len(sprites) >= len(bars), "Need at least as many sprites as bars"

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    bars_sorted = sorted(bars, key=lambda b: int(b["r_min"]))
    sprites_sorted = sorted(
        sprites,
        key=lambda p: (-int(p["area"]), int(p["r0"]), int(p["c0"])),
    )

    for bar, pat in zip(bars_sorted, sprites_sorted):
        transformed = np.array(pat["content"], dtype=int).copy()
        for rr in range(int(pat["h"])):
            for cc in range(int(pat["w"])):
                v = int(transformed[int(rr), int(cc)])
                if v == int(bg):
                    continue
                if v == int(pat["outer_color"]):
                    transformed[int(rr), int(cc)] = int(bar["border"])
                else:
                    transformed[int(rr), int(cc)] = int(bar["fill"])

        place_r = int(bar["r_max"] - pat["h"] + 1)
        place_c = int(pat["c0"])
        paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


# --------------- generator ---------------

GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.6
GEN_RETRIES = 400


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
        r0, r1 = r - h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {"fill": int(fills[i]), "border": int(borders[i]), "h": int(h), "r_min": int(r0), "r_max": int(r1)}
        r = r0 - 1
    return grid, bars


def _derangement(n: int) -> list[int]:
    if n == 2:
        return [1, 0]
    shift = random.choice([1, 2])
    return [int((i + shift) % n) for i in range(n)]


def _candidate_dims(max_h: int) -> list[tuple[int, int, int]]:
    dims: list[tuple[int, int, int]] = []
    for h in range(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], max_h) + 1):
        for w in range(SPRITE_W_RANGE[0], SPRITE_W_RANGE[1] + 1):
            dims.append((int(h * w), int(h), int(w)))
    random.shuffle(dims)
    dims.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    return dims


def _choose_descending_dims(bars: list[dict]) -> list[tuple[int, int, int]] | None:
    candidate_lists = [_candidate_dims(int(bar["h"])) for bar in bars]
    choice: list[tuple[int, int, int]] = []

    def backtrack(i: int, prev_area: int | None) -> bool:
        if i == len(bars):
            return True
        candidates = candidate_lists[i][:]
        random.shuffle(candidates)
        candidates.sort(key=lambda x: -x[0])
        for area, h, w in candidates:
            if prev_area is not None and int(area) >= int(prev_area):
                continue
            if any(int(area) == int(prev[0]) for prev in choice):
                continue
            choice.append((int(area), int(h), int(w)))
            if backtrack(i + 1, int(area)):
                return True
            choice.pop()
        return False

    if backtrack(0, None):
        return choice
    return None


def _make_patch(h: int, w: int, outer_color: int, inner_color: int | None) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    patch = np.full((int(h), int(w)), int(outer_color), dtype=int)
    inner_bbox = None
    if inner_color is not None and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, w - 1)
            if (w - iw) % 2 == 0 and int(ih * iw * 2) < int(h * w)
        ]
        if valid_inner:
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(inner_color)
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))
    return patch, inner_bbox


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], top_h: int) -> list[dict] | None:
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        candidates: list[tuple[int, int]] = []
        for c0 in range(1, GRID_W - w):
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, int(r0), int(c0), int(h), int(w)):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()
    for sp in placed:
        patch = np.array(sp["patch"], dtype=int).copy()
        target_bar = bars[int(sp["target_idx"])]
        paste(inp, patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

        transformed = patch.copy()
        for rr in range(int(sp["h"])):
            for cc in range(int(sp["w"])):
                v = int(transformed[int(rr), int(cc)])
                if v == int(bg):
                    continue
                if v == int(sp["outer_color"]):
                    transformed[int(rr), int(cc)] = int(target_bar["border"])
                else:
                    transformed[int(rr), int(cc)] = int(target_bar["fill"])
        paste(out, transformed, int(target_bar["r_max"] - sp["h"] + 1), int(sp["c0"]), transparent=int(bg))
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = n_bars

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(x) for x in palette[1 : 1 + n_bars]]
    borders = [int(x) for x in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    dims = _choose_descending_dims(bars)
    if dims is None:
        return None

    outer_perm = _derangement(n_bars)
    sprites: list[dict] = []
    for target_idx in range(n_sprites):
        area, h, w = dims[target_idx]
        outer_idx = int(outer_perm[target_idx])
        inner_choices = [i for i in range(n_bars) if i != outer_idx]
        if n_bars >= 3:
            non_target_choices = [i for i in inner_choices if i != target_idx]
            if non_target_choices:
                inner_choices = non_target_choices
        inner_idx = random.choice(inner_choices) if inner_choices else None
        patch, inner_bbox = _make_patch(int(h), int(w), int(bars[outer_idx]["fill"]), None if inner_idx is None else int(bars[inner_idx]["fill"]))
        sprites.append(
            {
                "target_idx": int(target_idx),
                "outer_idx": int(outer_idx),
                "outer_color": int(bars[outer_idx]["fill"]),
                "inner_idx": None if inner_idx is None else int(inner_idx),
                "patch": patch,
                "h": int(h),
                "w": int(w),
                "area": int(area),
                "inner_bbox": inner_bbox,
            }
        )

    random.shuffle(sprites)
    placed = _place_sprites(sprites, top_h)
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
