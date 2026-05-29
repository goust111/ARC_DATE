# Combo variant of 7c66cb00 (index 16)
# combo_spec: D2/1+D6/0+D13/2
# Name: 共享边框色匹配边框色后颜色保持

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The source task used sprite.outer -> bar.fill matching, then recolored
#    the docked sprite. Here the bars all share one border color, so a plain
#    D2 border-match would collapse: every sprite outer would match every bar.
#    The fused rule resolves that by keeping the shared border color as a
#    common outer shell while moving the discriminative key to the inner color.
#    Small example:
#    - Bars: [border=6, fill=2], [border=6, fill=4], [border=6, fill=8]
#    - Top sprite A uses outer=6 and inner=4 -> it docks into the fill-4 bar.
#    - Top sprite B uses outer=6 and inner=2 -> it docks into the fill-2 bar.
#    - Unlike the source task, neither sprite changes color after docking.
#    The visual difference from the original is therefore twofold:
#    all bars have the same thin edge color, and the transported sprite keeps
#    its border-colored shell plus fill-colored core unchanged in the output.
#
# b) Complexity calibration:
#    To avoid over-simplifying into trivial monochrome copies, every generated
#    sprite is forced to be two-tone. The bars keep distinct fill colors, and
#    2-3 sprites are sampled so multiple examples support "inner color chooses
#    the bar". We still limit the palette to one shared border color plus the
#    bar fills, which keeps the inference compact enough for ARC.
#
# c) Implicit assumptions audit:
#    - Original matching `outer_color == bar["fill"]` is invalid because outer
#      now equals the same shared border on every sprite.
#    - Original color-wise component search is invalid because the inner region
#      would be extracted as its own object. We must detect whole non-bg
#      connected sprites instead.
#    - Original recoloring logic is invalid because D6/0 removes all color
#      swapping; the pasted sprite must be unchanged.
#    - Bar detection by majority fill still works because the bar interior
#      remains wide and each bar row still has only one dominant non-bg color.
#
# d) Placement legality:
#    Generator guarantees each sprite height fits inside its target bar, every
#    x-span stays within cols 1..W-2, top sprites are pairwise separated by at
#    least one background cell, and docked sprites targeting the same bar do
#    not overlap horizontally.
#
# e) Visual observability:
#    The shared border color appears on all bar edges and on every sprite outer
#    shell, so a solver cannot use that color to distinguish destinations.
#    The only repeated distinguishing cue is the sprite inner color matching a
#    bar fill. Because colors are preserved after docking, that cue remains
#    visible in both input and output examples.
#
# f) Visual cleanliness:
#    Bars remain simple full-width bands separated by one background row, and
#    each top sprite is a compact bordered rectangle with a single inner block.
#    No extra inner sub-sprite outputs are rendered, so the output contains one
#    moved copy per input sprite and stays readable.
#
# g) Change inferability:
#    The fusion introduces a structured color system rather than arbitrary
#    recoloring: shared border color = common shell, unique bar fills = routing
#    labels. This keeps the rule inferable from a few examples because the same
#    inner color repeatedly selects the same fill-colored bar and survives the
#    move unchanged.
#
# New solver assumptions introduced here:
# - Top sprites are connected non-bg objects; generator enforces bg separation.
# - Each sprite has exactly one minority inner color; generator always builds a
#   border-colored frame around a single inner rectangle.
# - Bar fills are unique, so inner color identifies at most one bar.
# - All bars share one border color; generator enforces that explicitly.


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (3, 5)
SPRITE_W_RANGE = (3, 9)
GEN_RETRIES = 300


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    """Detect full-width horizontal bars in the lower portion."""
    h, w = g.shape
    bars: list[dict] = []
    current: dict | None = None

    for r in range(h):
        row = g[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if not non_bg:
            if current is not None:
                bars.append(current)
                current = None
            continue

        fill, count = Counter(non_bg).most_common(1)[0]
        if int(count) <= int(w * 0.7):
            if current is not None:
                bars.append(current)
                current = None
            continue

        left = int(row[0])
        right = int(row[-1])
        border = left if left != int(fill) else right

        if current and current["fill"] == int(fill) and current["border"] == int(border):
            current["r_max"] = int(r)
        else:
            if current is not None:
                bars.append(current)
            current = {
                "fill": int(fill),
                "border": int(border),
                "r_min": int(r),
                "r_max": int(r),
            }

    if current is not None:
        bars.append(current)

    return bars


def _find_sprites(top: np.ndarray, bg: int) -> list[dict]:
    """Detect top sprites as connected visual objects instead of per-color parts."""
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
        inner_colors = [int(color) for color in colors if int(color) != int(outer_color)]
        sprites.append(
            {
                "outer_color": int(outer_color),
                "inner_color": inner_colors[0] if inner_colors else None,
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
    Move two-tone sprites from the top area into matching lower bars.

    Input structure:
    - The background is the most common color.
    - The lower region contains 2-3 full-width horizontal bars whose edge
      columns share one common border color while their interior fill colors
      are distinct.
    - The upper region contains separated two-color sprites: a majority outer
      shell in that shared border color plus a minority inner block.

    Rule:
    - First verify the D2-style outer/border agreement: the sprite outer shell
      must use the same shared color as the bar borders.
    - Because all bars share that border color, use the sprite inner color to
      choose the bar whose fill color is the same.
    - Erase the original top sprites, then paste each sprite into its matched
      bar at the same x-position and bottom alignment.
    - Keep both sprite colors unchanged after placement.

    Key insight:
    The border-match cue from the D2 variant is still present: every movable
    sprite has the shared border color as its outer shell. Once that condition
    is satisfied for all bars, the inner color becomes the tie-breaker that
    identifies the destination fill. The move stays spatially the same, but
    the recoloring step is completely removed.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(g.flatten()).most_common(1)[0][0])

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    pattern_area_end = min(int(bar["r_min"]) for bar in bars)
    sprites = _find_sprites(g[:pattern_area_end, :], bg)
    shared_borders = {int(bar["border"]) for bar in bars}
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}
    shared_border = next(iter(shared_borders)) if len(shared_borders) == 1 else None

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] == int(bg), out[:pattern_area_end, :], int(bg))

    for sprite in sprites:
        if shared_border is None or int(sprite["outer_color"]) != int(shared_border):
            continue
        inner_color = sprite["inner_color"]
        if inner_color is None:
            continue
        bar = fill_to_bar.get(int(inner_color))
        if bar is None:
            continue

        place_r = int(bar["r_max"] - sprite["h"] + 1)
        paste(out, np.array(sprite["content"], dtype=int), place_r, int(sprite["c0"]), transparent=int(bg))

    return out


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], border: int, heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H

    for i in range(len(heights) - 1, -1, -1):
        bar_h = int(heights[i])
        r0, r1 = r - bar_h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(border)
        grid[r0 : r1 + 1, GRID_W - 1] = int(border)
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(border),
            "h": int(bar_h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = r0 - 1

    return grid, bars


def _make_sprite(target_idx: int, bars: list[dict], shared_border: int) -> dict:
    target_bar = bars[target_idx]
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(target_bar["h"])))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), int(shared_border), dtype=int)

    valid_inner = [
        (ih, iw)
        for ih in range(1, h - 1)
        for iw in range(1, w - 1)
        if (w - iw) % 2 == 0 and ih * iw * 2 < h * w
    ]
    assert valid_inner, "sprite dimensions must allow a minority inner block"

    ih, iw = random.choice(valid_inner)
    ir0 = random.randint(1, h - ih - 1)
    ic0 = (w - iw) // 2
    patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(target_bar["fill"])

    return {
        "target_idx": int(target_idx),
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": (int(ir0), int(ic0), int(ih), int(iw)),
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

    for sprite in sprites:
        h = int(sprite["h"])
        w = int(sprite["w"])
        target_idx = int(sprite["target_idx"])
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
        placed.append({**sprite, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for sprite in placed:
        patch = np.array(sprite["patch"], dtype=int)
        target_bar = bars[int(sprite["target_idx"])]
        h = int(sprite["h"])
        c0 = int(sprite["c0"])

        paste(inp, patch, int(sprite["r0"]), c0, transparent=int(bg))
        paste(out, patch, int(target_bar["r_max"] - h + 1), c0, transparent=int(bg))

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
    shared_border = int(palette[1])
    fills = [int(v) for v in palette[2 : 2 + n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, shared_border, heights)
    top_h = int(bars[0]["r_min"])

    targets = _choose_targets(n_bars, n_sprites)
    if len(set(int(t) for t in targets)) < 2:
        return None
    sprites = [_make_sprite(targets[i], bars, shared_border) for i in range(n_sprites)]
    sprites.sort(key=lambda sprite: (-int(sprite["w"]), -int(sprite["h"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

    if get_bg(inp) != int(bg):
        return None
    if len({int(bar["border"]) for bar in bars}) != 1:
        return None
    if len({int(sprite["target_idx"]) for sprite in placed}) < 2:
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
