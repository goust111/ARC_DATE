# Combo variant of 7c66cb00 (index 11)
# combo_spec: D3/0+D6/1+D11/2
# Name: 三色精灵子精灵保持在外层条统一重着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concretization + behavior trace
#    - Input keeps the original global layout: several full-width horizontal bars below,
#      separate sprites above them. The fused change is local to the sprite design and
#      to the docking rule.
#    - Each top sprite now has exactly 3 nested colors, all chosen from distinct bar
#      fill colors: outer shell, middle band, and core. The sprite silhouette is not a
#      plain rectangle; it has tapered rows so a later monochrome repaint still preserves
#      a visible shape.
#    - Counterfactual old split logic on one 3-color sprite would be:
#        outer(fill_A) -> bar A,
#        middle(fill_B) -> bar B,
#        core(fill_C) -> bar C,
#      creating multiple descendants across bars.
#    - Fused logic here is:
#        sprite(fill_A outside, fill_B middle, fill_C core)
#        -> match only by fill_A,
#        -> dock the whole sprite once into bar A at the original x,
#        -> recolor every non-bg pixel uniformly to border_A,
#        -> erase the source sprite from the top.
#      The middle/core colors matter only as internal evidence that the sprite is
#      three-layered; they no longer trigger independent docking.
# b) Complexity calibration
#    - We fix the scene to exactly 3 bars so every sample can support a true 3-color
#      sprite with distinct layer colors. This prevents accidental two-color fallbacks.
#    - Sprites use a small family of symmetric tapered silhouettes with widths 5/7/9.
#      That is enough variation to make samples distinct, while keeping the rule easy to
#      infer from 2-4 examples.
#    - Every sprite uses all 3 bar fills exactly once, so solvers must parse the nested
#      structure instead of relying on missing layers or optional cases.
# c) Hidden assumption audit against the original main()
#    - Original top-pattern discovery iterated by color component, which duplicates one
#      logical sprite once per color. With 3 nested colors that would triple-count the
#      same object and produce conflicting placements. The new solver first labels full
#      non-bg components, then analyzes each component's internal colors.
#    - Original code assumed at most one inner color and recolored outer/inner
#      differently. That assumption is false now: there are exactly 3 colors, but all of
#      them collapse to the same destination color.
#    - Original D11-style independent inner docking is intentionally invalid here. The
#      solver must ignore middle/core as destinations and keep them inside the outer-bar
#      placement.
# d) Placement legality
#    - Top sprites are 4-connected and separated by at least one background cell.
#    - Output occupancy is tracked only on the outer bar of each sprite, because the
#      fused rule produces exactly one docked copy per sprite.
#    - Width stays within cols 1..W-2 and sprite height fits the matched outer bar.
# e) Visual observability
#    - Human-visible clue: each input sprite shows 3 nested colors from 3 different bars,
#      but the output shows only one descendant per sprite and it is completely monochrome.
#      Across examples this makes the "inner layers stay with the outer layer" rule
#      observable without reading numeric color ids.
# f) Visual cleanliness
#    - No extra descendants are pasted into secondary bars, so the lower area remains
#      cleaner than the split variants.
#    - Tapered silhouettes keep output shapes distinguishable even after recoloring.
# g) Color-change inferability
#    - All sprite colors are bar fills; all output sprite pixels become the matched bar's
#      border color. The bar itself visually supplies the destination color cue.
# 9) New solver assumptions and generator guarantees
#    - main() assumes exactly 3 bars and exactly 3 sprite colors per connected sprite.
#      generate_input() enforces both.
#    - main() assumes exactly one sprite color touches the sprite bbox border and the
#      other 2 colors form strictly nested centered regions. Generation uses tapered
#      outer shells plus centered middle/core rectangles to guarantee this.
#    - main() assumes output overlap only matters within outer bars. The generator tracks
#      those intervals and rejects placements that would collide.


GRID_H = 30
GRID_W = 24
BAR_COUNT = 3
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (5, 8)
MAX_TOTAL_BAR_HEIGHT = 20
MIN_TOP_AREA = 7
SPRITE_H = 5
SPRITE_W_CHOICES = (5, 7, 9)
GEN_RETRIES = 400

OUTER_ROW_TEMPLATES: dict[int, tuple[tuple[int, ...], ...]] = {
    5: ((3, 5, 5, 5, 3),),
    7: (
        (5, 7, 7, 7, 5),
        (3, 5, 7, 5, 3),
        (5, 7, 5, 7, 5),
    ),
    9: (
        (7, 9, 9, 9, 7),
        (5, 7, 9, 7, 5),
        (7, 9, 7, 9, 7),
    ),
}


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) <= int(inner[0])
        and int(outer[1]) <= int(inner[1])
        and int(outer[2]) >= int(inner[2])
        and int(outer[3]) >= int(inner[3])
    )


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    current_bar: dict | None = None
    h, w = grid.shape
    for r in range(h):
        row = grid[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(w * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if current_bar is not None and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar is not None:
                        current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                        "h": 1,
                    }
            else:
                if current_bar is not None:
                    current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                    bars.append(current_bar)
                    current_bar = None
        else:
            if current_bar is not None:
                current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                bars.append(current_bar)
                current_bar = None
    if current_bar is not None:
        current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
        bars.append(current_bar)
    return bars


def _extract_layers(patch: np.ndarray, bg: int) -> dict[str, dict]:
    infos = []
    h, w = patch.shape
    for color in sorted(set(map(int, patch.flatten())) - {int(bg)}):
        mask = patch == int(color)
        coords = np.argwhere(mask)
        bbox = bbox_coords(coords)
        touches_border = bool(
            mask[0, :].any() or mask[h - 1, :].any() or mask[:, 0].any() or mask[:, w - 1].any()
        )
        infos.append(
            {
                "color": int(color),
                "bbox": tuple(map(int, bbox)),
                "bbox_area": _bbox_area(bbox),
                "area": int(mask.sum()),
                "touches_border": touches_border,
            }
        )

    assert len(infos) == 3, "Each sprite must have exactly three colors"

    outer_candidates = [info for info in infos if info["touches_border"]]
    assert len(outer_candidates) == 1, "Expected exactly one outer layer touching the sprite border"
    outer = outer_candidates[0]

    remaining = [info for info in infos if info["color"] != outer["color"]]
    remaining.sort(key=lambda info: (int(info["bbox_area"]), int(info["area"])), reverse=True)
    middle, core = remaining

    assert _contains(tuple(outer["bbox"]), tuple(middle["bbox"]))
    assert _contains(tuple(middle["bbox"]), tuple(core["bbox"]))
    return {"outer": outer, "middle": middle, "core": core}


def _recolor_monochrome(patch: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full_like(patch, int(bg))
    out[patch != int(bg)] = int(color)
    return out


def _build_outer_mask(row_widths: tuple[int, ...], width: int) -> np.ndarray:
    mask = np.zeros((SPRITE_H, width), dtype=bool)
    for r, row_w in enumerate(row_widths):
        c0 = int((width - row_w) // 2)
        mask[r, c0 : c0 + row_w] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock three-layer sprites into the bar matched by their outer color and repaint them solidly.

    Input structure:
    - Background is the most common color.
    - Bottom area contains exactly three full-width horizontal bars, each with a fill color and border color.
    - Top area contains separated three-color sprites: one outer shell plus two nested inner layers, all using bar fill colors.

    Rule:
    - For each sprite, find the bar whose fill color matches the sprite's outer layer color.
    - Remove the source sprite from the top and place the whole sprite once into that bar at the same x-position, bottom-aligned.
    - Recolor every non-background pixel of the placed sprite to the matched bar's border color; middle and core do not dock elsewhere.

    Key insight:
    - The extra sprite colors are descriptive, not destinations: only the outer layer decides the target bar, and the entire three-layer structure collapses into one monochrome copy there.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert len(bars) == BAR_COUNT, "Variant expects exactly three horizontal bars"
    bars_by_fill = {int(bar["fill"]): bar for bar in bars}

    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    top = g[:pattern_area_end, :]

    lab, n = label_mask(top != int(bg), connectivity=4)
    sprites = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        content = crop_bbox(top, (r0, c0, r1, c1))
        layers = _extract_layers(content, bg)
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "content": np.array(content, dtype=int),
                "layers": layers,
            }
        )

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for sprite in sprites:
        content = np.array(sprite["content"], dtype=int)
        c0 = int(sprite["c0"])
        h = int(content.shape[0])
        outer_color = int(sprite["layers"]["outer"]["color"])
        outer_bar = bars_by_fill[outer_color]
        transformed = _recolor_monochrome(content, int(outer_bar["border"]), bg)
        paste(out, transformed, int(outer_bar["r_max"] - h + 1), c0, transparent=int(bg))

    return out


def _sample_bar_heights() -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(BAR_COUNT)]
        if sum(heights) > MAX_TOTAL_BAR_HEIGHT:
            continue
        if GRID_H - sum(heights) - (BAR_COUNT - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[list-item]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
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
        r = int(r0 - 1)
    return grid, bars


def _make_sprite(order: tuple[int, int, int], bars: list[dict], bg: int) -> dict:
    outer_idx, middle_idx, core_idx = map(int, order)
    w = int(random.choice(SPRITE_W_CHOICES))
    row_widths = random.choice(OUTER_ROW_TEMPLATES[w])
    outer_mask = _build_outer_mask(row_widths, w)

    inner_capacity = int(min(row_widths[1:4]))
    middle_width_choices = [mw for mw in range(3, inner_capacity, 2)]
    assert middle_width_choices, "Sprite template must admit a middle layer"
    mw = int(random.choice(middle_width_choices))
    core_width_choices = [cw for cw in range(1, mw, 2)]
    cw = int(random.choice(core_width_choices))

    patch = np.full((SPRITE_H, w), int(bg), dtype=int)
    patch[outer_mask] = int(bars[outer_idx]["fill"])

    middle_bbox = (1, int((w - mw) // 2), 3, mw)
    mr0, mc0, mh, _ = middle_bbox
    patch[mr0 : mr0 + mh, mc0 : mc0 + mw] = int(bars[middle_idx]["fill"])

    core_bbox = (2, int((w - cw) // 2), 1, cw)
    cr0, cc0, ch, _ = core_bbox
    patch[cr0 : cr0 + ch, cc0 : cc0 + cw] = int(bars[core_idx]["fill"])

    return {
        "outer_idx": int(outer_idx),
        "middle_idx": int(middle_idx),
        "core_idx": int(core_idx),
        "patch": patch,
        "h": int(SPRITE_H),
        "w": int(w),
        "middle_bbox": tuple(map(int, middle_bbox)),
        "core_bbox": tuple(map(int, core_bbox)),
        "row_widths": tuple(map(int, row_widths)),
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[int(start) : int(start + length)].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in range(BAR_COUNT)]
    top_rects: list[tuple[int, int, int, int]] = []
    placed = []

    for sprite in sprites:
        h, w = int(sprite["h"]), int(sprite["w"])
        outer_idx = int(sprite["outer_idx"])
        candidates = []
        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_idx], c0, w):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][c0 : c0 + w] = True
        placed.append({**sprite, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()

    for sprite in placed:
        patch = np.array(sprite["patch"], dtype=int)
        r0, c0 = int(sprite["r0"]), int(sprite["c0"])
        paste(inp, patch, r0, c0, transparent=int(bg))

        outer_bar = bars[int(sprite["outer_idx"])]
        transformed = _recolor_monochrome(patch, int(outer_bar["border"]), bg)
        paste(out, transformed, int(outer_bar["r_max"] - patch.shape[0] + 1), c0, transparent=int(bg))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + BAR_COUNT]]
    borders = [int(v) for v in palette[1 + BAR_COUNT : 1 + 2 * BAR_COUNT]]

    heights = _sample_bar_heights()
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    orders = [tuple(random.sample(range(BAR_COUNT), BAR_COUNT)) for _ in range(n_sprites)]
    if n_sprites >= 2 and len({int(order[0]) for order in orders}) == 1:
        second = list(orders[1])
        second[0], second[1] = second[1], second[0]
        orders[1] = tuple(second)

    sprites = [_make_sprite(order, bars, bg) for order in orders]
    sprites.sort(key=lambda sprite: (-int(sprite["w"]), -sum(int(v) for v in sprite["row_widths"])))

    placed = _place_sprites(sprites, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

    if int(get_bg(inp)) != int(bg):
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
