# Variant of 7c66cb00
# Dimension: D3 — 源对象：精灵结构与颜色组成 (src)
# Change: 双色精灵：外层色（多数像素）包裹内层色（少数像素，可选），2x2到5x9尺寸，左右对称，在顶部区域互不重叠... → 三色精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

# Design analysis for the D3 variant
# ----------------------------------
# a) Concretization + behavior trace
#    - Input still has full-width horizontal bars in the lower portion and separated
#      sprites in the top portion, but each top sprite now has exactly 3 nested fill
#      colors taken from 3 different bars: outer shell, middle layer, core.
#    - We render sprites as LR-symmetric nested rectangles:
#        outer: full 5x(5/7/9) patch
#        middle: centered 3x(smaller odd width) rectangle
#        core: centered 1x(smaller odd width) segment inside the middle rectangle
#    - Original local logic on one 2-color sprite:
#        sprite(fill_A outside, fill_B inside)
#        -> dock full sprite into bar A with A->border_A and B->fill_A
#        -> also dock the inner region into bar B as border_B
#    - Variant local logic on one 3-color sprite:
#        sprite(fill_A outside, fill_B middle, fill_C core)
#        -> dock full sprite into bar A with A->border_A and {B,C}->fill_A
#        -> dock the middle+core sub-sprite into bar B with B->border_B and C->fill_B
#        -> dock the core sub-sprite into bar C as border_C
#      The difference is that one source object now splits recursively into 3 outputs.
# b) Complexity calibration
#    - We fix bar count to 3 so every sample has enough destinations for the 3 layers.
#    - Every sprite uses all 3 bar fill colors exactly once, which makes the new rule
#      visually obvious and avoids ambiguous "missing third layer" cases.
#    - Width choices are limited to odd values and nested centered rectangles so the
#      structure is easy to parse from 2-3 examples, but still varied enough to avoid
#      repetition.
# c) Hidden assumption audit against the original main()
#    - Original pattern discovery iterated per color component, then inferred a sprite
#      from each color's bbox. That is safe for the old task only because 1 inner color
#      created at most one useful duplicate. With 3 nested colors it would detect the
#      same sprite 3 times and over-paste conflicting outputs.
#    - Original solver assumed "dominant color = outer, at most one other color =
#      inner". That breaks once we have 3 colors; middle and core must be separated.
#    - Original recolor loop mapped exactly two source colors. The variant needs a
#      recursive recolor for outer and middle layers plus an independent core docking.
# d) Placement legality
#    - Top sprites remain non-overlapping connected components with a 1-cell gap.
#    - Generator tracks horizontal occupancy separately for each target bar using the
#      exact interval consumed by the outer, middle, and core dockings, so output bars
#      never receive overlapping pasted regions.
#    - All x positions stay within cols 1..W-2, and every layer's height fits its bar.
# e) Visual observability
#    - Human-visible clue: each input sprite clearly has 3 nested color bands, and in
#      the output the same source creates 3 aligned descendants across different bars.
#      This is inferable from structure, not from numeric color ids.
# f) Visual cleanliness
#    - There is no overdraw within a bar because per-bar x-intervals are reserved.
#    - The nested rectangles stay simple and symmetric, so the split relation remains
#      legible after docking.
# g) Color-change inferability
#    - Colors are constrained to 3 bar fills + 3 bar borders + bg.
#    - Outer and middle outputs preserve the original "border vs fill" cue, which tells
#      the solver that the target bar contributes the new colors.
# 9) New solver assumptions and generator guarantees
#    - main() assumes every top sprite is a single connected non-bg component.
#      generate_input() guarantees that by using fully filled nested rectangles.
#    - main() assumes exactly one color touches the sprite bbox border (outer layer),
#      and the other 2 colors have strictly nested bboxes. Generation enforces this.
#    - main() assumes all 3 sprite colors match distinct bar fill colors. Generation
#      uses a permutation of the 3 bar fills for every sprite.


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

    assert len(infos) == 3, "Each sprite must have exactly 3 colors in this variant"

    outer_candidates = [info for info in infos if info["touches_border"]]
    assert len(outer_candidates) == 1, "Expected a single outer layer touching the sprite border"
    outer = outer_candidates[0]

    remaining = [info for info in infos if info["color"] != outer["color"]]
    remaining.sort(key=lambda info: info["bbox_area"], reverse=True)
    middle, core = remaining

    assert _contains(outer["bbox"], middle["bbox"])
    assert _contains(middle["bbox"], core["bbox"])
    return {"outer": outer, "middle": middle, "core": core}


def _recolor_nested(patch: np.ndarray, owner_color: int, border_color: int, fill_color: int, bg: int) -> np.ndarray:
    out = np.full_like(patch, int(bg))
    fg = patch != int(bg)
    out[fg] = int(fill_color)
    out[patch == int(owner_color)] = int(border_color)
    return out


def _recolor_solid(patch: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full_like(patch, int(bg))
    out[patch != int(bg)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
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
                "content": content,
                "layers": layers,
            }
        )

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for sprite in sprites:
        content = sprite["content"]
        c0 = int(sprite["c0"])
        h = int(content.shape[0])
        layers = sprite["layers"]

        outer = layers["outer"]
        outer_bar = bars_by_fill[int(outer["color"])]
        outer_patch = _recolor_nested(content, int(outer["color"]), int(outer_bar["border"]), int(outer_bar["fill"]), bg)
        paste(out, outer_patch, int(outer_bar["r_max"] - h + 1), c0, transparent=int(bg))

        middle = layers["middle"]
        mr0, mc0, mr1, mc1 = map(int, middle["bbox"])
        middle_crop = crop_bbox(content, (mr0, mc0, mr1, mc1))
        middle_bar = bars_by_fill[int(middle["color"])]
        middle_patch = _recolor_nested(
            middle_crop,
            int(middle["color"]),
            int(middle_bar["border"]),
            int(middle_bar["fill"]),
            bg,
        )
        paste(
            out,
            middle_patch,
            int(middle_bar["r_max"] - middle_patch.shape[0] + 1),
            int(c0 + mc0),
            transparent=int(bg),
        )

        core = layers["core"]
        cr0, cc0, cr1, cc1 = map(int, core["bbox"])
        core_crop = crop_bbox(content, (cr0, cc0, cr1, cc1))
        core_bar = bars_by_fill[int(core["color"])]
        core_patch = _recolor_solid(core_crop, int(core_bar["border"]), bg)
        paste(
            out,
            core_patch,
            int(core_bar["r_max"] - core_patch.shape[0] + 1),
            int(c0 + cc0),
            transparent=int(bg),
        )

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


def _make_sprite(order: tuple[int, int, int], bars: list[dict]) -> dict:
    outer_idx, middle_idx, core_idx = map(int, order)
    w = int(random.choice(SPRITE_W_CHOICES))
    middle_width_choices = [mw for mw in (w - 2, w - 4) if mw >= 3]
    mw = int(random.choice(middle_width_choices))
    core_width_choices = [cw for cw in range(1, mw, 2) if cw <= mw - 2]
    cw = int(random.choice(core_width_choices))

    patch = np.full((SPRITE_H, w), int(bars[outer_idx]["fill"]), dtype=int)
    middle_bbox = (1, (w - mw) // 2, 3, mw)
    core_bbox = (2, (w - cw) // 2, 1, cw)

    mr0, mc0, mh, _ = middle_bbox
    cr0, cc0, ch, _ = core_bbox
    patch[mr0 : mr0 + mh, mc0 : mc0 + mw] = int(bars[middle_idx]["fill"])
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
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[int(start) : int(start + length)].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _placement_intervals(sprite: dict) -> list[tuple[int, int, int]]:
    _, middle_c0, _, middle_w = sprite["middle_bbox"]
    _, core_c0, _, core_w = sprite["core_bbox"]
    return [
        (int(sprite["outer_idx"]), 0, int(sprite["w"])),
        (int(sprite["middle_idx"]), int(middle_c0), int(middle_w)),
        (int(sprite["core_idx"]), int(core_c0), int(core_w)),
    ]


def _place_sprites(sprites: list[dict], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in range(BAR_COUNT)]
    top_rects: list[tuple[int, int, int, int]] = []
    placed = []

    for sprite in sprites:
        h, w = int(sprite["h"]), int(sprite["w"])
        intervals = _placement_intervals(sprite)
        candidates = []
        for c0 in range(1, GRID_W - w):
            if any(not _interval_free(bar_occ[bar_idx], c0 + off, width) for bar_idx, off, width in intervals):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        for bar_idx, off, width in intervals:
            bar_occ[bar_idx][int(c0 + off) : int(c0 + off + width)] = True

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
        outer_patch = _recolor_nested(patch, int(outer_bar["fill"]), int(outer_bar["border"]), int(outer_bar["fill"]), bg)
        paste(out, outer_patch, int(outer_bar["r_max"] - patch.shape[0] + 1), c0, transparent=int(bg))

        mr0, mc0, mh, mw = map(int, sprite["middle_bbox"])
        middle_crop = patch[mr0 : mr0 + mh, mc0 : mc0 + mw]
        middle_bar = bars[int(sprite["middle_idx"])]
        middle_patch = _recolor_nested(
            middle_crop,
            int(middle_bar["fill"]),
            int(middle_bar["border"]),
            int(middle_bar["fill"]),
            bg,
        )
        paste(
            out,
            middle_patch,
            int(middle_bar["r_max"] - middle_patch.shape[0] + 1),
            int(c0 + mc0),
            transparent=int(bg),
        )

        cr0, cc0, ch, cw = map(int, sprite["core_bbox"])
        core_crop = patch[cr0 : cr0 + ch, cc0 : cc0 + cw]
        core_bar = bars[int(sprite["core_idx"])]
        core_patch = _recolor_solid(core_crop, int(core_bar["border"]), bg)
        paste(
            out,
            core_patch,
            int(core_bar["r_max"] - core_patch.shape[0] + 1),
            int(c0 + cc0),
            transparent=int(bg),
        )

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
    if n_sprites >= 2 and len({order[0] for order in orders}) == 1:
        orders[1] = tuple([orders[1][1], orders[1][0], orders[1][2]])

    sprites = [_make_sprite(order, bars) for order in orders]
    sprites.sort(key=lambda sprite: (-int(sprite["w"]), -int(sprite["middle_bbox"][3]), -int(sprite["core_bbox"][3])))

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
