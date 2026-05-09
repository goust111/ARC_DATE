# Variant of 7c66cb00
# Dimension: D17 — 符号编码：隐式变换参数 (symbol_instructions)
# Change: 无显式符号编码，变换完全由颜色匹配规则隐式确定... → 顶部区域的图例定义颜色映射

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

# Design analysis for the D17 variant
# ----------------------------------
# source_task_alignment:
#   reference_task_58490d8a uses a dedicated legend region as an instruction surface:
#   the solver must first read a small symbolic color index and only then apply the rule.
#   This variant ports exactly that idea into 7c66cb00, but keeps the original bar/sprite
#   geometry instead of introducing a new box object.
#
# a) concretization_and_trace:
#   input_layout:
#   - Bottom area is unchanged: 2-3 full-width horizontal bars with fill at cols 1..W-2
#     and border colors at cols 0 and W-1.
#   - Top area still contains 2-3 small sprites.
#   - New element: a 2-column legend in the top-left corner, with one row per bar.
#     Row i is [bar_fill_i, mapped_output_color_i]. With 2 bars the legend is 2x2;
#     with 3 bars it is 3x2.
#   - A one-cell bg gap around the legend keeps it visually separate from the sprites.
#   output_layout:
#   - The legend stays unchanged.
#   - Sprites are erased from the top area.
#   - Each detected sprite component still docks into the bar whose fill matches its
#     dominant / outer color, bottom-aligned at the same x-position.
#   - Recoloring is no longer fixed "outer->target border, inner->target fill".
#     Instead every non-bg sprite color c is recolored by the legend lookup c -> map[c].
#   trace_example:
#   - Original rule: full sprite with colors A(outside), B(inside) matched to bar A becomes
#     [A->border(A), B->fill(A)] inside bar A; inner B also independently docks into bar B
#     as border(B) because the solver extracts color components separately.
#   - Variant rule: same matching / same separate component extraction, but recolor uses
#     legend rows, e.g. A->X and B->Y. Then the docked full sprite becomes [A->X, B->Y],
#     and the independent B sub-component docks into bar B as solid Y.
#   difference:
#   - Matching is still implicit from bar fill colors.
#   - The transform parameter is now explicit and externalized in the legend.
#
# b) complexity_calibration:
#   - The legend has only 2 or 3 rows, matching the number of bars, so humans can inspect
#     it quickly and each mapping is reused at least once.
#   - Target colors are taken from a non-identity permutation of the bar border colors.
#     This prevents accidental collapse back to the original fixed swap.
#   - All fill colors and border colors are globally distinct in generated samples so the
#     recolor is visually obvious and no mapped color disappears into a bar fill.
#
# c) implicit_assumption_audit_on_original_main:
#   - Original pattern scan iterated over all non-bg pixels in the top area. That would
#     incorrectly treat the new legend as movable sprites, so the solver must mask out the
#     legend before component extraction.
#   - Original top-area clearing erased every non-bg pixel above the bars. That would also
#     erase the new instruction symbol, so clearing must exclude the legend mask.
#   - Original recolor logic assumed exactly two semantic roles inside a sprite:
#     dominant outer color and optional inner color. D17 breaks that assumption; recoloring
#     must instead be per-color from an external lookup, implemented without in-place
#     cascading assumptions.
#   - Original correctness happened to tolerate direct replacement because outputs were tied
#     to bar-local colors. Here targets are independent legend outputs, so recolor must read
#     from the original patch when writing transformed colors.
#
# d) placement_legality:
#   - The generator reserves the legend rectangle in the top area before sprite placement,
#     and the existing 1-cell separation rule keeps sprites away from it.
#   - The bar occupancy checks from the original generator are kept so docked full sprites
#     and independently docked inner sub-sprites do not overlap on their target bars.
#   - Sprite x-ranges stay inside cols 1..W-2, and sprite height is capped by the matched
#     bar height exactly as in the original task.
#
# e) visual_observability:
#   - Because the mapping is written explicitly as color pairs and the permutation is
#     guaranteed non-identity, the outputs differ from the original swap in a visible way.
#   - At least one sprite is forced to have an inner region when possible, so examples show
#     that the legend acts on colors globally, not only on the matched outer bar.
#
# f) visual_cleanliness:
#   - The legend is small, corner-anchored, and separated by background, so it reads as an
#     annotation rather than another movable object.
#   - The rest of the scene preserves the original clean "top sprites / bottom bars" split.
#
# g) color_change_inferability:
#   - Only bar fill colors appear as legend inputs, one row each.
#   - Only bar border colors appear as legend outputs, one row each.
#   - Every bar fill is used by at least one sprite, so every legend row is exercised.


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.6
GEN_RETRIES = 300
LEGEND_COLS = 2


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    bars = []
    current_bar = None
    H, W = g.shape
    for r in range(H):
        row = g[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            counts = Counter(non_bg)
            fill_color, count = counts.most_common(1)[0]
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


def _legend_mask(shape: tuple[int, int], n_rows: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[: int(n_rows), :LEGEND_COLS] = True
    return mask


def _read_legend(g: np.ndarray, n_bars: int, bg: int) -> dict[int, int]:
    color_map: dict[int, int] = {}
    for rr in range(int(n_bars)):
        src = int(g[int(rr), 0])
        dst = int(g[int(rr), 1])
        if src != int(bg) and dst != int(bg):
            color_map[int(src)] = int(dst)
    return color_map


def _apply_color_map(patch: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = np.array(patch, dtype=int).copy()
    original = np.array(patch, dtype=int)
    for src, dst in color_map.items():
        out[original == int(src)] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Match sprites to bars exactly as in the original task.
    - Read a 2x2 or 3x2 legend from the top-left corner.
    - Recolor every sprite pixel by legend lookup before docking.
    - Preserve the legend; erase only the movable sprites.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    n_bars = len(bars)
    pattern_area_end = int(min(int(b["r_min"]) for b in bars))
    legend_mask = _legend_mask(g.shape, n_bars)
    color_map = _read_legend(g, n_bars, bg)

    top = g[:pattern_area_end, :].copy()
    top[legend_mask[:pattern_area_end, :]] = int(bg)

    patterns = []
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
            bbox = crop_bbox(top, (r0, c0, r1, c1))
            colors_in_bbox = Counter(map(int, bbox.flatten()))
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue
            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            patterns.append(
                {
                    "outer_color": outer_color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )

    clear_mask = (out[:pattern_area_end, :] != int(bg)) & (~legend_mask[:pattern_area_end, :])
    out[:pattern_area_end, :][clear_mask] = int(bg)

    for bar in bars:
        matching = [p for p in patterns if int(p["outer_color"]) == int(bar["fill"])]
        for pat in matching:
            transformed = _apply_color_map(pat["content"], color_map)
            place_r = int(bar["r_max"] - pat["h"] + 1)
            place_c = int(pat["c0"])
            paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


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
    bars = [None] * len(heights)
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = r - h, r - 1
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
        r = r0 - 1
    return grid, bars


def _render_legend(grid: np.ndarray, bars: list[dict], color_map: dict[int, int]) -> None:
    for rr, bar in enumerate(bars):
        grid[int(rr), 0] = int(bar["fill"])
        grid[int(rr), 1] = int(color_map[int(bar["fill"])])


def _make_sprite(outer_idx: int, bars: list[dict], force_inner: bool = False) -> dict:
    outer_fill = int(bars[outer_idx]["fill"])
    min_h = max(SPRITE_H_RANGE[0], 3 if force_inner else SPRITE_H_RANGE[0])
    min_w = max(SPRITE_W_RANGE[0], 3 if force_inner else SPRITE_W_RANGE[0])
    h = random.randint(min_h, min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"])))
    w = random.randint(min_w, SPRITE_W_RANGE[1])
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    can_have_inner = len(bars) >= 2 and h >= 3 and w >= 3
    want_inner = can_have_inner and (force_inner or random.random() < INNER_COLOR_PROB)
    if want_inner:
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
        "inner_idx": inner_idx,
        "patch": patch,
        "h": int(h),
        "w": int(w),
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
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    legend_rect = (0, 0, len(bars) - 1, LEGEND_COLS - 1)
    top_rects: list[tuple[int, int, int, int]] = [legend_rect]
    placed = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx = int(sp["outer_idx"])
        inner_idx = sp["inner_idx"]
        candidates = []
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


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int, color_map: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[int(sp["outer_idx"])]
        h, c0 = int(sp["h"]), int(sp["c0"])

        paste(inp, patch, int(sp["r0"]), c0, transparent=int(bg))

        transformed = _apply_color_map(patch, color_map)
        paste(out, transformed, int(outer_bar["r_max"] - h + 1), c0, transparent=int(bg))

        if sp["inner_idx"] is not None:
            _, ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[int(sp["inner_idx"])]
            inner_fill = int(inner_bar["fill"])
            inner_patch = np.full((int(ih), int(iw)), int(color_map[inner_fill]), dtype=int)
            paste(out, inner_patch, int(inner_bar["r_max"] - ih + 1), c0 + int(ic0))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = n_bars

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    shifted_borders = borders[1:] + borders[:1]
    color_map = {int(fill): int(dst) for fill, dst in zip(fills, shifted_borders)}
    _render_legend(base, bars, color_map)

    outer_order = list(range(n_bars))
    random.shuffle(outer_order)
    force_inner_idx = random.choice(outer_order)
    sprites = [_make_sprite(idx, bars, force_inner=(idx == force_inner_idx)) for idx in outer_order]
    sprites.sort(key=lambda s: (s["inner_idx"] is None, -int(s["w"]), -int(s["h"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg, color_map)

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
