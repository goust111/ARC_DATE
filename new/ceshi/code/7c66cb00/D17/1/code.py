# Variant of 7c66cb00
# Dimension: D17 — 符号编码：隐式变换参数 (symbol_instructions)
# Change: 无显式符号编码，变换完全由颜色匹配规则隐式确定
#   → 条高度编码颜色变换模式：偶数高条执行原始换色，奇数高条保持原色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the D17 variant
# ----------------------------------
# a) Concrete behavior and trace
# - The scene layout stays the same: 2-3 full-width horizontal bars in the lower
#   portion and 2-3 sprites above them.
# - Matching stays the same: a sprite still docks into the bar whose fill color
#   matches the sprite's outer color.
# - The new implicit instruction is bar-height parity:
#   even height  -> perform the original recolor swap
#   odd height   -> keep the sprite colors unchanged
# - Example trace:
#   sprite = outer red, inner blue; target bar = fill red, border yellow.
#   If the target bar height is 6, the docked sprite becomes yellow/red.
#   If the target bar height is 5, the docked sprite stays red/blue.
#   Because red equals the bar fill, the outer shell blends into the bar and the
#   blue inner region remains visible. This makes the odd/even distinction
#   observable in the output rather than purely semantic.
#
# b) Complexity calibration
# - The generator always includes both parity branches in each sample: at least
#   one sprite targets an even-height bar and at least one sprite targets an
#   odd-height bar.
# - Every odd-targeted sprite is forced to be two-tone, so "keep original color"
#   is visibly different from the original always-swap task.
#
# c) Audit of the original solver assumptions
# - Bar detection only assumes horizontally wide near-solid rows; this remains
#   true.
# - Pattern extraction assumes top objects are still color components whose bboxes
#   contain the relevant two-tone content; this remains true.
# - The only semantic dependency that breaks is the unconditional recolor inside
#   the placement loop. That logic is now guarded by bar-height parity.
# - The original solver's per-color connected-component behavior is preserved, so
#   inner monochrome subregions still dock independently into their own bars.
#
# d) Placement legality
# - The generator keeps the original occupancy discipline:
#   outer sprite width reserves space in its outer bar, and any inner subregion
#   reserves space in the inner-color bar as well. This prevents overlaps among
#   all independently docked pieces that the solver may produce.
#
# e/f/g) Observability and cleanliness
# - Samples are rejected unless mixed-parity bars are present and the parity-aware
#   output differs from the original always-swap output on the same input.
# - Border colors are chosen distinct from all fill colors, so even-height swaps
#   are visually crisp.
# - Odd-height bars never rely on invisible monochrome sprites; they always get a
#   visible inner motif.
#
# New solver assumptions introduced by the variant
# - At least one bar exists and each bar is still a full-width strip with a
#   dominant fill color and edge border color.
# - Each used odd-height bar should receive a two-tone outer sprite for the D17
#   signal to be human-visible; the generator enforces this.


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    """Detect horizontal bars exactly as in the original solver."""
    H, W = grid.shape
    bars: list[dict] = []
    current_bar: dict | None = None

    for r in range(H):
        row = grid[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
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

    for bar in bars:
        bar["h"] = int(bar["r_max"] - bar["r_min"] + 1)
        bar["swap"] = bool(bar["h"] % 2 == 0)
    return bars


def _extract_patterns(grid: np.ndarray, pattern_area_end: int, bg: int) -> list[dict]:
    """Reproduce the original top-area pattern extraction behavior."""
    patterns: list[dict] = []
    top = grid[:pattern_area_end, :]
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
            bbox = crop_bbox(grid, (r0, c0, r1, c1))

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
                }
            )
    return patterns


def _transform_patch_for_bar(
    patch: np.ndarray,
    outer_color: int,
    inner_color: int | None,
    bar: dict,
    *,
    respect_parity: bool,
) -> np.ndarray:
    """Apply the bar-controlled color rule to one docked patch."""
    transformed = np.array(patch, dtype=int).copy()
    if respect_parity and not bool(bar["swap"]):
        return transformed

    transformed[transformed == int(outer_color)] = int(bar["border"])
    if inner_color is not None:
        transformed[transformed == int(inner_color)] = int(bar["fill"])
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock color-matched sprites into bars, with the target bar's height parity
    implicitly choosing whether recoloring happens.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    bars = _find_bars(g, bg)
    assert bars, "No bars found — grid must contain at least one horizontal bar"

    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    patterns = _extract_patterns(g, pattern_area_end, bg)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for bar in bars:
        matching = [pat for pat in patterns if int(pat["outer_color"]) == int(bar["fill"])]
        for pat in matching:
            transformed = _transform_patch_for_bar(
                pat["content"],
                int(pat["outer_color"]),
                pat["inner_color"],
                bar,
                respect_parity=True,
            )
            place_r = int(bar["r_max"] - pat["h"] + 1)
            paste(out, transformed, place_r, int(pat["c0"]), transparent=int(bg))

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
INNER_COLOR_PROB = 0.45
FORCED_INNER_MIN_AREA = 3
GEN_RETRIES = 400


def _sample_bar_heights(n_bars: int) -> list[int]:
    """Sample heights with enough room and with both parities represented."""
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        if len({h % 2 for h in heights}) < 2:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Render horizontal bars bottom-up and annotate parity-controlled behavior."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
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
            "swap": bool(h % 2 == 0),
        }
        r = r0 - 1
    return grid, bars


def _choose_inner_bbox(h: int, w: int, *, min_area: int) -> tuple[int, int, int, int] | None:
    """Pick a minority inner rectangle centered horizontally."""
    valid = [
        (ih, iw)
        for ih in range(1, h - 1)
        for iw in range(1, w - 1)
        if (w - iw) % 2 == 0 and ih * iw * 2 < h * w and ih * iw >= min_area
    ]
    if not valid:
        return None
    ih, iw = random.choice(valid)
    ir0 = random.randint(1, h - ih - 1)
    ic0 = (w - iw) // 2
    return int(ir0), int(ic0), int(ih), int(iw)


def _make_sprite(outer_idx: int, bars: list[dict], *, force_inner: bool) -> dict:
    """Create one sprite whose outer color matches its target bar."""
    outer_fill = int(bars[outer_idx]["fill"])

    for _ in range(50):
        h_min = max(SPRITE_H_RANGE[0], 3 if force_inner else SPRITE_H_RANGE[0])
        h_max = min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"]))
        w_min = max(SPRITE_W_RANGE[0], 3 if force_inner else SPRITE_W_RANGE[0])
        w_max = SPRITE_W_RANGE[1]
        if h_min > h_max or w_min > w_max:
            continue

        h = random.randint(h_min, h_max)
        w = random.randint(w_min, w_max)
        patch = np.full((h, w), outer_fill, dtype=int)

        want_inner = force_inner or (
            len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB
        )
        inner_idx = None
        inner_bbox = None

        if want_inner and len(bars) >= 2 and h >= 3 and w >= 3:
            inner_choices = [i for i in range(len(bars)) if i != outer_idx]
            inner_idx = random.choice(inner_choices)
            min_area = FORCED_INNER_MIN_AREA if force_inner else 1
            inner_bbox = _choose_inner_bbox(h, w, min_area=min_area)
            if inner_bbox is None:
                if force_inner:
                    continue
                inner_idx = None
            else:
                ir0, ic0, ih, iw = inner_bbox
                patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[inner_idx]["fill"])

        return {
            "outer_idx": int(outer_idx),
            "inner_idx": inner_idx,
            "patch": patch,
            "h": int(h),
            "w": int(w),
            "inner_bbox": inner_bbox,
        }

    raise RuntimeError("failed to create a sprite satisfying parity-visibility constraints")


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[start : start + length].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    """Require at least one background cell gap between top sprites."""
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    """Find valid non-overlapping placements for all top sprites."""
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
                    candidates.append((r0, c0))

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


def _render_output(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int, *, respect_parity: bool) -> np.ndarray:
    """Render the expected output, optionally forcing the old always-swap rule."""
    out = base.copy()
    for sp in placed:
        patch = np.array(sp["patch"], dtype=int).copy()
        outer_bar = bars[int(sp["outer_idx"])]
        h = int(sp["h"])
        c0 = int(sp["c0"])

        transformed = _transform_patch_for_bar(
            patch,
            int(outer_bar["fill"]),
            int(bars[int(sp["inner_idx"])]["fill"]) if sp["inner_idx"] is not None else None,
            outer_bar,
            respect_parity=respect_parity,
        )
        paste(out, transformed, int(outer_bar["r_max"] - h + 1), c0, transparent=int(bg))

        if sp["inner_idx"] is not None:
            _, ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[int(sp["inner_idx"])]
            inner_color = int(inner_bar["border"]) if (not respect_parity or inner_bar["swap"]) else int(inner_bar["fill"])
            inner_patch = np.full((int(ih), int(iw)), inner_color, dtype=int)
            paste(out, inner_patch, int(inner_bar["r_max"] - ih + 1), int(c0 + ic0))

    return out


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render input plus both parity-aware and legacy always-swap outputs."""
    inp = base.copy()
    for sp in placed:
        paste(inp, sp["patch"], int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

    parity_out = _render_output(base, placed, bars, bg, respect_parity=True)
    legacy_out = _render_output(base, placed, bars, bg, respect_parity=False)
    return inp, parity_out, legacy_out


def _assign_outer_targets(bars: list[dict], n_sprites: int) -> list[int]:
    """Guarantee that both parity-controlled branches are exercised."""
    even_targets = [i for i, bar in enumerate(bars) if bool(bar["swap"])]
    odd_targets = [i for i, bar in enumerate(bars) if not bool(bar["swap"])]
    assert even_targets and odd_targets

    counts = [0] * len(bars)
    assigned = [random.choice(even_targets), random.choice(odd_targets)]
    counts[assigned[0]] += 1
    counts[assigned[1]] += 1

    while len(assigned) < n_sprites:
        choices = [i for i in range(len(bars)) if counts[i] < 2]
        idx = random.choice(choices)
        assigned.append(idx)
        counts[idx] += 1

    random.shuffle(assigned)
    return assigned


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt."""
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(c) for c in palette[1 : 1 + n_bars]]
    borders = [int(c) for c in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    outer_targets = _assign_outer_targets(bars, n_sprites)
    sprites = [
        _make_sprite(
            outer_idx,
            bars,
            force_inner=not bool(bars[outer_idx]["swap"]),
        )
        for outer_idx in outer_targets
    ]
    sprites.sort(key=lambda sp: (sp["inner_idx"] is None, -int(sp["w"]), -int(sp["h"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    used_parities = {int(bars[int(sp["outer_idx"])]["h"]) % 2 for sp in placed}
    if used_parities != {0, 1}:
        return None
    if any((not bool(bars[int(sp["outer_idx"])]["swap"])) and sp["inner_idx"] is None for sp in placed):
        return None

    inp, out, legacy_out = _render(base, placed, bars, bg)
    if np.array_equal(out, legacy_out):
        return None
    if get_bg(inp) != bg:
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
