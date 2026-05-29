# Variant of 7c66cb00
# Dimension: D17 — 符号编码：隐式变换参数 (symbol_instructions)
# Change: 无显式符号编码，变换完全由颜色匹配规则隐式确定... → 条边框上的标记像素指示精灵翻转

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# ---------------------------------------------------------------------------
# Design analysis for the D17 variant
#
# a) Concretization and behavior trace
# - Original bars already used a border/fill dichotomy. The D17 change reuses
#   that structure and embeds one extra marker pixel into each bar's border:
#   exactly one shared marker color appears on either col 0 or col W-1.
# - If the marker sits on the left border, every sprite matched to that bar is
#   horizontally mirrored before docking. If it sits on the right border, the
#   sprite keeps its orientation.
# - The original recolor rule stays intact after the optional flip:
#   sprite outer -> bar border, sprite inner -> bar fill.
# - Concrete trace:
#   input sprite = a fill-A rectangle with a smaller fill-B block shifted left.
#   Original task: dock it into bar A as-is, recolored.
#   Variant: if bar A has the marker on col 0, first mirror the patch so the
#   fill-B block shifts to the right, then recolor and dock. If the marker is
#   on col W-1, the block stays on the left.
#
# b) Complexity calibration
# - A single marker pixel is enough to add symbolic decoding, but only if the
#   flip is visually observable. To avoid an invisible change, every generated
#   full sprite is deliberately horizontally asymmetric via an off-center inner
#   block. This is the minimal non-D17 adaptation needed for visibility.
# - Each generated instance gives every bar at least one full sprite and uses a
#   mixed set of left/right markers, so both instruction branches are repeated
#   within one puzzle instead of relying on lucky cross-example coverage.
#
# c) Hidden-assumption audit against the original solver
# - Old assumption: `row[0]` or `row[-1]` directly reveals the bar border.
#   Broken now because one edge pixel can be the marker color.
#   Fix: detect bars by fill-majority as before, then infer the border from the
#   dominant color across both edge columns over the whole bar height.
# - Old assumption: sprite orientation does not matter because the generated
#   shapes are left-right symmetric. Broken now because D17 needs visible flip.
#   Fix: generator always creates a left-biased inner block.
# - Old assumption: some bars may have no matched full sprite and that is fine.
#   For D17 this weakens the symbol signal because unused markers teach nothing.
#   Fix: generator creates exactly one full sprite per bar.
# - Old assumption: recolor can be implemented by sequential whole-array
#   replacements. Unsafe here because a bar border color may equal some other
#   bar's fill color; overwriting one color first can corrupt the second rule.
#   Fix: recolor uses masks from the pre-recolor, already-flipped patch.
#
# d) Placement legality
# - Full sprites still live entirely in cols 1..W-2, so embedded markers on the
#   border are never overwritten.
# - Output overlap is controlled exactly as in the original generator: full
#   sprites reserve their full x-interval in the matched bar, and inner-color
#   subcomponents reserve their own narrower interval in their matched bar.
# - Heights are clamped so every full sprite fits inside its own target bar and
#   every inner subcomponent fits inside its secondary target bar.
#
# e) Visual observability
# - The only new symbol is the marker pixel color and its left/right placement.
# - All input full sprites share the same "left-biased" look; therefore the
#   output orientation difference is easy to compare across bars:
#   left marker -> right-biased output, right marker -> left-biased output.
# - The marker color is shared across bars and unused elsewhere, so humans can
#   isolate it as an instruction rather than a normal bar color.
#
# f) Visual cleanliness
# - Markers are only one pixel each and live on border columns, away from the
#   moving sprites. No extra overlay layer is added inside the docking region.
# - Full sprites remain simple rectangles with one off-center inner block, so
#   the before/after relation stays legible.
#
# g) Color-change inferability
# - Colors still follow the original structural cue: outer color chooses the
#   target bar via fill-color match, then recolors to that bar's border/fill.
# - The new branch is not color-coded by numeric IDs; it is encoded only by the
#   visible marker position on the border.
#
# h) Solver assumptions introduced by the variant
# - Each bar must contain exactly one edge pixel whose color differs from the
#   inferred border; that pixel is the marker. The generator guarantees this.
# - Full sprites must remain horizontally asymmetric often enough for flip to be
#   visible; the generator guarantees this by construction, not by color values.
# - Top-area non-background components are still separated by at least one cell,
#   so the original bbox-based pattern extraction remains valid.
# ---------------------------------------------------------------------------


def _find_bars_with_markers(g: np.ndarray, bg: int) -> list[dict]:
    """Detect full-width horizontal bars and decode their embedded flip marker."""
    h, w = g.shape
    bars: list[dict] = []
    current: dict | None = None

    for r in range(h):
        row = g[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(w * 0.7):
                if current and int(current["fill"]) == int(fill_color):
                    current["r_max"] = int(r)
                else:
                    if current:
                        bars.append(current)
                    current = {"fill": int(fill_color), "r_min": int(r), "r_max": int(r)}
            else:
                if current:
                    bars.append(current)
                    current = None
        else:
            if current:
                bars.append(current)
                current = None
    if current:
        bars.append(current)

    assert bars, "No bars found"

    for bar in bars:
        r0 = int(bar["r_min"])
        r1 = int(bar["r_max"])
        left_vals = [int(g[r, 0]) for r in range(r0, r1 + 1)]
        right_vals = [int(g[r, w - 1]) for r in range(r0, r1 + 1)]
        border = int(Counter(left_vals + right_vals).most_common(1)[0][0])
        left_marker_rows = [r for r in range(r0, r1 + 1) if int(g[r, 0]) != border]
        right_marker_rows = [r for r in range(r0, r1 + 1) if int(g[r, w - 1]) != border]

        assert len(left_marker_rows) + len(right_marker_rows) == 1, "Each bar must have exactly one marker pixel"
        assert not (left_marker_rows and right_marker_rows), "Marker cannot appear on both sides of one bar"

        if left_marker_rows:
            marker_side = "left"
            marker_r = int(left_marker_rows[0])
            marker_c = 0
            flip = True
        else:
            marker_side = "right"
            marker_r = int(right_marker_rows[0])
            marker_c = w - 1
            flip = False

        bar["border"] = border
        bar["flip"] = flip
        bar["marker_side"] = marker_side
        bar["marker_pos"] = (marker_r, marker_c)

    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched horizontal bars with a color swap.

    D17 variant:
    - Each bar carries one special marker pixel on its border.
    - Left-border marker => matched sprite is horizontally flipped first.
    - Right-border marker => matched sprite keeps its orientation.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars_with_markers(g, bg)
    pattern_area_end = int(min(int(b["r_min"]) for b in bars))

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
            other_colors = [int(c) for c in colors_in_bbox if int(c) != int(outer_color)]
            inner_color = int(other_colors[0]) if other_colors else None

            patterns.append(
                {
                    "outer_color": outer_color,
                    "inner_color": inner_color,
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "content": bbox,
                }
            )

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for bar in bars:
        matching = [p for p in patterns if int(p["outer_color"]) == int(bar["fill"])]
        for pat in matching:
            transformed = np.fliplr(pat["content"]) if bool(bar["flip"]) else np.array(pat["content"], dtype=int).copy()
            original = transformed.copy()
            transformed[original == int(pat["outer_color"])] = int(bar["border"])
            if pat["inner_color"] is not None:
                transformed[original == int(pat["inner_color"])] = int(bar["fill"])

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
SPRITE_H_RANGE = (3, 5)
SPRITE_W_RANGE = (4, 9)
GEN_RETRIES = 300


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _choose_flips(n_bars: int) -> list[bool]:
    while True:
        flips = [bool(random.getrandbits(1)) for _ in range(n_bars)]
        if any(flips) and not all(flips):
            return flips


def _build_bars(
    bg: int, fills: list[int], borders: list[int], heights: list[int], marker_color: int, flips: list[bool]
) -> tuple[np.ndarray, list[dict]]:
    """Render horizontal bars bottom-up and embed one marker pixel per bar."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = r - h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])

        marker_r = random.randint(r0, r1)
        marker_c = 0 if bool(flips[i]) else GRID_W - 1
        grid[marker_r, marker_c] = int(marker_color)

        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
            "flip": bool(flips[i]),
            "marker_pos": (int(marker_r), int(marker_c)),
        }
        r = r0 - 1
    return grid, bars


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    """
    Create a left-biased two-tone rectangle.

    The inner block is always shifted toward the left half so a horizontal flip
    changes the visible arrangement inside the bbox.
    """
    outer_fill = int(bars[outer_idx]["fill"])
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"])))
    w = random.randint(*SPRITE_W_RANGE)

    other_bar_indices = [i for i in range(len(bars)) if i != outer_idx]
    inner_idx = random.choice(other_bar_indices)
    inner_fill = int(bars[inner_idx]["fill"])

    patch = np.full((h, w), outer_fill, dtype=int)

    valid_shapes: list[tuple[int, int, int, int]] = []
    max_ih = max(1, h - 2)
    max_iw = max(1, w - 3)
    for ih in range(1, max_ih + 1):
        for iw in range(1, max_iw + 1):
            if ih * iw * 2 >= h * w:
                continue
            for ir0 in range(1, h - ih):
                for ic0 in range(1, w - iw):
                    left_margin = ic0
                    right_margin = w - (ic0 + iw)
                    if left_margin >= right_margin:
                        continue
                    valid_shapes.append((ir0, ic0, ih, iw))

    assert valid_shapes, "Expected at least one asymmetric inner placement"
    ir0, ic0, ih, iw = random.choice(valid_shapes)
    patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = inner_fill

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": int(inner_idx),
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

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx = int(sp["outer_idx"])
        inner_idx = int(sp["inner_idx"])
        _ir0, ic0, _ih, iw = sp["inner_bbox"]

        candidates = []
        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_idx], c0, w):
                continue
            if not _interval_free(bar_occ[inner_idx], c0 + ic0, iw):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][c0 : c0 + w] = True
        bar_occ[inner_idx][c0 + ic0 : c0 + ic0 + iw] = True
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _recolor_patch(patch: np.ndarray, source_outer: int, source_inner: int | None, target_border: int, target_fill: int) -> np.ndarray:
    out = np.array(patch, dtype=int).copy()
    original = out.copy()
    out[original == int(source_outer)] = int(target_border)
    if source_inner is not None:
        out[original == int(source_inner)] = int(target_fill)
    return out


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[int(sp["outer_idx"])]
        inner_bar = bars[int(sp["inner_idx"])]
        h = int(sp["h"])
        c0 = int(sp["c0"])

        paste(inp, patch, int(sp["r0"]), c0, transparent=bg)

        full_patch = np.fliplr(patch) if bool(outer_bar["flip"]) else patch.copy()
        full_patch = _recolor_patch(full_patch, int(outer_bar["fill"]), int(inner_bar["fill"]), int(outer_bar["border"]), int(outer_bar["fill"]))
        paste(out, full_patch, int(outer_bar["r_max"]) - h + 1, c0, transparent=bg)

        ir0, ic0, ih, iw = sp["inner_bbox"]
        inner_patch = patch[ir0 : ir0 + ih, ic0 : ic0 + iw].copy()
        if bool(inner_bar["flip"]):
            inner_patch = np.fliplr(inner_patch)
        inner_patch = _recolor_patch(inner_patch, int(inner_bar["fill"]), None, int(inner_bar["border"]), int(inner_bar["fill"]))
        paste(out, inner_patch, int(inner_bar["r_max"]) - ih + 1, c0 + ic0, transparent=bg)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]
    marker_color = int(palette[1 + 2 * n_bars])
    flips = _choose_flips(n_bars)

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights, marker_color, flips)
    top_h = int(bars[0]["r_min"])

    sprites = [_make_sprite(i, bars) for i in range(n_bars)]
    sprites.sort(key=lambda s: (-int(s["w"]), -int(s["h"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

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
