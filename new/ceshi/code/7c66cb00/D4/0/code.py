# Variant of 7c66cb00
# Dimension: D4 — 目标对象：水平条结构 (dst)
# Change: 全宽水平条，底部堆叠2-3个，每个条4-8行高，由填充色（内部）和边框色（首尾两列）组成，条间间隔1行背景... → 垂直条替代水平条

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the D4 variant
# ----------------------------------
# Source-task review:
# - No separate source_task was provided, so the baseline is the original 7c66cb00
#   solver/generator only.
#
# a) Concretized variant and behavior trace
# - Old scene:
#   - 2-3 full-width horizontal bars live in the lower part of the grid.
#   - A sprite above them keeps its original x-position and is bottom-aligned into
#     the matched bar.
# - New scene:
#   - 2-3 full-height vertical bars live flush on the LEFT or RIGHT side.
#   - Each bar is 4-8 columns wide.
#   - Inside a bar, rows 1..H-2 are fill color; row 0 and row H-1 are border color.
#   - Sprites occupy the opposite side of the grid and keep their original r-position.
#   - They dock horizontally into the matched bar:
#     - bars on the right  -> sprite right-aligns to the bar's right edge
#     - bars on the left   -> sprite left-aligns to the bar's left edge
# - Example trace:
#   - Suppose bar A is on the right, fill=2, border=5, columns 18..22.
#   - A 3x4 sprite at rows 10..12, cols 2..5 has outer color 2 and inner color 8.
#   - Old logic would have searched a horizontal bar with fill 2 and pasted at
#     row = bar.r_max - 3 + 1, col = 2.
#   - New logic searches a vertical bar with fill 2 and pastes at
#     row = 10, col = 22 - 4 + 1 = 19.
#   - Recolor remains the same: outer 2 -> 5, inner 8 -> 2.
#
# b) Complexity calibration
# - The bar count stays 2-3, matching colors stay one-to-one by fill color, and bars
#   retain the same two-color structure. This keeps the core reasoning load similar.
# - The side of the bar cluster varies (left/right) so the 90-degree layout change is
#   visually obvious and not a trivial transpose of one fixed template.
# - Sprite detection still happens per monochrome component, so inner-color regions are
#   independently dockable exactly as in the original. That preserves the original task's
#   "hidden extra objects" complexity.
#
# c) Original main() assumptions that break under D4
# - Scanning rows for bars no longer works: bars are now column-structured.
# - `pattern_area_end = min(bar.r_min)` assumed sprites are above bars; now sprites are
#   on the opposite side, so the solver must crop by columns, not rows.
# - `place_r = bar.r_max - h + 1` assumed vertical docking; the variant needs horizontal
#   docking while preserving the sprite's row.
# - Clearing `out[:pattern_area_end, :]` assumed the sprite zone is the top strip; now the
#   solver must clear the left or right side strip instead.
# - The original "fill dominates the row" heuristic becomes "fill dominates the column".
#
# d) Placement legality guarantees
# - Generator guarantees full-height bars with 1-column gaps and one extra background
#   separator between the bar cluster and the sprite side.
# - Every sprite width is clamped to its target bar width, so horizontal docking fits.
# - Every sprite row interval stays within rows 1..H-2 so it fits inside bar fill rows.
# - For each target bar, sprite row intervals do not overlap; inner sub-sprite row
#   intervals are also reserved in their own target bars.
# - Input-side sprites are pairwise separated by at least one background cell.
#
# e) Visual observability
# - Humans can see that the destination objects are now vertical strips rather than
#   bottom horizontal slabs.
# - The motion cue changes from "drop downward" to "slide sideways", because the sprite's
#   row is preserved while its column changes.
# - Left/right variation is visible but not ambiguous: the bars always touch one outer
#   edge, and sprites always live on the opposite side.
#
# f) Visual cleanliness
# - Bars occupy one side cluster only; sprites occupy the other side only.
# - A dedicated background separator prevents the side cluster from visually merging with
#   the sprite field.
# - Limiting sprites to 2-3 and enforcing 1-cell separation keeps the side field legible.
#
# g) Color-change inferability
# - The color rule is unchanged from the original:
#   - outer color identifies the target bar because it equals the bar fill
#   - outer recolors to the matched bar border
#   - inner recolors to the matched bar fill
# - Because bars remain structured objects with stable fill/border roles, the color swap
#   is still easy to infer from a few examples.
#
# h) Quantitative generator target
# - A valid generated sample must contain:
#   - 2-3 full-height vertical bars on one side
#   - 2-3 sprites on the opposite side
#   - at least one horizontally dockable sprite whose preserved row is meaningful
# - The validation step later checks both success rate and that the original horizontal
#   solver does not solve these inputs as-is.
#
# i) New main() assumptions and how generate_input() satisfies them
# - Assumption: every bar column has fill as the majority color and border at top/bottom.
#   Generator renders bars exactly that way.
# - Assumption: the bar cluster touches either the left edge or the right edge.
#   Generator always anchors the cluster flush to one outer side.
# - Assumption: sprites lie entirely outside the bar cluster and do not span into it.
#   Generator restricts sprite placement to the opposite-side field.
# - Assumption: matched sprite width fits the target bar width.
#   Generator clamps sprite width by target bar width.


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_WIDTH_RANGE = (4, 8)
MIN_SIDE_AREA = 7
SPRITE_H_RANGE = (2, 9)
SPRITE_W_RANGE = (2, 5)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 300


def _find_vertical_bars(grid: np.ndarray, bg: int) -> tuple[list[dict], str]:
    """Detect full-height vertical bars and infer which side they occupy."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bars: list[dict] = []
    current_bar: dict | None = None

    for c in range(W):
        col = g[:, c]
        non_bg = [int(v) for v in col if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(H * 0.7):
                border = int(col[0]) if int(col[0]) != int(fill_color) else int(col[-1])
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["c_max"] = int(c)
                else:
                    if current_bar:
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "c_min": int(c),
                        "c_max": int(c),
                    }
            elif current_bar:
                bars.append(current_bar)
                current_bar = None
        elif current_bar:
            bars.append(current_bar)
            current_bar = None

    if current_bar:
        bars.append(current_bar)

    if not bars:
        raise AssertionError("No bars found — grid must contain full-height vertical bars")

    side = "left" if int(bars[0]["c_min"]) == 0 else "right"
    if side == "left":
        assert int(bars[-1]["c_max"]) < W - 1, "Left-side bars must leave sprite space on the right"
    else:
        assert int(bars[-1]["c_max"]) == W - 1, "Right-side bars must touch the right edge"

    for bar in bars:
        bar["w"] = int(bar["c_max"] - bar["c_min"] + 1)
        bar["r_min"] = 0
        bar["r_max"] = H - 1

    return bars, side


def _extract_patterns(g: np.ndarray, bg: int, bars: list[dict], side: str) -> tuple[list[dict], slice]:
    """Find per-color sprite components in the side area opposite the bars."""
    H, W = g.shape
    if side == "left":
        pattern_slice = slice(int(bars[-1]["c_max"]) + 1, W)
        col_offset = int(bars[-1]["c_max"]) + 1
    else:
        pattern_slice = slice(0, int(bars[0]["c_min"]))
        col_offset = 0

    region = g[:, pattern_slice]
    patterns: list[dict] = []

    for col in sorted(set(map(int, region.flatten())) - {int(bg)}):
        mask = region == int(col)
        if not bool(mask.any()):
            continue
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = map(int, bbox_coords(coords))
            bbox = crop_bbox(region, (r0, c0, r1, c1))
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
                    "c0": int(c0 + col_offset),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )

    return patterns, pattern_slice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched vertical bars with a color swap.

    Input structure:
    - bg = most common color.
    - One side contains 2-3 full-height vertical bars.
    - The opposite side contains 2-3 small sprites made from bar fill colors.

    Rule:
    - Match each sprite to the vertical bar whose fill color equals the sprite's outer color.
    - Preserve the sprite's row and dock it horizontally into the target bar.
    - Recolor: sprite outer -> matched bar border; sprite inner -> matched bar fill.
    - Erase the original side-area sprites.

    As in the original task, per-color connected components are treated as independent
    dockable patterns, so inner-color regions also move to their own matching bars.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    bars, side = _find_vertical_bars(g, bg)
    patterns, pattern_slice = _extract_patterns(g, bg, bars, side)

    out[:, pattern_slice] = np.where(out[:, pattern_slice] != int(bg), int(bg), out[:, pattern_slice])

    for bar in bars:
        matching = [p for p in patterns if int(p["outer_color"]) == int(bar["fill"])]
        for pat in matching:
            transformed = np.array(pat["content"], dtype=int).copy()
            for rr in range(int(pat["h"])):
                for cc in range(int(pat["w"])):
                    v = int(transformed[rr, cc])
                    if v == int(pat["outer_color"]):
                        transformed[rr, cc] = int(bar["border"])
                    elif pat["inner_color"] is not None and v == int(pat["inner_color"]):
                        transformed[rr, cc] = int(bar["fill"])

            place_r = int(pat["r0"])
            if side == "left":
                place_c = int(bar["c_min"])
            else:
                place_c = int(bar["c_max"] - pat["w"] + 1)
            paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


def _sample_bar_widths(n_bars: int) -> list[int]:
    """Sample widths that leave room for a separator and a sprite field."""
    while True:
        widths = [random.randint(*BAR_WIDTH_RANGE) for _ in range(n_bars)]
        used = sum(widths) + (n_bars - 1) + 1
        if GRID_W - used < MIN_SIDE_AREA:
            continue
        return widths


def _build_bars(
    bg: int,
    fills: list[int],
    borders: list[int],
    widths: list[int],
    side: str,
) -> tuple[np.ndarray, list[dict], tuple[int, int]]:
    """Render side-anchored full-height vertical bars and return sprite-area bounds."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = []

    if side == "left":
        c = 0
        for fill, border, width in zip(fills, borders, widths):
            c0, c1 = c, c + width - 1
            grid[1:GRID_H - 1, c0:c1 + 1] = fill
            grid[0, c0:c1 + 1] = border
            grid[GRID_H - 1, c0:c1 + 1] = border
            bars.append({"fill": fill, "border": border, "c_min": c0, "c_max": c1, "w": width})
            c = c1 + 2
        sprite_bounds = (c, GRID_W - 1)
    else:
        c = GRID_W - 1
        for fill, border, width in reversed(list(zip(fills, borders, widths))):
            c1, c0 = c, c - width + 1
            grid[1:GRID_H - 1, c0:c1 + 1] = fill
            grid[0, c0:c1 + 1] = border
            grid[GRID_H - 1, c0:c1 + 1] = border
            bars.append({"fill": fill, "border": border, "c_min": c0, "c_max": c1, "w": width})
            c = c0 - 2
        bars.sort(key=lambda b: int(b["c_min"]))
        sprite_bounds = (0, c)

    for bar in bars:
        bar["r_min"] = 0
        bar["r_max"] = GRID_H - 1

    return grid, bars, sprite_bounds


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    """Create a side-area sprite whose width fits its target bar."""
    outer_fill = int(bars[outer_idx]["fill"])
    max_w = min(SPRITE_W_RANGE[1], int(bars[outer_idx]["w"]))
    w = random.randint(SPRITE_W_RANGE[0], max_w)
    h = random.randint(*SPRITE_H_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        inner_idx = random.choice(inner_choices)
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, w - 1)
            if (h - ih) % 2 == 0 and ih * iw * 2 < h * w
        ]
        if valid_inner:
            ih, iw = random.choice(valid_inner)
            ir0 = (h - ih) // 2
            ic0 = random.randint(1, w - iw - 1)
            patch[ir0:ir0 + ih, ic0:ic0 + iw] = int(bars[inner_idx]["fill"])
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
    return not occ[start:start + length].any()


def _rect_separated(
    rects: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(
    sprites: list[dict],
    bars: list[dict],
    sprite_bounds: tuple[int, int],
) -> list[dict] | None:
    """Place sprites in the side field while reserving row intervals in target bars."""
    sprite_c_min, sprite_c_max = map(int, sprite_bounds)
    bar_occ = [np.zeros(GRID_H, dtype=bool) for _ in bars]
    side_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx, inner_idx = int(sp["outer_idx"]), sp["inner_idx"]

        candidates = []
        for r0 in range(1, GRID_H - h):
            if not _interval_free(bar_occ[outer_idx], r0, h):
                continue
            if inner_idx is not None:
                ir0, _ic0, ih, _iw = sp["inner_bbox"]
                if not _interval_free(bar_occ[int(inner_idx)], r0 + ir0, ih):
                    continue
            for c0 in range(sprite_c_min, sprite_c_max - w + 2):
                if _rect_separated(side_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][r0:r0 + h] = True
        if inner_idx is not None:
            ir0, _ic0, ih, _iw = sp["inner_bbox"]
            bar_occ[int(inner_idx)][r0 + ir0:r0 + ir0 + ih] = True

        placed.append({**sp, "r0": r0, "c0": c0})
        side_rects.append((r0, c0, r0 + h - 1, c0 + w - 1))

    return placed


def _render(
    base: np.ndarray,
    placed: list[dict],
    bars: list[dict],
    bg: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Render input sprites in the side field and output sprites docked into bars."""
    inp, out = base.copy(), base.copy()

    for sp in placed:
        patch = sp["patch"].copy()
        outer_bar = bars[int(sp["outer_idx"])]
        h, w = int(sp["h"]), int(sp["w"])
        r0, c0 = int(sp["r0"]), int(sp["c0"])

        paste(inp, patch, r0, c0, transparent=bg)

        transformed = patch.copy()
        transformed[transformed == int(outer_bar["fill"])] = int(outer_bar["border"])
        if sp["inner_idx"] is not None:
            inner_bar = bars[int(sp["inner_idx"])]
            transformed[transformed == int(inner_bar["fill"])] = int(outer_bar["fill"])

        place_c = int(outer_bar["c_min"]) if side == "left" else int(outer_bar["c_max"] - w + 1)
        paste(out, transformed, r0, place_c, transparent=bg)

        if sp["inner_idx"] is not None:
            ir0, _ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[int(sp["inner_idx"])]
            inner_patch = np.full((ih, iw), int(inner_bar["border"]), dtype=int)
            inner_c = int(inner_bar["c_min"]) if side == "left" else int(inner_bar["c_max"] - iw + 1)
            paste(out, inner_patch, r0 + ir0, inner_c)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt."""
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)
    side = random.choice(["left", "right"])

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    fills = palette[1:1 + n_bars]
    borders = palette[1 + n_bars:1 + 2 * n_bars]

    widths = _sample_bar_widths(n_bars)
    base, bars, sprite_bounds = _build_bars(bg, fills, borders, widths, side)

    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)
    sprites = [_make_sprite(outer_slots[i], bars) for i in range(n_sprites)]
    sprites.sort(key=lambda s: (s["inner_idx"] is None, -s["h"], -s["w"]))

    placed = _place_sprites(sprites, bars, sprite_bounds)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg, side)

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
