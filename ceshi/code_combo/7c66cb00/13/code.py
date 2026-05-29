# Combo variant of 7c66cb00 (index 13)
# combo_spec: D3/2+D7/0+D16/1
# Name: 内层色编码x位置有洞精灵顶部对齐

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original 7c66cb00 keeps each sprite's x and bottom-aligns it in the matching
#    bar. Here the sprite still matches by outer color, but two stages change:
#    first, the patch keeps an interior bg hole; second, the holed inner accent's
#    horizontal position controls the destination x, so the same sprite can move
#    left/center/right even when its input x is unrelated. The pasted result is
#    top-aligned, and because the hole remains transparent the bar fill shows
#    through it.
# b) Complexity calibration:
#    Purely continuous x mapping is hard to infer from 2-4 ARC examples, so the
#    generator uses three repeated inner-accent bands (left/center/right) with a
#    punched hole centered inside each band. The solver computes x from the inner
#    accent's proportional horizontal centroid, while the visible hole reinforces
#    the same cue for the human solver.
# c) Implicit-assumption audit:
#    The original solver scanned top objects per color, which can split a two-tone
#    sprite and completely loses hole geometry. The new solver instead labels whole
#    foreground components in the top area, then reads full bboxes. It also drops
#    the old assumption that output x == input x and explicitly recovers the inner
#    accent color before decoding the target column.
# d) Placement legality:
#    The generator creates exactly one sprite per target bar, so decoded x values
#    cannot overlap inside a bar. Top-area rectangles are chosen with a 1-cell gap,
#    stay within cols 1..W-2, and are forced to differ from the decoded output x.
# e) Visual observability:
#    Every sprite has three cues at once: outer color names the bar, inner color
#    marks it as two-tone, and a single bg hole inside the inner region visibly
#    picks the left/center/right output band. The output keeps the hole, making the
#    "window" effect directly observable.
# f) Cleanliness / new assumptions:
#    The solver now assumes each top sprite has at least one interior bg hole and
#    that bars still occupy nearly whole rows. The generator guarantees those facts
#    by using rectangular ring-like sprites with a punched interior cell and the
#    original full-width bar layout.


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 8
SPRITE_H_RANGE = (3, 5)
SPRITE_W_RANGE = (7, 9)
GEN_RETRIES = 400


def _detect_bars(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    bars: list[dict] = []
    current: dict | None = None
    for r in range(H):
        row = g[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if not non_bg:
            if current is not None:
                bars.append(current)
                current = None
            continue

        fill, count = Counter(non_bg).most_common(1)[0]
        if int(count) <= int(W * 0.7):
            if current is not None:
                bars.append(current)
                current = None
            continue

        border = int(row[0]) if int(row[0]) != int(fill) else int(row[-1])
        if current is not None and int(current["fill"]) == int(fill):
            current["r_max"] = int(r)
        else:
            if current is not None:
                bars.append(current)
            current = {"fill": int(fill), "border": int(border), "r_min": int(r), "r_max": int(r)}

    if current is not None:
        bars.append(current)
    return bars


def _dominant_outer_color(patch: np.ndarray, bg: int) -> int | None:
    colors = Counter(int(v) for v in patch.flatten() if int(v) != int(bg))
    if not colors:
        return None
    return int(colors.most_common(1)[0][0])


def _inner_rel_col(patch: np.ndarray, bg: int, outer_color: int | None = None) -> float:
    if outer_color is None:
        outer_color = _dominant_outer_color(patch, bg)
    if outer_color is None:
        return 0.5

    colors = Counter(int(v) for v in patch.flatten() if int(v) != int(bg))
    inner_colors = [(count, color) for color, count in colors.items() if int(color) != int(outer_color)]
    if inner_colors:
        _, inner_color = min(inner_colors)
        inner_coords = np.argwhere(patch == int(inner_color))
        if inner_coords.size > 0:
            denom = max(1, int(patch.shape[1]) - 1)
            return float(np.mean(inner_coords[:, 1]) / denom)

    h, w = patch.shape
    if h <= 2 or w <= 2:
        return 0.5
    interior = patch[1 : h - 1, 1 : w - 1]
    hole_coords = np.argwhere(interior == int(bg))
    if hole_coords.size == 0:
        return 0.5
    hole_cols = hole_coords[:, 1] + 1
    denom = max(1, w - 1)
    return float(np.mean(hole_cols) / denom)


def _encoded_target_col(patch: np.ndarray, bg: int, grid_w: int, outer_color: int | None = None) -> int:
    w = int(patch.shape[1])
    left_span = max(0, int(grid_w - 2 - w))
    rel = min(1.0, max(0.0, _inner_rel_col(patch, bg, outer_color)))
    return int(1 + round(rel * left_span))


def _extract_sprites(top: np.ndarray, bg: int) -> list[dict]:
    mask = top != int(bg)
    lab, n = label_mask(mask, connectivity=4)
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
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "content": bbox,
                "outer_color": outer_color,
            }
        )
    return sprites


def _transform_patch(patch: np.ndarray, outer_color: int, bar_fill: int, bar_border: int, bg: int) -> np.ndarray:
    transformed = np.array(patch, dtype=int).copy()
    transformed[transformed == int(outer_color)] = int(bar_border)
    transformed[(transformed != int(bg)) & (transformed != int(bar_border))] = int(bar_fill)
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock holed two-tone sprites into their color-matched bars, with x decoded from the inner accent.

    Input structure:
    - The background is the most common color.
    - The lower region contains 2-3 full-width horizontal bars with a fill color and edge border color.
    - Above the bars sit small connected sprites whose dominant outer color matches one bar fill; each sprite
      also contains an inner accent and at least one bg hole inside its bbox.

    Rule:
    - Remove the source sprites from the top area.
    - For each sprite, find the bar whose fill color equals the sprite's outer color.
    - Recolor outer pixels to that bar's border color and recolor the inner non-bg pixels to the bar fill.
    - Paste the recolored sprite top-aligned in the matched bar, with destination x computed from the inner
      accent's horizontal centroid rather than from the sprite's original x; keep the hole transparent.

    Key insight:
    - The inner accent is the actual x-code, and the hole is a visible window punched inside that accent. The
      sprite still chooses its bar by outer color, but its left/center/right landing position is inferred from
      the inner structure rather than the source x.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    bars = _detect_bars(g, bg)
    assert bars, "No bars found"
    top_end = int(min(int(bar["r_min"]) for bar in bars))
    sprites = _extract_sprites(g[:top_end, :], bg)
    bar_by_fill = {int(bar["fill"]): bar for bar in bars}

    out[:top_end, :] = int(bg)

    for sprite in sprites:
        outer = int(sprite["outer_color"])
        if outer not in bar_by_fill:
            continue
        bar = bar_by_fill[outer]
        patch = np.array(sprite["content"], dtype=int)
        transformed = _transform_patch(patch, outer, int(bar["fill"]), int(bar["border"]), bg)
        place_r = int(bar["r_min"])
        place_c = _encoded_target_col(patch, bg, W, outer)
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
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[list-item]
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


def _slot_hole_col(slot: int, w: int) -> int:
    if slot == 0:
        return 2
    if slot == 1:
        return w // 2
    return w - 3


def _make_sprite(outer_idx: int, inner_idx: int, bars: list[dict], slot: int, bg: int) -> dict:
    bar_h = int(bars[outer_idx]["h"])
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], bar_h))
    w = random.randint(*SPRITE_W_RANGE)

    outer_fill = int(bars[outer_idx]["fill"])
    inner_fill = int(bars[inner_idx]["fill"])
    patch = np.full((h, w), int(outer_fill), dtype=int)

    hole_c = _slot_hole_col(slot, w)
    ic0 = max(1, hole_c - 1)
    ic1 = min(w - 2, hole_c + 1)
    patch[1 : h - 1, ic0 : ic1 + 1] = int(inner_fill)

    hole_r = h // 2
    patch[hole_r, hole_c] = int(bg)

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": int(inner_idx),
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "slot": int(slot),
    }


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_top_sprites(sprites: list[dict], top_h: int, bg: int) -> list[dict] | None:
    rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []
    for sprite in sorted(sprites, key=lambda s: (-int(s["w"]), -int(s["h"]))):
        h = int(sprite["h"])
        w = int(sprite["w"])
        out_c = _encoded_target_col(np.array(sprite["patch"], dtype=int), bg, GRID_W)
        candidates: list[tuple[int, int]] = []
        for c0 in range(1, GRID_W - w):
            if abs(int(c0) - int(out_c)) < 2:
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(rects, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        rects.append((r0, c0, r0 + h - 1, c0 + w - 1))
        placed.append({**sprite, "r0": int(r0), "c0": int(c0), "out_c": int(out_c)})
    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()
    for sprite in placed:
        patch = np.array(sprite["patch"], dtype=int)
        outer_idx = int(sprite["outer_idx"])
        bar = bars[outer_idx]
        paste(inp, patch, int(sprite["r0"]), int(sprite["c0"]), transparent=int(bg))

        transformed = _transform_patch(patch, int(bar["fill"]), int(bar["fill"]), int(bar["border"]), bg)
        paste(out, transformed, int(bar["r_min"]), int(sprite["out_c"]), transparent=int(bg))
    return inp, out


def _expected_shift_metric(placed: list[dict]) -> int:
    return int(sum(abs(int(sprite["c0"]) - int(sprite["out_c"])) for sprite in placed))


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
    if top_h < MIN_TOP_AREA:
        return None

    outer_order = list(range(n_bars))
    random.shuffle(outer_order)
    slots = [0, 1, 2]
    random.shuffle(slots)
    slots = slots[:n_sprites]

    sprites: list[dict] = []
    for i, outer_idx in enumerate(outer_order[:n_sprites]):
        inner_choices = [idx for idx in range(n_bars) if idx != outer_idx]
        inner_idx = random.choice(inner_choices)
        sprites.append(_make_sprite(outer_idx, inner_idx, bars, int(slots[i]), bg))

    placed = _place_top_sprites(sprites, top_h, bg)
    if placed is None:
        return None
    if _expected_shift_metric(placed) <= 0:
        return None
    if len({int(sprite["out_c"]) for sprite in placed}) != len(placed):
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
