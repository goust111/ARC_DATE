# Combo variant of 7c66cb00 (index 7)
# combo_spec: D3/2+D8/1+D18/0
# Name: 有洞精灵擦除留轮廓与移动轨迹可见

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original: each top sprite vanishes completely, then reappears bottom-aligned
#    in the bar whose fill matches the sprite's outer color. Here the sprite still
#    docks the same way, but its source location is not blanked to pure bg: only a
#    1-pixel outline remains, and a vertical trail connects source to destination.
#    The sprite itself now contains true bg holes, so the bar fill shows through
#    those holes after docking.
# b) Complexity calibration:
#    Holes are mandatory and sprites are always connected two-color objects. The
#    trail uses exactly one column per sprite, so the new cue is strong but not
#    noisy. Color roles stay low-entropy: outer color identifies the target bar,
#    bar border recolors the shell, and bar fill is both the inner recolor and the
#    visible trail color.
# c) Broken assumptions from the original code:
#    - "Top pattern == one monochrome component" no longer holds, because a sprite
#      must be detected as one multi-color connected object with bg holes.
#    - "Erase top area to bg" no longer holds; we now re-draw a boundary residue.
#    - "No intermediate path" no longer holds; we must reserve and render a trail.
#    - Sequential recolor is unsafe because another bar's fill may equal the target
#      bar border. Recoloring therefore uses masks from the original patch.
# d) Placement legality:
#    The generator reserves horizontal intervals inside target bars, keeps top
#    sprites separated by bg, enforces distinct trail columns, and forbids any
#    trail column from passing through another source bbox.
# e) Visual observability:
#    Every sprite has at least one true hole, so "holes survive docking" is visible.
#    The source outline and the vertical trail create explicit before/after linkage
#    from the top object to the docked object.
# f) Visual cleanliness:
#    Trails are single-column and source sprites are spaced apart. This avoids
#    dense overdraw while still allowing some trails to cross unrelated bars.
# g) Inferability of color changes:
#    Colors are not keyed by numeric ids. Humans can infer that the outer color
#    selects the bar because the trail and the original shell use that same fill
#    color, while the docked shell switches to the bar border.


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    """Detect full-width horizontal bars with border cells at both ends."""
    H, W = g.shape
    bars: list[dict] = []
    current: dict | None = None
    for r in range(H):
        row = g[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(W * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if current is not None and int(current["fill"]) == int(fill_color):
                    current["r_max"] = int(r)
                else:
                    if current is not None:
                        bars.append(current)
                    current = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                    }
            elif current is not None:
                bars.append(current)
                current = None
        elif current is not None:
            bars.append(current)
            current = None
    if current is not None:
        bars.append(current)
    return bars


def _boundary_mask(patch: np.ndarray, bg: int) -> np.ndarray:
    """1-pixel outer contour of non-bg cells; enclosed hole edges are excluded."""
    h, w = patch.shape
    solid = patch != int(bg)
    exterior_bg = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if bool(solid[r, c]) or bool(exterior_bg[r, c]):
                continue
            exterior_bg[r, c] = True
            stack.append((int(r), int(c)))

    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if bool(solid[nr, nc]) or bool(exterior_bg[nr, nc]):
                continue
            exterior_bg[nr, nc] = True
            stack.append((nr, nc))

    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(solid[r, c]):
                continue
            if (
                r == 0
                or r == h - 1
                or c == 0
                or c == w - 1
                or bool(exterior_bg[r - 1, c])
                or bool(exterior_bg[r + 1, c])
                or bool(exterior_bg[r, c - 1])
                or bool(exterior_bg[r, c + 1])
            ):
                out[r, c] = True
    return out


def _count_true_holes(patch: np.ndarray, bg: int) -> int:
    """Count enclosed bg cells inside a sprite bbox."""
    h, w = patch.shape
    holes = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(patch[r, c]) != int(bg):
                continue
            if (
                int(patch[r - 1, c]) != int(bg)
                and int(patch[r + 1, c]) != int(bg)
                and int(patch[r, c - 1]) != int(bg)
                and int(patch[r, c + 1]) != int(bg)
            ):
                holes += 1
    return int(holes)


def _extract_sprites(g: np.ndarray, bars: list[dict], bg: int) -> list[dict]:
    """Detect top multi-color sprites as connected non-bg components."""
    pattern_area_end = int(min(int(b["r_min"]) for b in bars))
    top = g[:pattern_area_end, :]
    sprites: list[dict] = []
    for coords in components_mask(top != int(bg), connectivity=4):
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        patch = crop_bbox(g, (r0, c0, r1, c1))
        colors = Counter(int(v) for v in patch.flatten() if int(v) != int(bg))
        if not colors:
            continue
        outer_color = int(colors.most_common(1)[0][0])
        inner_candidates = [int(c) for c in colors.keys() if int(c) != int(outer_color)]
        inner_color = int(inner_candidates[0]) if inner_candidates else None
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "content": patch,
                "outer_color": int(outer_color),
                "inner_color": inner_color,
                "outline_mask": _boundary_mask(patch, bg),
                "trail_c": int(c0 + patch.shape[1] // 2),
                "hole_count": _count_true_holes(patch, bg),
            }
        )
    sprites.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    return sprites


def _solve_core(grid: np.ndarray, *, keep_outline: bool, keep_trail: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "expected at least one bar"
    sprites = _extract_sprites(g, bars, bg)
    pattern_area_end = int(min(int(b["r_min"]) for b in bars))

    # Remove original top sprites.
    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    if keep_outline:
        for sp in sprites:
            outline_patch = np.full((int(sp["h"]), int(sp["w"])), int(bg), dtype=int)
            outline_patch[np.asarray(sp["outline_mask"], dtype=bool)] = int(sp["outer_color"])
            paste(out, outline_patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

    # Bottom-up rendering lets upper docked sprites cover lower trails if they cross.
    for bar in reversed(bars):
        matching = [sp for sp in sprites if int(sp["outer_color"]) == int(bar["fill"])]
        matching.sort(key=lambda sp: int(sp["c0"]))
        for sp in matching:
            content = np.array(sp["content"], dtype=int)
            transformed = content.copy()
            transformed[content == int(sp["outer_color"])] = int(bar["border"])
            if sp["inner_color"] is not None:
                transformed[content == int(sp["inner_color"])] = int(bar["fill"])

            place_r = int(bar["r_max"] - int(sp["h"]) + 1)
            place_c = int(sp["c0"])

            if keep_trail:
                trail_c = int(sp["trail_c"])
                line_r0 = int(sp["r0"] + int(sp["h"]))
                line_r1 = int(place_r - 1)
                if line_r0 <= line_r1:
                    out[line_r0 : line_r1 + 1, trail_c] = int(bar["fill"])

            paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


def baseline_without_marks(grid: np.ndarray) -> np.ndarray:
    """Variant baseline: dock holey sprites but omit outline residue and trails."""
    return _solve_core(grid, keep_outline=False, keep_trail=False)


def fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict:
    """Quantify whether the combo behavior is materially present."""
    bg = int(get_bg(inp))
    bars = _find_bars(np.array(inp, dtype=int), bg)
    sprites = _extract_sprites(np.array(inp, dtype=int), bars, bg)
    baseline = baseline_without_marks(inp)

    outline_pixels = 0
    trail_pixels = 0
    for sp in sprites:
        outline_pixels += int(np.count_nonzero(np.asarray(sp["outline_mask"], dtype=bool)))
        target_bar = next(b for b in bars if int(b["fill"]) == int(sp["outer_color"]))
        place_r = int(target_bar["r_max"] - int(sp["h"]) + 1)
        line_r0 = int(sp["r0"] + int(sp["h"]))
        line_r1 = int(place_r - 1)
        if line_r0 <= line_r1:
            col = int(sp["trail_c"])
            trail_pixels += int(np.count_nonzero(out[line_r0 : line_r1 + 1, col] == int(target_bar["fill"])))

    return {
        "sprite_count": int(len(sprites)),
        "hole_cells": int(sum(int(sp["hole_count"]) for sp in sprites)),
        "outline_pixels": int(outline_pixels),
        "trail_pixels": int(trail_pixels),
        "delta_vs_baseline": int(np.count_nonzero(np.array(out, dtype=int) != np.array(baseline, dtype=int))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move holey two-color sprites into their color-matched bars, but keep source clues behind.

    Input structure:
    - The most common color is the background.
    - The upper region contains connected two-color sprites whose shell color matches one bar fill.
    - Each sprite may contain real background holes inside its bbox.
    - The lower region contains 2-3 full-width horizontal bars with a fill and a border color.

    Rule:
    - Detect each whole top sprite as one connected object, even if it has multiple colors and bg holes.
    - Match a sprite to the bar whose fill equals the sprite's outer color.
    - Dock it at the same x-position, bottom-aligned in that bar, recoloring shell -> bar border
      and inner color -> bar fill, while preserving bg holes so the bar shows through.
    - Erase the original sprite body, then leave a 1-pixel outline in the original outer color and
      draw a vertical trail in the target bar's fill color from source to destination.

    Key insight:
    The outer color still identifies the target bar, but the transformation is no longer pure
    disappearance: the source leaves an outline residue, and the motion becomes explicit through
    a vertical trail while the docked sprite keeps its internal holes.
    """
    return _solve_core(grid, keep_outline=True, keep_trail=True)


# --------------- generator ---------------

GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 9
SPRITE_H_CHOICES = (4, 5)
SPRITE_W_CHOICES = (5, 7, 9)
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
    bars = [None] * len(heights)
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {"fill": int(fills[i]), "border": int(borders[i]), "h": int(h), "r_min": int(r0), "r_max": int(r1)}
        r = int(r0 - 1)
    return grid, bars


def _make_holey_sprite(outer_idx: int, bars: list[dict], bg: int) -> dict:
    outer_fill = int(bars[outer_idx]["fill"])
    h_choices = [h for h in SPRITE_H_CHOICES if int(h) <= int(bars[outer_idx]["h"])]
    if not h_choices:
        raise ValueError("target bar too short for holey sprite")
    h = int(random.choice(h_choices))
    w = int(random.choice(SPRITE_W_CHOICES))

    inner_idx = random.choice([i for i in range(len(bars)) if int(i) != int(outer_idx)])
    inner_fill = int(bars[inner_idx]["fill"])

    patch = np.full((h, w), outer_fill, dtype=int)
    ih = 2 if h == 4 else random.choice([2, 3])
    iw_choices = [iw for iw in (3, 5) if int(iw) <= int(w - 2)]
    iw = int(random.choice(iw_choices))
    ir0 = 1 if h - ih == 2 else random.randint(1, h - ih - 1)
    ic0 = (w - iw) // 2
    patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = inner_fill

    # Carve a true hole fully enclosed by non-bg pixels.
    hole_r = int(ir0 if ih == 2 else ir0 + 1)
    hole_c = int(ic0 + iw // 2)
    patch[hole_r, hole_c] = int(bg)

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": int(inner_idx),
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "trail_offset": int(w // 2),
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[start : start + length].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _assign_outer_targets(n_bars: int, n_sprites: int) -> list[int]:
    slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(slots)
    targets = slots[:n_sprites]
    if n_sprites >= 2 and n_bars >= 2 and len(set(targets)) == 1:
        other = random.choice([i for i in range(n_bars) if int(i) != int(targets[0])])
        targets[-1] = int(other)
    random.shuffle(targets)
    return [int(t) for t in targets]


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    track_cols: list[int] = []
    placed: list[dict] = []

    for sp in sorted(sprites, key=lambda s: (-int(s["w"]), -int(s["h"]))):
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx = int(sp["outer_idx"])
        candidates: list[tuple[int, int]] = []
        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_idx], c0, w):
                continue
            trail_c = int(c0 + int(sp["trail_offset"]))
            c1 = int(c0 + w - 1)
            if trail_c in track_cols:
                continue
            if any(int(cc0) <= trail_c <= int(cc1) for _, cc0, _, cc1 in top_rects):
                continue
            if any(c0 <= int(tc) <= c1 for tc in track_cols):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][c0 : c0 + w] = True
        placed_sp = {**sp, "r0": int(r0), "c0": int(c0), "trail_c": int(c0 + int(sp["trail_offset"]))}
        placed.append(placed_sp)
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
        track_cols.append(int(placed_sp["trail_c"]))

    placed.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    return placed


def _render_pair(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for sp in placed:
        paste(inp, np.array(sp["patch"], dtype=int), int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

    for sp in placed:
        outline_patch = np.full((int(sp["h"]), int(sp["w"])), int(bg), dtype=int)
        outline_patch[_boundary_mask(np.array(sp["patch"], dtype=int), bg)] = int(bars[int(sp["outer_idx"])]["fill"])
        paste(out, outline_patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

    for bar_idx in range(len(bars) - 1, -1, -1):
        bar = bars[bar_idx]
        matching = [sp for sp in placed if int(sp["outer_idx"]) == int(bar_idx)]
        matching.sort(key=lambda sp: int(sp["c0"]))
        for sp in matching:
            content = np.array(sp["patch"], dtype=int)
            outer_fill = int(bar["fill"])
            inner_fill = int(bars[int(sp["inner_idx"])]["fill"])
            transformed = content.copy()
            transformed[content == int(outer_fill)] = int(bar["border"])
            transformed[content == int(inner_fill)] = int(bar["fill"])

            place_r = int(bar["r_max"] - int(sp["h"]) + 1)
            line_r0 = int(sp["r0"] + int(sp["h"]))
            line_r1 = int(place_r - 1)
            if line_r0 <= line_r1:
                out[line_r0 : line_r1 + 1, int(sp["trail_c"])] = int(bar["fill"])
            paste(out, transformed, place_r, int(sp["c0"]), transparent=int(bg))

    return inp, out


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

    sprites = [_make_holey_sprite(outer_idx, bars, bg) for outer_idx in _assign_outer_targets(n_bars, n_sprites)]
    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render_pair(base, placed, bars, bg)
    got = main(inp)
    if not np.array_equal(got, out):
        return None

    metrics = fusion_metrics(inp, out)
    if int(metrics["sprite_count"]) != int(n_sprites):
        return None
    if int(metrics["hole_cells"]) < int(n_sprites):
        return None
    if int(metrics["outline_pixels"]) <= 0:
        return None
    if int(metrics["trail_pixels"]) < int(n_sprites):
        return None
    if int(metrics["delta_vs_baseline"]) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
