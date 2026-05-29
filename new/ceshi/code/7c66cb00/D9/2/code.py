# Variant of 7c66cb00
# Dimension: D9 — 源对象：精灵数量与条的对应基数 (meta)
# Change: 2-3个精灵，2-3个条；每个精灵匹配一个条（通过颜色），一个条可接收0-2个精灵... → 含干扰精灵（颜色不匹配任何条）

from __future__ import annotations

"""
Design analysis for the D9 variant
==================================

variant_spec:
  core_change:
    - Keep the original 30x24 layout, bottom horizontal bars, bottom-aligned docking,
      x-position preservation, and color-swap rule for valid sprites.
    - Add 1-2 extra top sprites whose dominant outer color does not match any bar fill.
    - These distractor sprites are erased from the top area in the output and never dock
      into any bar.
  concrete_pixel_behavior:
    - Input top area now contains 3-5 visually separated non-bg objects:
      2-3 valid sprites and 1-2 distractors.
    - Valid sprite: dominant outer color equals one bar fill; optional minority inner
      patch may use another bar fill.
    - Distractor sprite: same small sprite-like appearance, but all its non-bg colors are
      drawn from colors unused by any bar fill or border, so it looks like a plausible
      sprite while remaining unmatched.
    - Output clears all top sprites. Only valid sprites reappear inside bars.
  behavior_trace:
    - Original pipeline:
      1. Find bars.
      2. Scan top non-bg color-components.
      3. Any component whose dominant color matches a bar fill gets docked.
    - Variant pipeline:
      1. Find bars.
      2. First detect whole top sprites as connected non-bg objects.
      3. Classify each whole object by its dominant outer color.
      4. If outer color matches a bar fill, dock it with the original recolor logic.
      5. Otherwise erase it only.
    - Example:
      * Bars fills are {2, 4}. Top contains sprites with dominant colors [2, 7, 4].
      * Original task had only [2, 4], so every top object docked.
      * Variant adds dominant color 7: it is cleared from the top in the output but
        produces no bar placement.
  complexity_calibration:
    - The new signal is visible because every sample contains at least one top sprite
      whose colors never appear in the bars, yet it disappears in the output.
    - Valid sprites remain 2-3, so the original matching pattern is still repeated enough
      for a human to infer the main rule.
    - Distractor count is capped at 1-2 to avoid overwhelming the top area.
  implicit_assumption_review:
    - The original solver iterated per-color connected components and implicitly relied on
      every top component belonging to a valid dockable sprite.
    - That is fragile once unmatched sprites exist: the solver should decide validity at
      the whole-object level, not per color-fragment.
    - The new solver therefore labels connected components on (top != bg), crops each
      whole object, computes its dominant outer color, and ignores the entire object if
      that outer color is not one of the bar fills.
    - After an object is accepted, the original sub-sprite behavior is preserved: inner
      monochrome regions that use another bar fill still dock independently into that
      other bar.
  placement_legality:
    - Generator keeps the original bar occupancy checks for valid sprites so the docked
      outputs do not overlap within a target bar.
    - Distractors do not reserve bar space because they never appear in the output, but
      all top sprites still require a 1-cell gap so they remain visually separate.
    - Required top-area height grows with the total number of top sprites to keep
      placement feasible and readable.
  visual_observability:
    - A human can see that some top sprites have colors unrelated to any bar fill and
      disappear completely, while others still move into matching bars.
    - The rule remains based on visible color matching, not numeric color ids.
  cleanliness:
    - No extra markers are added.
    - Distractors stay in the same top region and use the same general sprite size range,
      so the grid still reads as "a group of sprites above several bars".
  color_inference:
    - Valid matching remains structurally anchored by the bars' fill colors.
    - Distractor colors are taken from unused palette colors, limiting ambiguity.
  source_task:
    - None provided, so no extra reference solver was consulted.
"""

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
MATCHED_SPRITE_COUNT_RANGE = (2, 3)
DISTRACTOR_COUNT_RANGE = (1, 2)
BAR_HEIGHT_RANGE = (4, 8)
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 500


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    """Detect full-width horizontal bars with edge borders."""
    bars: list[dict] = []
    current_bar: dict | None = None
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(w * 0.7):
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
            elif current_bar:
                bars.append(current_bar)
                current_bar = None
        elif current_bar:
            bars.append(current_bar)
            current_bar = None
    if current_bar:
        bars.append(current_bar)
    return bars


def _extract_top_sprites(g: np.ndarray, pattern_area_end: int, bg: int) -> list[dict]:
    """Detect whole top sprites as connected non-bg objects."""
    top = g[:pattern_area_end, :]
    mask = top != int(bg)
    if not mask.any():
        return []
    lab, n = label_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        content = crop_bbox(g, (r0, c0, r1, c1))
        colors = Counter(map(int, content.flatten()))
        colors.pop(int(bg), None)
        if not colors:
            continue
        outer_color = int(colors.most_common(1)[0][0])
        other_colors = [int(c) for c in colors if int(c) != int(outer_color)]
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "h": int(content.shape[0]),
                "w": int(content.shape[1]),
                "content": np.array(content, dtype=int),
                "outer_color": int(outer_color),
                "other_colors": other_colors,
            }
        )
    return sprites


def _dock_valid_sprite(out: np.ndarray, sprite: dict, fill_to_bar: dict[int, dict], bg: int) -> None:
    """Dock one accepted sprite, preserving the original inner-sub-sprite behavior."""
    outer_bar = fill_to_bar[int(sprite["outer_color"])]
    transformed = np.array(sprite["content"], dtype=int).copy()
    transformed[transformed == int(sprite["outer_color"])] = int(outer_bar["border"])
    transformed[(transformed != int(bg)) & (transformed != int(outer_bar["border"]))] = int(outer_bar["fill"])
    paste(out, transformed, int(outer_bar["r_max"] - sprite["h"] + 1), int(sprite["c0"]), transparent=int(bg))

    for inner_color in sprite["other_colors"]:
        if int(inner_color) not in fill_to_bar:
            continue
        inner_bar = fill_to_bar[int(inner_color)]
        inner_mask = np.array(sprite["content"], dtype=int) == int(inner_color)
        inner_lab, inner_n = label_mask(inner_mask, connectivity=4)
        for region_id in range(1, int(inner_n) + 1):
            coords = np.argwhere(inner_lab == int(region_id))
            if coords.size == 0:
                continue
            lr0, lc0, lr1, lc1 = map(int, bbox_coords(coords))
            inner_patch = crop_bbox(sprite["content"], (lr0, lc0, lr1, lc1))
            inner_patch = np.where(inner_patch == int(inner_color), int(inner_bar["border"]), int(bg))
            place_r = int(inner_bar["r_max"] - inner_patch.shape[0] + 1)
            place_c = int(sprite["c0"] + lc0)
            paste(out, inner_patch, place_r, place_c, transparent=int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock valid two-tone sprites into color-matched bars and erase distractors.

    Variant rule:
    - Top area now may contain 1-2 distractor sprites whose dominant outer color does not
      match any bar fill.
    - Every top sprite is erased from the source area.
    - Only sprites whose dominant outer color matches a bar fill are docked into bars.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found — grid must contain at least one horizontal bar"
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}
    pattern_area_end = min(int(bar["r_min"]) for bar in bars)

    sprites = _extract_top_sprites(g, pattern_area_end, bg)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for sprite in sprites:
        if int(sprite["outer_color"]) not in fill_to_bar:
            continue
        _dock_valid_sprite(out, sprite, fill_to_bar, bg)

    return out


def _required_top_area(total_sprites: int) -> int:
    """Increase headroom when the variant adds extra distractors."""
    return 7 + 2 * max(0, int(total_sprites) - 3)


def _sample_bar_heights(n_bars: int, total_sprites: int) -> list[int]:
    """Sample bar heights that leave enough room for 3-5 top sprites."""
    min_top_area = _required_top_area(total_sprites)
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if GRID_H - sum(heights) - (n_bars - 1) < min_top_area:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Render bottom bars and return the base grid plus bar metadata."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars = [None] * len(heights)
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = heights[i]
        r0, r1 = r - h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = fills[i]
        grid[r0 : r1 + 1, 0] = borders[i]
        grid[r0 : r1 + 1, GRID_W - 1] = borders[i]
        bars[i] = {"fill": fills[i], "border": borders[i], "h": h, "r_min": r0, "r_max": r1}
        r = r0 - 1
    return grid, bars


def _sample_inner_bbox(h: int, w: int) -> tuple[int, int, int, int] | None:
    """Sample a centered-width minority inner region."""
    valid_inner = [
        (ih, iw)
        for ih in range(1, h - 1)
        for iw in range(1, w - 1)
        if (w - iw) % 2 == 0 and ih * iw * 2 < h * w
    ]
    if not valid_inner:
        return None
    ih, iw = random.choice(valid_inner)
    ir0 = random.randint(1, h - ih - 1)
    ic0 = (w - iw) // 2
    return ir0, ic0, ih, iw


def _build_sprite_patch(outer_color: int, height_cap: int, inner_choices: list[int]) -> tuple[np.ndarray, int | None, tuple[int, int, int, int] | None]:
    """Create a small outer-color sprite with an optional minority inner region."""
    h = random.randint(SPRITE_H_RANGE[0], max(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], height_cap)))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), int(outer_color), dtype=int)

    inner_color = None
    inner_bbox = None
    if inner_choices and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_bbox = _sample_inner_bbox(h, w)
        if inner_bbox is not None:
            inner_color = int(random.choice(inner_choices))
            ir0, ic0, ih, iw = inner_bbox
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(inner_color)
        else:
            inner_bbox = None
    return patch, inner_color, inner_bbox


def _make_matched_sprite(outer_idx: int, bars: list[dict]) -> dict:
    """Create a valid sprite whose dominant color matches one bar fill."""
    patch, inner_color, inner_bbox = _build_sprite_patch(
        outer_color=int(bars[outer_idx]["fill"]),
        height_cap=int(bars[outer_idx]["h"]),
        inner_choices=[int(b["fill"]) for i, b in enumerate(bars) if i != int(outer_idx)],
    )
    inner_idx = None
    if inner_color is not None:
        for idx, bar in enumerate(bars):
            if int(bar["fill"]) == int(inner_color):
                inner_idx = int(idx)
                break
    return {
        "matched": True,
        "outer_idx": int(outer_idx),
        "inner_idx": inner_idx,
        "patch": patch,
        "h": int(patch.shape[0]),
        "w": int(patch.shape[1]),
        "inner_bbox": inner_bbox,
    }


def _make_distractor_sprite(distractor_colors: list[int], max_bar_height: int) -> dict:
    """Create an unmatched top sprite using colors unused by any bar."""
    outer_color = int(random.choice(distractor_colors))
    inner_choices = [int(c) for c in distractor_colors if int(c) != int(outer_color)]
    patch, inner_color, inner_bbox = _build_sprite_patch(
        outer_color=outer_color,
        height_cap=max(int(max_bar_height), SPRITE_H_RANGE[1]),
        inner_choices=inner_choices,
    )
    return {
        "matched": False,
        "outer_idx": None,
        "inner_idx": None,
        "patch": patch,
        "h": int(patch.shape[0]),
        "w": int(patch.shape[1]),
        "inner_bbox": inner_bbox,
        "inner_color": inner_color,
    }


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
    """Place all top sprites while preserving valid-sprite docking capacity."""
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        candidates: list[tuple[int, int]] = []
        for c0 in range(1, GRID_W - w):
            if sp["matched"]:
                outer_idx = int(sp["outer_idx"])
                if not _interval_free(bar_occ[outer_idx], c0, w):
                    continue
                if sp["inner_idx"] is not None:
                    _, ic0, _, iw = sp["inner_bbox"]
                    if not _interval_free(bar_occ[int(sp["inner_idx"])], c0 + ic0, iw):
                        continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        if sp["matched"]:
            outer_idx = int(sp["outer_idx"])
            bar_occ[outer_idx][c0 : c0 + w] = True
            if sp["inner_idx"] is not None:
                _, ic0, _, iw = sp["inner_bbox"]
                bar_occ[int(sp["inner_idx"])][c0 + ic0 : c0 + ic0 + iw] = True
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Render input with all top sprites and output with only valid sprites docked."""
    inp, out = base.copy(), base.copy()
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}

    for sp in placed:
        paste(inp, np.array(sp["patch"], dtype=int), int(sp["r0"]), int(sp["c0"]), transparent=int(bg))
        if not sp["matched"]:
            continue
        sprite = {
            "content": np.array(sp["patch"], dtype=int),
            "outer_color": int(bars[int(sp["outer_idx"])]["fill"]),
            "other_colors": [int(bars[int(sp["inner_idx"])]["fill"])] if sp["inner_idx"] is not None else [],
            "h": int(sp["h"]),
            "c0": int(sp["c0"]),
        }
        _dock_valid_sprite(out, sprite, fill_to_bar, bg)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    """One output-first generation attempt."""
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_matched = random.randint(*MATCHED_SPRITE_COUNT_RANGE)
    n_distractors = random.randint(*DISTRACTOR_COUNT_RANGE)
    total_sprites = n_matched + n_distractors

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(c) for c in palette[1 : 1 + n_bars]]
    borders = [int(c) for c in palette[1 + n_bars : 1 + 2 * n_bars]]
    distractor_colors = [int(c) for c in palette[1 + 2 * n_bars :]]
    if not distractor_colors:
        return None

    heights = _sample_bar_heights(n_bars, total_sprites)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)
    matched_sprites = [_make_matched_sprite(int(outer_slots[i]), bars) for i in range(n_matched)]
    distractor_sprites = [_make_distractor_sprite(distractor_colors, max(int(bar["h"]) for bar in bars)) for _ in range(n_distractors)]
    sprites = matched_sprites + distractor_sprites
    sprites.sort(key=lambda s: (not s["matched"], -int(s["w"]), -int(s["h"])))

    placed = _place_sprites(sprites, bars, top_h)
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
