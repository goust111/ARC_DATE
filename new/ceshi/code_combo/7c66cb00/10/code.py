# Combo variant of 7c66cb00 (index 10)
# combo_spec: D4/1+D7/1+D10/1
# Name: 精灵在两侧对接非全宽条居中对齐

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

"""
Design analysis
===============

a) Concrete fusion behavior + trace
  Fused spec in pixels:
    - The grid is still 30x24 with 2-3 horizontal bars near the bottom, but every bar
      is shorter than the canvas width and centered horizontally.
    - Sprites no longer sit in a top strip. They sit in the far-left or far-right side
      lanes, with their rows chosen from inside the row span of the bar they belong to.
    - Matching is still color-driven: a sprite's outer/majority color equals the fill
      color of its destination bar.
    - Output erases the side sprites, swaps colors as before, then moves each sprite
      horizontally into the middle of the matched bar. Left sprites dock against the
      left side of the bar's center split; right sprites dock against the right side.
    - Vertical placement is no longer bottom-aligned. The sprite bbox is centered inside
      the bar height by `r_min + (bar_h - sprite_h) // 2`.

  Concrete trace:
    - Original task: a sprite above the bars at x=8 keeps x=8, drops into a full-width
      bar, and lands flush with the bar bottom.
    - Fused task: suppose a 3x4 left-lane sprite uses outer color 2, and the centered
      bar with fill 2 spans cols 6..17 and rows 20..26.
        * input: sprite might sit at cols 0..3 and rows 21..23
        * output: x no longer stays 0..3 or original c0; because it came from the left,
          it is pasted with its right edge meeting the bar's central split
        * output y is `20 + (7 - 3) // 2 = 22`, not bottom row 24
    - If a right-lane sprite targets the same bar, it is pasted from the other side of
      the same split, so the pair visually meets in the bar's center region.

b) Complexity calibration
  The variant stays learnable because each freedom is low-entropy and visibly repeated:
    - side choice is binary: left or right
    - bars are always short and centered, so the lateral relationship is stable
    - destination bar is still identified by fill color, not by row order or color id
    - at least one bar receives both a left sprite and a right sprite, which makes the
      "meet in the middle" rule visible instead of hypothetical
  This avoids a too-simple degeneration back to D10 edge docking or back to the original
  x-preserving drop rule.

c) Hidden assumptions from the original main() that break here
  The original solver depended on several premises that are all false after fusion:
    - "bars are almost full-width" (`count > W * 0.7`) is false because bars are width
      10-14 and centered.
    - "bar borders touch canvas edges" is false because both outer margins remain bg.
    - "all sprites are above the first bar" is false because sprites share rows with the
      bars and live only in side lanes.
    - "input x-position is preserved in output" is false because the new x-rule depends
      on side-of-entry and the bar midpoint.
    - "vertical placement is bottom alignment" is false because placement now depends on
      centered row offset inside the bar.
  The fused solver therefore:
    - detects bars as wide rectangular non-bg components
    - removes those bar rectangles before searching for sprites
    - classifies each sprite by left/right side relative to the bar block
    - computes a center-split placement instead of reusing input c0

d) Placement legality
  Generator guarantees:
    - every bar width is in 10..14, so it is visibly non-full-width and leaves bg on
      both outer sides
    - every sprite width is at most 4, so it fits strictly inside one half of its bar
      and does not collapse to edge docking
    - every sprite height is strictly smaller than its target bar height, so vertical
      centering differs from bottom alignment on every placement
    - each `(bar, side)` slot is used at most once, so output pastes never overlap
    - side-lane sprites stay fully inside cols 0..3 or 20..23 and therefore never touch
      the bars in the input

e) Visual observability
  A human can infer the fused rule from 2-3 examples because:
    - the top area is empty, so sprites clearly are not "falling from above"
    - every bar is inset from both sides, so partial-width is obvious by eye
    - left/right source positions recur across examples
    - one bar visibly receives a left sprite and a right sprite that end up facing each
      other around the middle, which distinguishes this from simple edge docking
    - color swap remains structurally cued by bar fill/border roles

f) Visual cleanliness
  The layout is intentionally constrained:
    - bars are stacked vertically with 1-row bg gaps, not arbitrarily packed
    - all bars share the same horizontal center, so the "middle zone" is consistent
    - side sprites live in dedicated outer lanes, which keeps them visually separated
      from the bars in the input
    - output inserts the sprites into empty interior regions of the bars, not on top of
      the bar borders, so the resulting shapes stay legible

g) Change inferability
  The recolor rule stays limited to two roles:
    - outer sprite color becomes the matched bar border
    - inner sprite color, if present, becomes the matched bar fill
  Because sprite colors are always chosen from the set of bar fills, the solver can
  infer bar identity structurally from the input alone. No rule depends on numeric color
  ordering or any non-visual property.

New solver assumptions audited against the generator
  - Wide rectangular components of width >= 10 are exactly the bars. Generator keeps all
    sprites width <= 4 and places them outside the bar block.
  - Each bar rectangle uses exactly two colors in a strict border/fill pattern.
  - Every non-bar component lies completely left or right of the bar block, so side
    classification is unambiguous.
  - There is at most one sprite per `(bar, side)` slot, so center-split placements do
    not collide.
  - Sprite height is always strictly smaller than target bar height, so the centered-row
    formula meaningfully differs from bottom alignment.
"""


GRID_H = 30
GRID_W = 24

BAR_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
BAR_WIDTH_RANGE = (10, 14)
MAX_THREE_BAR_TOTAL = 20
BAR_GAP = 1
MIN_TOP_MARGIN = 2

SIDE_LANE_W = 4
SPRITE_COUNT_RANGE = (2, 3)
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 4)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 400


def _extract_bars(g: np.ndarray, bg: int) -> list[dict]:
    """Detect centered partial-width bars as large rectangular components."""
    lab, n = label_mask(g != int(bg), connectivity=4)
    bars: list[dict] = []

    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        patch = crop_bbox(g, (r0, c0, r1, c1))
        h, w = map(int, patch.shape)

        if h < BAR_HEIGHT_RANGE[0] or w < BAR_WIDTH_RANGE[0] or w < 3:
            continue
        if np.any(patch == int(bg)):
            continue

        colors = sorted(set(map(int, patch.flatten())))
        if len(colors) != 2:
            continue
        if len(set(map(int, patch[:, 0]))) != 1 or len(set(map(int, patch[:, -1]))) != 1:
            continue

        border = int(patch[0, 0])
        if not np.all(patch[:, 0] == int(border)) or not np.all(patch[:, -1] == int(border)):
            continue

        inner = patch[:, 1:-1]
        if inner.size == 0 or len(set(map(int, inner.flatten()))) != 1:
            continue
        fill = int(inner[0, 0])
        if fill == border or not np.all(inner == int(fill)):
            continue

        bars.append(
            {
                "fill": fill,
                "border": border,
                "r_min": r0,
                "r_max": r1,
                "c_min": c0,
                "c_max": c1,
                "h": h,
                "w": w,
            }
        )

    bars.sort(key=lambda b: (int(b["r_min"]), int(b["c_min"])))
    assert bars, "No centered partial-width bars found"
    return bars


def _remove_bars(g: np.ndarray, bars: list[dict], bg: int) -> np.ndarray:
    """Blank out all bar rectangles so only side-lane sprites remain."""
    sprite_grid = np.array(g, dtype=int).copy()
    for bar in bars:
        sprite_grid[int(bar["r_min"]) : int(bar["r_max"]) + 1, int(bar["c_min"]) : int(bar["c_max"]) + 1] = int(bg)
    return sprite_grid


def _extract_sprites(sprite_grid: np.ndarray, bars: list[dict], bg: int) -> list[dict]:
    """Extract connected side-lane sprites and classify them by entry side."""
    left_edge = min(int(bar["c_min"]) for bar in bars)
    right_edge = max(int(bar["c_max"]) for bar in bars)

    lab, n = label_mask(sprite_grid != int(bg), connectivity=4)
    sprites: list[dict] = []

    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))

        side = None
        if c1 < left_edge:
            side = "left"
        elif c0 > right_edge:
            side = "right"
        if side is None:
            continue

        patch = crop_bbox(sprite_grid, (r0, c0, r1, c1))
        colors = Counter(int(v) for v in patch.flatten())
        colors.pop(int(bg), None)
        if not colors:
            continue

        outer_color = int(colors.most_common(1)[0][0])
        inner_candidates = [int(color) for color in colors if int(color) != int(outer_color)]

        sprites.append(
            {
                "outer_color": outer_color,
                "inner_color": int(inner_candidates[0]) if inner_candidates else None,
                "side": side,
                "r0": r0,
                "c0": c0,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "content": patch,
            }
        )

    sprites.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    return sprites


def _center_row(bar: dict, sprite_h: int) -> int:
    return int(bar["r_min"] + (int(bar["h"]) - int(sprite_h)) // 2)


def _meeting_col(bar: dict, side: str, sprite_w: int) -> int:
    split = int(bar["c_min"] + int(bar["w"]) // 2)
    if side == "left":
        return int(split - int(sprite_w))
    return int(split)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move side-lane two-tone sprites into centered partial-width bars with a color swap.

    Input structure:
    - Background is the most common color.
    - The lower portion contains 2-3 centered horizontal bars that do not span the full
      width; each bar has a single border color on its outer columns and a fill color in
      the interior.
    - Small 1-2 color sprites sit in the far-left or far-right lanes, and each sprite's
      rows lie within the row span of its target bar.

    Rule:
    - Match each sprite to the bar whose fill color equals the sprite's outer color.
    - Erase the original side-lane sprite.
    - Recolor the sprite: outer color becomes the matched bar's border, inner color
      becomes that bar's fill.
    - If the sprite came from the left, place it against the left side of the bar's
      central split; if it came from the right, place it against the right side.
    - Vertically center the sprite inside the target bar instead of bottom-aligning it.

    Key insight:
    - The destination bar is still chosen by color, but the important spatial cue is no
      longer the original x-position. Side-of-entry determines which half of the bar the
      sprite occupies, and both sides converge toward the same middle seam.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _extract_bars(g, bg)
    sprite_grid = _remove_bars(g, bars, bg)
    sprites = _extract_sprites(sprite_grid, bars, bg)

    out[sprite_grid != int(bg)] = int(bg)

    bar_by_fill = {int(bar["fill"]): bar for bar in bars}
    for sp in sprites:
        bar = bar_by_fill.get(int(sp["outer_color"]))
        if bar is None:
            continue

        transformed = np.array(sp["content"], dtype=int).copy()
        transformed[transformed == int(sp["outer_color"])] = int(bar["border"])
        if sp["inner_color"] is not None:
            transformed[transformed == int(sp["inner_color"])] = int(bar["fill"])

        place_r = _center_row(bar, int(sp["h"]))
        place_c = _meeting_col(bar, str(sp["side"]), int(sp["w"]))
        paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        top_margin = GRID_H - sum(heights) - (n_bars - 1) * BAR_GAP
        if top_margin < MIN_TOP_MARGIN:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Render centered short bars stacked bottom-up with 1-row gaps."""
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars = [None] * len(heights)
    r = GRID_H

    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
        w = random.randint(*BAR_WIDTH_RANGE)
        c0 = int((GRID_W - w) // 2)
        c1 = int(c0 + w - 1)

        grid[r0 : r1 + 1, c0] = int(borders[i])
        grid[r0 : r1 + 1, c1] = int(borders[i])
        grid[r0 : r1 + 1, c0 + 1 : c1] = int(fills[i])

        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": h,
            "w": w,
            "r_min": r0,
            "r_max": r1,
            "c_min": c0,
            "c_max": c1,
        }
        r = r0 - BAR_GAP

    return grid, bars


def _make_sprite(outer_idx: int, side: str, bars: list[dict]) -> dict:
    """Create a compact two-tone sprite that fits a side lane and one half of its bar."""
    outer_bar = bars[int(outer_idx)]
    max_h = min(SPRITE_H_RANGE[1], int(outer_bar["h"]) - 1)
    h = random.randint(SPRITE_H_RANGE[0], max_h)
    max_w = min(SPRITE_W_RANGE[1], SIDE_LANE_W, int(outer_bar["w"]) // 2 - 1)
    w = random.randint(SPRITE_W_RANGE[0], max_w)

    patch = np.full((h, w), int(outer_bar["fill"]), dtype=int)

    inner_idx = None
    inner_bbox = None
    other_bars = [i for i in range(len(bars)) if i != int(outer_idx)]
    if other_bars and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, w - 1)
            if ih * iw * 2 < h * w
        ]
        if valid_inner:
            inner_idx = random.choice(other_bars)
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[int(inner_idx)]["fill"])
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": inner_idx,
        "side": str(side),
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _choose_sprites(n_sprites: int, bars: list[dict]) -> list[dict]:
    """Force one left/right pair on the same bar so middle convergence is visible."""
    pair_bar = random.randrange(len(bars))
    slots = [(int(pair_bar), "left"), (int(pair_bar), "right")]

    other_slots = [(i, side) for i in range(len(bars)) for side in ("left", "right") if i != int(pair_bar)]
    random.shuffle(other_slots)
    while len(slots) < n_sprites:
        slots.append(other_slots.pop())

    random.shuffle(slots)
    return [_make_sprite(int(outer_idx), str(side), bars) for outer_idx, side in slots]


def _place_sprites(sprites: list[dict], bars: list[dict]) -> list[dict]:
    """Place each sprite in the far-left or far-right lane within its target bar's rows."""
    placed: list[dict] = []
    for sp in sprites:
        bar = bars[int(sp["outer_idx"])]
        row_min = int(bar["r_min"])
        row_max = int(bar["r_max"] - sp["h"] + 1)
        center_r = _center_row(bar, int(sp["h"]))
        candidates = [r for r in range(row_min, row_max + 1) if int(r) != int(center_r)]
        r0 = random.choice(candidates) if candidates else center_r
        if sp["side"] == "left":
            c0 = random.randint(0, SIDE_LANE_W - int(sp["w"]))
        else:
            c0 = random.randint(GRID_W - SIDE_LANE_W, GRID_W - int(sp["w"]))

        placed.append({**sp, "r0": int(r0), "c0": int(c0)})

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Render input with side sprites and output with centered docking."""
    inp = base.copy()
    out = base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[int(sp["outer_idx"])]
        paste(inp, patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

        transformed = patch.copy()
        transformed[patch == int(outer_bar["fill"])] = int(outer_bar["border"])
        if sp["inner_idx"] is not None:
            inner_fill = int(bars[int(sp["inner_idx"])]["fill"])
            transformed[patch == int(inner_fill)] = int(outer_bar["fill"])

        out_r = _center_row(outer_bar, int(sp["h"]))
        out_c = _meeting_col(outer_bar, str(sp["side"]), int(sp["w"]))
        paste(out, transformed, out_r, out_c, transparent=int(bg))

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
    sprites = _choose_sprites(n_sprites, bars)
    placed = _place_sprites(sprites, bars)

    inp, out = _render(base, placed, bars, bg)
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
