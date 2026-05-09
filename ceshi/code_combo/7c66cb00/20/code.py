# Combo variant of 7c66cb00 (index 20)
# combo_spec: D4/1+D5/2+D10/2
# Name: 居中非全宽条间散落精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

"""
Design analysis
===============

a) Concrete fusion behavior
  Base task:
    - small 1-2 color sprites sit in one top zone
    - a sprite is matched to the bar whose fill color equals the sprite's outer color
    - every docked sprite is bottom-aligned inside that bar
  Fused variant:
    - bars are partial-width rectangles, not full-width strips
    - the whole bar stack is vertically centered, so there is bg above and below it
    - visible sprites no longer live in one top block; they are placed only in the internal gaps between consecutive bars
    - a sprite still belongs to the bar identified by its outer fill color, but now the gap position determines which face of that bar it docks to:
      if the sprite is above its target bar it docks to the bar's top edge; if it is below, it docks to the bottom edge
    - because the generator places each sprite strictly closer to its own colored bar than to the bar on the other side of the gap,
      the intended reading is “move to the nearest bar above/below”
  Small trace:
    - suppose bar A is rows 6..10 and bar B is rows 15..19
    - a sprite with outer color = fill(A) sits in rows 11..13, so it is below A and above B, but closer to A
    - output: erase it from the gap, recolor its outer cells to border(A), recolor its inner cells to fill(A), and bottom-align it in A
    - if another sprite with outer color = fill(B) sits in the same gap but rows 14..14, it is closer to B, so it top-aligns in B
  Difference from the original:
    - the solver can no longer crop “everything above the first bar”
    - the docking row is no longer always `bar.r_max - h + 1`

b) Complexity calibration
  The variant keeps the original color-to-bar identity so samples stay learnable from a few examples.
  The visible change is still strong: bars are cropped horizontally, centered vertically, and sprites are interleaved between them.
  To avoid overcomplicating the reading, bars remain one per horizontal band; this preserves clean vertical gaps, which are essential for the
  “nearest bar above/below” cue to be visible.

c) Hidden assumptions from the original main() that fail here
  - “bar rows cover most of the width”: false for partial-width bars
  - “bar borders touch column 0 / W-1”: false because bars float inside the canvas
  - “all sprites are above the first bar”: false because sprites are inside internal bar gaps
  - “every sprite docks to the bottom of its target bar”: false because side-of-approach now matters
  The fix is:
    - detect bars as large rectangular components with side borders
    - remove those bar cells from the grid
    - find sprite patterns in the remaining non-bg cells across the full canvas
    - determine docking edge from each pattern's position relative to its matched bar

d) Placement legality
  Generator guarantees:
    - every visible sprite lies wholly inside one internal gap between consecutive bars
    - each sprite is strictly nearer to its own target bar than to the other bar bordering that gap
    - each sprite fits horizontally inside its outer target bar
    - if an inner-color subcomponent exists, its x-interval also fits inside its own target bar
    - input sprites are separated by at least one bg cell
    - docked intervals do not overlap inside any bar

e) Visual observability
  A human sees three stable cues:
    - bars are clearly partial-width framed rectangles
    - sprites are clearly separate small components sitting between bars, not inside them
    - outputs show some sprites inserted at top edges and some at bottom edges of bars
  This is inferable from geometry, not from hidden numeric properties of colors.

f) Visual cleanliness
  Using one bar per row band avoids tangled overlaps between bars and gap sprites.
  Internal gaps are 5 rows tall, enough for sprites to be isolated and for “closer to upper vs lower bar” to be visible.
  Bars are kept roughly center-aligned horizontally so the grid does not become noisy.

g) Color-change inferability
  The recolor rule stays structural:
    - outer sprite color becomes the border color of its matched bar
    - inner sprite color becomes the fill color of that same matched bar
  Keeping the visible sprite connected avoids the harder-to-infer behavior where one two-tone input object would split across multiple bars.

New solver assumptions audited against the generator
  - Bars are the only non-bg components with width >= 10 and a clean border/fill rectangle. Generator enforces sprite width <= 8.
  - Every non-bar color used by sprites is some bar fill color. Generator enforces this exactly.
  - Fill colors are unique across bars, so color matching identifies one destination bar. Generator enforces unique fills.
  - Every visible sprite sits strictly above or strictly below its matched bar, never overlapping it. Generator places sprites only in internal gaps.
"""


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 5)
BAR_WIDTH_RANGE = (10, 13)
GAP_HEIGHT_RANGE = (5, 5)
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 8)
INNER_COLOR_PROB = 0.45
MAX_THREE_BAR_TOTAL = 13
MIN_MARGIN = 2
MIN_ADJ_OVERLAP = 4
GEN_RETRIES = 600


def _extract_bars(g: np.ndarray, bg: int) -> list[dict]:
    """Detect centered partial-width bars as rectangular non-bg components."""
    lab, n = label_mask(g != int(bg), connectivity=4)
    bars: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        patch = crop_bbox(g, (r0, c0, r1, c1))
        h, w = map(int, patch.shape)
        if h < BAR_HEIGHT_RANGE[0] or w < 10:
            continue
        if np.any(patch == int(bg)) or w < 3:
            continue
        if len(set(map(int, patch[:, 0].flatten()))) != 1:
            continue
        if len(set(map(int, patch[:, -1].flatten()))) != 1:
            continue
        border = int(patch[0, 0])
        if border != int(patch[0, -1]):
            continue
        if not np.all(patch[:, 0] == int(border)) or not np.all(patch[:, -1] == int(border)):
            continue
        inner = patch[:, 1:-1]
        if inner.size == 0:
            continue
        colors = set(map(int, patch.flatten()))
        if len(colors) != 2:
            continue
        if len(set(map(int, inner.flatten()))) != 1:
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
    bars.sort(key=lambda b: int(b["r_min"]))
    assert bars, "No centered partial-width bars found"
    return bars


def _bar_mask(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for bar in bars:
        mask[int(bar["r_min"]) : int(bar["r_max"]) + 1, int(bar["c_min"]) : int(bar["c_max"]) + 1] = True
    return mask


def _extract_patterns(g: np.ndarray, bg: int, bars: list[dict]) -> list[dict]:
    """Extract connected 1-2 color sprites from the non-bar area."""
    patterns = []
    mask = _bar_mask(g.shape, bars)
    sprite_zone = np.where(mask, int(bg), g)
    lab, n = label_mask(sprite_zone != int(bg), connectivity=4)
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        bbox = crop_bbox(sprite_zone, (r0, c0, r1, c1))
        colors_in_bbox = Counter(map(int, bbox.flatten()))
        colors_in_bbox.pop(int(bg), None)
        if not colors_in_bbox:
            continue
        outer_color = int(colors_in_bbox.most_common(1)[0][0])
        inner_colors = [int(c) for c in colors_in_bbox if int(c) != int(outer_color)]
        patterns.append(
            {
                "outer_color": outer_color,
                "inner_color": int(inner_colors[0]) if inner_colors else None,
                "r0": r0,
                "c0": c0,
                "h": int(bbox.shape[0]),
                "w": int(bbox.shape[1]),
                "content": bbox,
            }
        )
    patterns.sort(key=lambda p: (int(p["r0"]), int(p["c0"]), -int(p["h"]) * int(p["w"])))
    return patterns


def _gap_context(r0: int, h: int, bars: list[dict]) -> tuple[dict, dict, int, int] | None:
    """Return the consecutive bars whose gap contains the sprite, plus distances to each bar."""
    r1 = int(r0 + h - 1)
    for gap_idx in range(len(bars) - 1):
        upper = bars[gap_idx]
        lower = bars[gap_idx + 1]
        if r0 > int(upper["r_max"]) and r1 < int(lower["r_min"]):
            dist_up = int(r0 - (int(upper["r_max"]) + 1))
            dist_down = int((int(lower["r_min"]) - 1) - r1)
            return upper, lower, dist_up, dist_down
    return None


def _extract_visual_sprites(g: np.ndarray, bg: int, bars: list[dict]) -> list[dict]:
    """Connected visible sprites, used only for generator auditing."""
    mask = _bar_mask(g.shape, bars)
    sprite_zone = np.where(mask, int(bg), g)
    lab, n = label_mask(sprite_zone != int(bg), connectivity=4)
    sprites: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        bbox = crop_bbox(sprite_zone, (r0, c0, r1, c1))
        colors = Counter(map(int, bbox.flatten()))
        colors.pop(int(bg), None)
        if not colors:
            continue
        sprites.append(
            {
                "outer_color": int(colors.most_common(1)[0][0]),
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "h": int(bbox.shape[0]),
                "w": int(bbox.shape[1]),
            }
        )
    sprites.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    return sprites


def _relation_to_bar(r0: int, h: int, bar: dict) -> str:
    r1 = int(r0 + h - 1)
    if r1 < int(bar["r_min"]):
        return "above"
    if int(r0) > int(bar["r_max"]):
        return "below"
    raise AssertionError("Sprite overlaps its target bar")


def _dock_row(bar: dict, h: int, relation: str) -> int:
    if relation == "above":
        return int(bar["r_min"])
    if relation == "below":
        return int(bar["r_max"] - h + 1)
    raise ValueError(f"unknown relation {relation}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move inter-gap two-tone sprites into color-matched centered bars and dock on the nearest-facing edge.

    Input structure:
    - Background is the most common color.
    - Middle of the grid: 2-3 partial-width horizontal bars, each a rectangle with border color on the two side columns and fill color inside.
    - Between consecutive bars: small 1-2 color sprites using bar fill colors; these sprites are separated from the bars by background rows.

    Rule:
    - Match each sprite to the unique bar whose fill color equals the sprite's outer color.
    - Erase the sprite from the gap area.
    - Recolor outer cells to the matched bar's border color and inner cells to that bar's fill color.
    - If the sprite started above its target bar, dock it to the bar's top edge; if it started below, dock it to the bar's bottom edge.

    Key insight:
    - The original color-based ownership is preserved, but the spatial variable has changed: sprites are no longer all above the bar stack,
      so the solver must detect whether each matched sprite approaches its bar from above or below and choose the docking edge accordingly.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _extract_bars(g, bg)
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}
    patterns = _extract_patterns(g, bg, bars)
    mask = _bar_mask(g.shape, bars)

    out[~mask] = np.where(out[~mask] != int(bg), int(bg), out[~mask])

    for pat in patterns:
        bar = fill_to_bar.get(int(pat["outer_color"]))
        gap_ctx = _gap_context(int(pat["r0"]), int(pat["h"]), bars)
        if gap_ctx is not None:
            upper, lower, dist_up, dist_down = gap_ctx
            if dist_up < dist_down:
                bar = upper
            elif dist_down < dist_up:
                bar = lower
            elif bar is None:
                bar = upper
            elif bar not in (upper, lower):
                bar = upper
        assert bar is not None, "Every sprite must resolve to a target bar"
        relation = _relation_to_bar(int(pat["r0"]), int(pat["h"]), bar)
        place_r = _dock_row(bar, int(pat["h"]), relation)
        place_c = int(pat["c0"])
        assert int(bar["c_min"]) <= place_c
        assert place_c + int(pat["w"]) - 1 <= int(bar["c_max"])

        src = np.array(pat["content"], dtype=int)
        transformed = src.copy()
        transformed[src == int(pat["outer_color"])] = int(bar["border"])
        if pat["inner_color"] is not None:
            transformed[src == int(pat["inner_color"])] = int(bar["fill"])
        paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


def _sample_layout(n_bars: int) -> tuple[list[int], list[int], int, int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        gaps = [random.randint(*GAP_HEIGHT_RANGE) for _ in range(n_bars - 1)]
        total = sum(heights) + sum(gaps)
        free = GRID_H - total
        if free < 2 * MIN_MARGIN:
            continue
        top_margin = free // 2
        bottom_margin = free - top_margin
        if top_margin < MIN_MARGIN or bottom_margin < MIN_MARGIN:
            continue
        return heights, gaps, top_margin, bottom_margin


def _sample_bar_spans(n_bars: int) -> list[tuple[int, int]]:
    while True:
        spans: list[tuple[int, int]] = []
        for _ in range(n_bars):
            w = random.randint(*BAR_WIDTH_RANGE)
            centered = (GRID_W - w) // 2 + random.randint(-2, 2)
            c0 = min(max(1, centered), GRID_W - w - 2)
            spans.append((int(c0), int(c0 + w - 1)))
        ok = True
        for i in range(n_bars - 1):
            overlap = min(int(spans[i][1]), int(spans[i + 1][1])) - max(int(spans[i][0]), int(spans[i + 1][0])) + 1
            if overlap < MIN_ADJ_OVERLAP:
                ok = False
                break
        if ok:
            return spans


def _build_bars(bg: int, fills: list[int], borders: list[int]) -> tuple[np.ndarray, list[dict]]:
    n_bars = len(fills)
    heights, gaps, top_margin, _ = _sample_layout(n_bars)
    spans = _sample_bar_spans(n_bars)

    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = []
    r = int(top_margin)
    for i in range(n_bars):
        h = int(heights[i])
        r0, r1 = int(r), int(r + h - 1)
        c0, c1 = spans[i]
        fill = int(fills[i])
        border = int(borders[i])
        grid[r0 : r1 + 1, c0] = border
        grid[r0 : r1 + 1, c1] = border
        grid[r0 : r1 + 1, c0 + 1 : c1] = fill
        bars.append(
            {
                "fill": fill,
                "border": border,
                "r_min": r0,
                "r_max": r1,
                "c_min": c0,
                "c_max": c1,
                "h": h,
                "w": int(c1 - c0 + 1),
            }
        )
        r = r1 + 1
        if i < n_bars - 1:
            r += int(gaps[i])
    return grid, bars


def _sample_assignments(n_sprites: int, n_bars: int) -> list[tuple[int, int]]:
    upper_choices = [(gap_idx, gap_idx) for gap_idx in range(n_bars - 1)]
    lower_choices = [(gap_idx, gap_idx + 1) for gap_idx in range(n_bars - 1)]
    all_choices = upper_choices + lower_choices
    counts = {i: 0 for i in range(n_bars)}
    chosen: list[tuple[int, int]] = []

    if n_sprites >= 2:
        first = random.choice(upper_choices)
        second = random.choice(lower_choices)
        for spec in (first, second):
            chosen.append(spec)
            counts[int(spec[1])] += 1

    while len(chosen) < n_sprites:
        spec = random.choice(all_choices)
        if counts[int(spec[1])] >= 2:
            continue
        chosen.append(spec)
        counts[int(spec[1])] += 1

    random.shuffle(chosen)
    return chosen


def _make_sprite(outer_idx: int, inner_idx_candidate: int, bars: list[dict], gap_h: int) -> dict | None:
    outer_bar = bars[outer_idx]
    h_hi = min(SPRITE_H_RANGE[1], int(outer_bar["h"]), int(gap_h - 2))
    if h_hi < SPRITE_H_RANGE[0]:
        return None
    h = random.randint(SPRITE_H_RANGE[0], h_hi)
    w = random.randint(SPRITE_W_RANGE[0], min(SPRITE_W_RANGE[1], int(outer_bar["w"])))
    patch = np.full((h, w), int(outer_bar["fill"]), dtype=int)

    inner_idx = None
    inner_bbox = None
    inner_bar = bars[inner_idx_candidate]
    overlap = min(int(outer_bar["c_max"]), int(inner_bar["c_max"])) - max(int(outer_bar["c_min"]), int(inner_bar["c_min"])) + 1
    if overlap >= 3 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, min(w - 1, overlap))
            if (w - iw) % 2 == 0 and ih * iw * 2 < h * w
        ]
        if valid_inner:
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(inner_bar["fill"])
            inner_idx = int(inner_idx_candidate)
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))

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


def _rows_for_target(gap_top: int, gap_bottom: int, h: int, target_is_upper: bool) -> list[int]:
    rows = []
    for r0 in range(int(gap_top), int(gap_bottom - h + 2)):
        r1 = int(r0 + h - 1)
        dist_up = int(r0 - gap_top)
        dist_down = int(gap_bottom - r1)
        if dist_up < 1 or dist_down < 1:
            continue
        if target_is_upper and dist_up < dist_down:
            rows.append(int(r0))
        if (not target_is_upper) and dist_down < dist_up:
            rows.append(int(r0))
    return rows


def _place_sprites(sprites: list[dict], bars: list[dict]) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    rects: list[tuple[int, int, int, int]] = []
    placed = []

    for sp in sprites:
        gap_idx = int(sp["gap_idx"])
        upper_bar = bars[gap_idx]
        lower_bar = bars[gap_idx + 1]
        outer_bar = bars[int(sp["outer_idx"])]
        h, w = int(sp["h"]), int(sp["w"])
        gap_top = int(upper_bar["r_max"] + 1)
        gap_bottom = int(lower_bar["r_min"] - 1)
        row_candidates = _rows_for_target(gap_top, gap_bottom, h, int(sp["outer_idx"]) == gap_idx)
        if not row_candidates:
            return None

        c0_lo = int(outer_bar["c_min"])
        c0_hi = int(outer_bar["c_max"]) - w + 1
        if c0_lo > c0_hi:
            return None

        candidates = []
        for c0 in range(int(c0_lo), int(c0_hi) + 1):
            if not _interval_free(bar_occ[int(sp["outer_idx"])], c0, w):
                continue
            for r0 in row_candidates:
                if _rect_separated(rects, int(r0), int(c0), h, w):
                    candidates.append((int(r0), int(c0)))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[int(sp["outer_idx"])][c0 : c0 + w] = True

        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[int(sp["outer_idx"])]
        relation = _relation_to_bar(int(sp["r0"]), int(sp["h"]), outer_bar)
        paste(inp, patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

        transformed = patch.copy()
        transformed[patch == int(outer_bar["fill"])] = int(outer_bar["border"])
        if sp["inner_idx"] is not None:
            inner_bar = bars[int(sp["inner_idx"])]
            transformed[patch == int(inner_bar["fill"])] = int(outer_bar["fill"])
        paste(out, transformed, _dock_row(outer_bar, int(sp["h"]), relation), int(sp["c0"]), transparent=int(bg))

    return inp, out


def _analyze_generated_input(grid: np.ndarray) -> dict:
    """Quantify whether the generated sample really uses the fused behavior."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    bars = _extract_bars(g, bg)
    sprites = _extract_visual_sprites(g, bg, bars)
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}

    top_dock = 0
    bottom_dock = 0
    in_internal_gap = 0
    nearest_ok = 0

    for sp in sprites:
        bar = fill_to_bar[int(sp["outer_color"])]
        relation = _relation_to_bar(int(sp["r0"]), int(sp["h"]), bar)
        if relation == "above":
            top_dock += 1
        else:
            bottom_dock += 1

        for gap_idx in range(len(bars) - 1):
            upper = bars[gap_idx]
            lower = bars[gap_idx + 1]
            if int(sp["r0"]) > int(upper["r_max"]) and int(sp["r1"]) < int(lower["r_min"]):
                in_internal_gap += 1
                dist_up = int(sp["r0"] - (upper["r_max"] + 1))
                dist_down = int((lower["r_min"] - 1) - sp["r1"])
                expected_fill = int(upper["fill"]) if dist_up < dist_down else int(lower["fill"])
                if dist_up != dist_down and expected_fill == int(sp["outer_color"]):
                    nearest_ok += 1
                break

    return {
        "sprite_count": len(sprites),
        "top_dock": top_dock,
        "bottom_dock": bottom_dock,
        "in_internal_gap": in_internal_gap,
        "nearest_ok": nearest_ok,
    }


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(x) for x in palette[1 : 1 + n_bars]]
    borders = [int(x) for x in palette[1 + n_bars : 1 + 2 * n_bars]]

    base, bars = _build_bars(bg, fills, borders)
    assignments = _sample_assignments(n_sprites, n_bars)
    sprite_specs = []
    for gap_idx, outer_idx in assignments:
        opposite_idx = int(gap_idx + 1) if int(outer_idx) == int(gap_idx) else int(gap_idx)
        gap_h = int(bars[gap_idx + 1]["r_min"] - bars[gap_idx]["r_max"] - 1)
        sp = _make_sprite(int(outer_idx), opposite_idx, bars, gap_h)
        if sp is None:
            return None
        sprite_specs.append({**sp, "gap_idx": int(gap_idx)})

    sprite_specs.sort(key=lambda s: (s["inner_idx"] is None, -int(s["w"]), -int(s["h"])))
    placed = _place_sprites(sprite_specs, bars)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

    if get_bg(inp) != int(bg):
        return None
    metrics = _analyze_generated_input(inp)
    if metrics["sprite_count"] != n_sprites:
        return None
    if metrics["in_internal_gap"] != n_sprites:
        return None
    if metrics["nearest_ok"] != n_sprites:
        return None
    if n_sprites >= 2 and (metrics["top_dock"] == 0 or metrics["bottom_dock"] == 0):
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
