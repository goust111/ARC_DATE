# Combo variant of 7c66cb00 (index 3)
# combo_spec: D4/0+D7/0+D16/0
# Name: 垂直条顶部对齐与水平滑动

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
# - Baseline scene:
#   - 2-3 full-width horizontal bars are stacked at the bottom.
#   - Sprites are in the top strip.
#   - Each sprite matches a bar by outer_color == bar.fill, keeps its x-position,
#     bottom-aligns, and swaps colors.
# - Combo scene:
#   - The whole layout is rotated conceptually into "left bars / right sprites".
#   - Bars become 2-3 full-height vertical strips anchored on the LEFT.
#   - Each bar has fill in rows 1..H-2 and border on row 0 and row H-1.
#   - Sprites are detected only in the RIGHT-side field.
#   - A matching pattern no longer preserves either input coordinate:
#     it top-aligns to row 0 and joins a right-aligned horizontal packed block
#     inside its target bar.
# - Concrete trace:
#   - Suppose bar A spans cols 5..9 with fill=3 and border=7.
#   - A sprite on the right at rows 11..14, cols 18..20 has outer=3.
#   - Original task would place it in a bottom bar at row = bar.r_max-h+1 and
#     keep c0=18.
#   - This combo places it at row 0 inside the vertical bar. If bar A has two
#     matching patterns with widths 2 and 3, their total width is 5, so the packed
#     block starts at col 5 and ends at col 9. The sprite order is derived from the
#     input-side top-to-bottom order because that is the visible ordering cue in the
#     right strip.
#
# b) Complexity calibration
# - The destination structure changes visibly: wide bottom bars become narrow side bars.
# - The movement changes in two independent observable ways:
#   - vertical: every matched pattern rises to the top of its target bar
#   - horizontal: every matched pattern slides leftward into a right-flush block
# - Packing uses only one visible grouping rule: per bar, order by input r0. This is
#   enough structure to avoid ambiguity without introducing hidden metadata.
#
# c) Implicit assumptions audited
# - Original row-scan bar detection fails because bars are now column-major objects.
# - Original `pattern_area_end` row crop fails because sprites are no longer above bars;
#   the solver must crop by columns using the bar cluster extent.
# - Original `place_r = bar["r_max"] - h + 1` fails under D7; combo needs `place_r = 0`.
# - Original `place_c = pat["c0"]` fails under D16; combo recomputes `place_c` from a
#   per-bar packed block width.
# - The original generator reserved horizontal occupancy in bars. That assumption breaks
#   because patterns now share the same top rows and only need non-overlapping widths.
# - New assumption: bars are always on the LEFT and sprites are always on the RIGHT.
#   The generator enforces that exact layout.
#
# d) Placement legality
# - Generator leaves one background column between adjacent bars and one more background
#   column between the bar cluster and the sprite field.
# - Each outer sprite width is clamped to its outer target bar width.
# - Each inner monochrome bbox width is clamped to its own target bar width.
# - For every bar, the generator reserves the total packed width contributed by all
#   patterns that will dock there, including inner sub-patterns detected separately by
#   the solver. Samples that would overflow a bar are rejected.
# - Right-side input sprites are pairwise separated by at least one background cell.
#
# e) Visual observability
# - A human can see bars on the left and sprites on the right immediately; this is much
#   more obvious than a hidden coordinate rule.
# - Top alignment is visible because sprites in the output all touch row 0 of their bars.
# - Right-edge packing is visible because each used bar's occupied columns end exactly at
#   that bar's right boundary.
#
# f) Visual cleanliness
# - The bar cluster and sprite field never overlap.
# - Only 2-3 sprites are used, preserving enough empty space for clean separation.
# - Packing within each bar is one-dimensional and contiguous, so output bars remain easy
#   to parse even when inner sub-patterns also dock independently.
#
# g) Change inferability
# - Colors keep their original semantic roles:
#   - sprite outer color identifies the target bar because it equals bar.fill
#   - sprite outer recolors to bar.border
#   - sprite inner recolors to bar.fill
# - No hidden use of absolute color ids is introduced. All ordering decisions come from
#   visible geometry: which side contains sprites, and the top-to-bottom order of those
#   sprites in the input strip.
#
# h) Fusion-specific anti-degeneracy target
# - A good sample should exercise all three combo aspects together:
#   - left-side vertical bars
#   - at least one pattern with non-zero upward motion (`r0 > 0`)
#   - at least one pattern with non-zero horizontal slide into a right-flush block
#   - at least two distinct target bars used by outer sprites, so color-based matching is
#     observable without relying on a single destination strip
# - The generator measures these metrics and rejects samples that do not show these cues.
#
# i) New main() assumptions and generator guarantees
# - Assumption: full-height left-anchored columns with top/bottom border dominate the
#   bar cluster. `_build_bars()` renders exactly that.
# - Assumption: all movable patterns lie strictly to the right of the final bar cluster.
#   `_place_sprites()` confines them to the right field only.
# - Assumption: total packed width per target bar fits inside that bar.
#   `_reservations_allow()` enforces that before any sprite is accepted.


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_WIDTH_RANGE = (4, 8)
MIN_SIDE_AREA = 7
SPRITE_H_RANGE = (2, 6)
SPRITE_W_RANGE = (2, 5)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 500


def _find_vertical_bars(grid: np.ndarray, bg: int) -> list[dict]:
    """Detect left-anchored full-height vertical bars."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bars: list[dict] = []
    current_bar: dict | None = None

    for c in range(w):
        col = g[:, c]
        non_bg = [int(v) for v in col if int(v) != int(bg)]
        if not non_bg:
            if current_bar is not None:
                bars.append(current_bar)
                current_bar = None
            continue

        fill_color, count = Counter(non_bg).most_common(1)[0]
        if int(count) <= int(h * 0.7):
            if current_bar is not None:
                bars.append(current_bar)
                current_bar = None
            continue

        border = int(col[0]) if int(col[0]) != int(fill_color) else int(col[-1])
        if current_bar is not None and int(current_bar["fill"]) == int(fill_color):
            current_bar["c_max"] = int(c)
        else:
            if current_bar is not None:
                bars.append(current_bar)
            current_bar = {
                "fill": int(fill_color),
                "border": int(border),
                "c_min": int(c),
                "c_max": int(c),
            }

    if current_bar is not None:
        bars.append(current_bar)

    assert bars, "No bars found — grid must contain left-side vertical bars"
    assert int(bars[0]["c_min"]) == 0, "Combo layout expects the bar cluster on the left edge"

    for bar in bars:
        bar["w"] = int(bar["c_max"] - bar["c_min"] + 1)
        bar["r_min"] = 0
        bar["r_max"] = h - 1

    return bars


def _extract_patterns(g: np.ndarray, bg: int, bars: list[dict]) -> tuple[list[dict], slice]:
    """Extract per-color connected patterns from the right-side sprite field."""
    field_slice = slice(int(bars[-1]["c_max"]) + 1, g.shape[1])
    col_offset = int(bars[-1]["c_max"]) + 1
    region = g[:, field_slice]
    patterns: list[dict] = []

    for color in sorted(set(map(int, region.flatten())) - {int(bg)}):
        mask = region == int(color)
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
            inner_colors = [int(v) for v in colors_in_bbox if int(v) != int(outer_color)]
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

    return patterns, field_slice


def _transform_pattern(pattern: dict, bar: dict) -> np.ndarray:
    transformed = np.array(pattern["content"], dtype=int).copy()
    transformed[transformed == int(pattern["outer_color"])] = int(bar["border"])
    if pattern["inner_color"] is not None:
        transformed[transformed == int(pattern["inner_color"])] = int(bar["fill"])
    return transformed


def _plan_pattern_placements(patterns: list[dict], bars: list[dict]) -> list[dict]:
    """Top-align and right-pack all patterns that match each bar."""
    placements: list[dict] = []
    for bar in bars:
        matching = sorted(
            [p for p in patterns if int(p["outer_color"]) == int(bar["fill"])],
            key=lambda p: (int(p["r0"]), int(p["c0"]), int(p["h"]), int(p["w"])),
        )
        total_w = sum(int(p["w"]) for p in matching)
        if total_w > int(bar["w"]):
            raise ValueError("packed patterns do not fit inside target vertical bar")

        cursor = int(bar["c_max"] - total_w + 1)
        for pattern in matching:
            placements.append(
                {
                    "pattern": pattern,
                    "bar": bar,
                    "place_r": 0,
                    "place_c": int(cursor),
                }
            )
            cursor += int(pattern["w"])

    return placements


def _solve_core(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    bars = _find_vertical_bars(g, bg)
    patterns, field_slice = _extract_patterns(g, bg, bars)

    out[:, field_slice] = np.where(out[:, field_slice] != int(bg), int(bg), out[:, field_slice])

    for placement in _plan_pattern_placements(patterns, bars):
        transformed = _transform_pattern(placement["pattern"], placement["bar"])
        paste(out, transformed, int(placement["place_r"]), int(placement["place_c"]), transparent=int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide right-side two-tone sprites into left-side vertical bars with top docking.

    Input structure:
    - `bg` is the most common color.
    - The left edge holds 2-3 full-height vertical bars separated by background columns.
    - The right-side field contains 2-3 small sprites made from those bar fill colors.

    Rule:
    - Match each detected sprite pattern to the bar whose fill color equals the pattern's outer color.
    - Erase the original right-side sprites.
    - Recolor outer pixels to the matched bar border color and recolor inner pixels to the matched bar fill color.
    - Place every matched pattern top-aligned at row 0 and pack all patterns for the same bar into a single block whose right edge touches that bar's right boundary.

    Key insight:
    - The original bar-matching color rule stays intact, but the spatial semantics change together:
      bars are vertical on the left, sprites come from the right, docking happens at the top, and x-placement is recomputed by right-edge packing instead of preserving the input column.
    """
    return _solve_core(grid)


def _sample_bar_widths(n_bars: int) -> list[int]:
    """Sample left-bar widths while leaving a right-side sprite field."""
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
) -> tuple[np.ndarray, list[dict], tuple[int, int]]:
    """Render the left-side vertical bars and return the right-side sprite bounds."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = []
    c = 0

    for fill, border, width in zip(fills, borders, widths):
        c0, c1 = int(c), int(c + width - 1)
        grid[1 : GRID_H - 1, c0 : c1 + 1] = int(fill)
        grid[0, c0 : c1 + 1] = int(border)
        grid[GRID_H - 1, c0 : c1 + 1] = int(border)
        bars.append(
            {
                "fill": int(fill),
                "border": int(border),
                "c_min": int(c0),
                "c_max": int(c1),
                "w": int(width),
                "r_min": 0,
                "r_max": GRID_H - 1,
            }
        )
        c = c1 + 2

    sprite_bounds = (int(c), GRID_W - 1)
    return grid, bars, sprite_bounds


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    """Create a right-side sprite whose outer color matches one target bar."""
    outer_fill = int(bars[outer_idx]["fill"])
    max_w = min(SPRITE_W_RANGE[1], int(bars[outer_idx]["w"]))
    w = random.randint(SPRITE_W_RANGE[0], max_w)
    h = random.randint(*SPRITE_H_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        random.shuffle(inner_choices)
        for candidate_idx in inner_choices:
            valid_inner = [
                (ih, iw)
                for ih in range(1, h - 1)
                for iw in range(1, w - 1)
                if iw <= int(bars[candidate_idx]["w"]) and ih * iw * 2 < h * w
            ]
            if not valid_inner:
                continue
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = random.randint(1, w - iw - 1)
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[candidate_idx]["fill"])
            inner_idx = int(candidate_idx)
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))
            break

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": inner_idx,
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _rect_separated(
    rects: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> bool:
    """Require at least one background cell between right-side sprites."""
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _sprite_contributions(sprite: dict) -> list[tuple[int, int]]:
    """Return (bar_idx, width) contributions after per-color pattern extraction."""
    parts = [(int(sprite["outer_idx"]), int(sprite["w"]))]
    if sprite["inner_idx"] is not None:
        _ir0, _ic0, _ih, iw = sprite["inner_bbox"]
        parts.append((int(sprite["inner_idx"]), int(iw)))
    return parts


def _reservations_allow(reserved: list[int], sprite: dict, bars: list[dict]) -> bool:
    trial = list(map(int, reserved))
    for bar_idx, width in _sprite_contributions(sprite):
        trial[bar_idx] += int(width)
    return all(int(trial[i]) <= int(bars[i]["w"]) for i in range(len(bars)))


def _apply_reservations(reserved: list[int], sprite: dict) -> None:
    for bar_idx, width in _sprite_contributions(sprite):
        reserved[bar_idx] += int(width)


def _place_sprites(
    sprites: list[dict],
    sprite_bounds: tuple[int, int],
) -> list[dict] | None:
    """Place all input sprites inside the right-side field."""
    c_min, c_max = map(int, sprite_bounds)
    right_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sprite in sorted(sprites, key=lambda s: (-int(s["h"]) * int(s["w"]), int(s["outer_idx"]))):
        h, w = int(sprite["h"]), int(sprite["w"])
        if c_min > c_max - w + 1:
            return None

        candidates: list[tuple[int, int]] = []
        rows = list(range(0, GRID_H - h + 1))
        cols = list(range(c_min, c_max - w + 2))
        random.shuffle(rows)
        random.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                if _rect_separated(right_rects, r0, c0, h, w):
                    candidates.append((int(r0), int(c0)))
            if candidates:
                break

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        placed.append({**sprite, "r0": int(r0), "c0": int(c0)})
        right_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render_input(base: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    inp = base.copy()
    for sprite in placed:
        paste(inp, sprite["patch"], int(sprite["r0"]), int(sprite["c0"]), transparent=int(bg))
    return inp


def _fusion_metrics(inp: np.ndarray) -> dict:
    """Quantify whether the generated sample actually exercises the combo behavior."""
    bg = int(get_bg(inp))
    bars = _find_vertical_bars(inp, bg)
    patterns, _field_slice = _extract_patterns(inp, bg, bars)
    placements = _plan_pattern_placements(patterns, bars)

    horizontal_shifts = [abs(int(item["place_c"]) - int(item["pattern"]["c0"])) for item in placements]
    vertical_shifts = [abs(int(item["place_r"]) - int(item["pattern"]["r0"])) for item in placements]

    right_gap_total = 0
    for bar in bars:
        matching = [item for item in placements if int(item["bar"]["fill"]) == int(bar["fill"])]
        if not matching:
            continue
        rightmost = max(int(item["place_c"]) + int(item["pattern"]["w"]) - 1 for item in matching)
        right_gap_total += int(bar["c_max"]) - rightmost

    return {
        "pattern_count": len(placements),
        "horizontal_shift_count": sum(1 for v in horizontal_shifts if v > 0),
        "vertical_shift_count": sum(1 for v in vertical_shifts if v > 0),
        "total_horizontal_shift": sum(horizontal_shifts),
        "total_vertical_shift": sum(vertical_shifts),
        "right_gap_total": int(right_gap_total),
        "bars_used": sum(
            1
            for bar in bars
            if any(int(item["bar"]["fill"]) == int(bar["fill"]) for item in placements)
        ),
    }


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]

    widths = _sample_bar_widths(n_bars)
    base, bars, sprite_bounds = _build_bars(bg, fills, borders, widths)

    distinct_outer_targets = random.sample(range(n_bars), k=2)
    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    for target in distinct_outer_targets:
        outer_slots.remove(target)
    random.shuffle(outer_slots)
    outer_sequence = distinct_outer_targets + outer_slots[: n_sprites - len(distinct_outer_targets)]
    random.shuffle(outer_sequence)
    reserved = [0 for _ in bars]
    sprites: list[dict] = []

    for i in range(n_sprites):
        outer_idx = int(outer_sequence[i])
        built = None
        for _ in range(80):
            sprite = _make_sprite(outer_idx, bars)
            if _reservations_allow(reserved, sprite, bars):
                _apply_reservations(reserved, sprite)
                built = sprite
                break
        if built is None:
            return None
        sprites.append(built)

    placed = _place_sprites(sprites, sprite_bounds)
    if placed is None:
        return None

    inp = _render_input(base, placed, bg)
    try:
        metrics = _fusion_metrics(inp)
    except Exception:
        return None
    if metrics["pattern_count"] == 0:
        return None
    if metrics["horizontal_shift_count"] == 0 or metrics["vertical_shift_count"] == 0:
        return None
    if metrics["bars_used"] < 2:
        return None
    if metrics["right_gap_total"] != 0:
        return None
    if get_bg(inp) != bg:
        return None

    try:
        _ = _solve_core(inp)
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
