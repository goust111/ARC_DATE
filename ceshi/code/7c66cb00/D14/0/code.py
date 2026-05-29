# Variant of 7c66cb00
# Dimension: D14 — 多步序列：处理顺序的影响 (multi_step)
# Change: 各精灵独立处理，处理顺序不影响最终结果... → 有序处理（从上到下的条优先）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

# Design analysis
# ---------------
# variant_spec:
#   preserved_structure:
#     - 30x24 grid.
#     - bottom area still contains 2-3 full-width horizontal bars with border color
#       on columns 0 and 23 and fill color in columns 1..22.
#     - top area still contains 3 connected sprites made from bar fill colors.
#     - color logic is unchanged: a detected pattern whose outer color equals a bar
#       fill docks into that bar and swaps colors outer->border, inner->fill.
#     - the original solver's "inner-color component is also its own detected pattern"
#       behavior is preserved, so a two-tone sprite may produce one full docking plus
#       one smaller inner docking in another bar.
#   D14 change:
#     - original behavior: every detected pattern docked at its original c0, so the
#       final result was order-independent.
#     - new behavior: detected patterns are processed sequentially. Priority is:
#         1. target bar from top to bottom,
#         2. within the same target bar, source pattern from top to bottom,
#         3. then left to right.
#     - when a pattern wants to dock into a bar, it first tries its original c0.
#       If that horizontal interval is already occupied inside that bar, it slides
#       right to the first free interval wide enough for its full bbox width.
#       Earlier placements never move.
#   concrete_trace:
#     - Suppose sprite A and sprite B both match the top bar and their input x-ranges
#       overlap. In the original task both would be pasted at their own c0 values,
#       causing overlap or silent overwrite. In this variant A is processed first,
#       keeps its c0, and B is shifted right until its bbox clears A.
#     - Suppose sprite A targets the top bar but contains an inner rectangle whose
#       color matches the second bar. The inner rectangle is still detected as its own
#       pattern. Because bar-1 patterns are processed before bar-2 patterns, that
#       smaller pattern reserves space in bar 2 before the bar-2 full sprite arrives.
#   complexity_calibration:
#     - The generator forces at least one visible right-shift of a full sprite.
#     - It also uses one two-tone sprite so the lower-bar reservation effect can
#       appear, but keeps the number of sprites fixed at 3 to avoid clutter.
#     - Bars remain only 2-3 and all matching still comes from visible fill colors,
#       so the task stays inferable from a few examples.
#   implicit_assumptions_audit:
#     - Original `place_c = pat["c0"]` assumed independent docking; this no longer
#       holds once multiple patterns can want the same bar interval. It is replaced by
#       first-fit placement with occupancy tracking.
#     - Original bar loop grouped by bar but did not define ordering within a bar.
#       The variant now sorts patterns explicitly by target bar index and source row.
#     - Vectorized recoloring would be wrong if a bar border color equals another bar
#       fill color. Recoloring therefore uses masks from the original patch values.
#   legality_and_generator_guarantees:
#     - The generator leaves enough top-space for 3 vertically separated sprites.
#     - Sprite A and B intentionally overlap in x and match the top bar, so B must
#       shift right in the output.
#     - Sprite A also contains an inner rectangle matching the second bar, and sprite
#       C is placed so its target interval overlaps that inner docking, so C may also
#       shift right.
#     - All required shifted destinations are checked to fit before the sample is
#       accepted.
#   visual_observability:
#     - The ordering cue is visible because the upper source sprite in a bar keeps its
#       x-position while the lower one is displaced rightward.
#     - The rule depends only on visible bar order and visible source positions, never
#       on hidden numeric color ids.
#   tidy_visuals:
#     - Only three visual sprites are used.
#     - Bars are short and the top area is taller than in the seed task so stacked
#       source sprites remain easy to separate by eye.
#   new_main_assumptions_checked_against_generator:
#     - Assumption: every detected outer color corresponds to some bar fill. The
#       generator only uses fill colors inside sprites.
#     - Assumption: every forced collision has a free slot to the right. The generator
#       checks this with explicit width constraints before accepting a sample.
#     - Assumption: inner-color regions are single connected rectangles. The generator
#       constructs them that way.

# --- Configuration constants ---
GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 5)
MAX_THREE_BAR_TOTAL = 13
MIN_TOP_AREA = 14
SPRITE_COUNT = 3
GEN_RETRIES = 400


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    current_bar = None
    H, W = grid.shape
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
                        "h": 1,
                    }
            else:
                if current_bar:
                    current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                    bars.append(current_bar)
                    current_bar = None
        else:
            if current_bar:
                current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                bars.append(current_bar)
                current_bar = None
    if current_bar:
        current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
        bars.append(current_bar)
    return bars


def _extract_patterns(grid: np.ndarray, pattern_area_end: int, bg: int) -> list[dict]:
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
            patterns.append(
                {
                    "outer_color": int(outer_color),
                    "inner_color": int(inner_colors[0]) if inner_colors else None,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": np.array(bbox, dtype=int),
                    "kind": "detected",
                }
            )
    return patterns


def _ordered_patterns(patterns: list[dict], bars: list[dict]) -> list[dict]:
    fill_to_bar = {int(bar["fill"]): idx for idx, bar in enumerate(bars)}
    filtered = [pat for pat in patterns if int(pat["outer_color"]) in fill_to_bar]
    return sorted(
        filtered,
        key=lambda pat: (
            int(fill_to_bar[int(pat["outer_color"])]),
            int(pat["r0"]),
            int(pat["c0"]),
            int(pat["h"]),
            int(pat["w"]),
        ),
    )


def _transform_pattern(pat: dict, bar: dict) -> np.ndarray:
    src = np.array(pat["content"], dtype=int)
    transformed = src.copy()
    transformed[src == int(pat["outer_color"])] = int(bar["border"])
    if pat["inner_color"] is not None:
        transformed[src == int(pat["inner_color"])] = int(bar["fill"])
    return transformed


def _first_fit_right(occ: np.ndarray, start: int, width: int) -> int | None:
    max_c = GRID_W - int(width) - 1
    for c0 in range(int(start), max_c + 1):
        if not occ[c0 : c0 + int(width)].any():
            return int(c0)
    return None


def _apply_ordered_docking(
    out: np.ndarray, patterns: list[dict], bars: list[dict], bg: int
) -> list[dict]:
    fill_to_bar = {int(bar["fill"]): idx for idx, bar in enumerate(bars)}
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    placements: list[dict] = []

    for pat in _ordered_patterns(patterns, bars):
        bar_idx = int(fill_to_bar[int(pat["outer_color"])])
        bar = bars[bar_idx]
        place_c = _first_fit_right(bar_occ[bar_idx], int(pat["c0"]), int(pat["w"]))
        if place_c is None:
            raise AssertionError("No rightward docking slot available for ordered placement")
        place_r = int(bar["r_max"] - pat["h"] + 1)
        transformed = _transform_pattern(pat, bar)
        paste(out, transformed, place_r, place_c, transparent=int(bg))
        bar_occ[bar_idx][place_c : place_c + int(pat["w"])] = True
        placements.append(
            {
                "outer_color": int(pat["outer_color"]),
                "kind": pat.get("kind", "detected"),
                "source_r0": int(pat["r0"]),
                "source_c0": int(pat["c0"]),
                "placed_c0": int(place_c),
                "bar_idx": int(bar_idx),
                "shifted": bool(int(place_c) != int(pat["c0"])),
            }
        )

    return placements


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    patterns = _extract_patterns(g, pattern_area_end, bg)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])
    _apply_ordered_docking(out, patterns, bars, bg)
    return out


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        top_h = GRID_H - sum(heights) - (n_bars - 1)
        if top_h < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars = [None] * len(heights)
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
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
        r = int(r0 - 1)
    return grid, bars


def _make_rect_sprite(
    outer_fill: int,
    h: int,
    w: int,
    *,
    inner_fill: int | None = None,
    inner_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    patch = np.full((int(h), int(w)), int(outer_fill), dtype=int)
    if inner_fill is not None and inner_bbox is not None:
        ir0, ic0, ih, iw = map(int, inner_bbox)
        patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(inner_fill)
    return patch


def _pattern_from_sprite(sp: dict, bars: list[dict]) -> list[dict]:
    patterns = [
        {
            "outer_color": int(bars[sp["outer_idx"]]["fill"]),
            "inner_color": int(bars[sp["inner_idx"]]["fill"]) if sp["inner_idx"] is not None else None,
            "r0": int(sp["r0"]),
            "c0": int(sp["c0"]),
            "h": int(sp["h"]),
            "w": int(sp["w"]),
            "content": np.array(sp["patch"], dtype=int),
            "kind": f"{sp['name']}:full",
        }
    ]
    if sp["inner_idx"] is not None and sp["inner_bbox"] is not None:
        ir0, ic0, ih, iw = map(int, sp["inner_bbox"])
        patterns.append(
            {
                "outer_color": int(bars[sp["inner_idx"]]["fill"]),
                "inner_color": None,
                "r0": int(sp["r0"] + ir0),
                "c0": int(sp["c0"] + ic0),
                "h": int(ih),
                "w": int(iw),
                "content": np.full((ih, iw), int(bars[sp["inner_idx"]]["fill"]), dtype=int),
                "kind": f"{sp['name']}:inner",
            }
        )
    return patterns


def _generate_sprite_set(bars: list[dict], top_h: int) -> list[dict] | None:
    top_bar = bars[0]
    lower_bar = bars[1]

    h_a = random.randint(3, min(5, int(top_bar["h"])))
    w_a = random.randint(5, 8)
    h_b = random.randint(2, min(5, int(top_bar["h"])))
    w_b = random.randint(3, 7)
    h_c = random.randint(2, min(5, int(lower_bar["h"])))
    w_c = random.randint(3, 6)

    total_h = h_a + h_b + h_c + 2
    if total_h > top_h:
        return None

    inner_h = random.randint(1, h_a - 2)
    inner_w_choices = [iw for iw in range(2, w_a - 1) if inner_h * iw * 2 < h_a * w_a]
    if not inner_w_choices:
        return None
    inner_w = random.choice(inner_w_choices)
    ir0 = random.randint(1, h_a - inner_h - 1)
    ic0 = random.randint(1, w_a - inner_w - 1)

    max_c_a = min(
        GRID_W - w_a - w_b - 1,
        GRID_W - ic0 - inner_w - w_c - 1,
    )
    if max_c_a < 1:
        return None
    c_a = random.randint(1, max_c_a)
    c_b = random.randint(c_a, min(c_a + w_a - 1, GRID_W - w_b - 1))
    inner_start = c_a + ic0
    c_c = random.randint(inner_start, min(inner_start + inner_w - 1, GRID_W - w_c - 1))

    top_offset = random.randint(0, top_h - total_h)
    r_a = top_offset
    r_b = r_a + h_a + 1
    r_c = r_b + h_b + 1

    sprite_a = {
        "name": "A",
        "outer_idx": 0,
        "inner_idx": 1,
        "h": int(h_a),
        "w": int(w_a),
        "r0": int(r_a),
        "c0": int(c_a),
        "inner_bbox": (int(ir0), int(ic0), int(inner_h), int(inner_w)),
    }
    sprite_a["patch"] = _make_rect_sprite(
        int(top_bar["fill"]),
        int(h_a),
        int(w_a),
        inner_fill=int(lower_bar["fill"]),
        inner_bbox=sprite_a["inner_bbox"],
    )

    sprite_b = {
        "name": "B",
        "outer_idx": 0,
        "inner_idx": None,
        "h": int(h_b),
        "w": int(w_b),
        "r0": int(r_b),
        "c0": int(c_b),
        "inner_bbox": None,
        "patch": _make_rect_sprite(int(top_bar["fill"]), int(h_b), int(w_b)),
    }

    sprite_c = {
        "name": "C",
        "outer_idx": 1,
        "inner_idx": None,
        "h": int(h_c),
        "w": int(w_c),
        "r0": int(r_c),
        "c0": int(c_c),
        "inner_bbox": None,
        "patch": _make_rect_sprite(int(lower_bar["fill"]), int(h_c), int(w_c)),
    }

    return [sprite_a, sprite_b, sprite_c]


def _render_case(base: np.ndarray, sprites: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    inp = base.copy()
    out = base.copy()

    patterns: list[dict] = []
    for sp in sprites:
        paste(inp, sp["patch"], int(sp["r0"]), int(sp["c0"]), transparent=int(bg))
        patterns.extend(_pattern_from_sprite(sp, bars))

    placements = _apply_ordered_docking(out, patterns, bars, bg)
    return inp, out, placements


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    sprites = _generate_sprite_set(bars, top_h)
    if sprites is None:
        return None

    inp, expected, placements = _render_case(base, sprites, bars, bg)

    full_shifts = [p for p in placements if p["kind"].endswith(":full") and p["shifted"]]
    if not full_shifts:
        return None
    if not any(p["kind"] == "B:full" and p["shifted"] for p in placements):
        return None

    if get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
