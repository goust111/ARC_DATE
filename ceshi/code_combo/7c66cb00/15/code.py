# Combo variant of 7c66cb00 (index 15)
# combo_spec: D8/0+D14/0+D18/2
# Name: 逐行下移渐变有序处理保留源

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

# Design analysis
# ---------------
# variant_spec:
#   preserved_structure:
#     - Grid size stays 30x24.
#     - The lower area still contains full-width horizontal bars whose fill colors
#       are the matching keys for the top-area sprites.
#     - Top sprites are still built only from bar fill colors, and the solver still
#       uses the seed task's color-component detection. That means a two-tone sprite
#       yields both one full bbox pattern and one inner-color sub-pattern.
#     - Recoloring at the destination is still the same semantic swap as the seed:
#       outer color -> target bar border, inner color -> target bar fill.
#   fused_changes:
#     - D8/0 keep_source: the output no longer clears the top region. The original
#       sprites remain visible exactly where they started.
#     - D14/0 ordered placement: patterns are processed by target bar top-to-bottom,
#       then by source row top-to-bottom, then left-to-right. Within each bar, a
#       pattern first tries its source c0 and otherwise slides right to the first
#       free interval. Earlier placements reserve space permanently.
#     - D18/2 visible trajectory: before the docked copy appears, the pattern leaves
#       row-by-row afterimages on the path to its chosen landing spot. The trail
#       starts in source colors, then uses the swapped destination colors, then fades
#       by turning many cells back into background.
#   concrete_trace:
#     - Example with visible sprites A, B, C:
#       A is the upper sprite targeting bar 1 and contains an inner rectangle for
#       bar 2. B is a lower sprite also targeting bar 1, with overlapping x-range.
#       C targets bar 2 and overlaps the inner rectangle's x-range.
#     - Seed task:
#       A and B would both paste into bar 1 at their own c0 values, source sprites
#       would be erased, and there would be no visible path.
#     - This combo:
#       A docks first into bar 1, keeping its x-position and leaving a fading trail.
#       B is processed later, sees A's occupied interval, shifts right, and its trail
#       bends toward that shifted landing column. A's inner component reserves space
#       in bar 2 before C is processed, so C may shift right there as well.
#       The final grid visibly contains the original top sprites, the lower docked
#       copies, and the intermediate trails between them.
#   complexity_calibration:
#     - The generator always uses three visible sprites so the ordered effect has
#       repeated support without overcrowding the board.
#     - It fixes the puzzle to two bars. This reduces visual clutter from the trails
#       while preserving the seed task's matching logic and bar-based reasoning.
#     - At least one full sprite is forced to shift right, and a two-tone sprite is
#       always present so the hidden sub-pattern behavior still matters.
#     - Trail rendering uses only source colors, destination bar colors, and bg. No
#       arbitrary palette remapping is introduced.
#   implicit_assumptions_audit:
#     - Seed assumption broken: `out[:pattern_area_end] = bg` is no longer valid,
#       because source sprites must remain. The combo therefore starts from `out=g`.
#     - Seed assumption broken: `place_c = pat["c0"]` is no longer valid once order
#       matters. A first-fit-right search with bar occupancy is required.
#     - New assumption: trail drawing must not destroy the kept source sprites or the
#       bars themselves. To guarantee that, trails are first painted onto a separate
#       background-only layer and then merged back only onto bg cells of the output.
#     - New assumption: recoloring must be based on masks from the original pattern,
#       not from already-modified values, otherwise bar-border colors that equal some
#       other fill could cascade incorrectly.
#     - New assumption: a target bar exists for every detected outer color. The
#       generator guarantees that sprites only use bar fill colors.
#   legality_and_generator_guarantees:
#     - All source sprites fit above the first bar with at least one row of spacing.
#     - Forced x-overlaps are constructed so rightward fallback destinations exist.
#     - Every target paste stays within columns 1..22 and within the chosen bar's
#       height. Inner rectangles are single connected boxes so the solver's component
#       extraction matches the generator's intended sub-pattern.
#     - Trails may geometrically pass through source rows, but the merge step writes
#       trail cells only into background, so sources remain legible and unchanged.
#   visual_observability:
#     - Source preservation is obvious because non-bg sprites still sit in the top
#       area of the output.
#     - Ordered processing is visible because the upper sprite in a bar keeps the
#       earlier slot while the lower one is displaced rightward.
#     - The trajectory is visible as a vertical/diagonal band in the empty corridor,
#       with color phases source -> swapped destination -> sparse fade toward bg.
#     - All ordering cues come from visible top-to-bottom bar order and visible
#       sprite positions, never from numeric color ids.
#   tidy_visuals:
#     - Bars are short and the top area is tall, leaving a wide corridor for trails.
#     - Sprite widths are moderate so the trail bands stay readable.
#     - Trails are clipped by the bg-only merge rule, preventing them from muddying
#       the source sprites or filling the solid bar interiors.
#   new_main_assumptions_checked_against_generator:
#     - Assumption: there is enough free room to the right for every forced shift.
#       The generator enforces explicit width inequalities before accepting a case.
#     - Assumption: at least one trail row stays visible outside the bars. The tall
#       top area and mandatory trail distance guarantee this.
#     - Assumption: full-sprite and inner-component ordering can both be inferred
#       from geometry alone. The generator uses one nested sprite plus two plain
#       sprites so examples repeatedly expose the pattern.

# --- Configuration constants ---
GRID_H = 30
GRID_W = 24
BAR_COUNT = 2
BAR_HEIGHT_RANGE = (4, 5)
MIN_TOP_AREA = 17
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
                        current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                        "h": 1,
                    }
            elif current_bar:
                current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                bars.append(current_bar)
                current_bar = None
        elif current_bar:
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


def _transform_pattern(pat: dict, bar: dict, bg: int) -> np.ndarray:
    src = np.array(pat["content"], dtype=int)
    transformed = src.copy()
    transformed[src == int(pat["outer_color"])] = int(bar["border"])
    if pat["inner_color"] is not None:
        transformed[src == int(pat["inner_color"])] = int(bar["fill"])
    transformed[src == int(bg)] = int(bg)
    return transformed


def _trail_patch(pat: dict, bar: dict, step: int, total_steps: int, bg: int) -> np.ndarray:
    src = np.array(pat["content"], dtype=int)
    transformed = _transform_pattern(pat, bar, bg)
    if total_steps <= 0:
        return np.full_like(src, int(bg))

    progress = float(step) / float(total_steps)
    if progress < 0.34:
        return src.copy()
    if progress < 0.67:
        return transformed

    trail = transformed.copy()
    rr, cc = np.indices(src.shape)
    if progress < 0.84:
        density = 2
    elif progress < 0.93:
        density = 3
    else:
        density = 4
    keep_mask = ((rr + cc + int(step)) % int(density)) == 0
    trail[(src != int(bg)) & (~keep_mask)] = int(bg)
    return trail


def _first_fit_right(occ: np.ndarray, start: int, width: int) -> int | None:
    max_c = GRID_W - int(width) - 1
    for c0 in range(int(start), max_c + 1):
        if not occ[c0 : c0 + int(width)].any():
            return int(c0)
    return None


def _emit_trail(
    trail_layer: np.ndarray,
    pat: dict,
    bar: dict,
    place_r: int,
    place_c: int,
    bg: int,
) -> None:
    start_r = int(pat["r0"])
    total_steps = int(place_r - start_r)
    if total_steps <= 1:
        return
    for step in range(1, total_steps):
        trail_r = int(start_r + step)
        raw_c = float(pat["c0"]) + (float(place_c) - float(pat["c0"])) * float(step) / float(total_steps)
        trail_c = int(round(raw_c))
        patch = _trail_patch(pat, bar, int(step), int(total_steps), int(bg))
        paste(trail_layer, patch, trail_r, trail_c, transparent=int(bg))


def _apply_combo(
    base: np.ndarray,
    patterns: list[dict],
    bars: list[dict],
    bg: int,
    *,
    add_trail: bool,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    out = np.array(base, dtype=int).copy()
    fill_to_bar = {int(bar["fill"]): idx for idx, bar in enumerate(bars)}
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    trail_layer = np.full_like(out, int(bg))
    placements: list[dict] = []

    for pat in _ordered_patterns(patterns, bars):
        bar_idx = int(fill_to_bar[int(pat["outer_color"])])
        bar = bars[bar_idx]
        place_c = _first_fit_right(bar_occ[bar_idx], int(pat["c0"]), int(pat["w"]))
        if place_c is None:
            raise AssertionError("No rightward docking slot available for ordered placement")
        place_r = int(bar["r_max"] - pat["h"] + 1)
        if add_trail:
            _emit_trail(trail_layer, pat, bar, int(place_r), int(place_c), int(bg))
        transformed = _transform_pattern(pat, bar, bg)
        paste(out, transformed, int(place_r), int(place_c), transparent=int(bg))
        bar_occ[bar_idx][int(place_c) : int(place_c) + int(pat["w"])] = True
        placements.append(
            {
                "kind": pat.get("kind", "detected"),
                "bar_idx": int(bar_idx),
                "source_r0": int(pat["r0"]),
                "source_c0": int(pat["c0"]),
                "placed_r0": int(place_r),
                "placed_c0": int(place_c),
                "w": int(pat["w"]),
                "h": int(pat["h"]),
                "shifted": bool(int(place_c) != int(pat["c0"])),
            }
        )

    if add_trail:
        mask = (out == int(bg)) & (trail_layer != int(bg))
        out[mask] = trail_layer[mask]
    return out, placements, trail_layer


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    patterns = _extract_patterns(g, pattern_area_end, bg)
    out, placements, trail_layer = _apply_combo(g, patterns, bars, bg, add_trail=True)
    return out, {
        "bg": int(bg),
        "bars": bars,
        "patterns": patterns,
        "placements": placements,
        "pattern_area_end": int(pattern_area_end),
        "trail_layer": trail_layer,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy color-matched sprites into bars while ordered docking leaves a fading descent trail.

    Input structure:
    - The lower zone contains two full-width horizontal bars with border colors on the outer columns and fill colors inside.
    - The top zone contains small sprites made from those bar fill colors; a two-tone sprite may contain a nested inner region of another bar's fill color.
    - Background is the globally most common color.

    Rule:
    - Detect each top-area color component, crop its full bbox, and match it to the bar whose fill matches that component's outer color.
    - Process matches in visible order: target bars from top to bottom, then source patterns from top to bottom, sliding right when an earlier docking already occupies the desired slot.
    - Keep the source sprite in place, place the recolored docked copy in the matched bar, and draw row-by-row intermediate copies on the way down so the path fades from sprite colors to destination colors and then toward background.

    Key insight:
    - The final landing slot is not independent anymore: ordered first-fit placement decides both where the docked copy ends up and where its visible fading trail bends, while the original source remains as a persistent reference.
    """
    out, _ = _solve_details(grid)
    return out


def _sample_bar_heights() -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(BAR_COUNT)]
        top_h = GRID_H - sum(heights) - (BAR_COUNT - 1)
        if top_h >= MIN_TOP_AREA:
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

    h_a = random.randint(3, min(4, int(top_bar["h"])))
    w_a = random.randint(4, 6)
    h_b = random.randint(2, min(4, int(top_bar["h"])))
    w_b = random.randint(3, 5)
    h_c = random.randint(2, min(4, int(lower_bar["h"])))
    w_c = random.randint(3, 4)

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

    min_overlap_ab = min(2, w_b, w_a - 1)
    max_overlap_ab = min(w_b, w_a - 1)
    if min_overlap_ab > max_overlap_ab:
        return None
    overlap_ab = random.randint(min_overlap_ab, max_overlap_ab)
    c_b = int(c_a + w_a - overlap_ab)
    if c_b + w_b > GRID_W - 1:
        return None

    inner_start = int(c_a + ic0)
    max_overlap_ic = min(w_c, inner_w)
    if max_overlap_ic < 1:
        return None
    overlap_ic = random.randint(1, max_overlap_ic)
    c_c = int(inner_start + inner_w - overlap_ic)
    if c_c < 1 or c_c + w_c > GRID_W - 1:
        return None

    top_offset = random.randint(0, top_h - total_h)
    r_a = int(top_offset)
    r_b = int(r_a + h_a + 1)
    r_c = int(r_b + h_b + 1)

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


def _render_case(
    base: np.ndarray,
    sprites: list[dict],
    bars: list[dict],
    bg: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    inp = np.array(base, dtype=int).copy()
    patterns: list[dict] = []
    for sp in sprites:
        paste(inp, sp["patch"], int(sp["r0"]), int(sp["c0"]), transparent=int(bg))
        patterns.extend(_pattern_from_sprite(sp, bars))
    expected, placements, trail_layer = _apply_combo(inp, patterns, bars, bg, add_trail=True)
    return inp, expected, placements, trail_layer


def _behavior_metrics(
    inp: np.ndarray,
    out: np.ndarray,
    bars: list[dict],
    patterns: list[dict],
    placements: list[dict],
    trail_layer: np.ndarray,
    bg: int,
) -> dict:
    no_trail, _, _ = _apply_combo(inp, patterns, bars, bg, add_trail=False)
    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    top_in = inp[:pattern_area_end, :]
    top_out = out[:pattern_area_end, :]
    trail_only_mask = (out != no_trail) & (no_trail == int(bg))
    return {
        "shifted_full_count": int(sum(p["kind"].endswith(":full") and p["shifted"] for p in placements)),
        "shifted_any_count": int(sum(p["shifted"] for p in placements)),
        "trail_visible_cells": int(np.count_nonzero(trail_only_mask)),
        "trail_visible_rows": int(np.count_nonzero(np.any(trail_only_mask, axis=1))),
        "retained_source_cells": int(np.count_nonzero((top_in != int(bg)) & (top_out == top_in))),
        "source_cells": int(np.count_nonzero(top_in != int(bg))),
        "trail_layer_cells": int(np.count_nonzero(trail_layer != int(bg))),
    }


def _gen_attempt() -> np.ndarray | None:
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + BAR_COUNT]]
    borders = [int(v) for v in palette[1 + BAR_COUNT : 1 + 2 * BAR_COUNT]]

    heights = _sample_bar_heights()
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    sprites = _generate_sprite_set(bars, top_h)
    if sprites is None:
        return None

    inp, expected, placements, trail_layer = _render_case(base, sprites, bars, bg)
    patterns: list[dict] = []
    for sp in sprites:
        patterns.extend(_pattern_from_sprite(sp, bars))
    metrics = _behavior_metrics(inp, expected, bars, patterns, placements, trail_layer, bg)

    if metrics["shifted_full_count"] < 1:
        return None
    if metrics["trail_visible_cells"] < 12 or metrics["trail_visible_rows"] < 4:
        return None
    if metrics["retained_source_cells"] != metrics["source_cells"]:
        return None
    if metrics["trail_layer_cells"] < metrics["trail_visible_cells"]:
        return None
    solved = main(inp)
    if get_bg(inp) != bg:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
