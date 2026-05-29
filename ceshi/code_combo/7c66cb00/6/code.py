# Combo variant of 7c66cb00 (index 6)
# combo_spec: D6/3+D14/0+D17/1
# Name: 条高度编码奇偶交换与有序处理

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis
# ---------------
# a) Concrete behavior and trace
# - Base scene: bottom horizontal bars, top detached sprites made from bar fill
#   colors. Matching still uses "sprite outer color == target bar fill color".
# - D6 contributes the fact that recoloring is no longer unconditional. In the
#   fused rule, D17 supplies the observable controller for that conditional:
#   target bar height parity. So the final behavior is:
#     even-height target bar -> swap outer->border and inner->fill
#     odd-height target bar  -> keep the detected patch colors unchanged
# - D14 contributes ordered docking with collision avoidance. Pieces are not
#   independent anymore. Every detected pattern is processed one at a time:
#     1. target bar from top to bottom,
#     2. inside the same bar, source pattern from top to bottom,
#     3. then left to right.
#   Each pattern tries its source c0 first and otherwise slides right to the
#   first free interval large enough for its bbox width.
# - Concrete fused trace used by the generator:
#   sprite A targets the upper bar and contains a smaller rectangle colored like
#   the lower bar. The solver therefore detects:
#     A:full  -> upper bar
#     A:inner -> lower bar
#   sprite B also targets the upper bar and starts under A with overlapping x.
#   So B is processed after A:full and must shift right.
#   sprite C targets the lower bar and starts inside A:inner's horizontal range.
#   So C often shifts right inside the lower bar as well.
#   Whether the docked colors actually swap depends on each target bar's height
#   parity, not on the sprite's own shape or area.
#
# b) Complexity calibration
# - Only 2 bars and 3 visual sprites are used. That is enough to show both
#   "ordered right-shift" and "parity-controlled recolor" without clutter.
# - Every sample contains one even-height bar and one odd-height bar, and at
#   least one full sprite targets each parity branch.
# - Any full sprite targeting an odd bar is forced to be two-tone, otherwise the
#   no-swap branch could disappear into the bar fill and become hard to infer.
#
# c) Audit of original assumptions
# - Original `place_c = pat["c0"]` is invalid once order matters; it is replaced
#   by occupancy tracking plus first-fit rightward search.
# - Original unconditional recolor is invalid once D6/D17 make recolor
#   conditional; recolor is now guarded by `bar["swap"]`.
# - Original solver already treated inner-color components as separate patterns.
#   That behavior is preserved and becomes important because those smaller pieces
#   now reserve space before later patterns dock.
# - Recoloring is applied using masks from the original patch values, so border
#   colors can safely equal some other bar's fill color without cascading writes.
#
# d) Placement legality
# - Bars are full-width with one background row between them.
# - Top sprites are vertically separated by one background row and remain inside
#   the top area.
# - Generator geometry guarantees enough room to the right for the forced shift
#   of B, and normally for C as well; final acceptance still verifies docking by
#   running the same ordered placement logic used by `main()`.
#
# e) Visual observability
# - Parity is visible because samples include both an even and an odd bar, and
#   the odd-targeted full sprite is always two-tone.
# - Parity is not tied to fixed bar position: some samples have upper-even /
#   lower-odd, others upper-odd / lower-even.
# - Order is visible because at least one full sprite is forced to shift right.
#
# f) Visual cleanliness
# - The scene uses only 3 visual sprites.
# - Small independently docked inner pieces exist, but they stay rectangular and
#   use the same bar colors, so they read as structured sub-patterns rather than
#   noise.
#
# g) Change inferability
# - The only color change question the solver must infer is binary:
#   even-height bar swaps, odd-height bar does not.
# - That binary rule is repeated in every sample and supported by both bar
#   parities appearing in the same grid.
#
# New `main()` assumptions checked against the generator
# - Every non-background color in the top area is one of the bar fill colors.
# - Every detected outer color maps to exactly one bar fill.
# - Ordered first-fit docking always has a legal slot; the generator rejects any
#   sample where the constructed patterns do not dock successfully.


GRID_H = 30
GRID_W = 24
GEN_RETRIES = 400

EVEN_BAR_HEIGHTS = (4, 6)
ODD_BAR_HEIGHTS = (5, 7)


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    """Detect full-width horizontal bars and annotate height parity."""
    H, W = grid.shape
    bars: list[dict] = []
    current_bar: dict | None = None

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
                    }
            else:
                if current_bar:
                    bars.append(current_bar)
                    current_bar = None
        else:
            if current_bar:
                bars.append(current_bar)
                current_bar = None

    if current_bar:
        bars.append(current_bar)

    for bar in bars:
        bar["h"] = int(bar["r_max"] - bar["r_min"] + 1)
        bar["swap"] = bool(int(bar["h"]) % 2 == 0)
    return bars


def _extract_patterns(grid: np.ndarray, pattern_area_end: int, bg: int) -> list[dict]:
    """Extract top-area patterns the same way the original task does."""
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


def _transform_pattern(
    pat: dict,
    bar: dict,
    *,
    respect_parity: bool,
) -> np.ndarray:
    """Apply the target-bar recolor rule to one detected patch."""
    src = np.array(pat["content"], dtype=int)
    transformed = src.copy()
    if respect_parity and not bool(bar["swap"]):
        return transformed

    transformed[src == int(pat["outer_color"])] = int(bar["border"])
    if pat["inner_color"] is not None:
        transformed[src == int(pat["inner_color"])] = int(bar["fill"])
    return transformed


def _first_fit_right(occ: np.ndarray, start: int, width: int, grid_w: int) -> int | None:
    max_c = int(grid_w) - int(width) - 1
    for c0 in range(int(start), max_c + 1):
        if not occ[c0 : c0 + int(width)].any():
            return int(c0)
    return None


def _apply_ordered_docking(
    out: np.ndarray,
    patterns: list[dict],
    bars: list[dict],
    bg: int,
    *,
    respect_parity: bool,
) -> list[dict]:
    """Dock patterns sequentially while reserving occupied intervals per bar."""
    grid_w = int(out.shape[1])
    fill_to_bar = {int(bar["fill"]): idx for idx, bar in enumerate(bars)}
    bar_occ = [np.zeros(grid_w, dtype=bool) for _ in bars]
    placements: list[dict] = []

    for pat in _ordered_patterns(patterns, bars):
        bar_idx = int(fill_to_bar[int(pat["outer_color"])])
        bar = bars[bar_idx]
        place_c = _first_fit_right(bar_occ[bar_idx], int(pat["c0"]), int(pat["w"]), grid_w)
        if place_c is None:
            raise AssertionError("No rightward docking slot available")

        place_r = int(bar["r_max"] - pat["h"] + 1)
        transformed = _transform_pattern(pat, bar, respect_parity=respect_parity)
        paste(out, transformed, place_r, place_c, transparent=int(bg))
        bar_occ[bar_idx][place_c : place_c + int(pat["w"])] = True
        placements.append(
            {
                "kind": pat.get("kind", "detected"),
                "outer_color": int(pat["outer_color"]),
                "inner_color": pat.get("inner_color"),
                "bar_idx": int(bar_idx),
                "source_r0": int(pat["r0"]),
                "source_c0": int(pat["c0"]),
                "placed_c0": int(place_c),
                "shifted": bool(int(place_c) != int(pat["c0"])),
                "swap_applied": bool(bar["swap"]) if respect_parity else True,
            }
        )

    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock top sprites into their color-matched bars one piece at a time, with
    the target bar's height parity deciding whether recoloring happens.

    Input structure:
    - The background is the most common color.
    - The lower region contains full-width horizontal bars; each bar has a fill
      color across the interior columns and a border color on the two side
      columns.
    - The upper region contains detached sprites made only from bar fill
      colors, and a two-tone sprite may also contribute a smaller inner-color
      component with its own bounding box.

    Rule:
    - Detect every top-area color component, recover its bbox content, and
      match it to the bar whose fill color equals that pattern's outer color.
    - Process the matches in order of target bar from top to bottom, then by
      source row and source column. Each piece tries to dock at its original
      x-position and slides right to the first free slot if needed.
    - Dock each piece bottom-aligned inside its target bar and erase the
      original top sprites.
    - If the target bar has even height, recolor outer -> bar border and inner
      -> bar fill. If the target bar has odd height, keep the pattern colors as
      they were detected.

    Key insight:
    - The bars encode the transformation mode through height parity, so the
      solver must read bar geometry before recoloring.
    - Because inner color components are detected as their own patterns, early
      placements reserve space and can force later patterns in the same target
      bar to shift right.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    patterns = _extract_patterns(g, pattern_area_end, bg)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])
    _apply_ordered_docking(out, patterns, bars, bg, respect_parity=True)
    return out


def _sample_bar_heights() -> list[int]:
    """Return two heights with opposite parity; order is randomized."""
    if random.random() < 0.5:
        return [random.choice(EVEN_BAR_HEIGHTS), random.choice(ODD_BAR_HEIGHTS)]
    return [random.choice(ODD_BAR_HEIGHTS), random.choice(EVEN_BAR_HEIGHTS)]


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Render the two bars bottom-up."""
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
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
            "swap": bool(h % 2 == 0),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = int(r0 - 1)
    return grid, bars


def _choose_inner_bbox(
    h: int,
    w: int,
    *,
    min_area: int = 1,
    anchor: str = "random",
) -> tuple[int, int, int, int] | None:
    valid = [
        (ih, iw)
        for ih in range(1, h - 1)
        for iw in range(2, w - 1)
        if ih * iw >= int(min_area) and ih * iw * 2 < h * w
    ]
    if not valid:
        return None

    ih, iw = random.choice(valid)
    ir0 = random.randint(1, h - ih - 1)
    if anchor == "right":
        ic0 = int(w - iw - 1)
    elif anchor == "left":
        ic0 = 1
    else:
        ic0 = random.randint(1, w - iw - 1)
    return int(ir0), int(ic0), int(ih), int(iw)


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
            "outer_color": int(bars[int(sp["outer_idx"])]["fill"]),
            "inner_color": int(bars[int(sp["inner_idx"])]["fill"]) if sp["inner_idx"] is not None else None,
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
                "outer_color": int(bars[int(sp["inner_idx"])]["fill"]),
                "inner_color": None,
                "r0": int(sp["r0"] + ir0),
                "c0": int(sp["c0"] + ic0),
                "h": int(ih),
                "w": int(iw),
                "content": np.full((ih, iw), int(bars[int(sp["inner_idx"])]["fill"]), dtype=int),
                "kind": f"{sp['name']}:inner",
            }
        )
    return patterns


def _generate_sprite_set(bars: list[dict], top_h: int) -> list[dict] | None:
    """Build three vertically separated sprites with forced ordering pressure."""
    top_bar = bars[0]
    lower_bar = bars[1]
    top_force_inner = not bool(top_bar["swap"])
    lower_force_inner = not bool(lower_bar["swap"])

    for _ in range(80):
        h_a = random.randint(3, min(5, int(top_bar["h"])))
        w_a = random.randint(6, 8)
        h_b = random.randint(2, min(4, int(top_bar["h"])))
        w_b = random.randint(3, 5)

        h_c_min = 3 if lower_force_inner else 2
        w_c_min = 4 if lower_force_inner else 3
        h_c = random.randint(h_c_min, min(5, int(lower_bar["h"])))
        w_c = random.randint(w_c_min, 7)

        total_h = int(h_a + h_b + h_c + 2)
        if total_h > int(top_h):
            continue

        inner_a = _choose_inner_bbox(h_a, w_a, min_area=2)
        if inner_a is None:
            continue
        ir_a, ic_a, ih_a, iw_a = inner_a

        inner_b = None
        if top_force_inner:
            inner_b = _choose_inner_bbox(h_b, w_b, min_area=1, anchor="right")
            if inner_b is None:
                continue

        inner_c = None
        if lower_force_inner:
            inner_c = _choose_inner_bbox(h_c, w_c, min_area=2, anchor="right")
            if inner_c is None:
                continue

        max_c_a = min(
            GRID_W - w_a - w_b - 1,
            GRID_W - ic_a - iw_a - w_c - 1,
        )
        if max_c_a < 1:
            continue

        c_a = random.randint(1, int(max_c_a))
        c_b = random.randint(c_a, min(c_a + w_a - 1, GRID_W - w_b - 1))
        inner_start = int(c_a + ic_a)
        c_c = random.randint(inner_start, min(inner_start + iw_a - 1, GRID_W - w_c - 1))

        top_offset = random.randint(0, int(top_h - total_h))
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
            "inner_bbox": tuple(map(int, inner_a)),
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
            "inner_idx": 1 if top_force_inner else None,
            "h": int(h_b),
            "w": int(w_b),
            "r0": int(r_b),
            "c0": int(c_b),
            "inner_bbox": tuple(map(int, inner_b)) if inner_b is not None else None,
        }
        sprite_b["patch"] = _make_rect_sprite(
            int(top_bar["fill"]),
            int(h_b),
            int(w_b),
            inner_fill=int(lower_bar["fill"]) if inner_b is not None else None,
            inner_bbox=sprite_b["inner_bbox"],
        )

        sprite_c = {
            "name": "C",
            "outer_idx": 1,
            "inner_idx": 0 if lower_force_inner else None,
            "h": int(h_c),
            "w": int(w_c),
            "r0": int(r_c),
            "c0": int(c_c),
            "inner_bbox": tuple(map(int, inner_c)) if inner_c is not None else None,
        }
        sprite_c["patch"] = _make_rect_sprite(
            int(lower_bar["fill"]),
            int(h_c),
            int(w_c),
            inner_fill=int(top_bar["fill"]) if inner_c is not None else None,
            inner_bbox=sprite_c["inner_bbox"],
        )

        return [sprite_a, sprite_b, sprite_c]

    return None


def _render_case(
    base: np.ndarray,
    sprites: list[dict],
    bars: list[dict],
    bg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Render input and both fused / legacy outputs for filtering."""
    inp = base.copy()
    parity_out = base.copy()
    legacy_out = base.copy()
    patterns: list[dict] = []

    for sp in sprites:
        paste(inp, sp["patch"], int(sp["r0"]), int(sp["c0"]), transparent=int(bg))
        patterns.extend(_pattern_from_sprite(sp, bars))

    parity_placements = _apply_ordered_docking(parity_out, patterns, bars, bg, respect_parity=True)
    legacy_placements = _apply_ordered_docking(legacy_out, patterns, bars, bg, respect_parity=False)
    return inp, parity_out, legacy_out, parity_placements, legacy_placements


def _gen_attempt() -> np.ndarray | None:
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1:3]]
    borders = [int(v) for v in palette[3:5]]

    heights = _sample_bar_heights()
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    sprites = _generate_sprite_set(bars, top_h)
    if sprites is None:
        return None

    try:
        inp, expected, legacy_out, placements, _ = _render_case(base, sprites, bars, bg)
    except AssertionError:
        return None

    if get_bg(inp) != bg:
        return None

    odd_targets = {
        int(sp["outer_idx"])
        for sp in sprites
        if not bool(bars[int(sp["outer_idx"])]["swap"])
    }
    if any(sp["inner_idx"] is None for sp in sprites if int(sp["outer_idx"]) in odd_targets):
        return None

    if not any(p["kind"] == "B:full" and p["shifted"] for p in placements):
        return None
    if not any(p["kind"].endswith(":full") and p["shifted"] for p in placements):
        return None
    if not any(bool(bar["swap"]) for bar in bars) or not any(not bool(bar["swap"]) for bar in bars):
        return None
    if np.array_equal(expected, legacy_out):
        return None
    if np.array_equal(main(inp), expected) is False:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
