# Variant of 7c66cb00
# Dimension: D1 — 匹配条件：精灵与条的配对方式 (match_cond)
# Change: 精灵的外层颜色（多数色）等于目标条的填充色：s.outer_color == d.fill_color... → 形状匹配替代颜色匹配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


"""
design_analysis:
  variant_goal:
    keep:
      - 30x24 canvas, bottom stacked horizontal bars, top sprites, erase-top then dock-into-bar output flow
      - recolor rule: sprite outer -> target bar border, sprite inner -> target bar fill
    change:
      - target bar is no longer chosen by sprite outer_color
      - each bar contains one bg notch whose occupied-cell silhouette matches exactly one top sprite silhouette
      - matching key is sprite non-bg mask == notch mask

  a_concretization_and_trace:
    input_structure:
      - top area contains 2-3 disconnected sprites, each a single non-bg component with 1-2 colors
      - bottom area contains 2-3 full-width bars with side borders and one bg cutout per bar
      - notch and sprite share the same outer silhouette, but sprite colors are intentionally decoupled from its target bar colors
    output_structure:
      - top sprites are erased
      - each sprite is pasted into the matched bar, bottom-aligned at its original x
      - generator fixes all sprites to a shared x that also equals all notch x positions, so the original-x rule is preserved and the matched notch gets filled
    concrete_trace:
      - original rule: sprite outer color 6 -> choose bar with fill 6
      - new rule: sprite silhouette shaped like a hook -> choose bar whose bg notch is that same hook, even if sprite outer color is 4 and target bar fill is 7
      - difference: color no longer identifies the bar; the visible hole shape does

  b_complexity_calibration:
    - one sprite per bar avoids ambiguous many-to-one notch assignment
    - each sample uses 2-3 distinct silhouettes chosen from a small library; this is enough repetition for humans while keeping branching low
    - outer colors are cyclically shifted away from their target bars, so the old color rule systematically disagrees with the new rule

  c_implicit_assumption_audit:
    original_main_dependencies:
      - bar detection assumed rows were almost entirely fill-colored
      - sprite detection labeled per-color regions, which accidentally split inner regions into separate pseudo-sprites
      - matching depended on outer_color == bar.fill
    variant_breakage:
      - bar rows now contain bg holes, so bar detection must tolerate missing fill cells
      - per-color labeling is invalid because notch matching needs the whole sprite silhouette, not isolated color islands
      - color-based matching must be replaced by silhouette equality
    fix:
      - bars are recognized by persistent non-bg edge borders plus a dominant interior fill
      - sprites are extracted from top-area non-bg connected components
      - bars expose one bg component per bbox; that component's mask is the notch signature

  d_placement_legality:
    - every sprite shape height is <= its target bar height
    - all sprites and all notches share one legal interior x-start so no x-based pairing cue appears
    - top sprites are stacked vertically with 1-row gaps, so they never touch
    - notch carving and output placement use the same bottom-aligned coordinates, so filling is exact and in-bounds

  e_visual_observability:
    - because outer colors are always mismatched relative to the target bar fill, a solver must use shape
    - all bars show explicit missing silhouettes; in the outputs those holes are filled by recolored copies
    - shared x prevents a cheap "same column => same bar" shortcut

  f_visual_cleanliness:
    - only one notch per bar and one sprite per bar
    - no overlapping top sprites, no overlapping bar fillings, and no extra sub-sprite side effects

  g_color_change_inference:
    - palette remains small: bg + distinct fills + distinct borders
    - inner color, when present, is a single minority patch; humans can still see that the inserted sprite adopts the target bar's border/fill colors

  solver_assumptions_to_be_guaranteed_by_generator:
    - each bar has exactly one bg notch component inside its bbox
    - each top sprite is exactly one non-bg component
    - notch silhouettes are unique within a grid
    - sprite original x equals notch x for every bar
"""


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 6)
MAX_THREE_BAR_TOTAL = 16
MIN_TOP_AREA = 12
SPRITE_GAP = 1
GEN_RETRIES = 400


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return tuple((int(r), int(c)) for r, c in coords)


def _dominant_non_bg(row: np.ndarray, bg: int) -> tuple[int, int] | None:
    vals = [int(v) for v in row if int(v) != int(bg)]
    if not vals:
        return None
    color, count = Counter(vals).most_common(1)[0]
    return int(color), int(count)


def _detect_bars(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    bars: list[dict] = []
    current = None
    min_fill_count = max(4, (W - 2) // 3)

    for r in range(H):
        row = g[r, :]
        border = int(row[0])
        candidate = None
        if border != int(bg) and int(row[-1]) == border:
            dom = _dominant_non_bg(row[1:-1], bg)
            if dom is not None:
                fill, count = dom
                if fill != border and count >= min_fill_count:
                    candidate = {"fill": int(fill), "border": int(border), "r_min": int(r), "r_max": int(r)}

        if candidate is None:
            if current is not None:
                bars.append(current)
                current = None
            continue

        if current and int(current["fill"]) == int(candidate["fill"]) and int(current["border"]) == int(candidate["border"]):
            current["r_max"] = int(r)
        else:
            if current is not None:
                bars.append(current)
            current = candidate

    if current is not None:
        bars.append(current)
    return bars


def _extract_sprites(g: np.ndarray, top_end: int, bg: int) -> list[dict]:
    top = g[:top_end, :]
    lab, n = label_mask(top != int(bg), connectivity=4)
    sprites: list[dict] = []

    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        bbox = crop_bbox(g, (r0, c0, r1, c1))
        mask = bbox != int(bg)
        colors = Counter(int(v) for v in bbox[mask].flatten())
        outer_color = int(colors.most_common(1)[0][0])
        inner_colors = [int(c) for c in colors if int(c) != int(outer_color)]
        inner_color = int(inner_colors[0]) if inner_colors else None
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "h": int(bbox.shape[0]),
                "w": int(bbox.shape[1]),
                "content": bbox,
                "mask": mask,
                "shape_sig": _shape_signature(mask),
                "outer_color": int(outer_color),
                "inner_color": inner_color,
            }
        )

    sprites.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    return sprites


def _extract_notch_signature(g: np.ndarray, bar: dict, bg: int) -> tuple[tuple[int, int], ...]:
    crop = g[int(bar["r_min"]) : int(bar["r_max"]) + 1, :]
    lab, n = label_mask(crop == int(bg), connectivity=4)
    assert n >= 1, "Each bar must contain a bg notch"
    comps = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        comps.append(coords)
    coords = max(comps, key=lambda arr: int(arr.shape[0]))
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in coords:
        mask[int(rr) - int(r0), int(cc) - int(c0)] = True
    return _shape_signature(mask)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _detect_bars(g, bg)
    assert bars, "No bars found"

    top_end = min(int(bar["r_min"]) for bar in bars)
    sprites = _extract_sprites(g, top_end, bg)

    out[:top_end, :] = int(bg)

    sig_to_bar: dict[tuple[tuple[int, int], ...], dict] = {}
    for bar in bars:
        sig = _extract_notch_signature(g, bar, bg)
        assert sig not in sig_to_bar, "Notch shapes must be unique"
        sig_to_bar[sig] = bar

    for sprite in sprites:
        assert sprite["shape_sig"] in sig_to_bar, "Every sprite must match one notch shape"
        bar = sig_to_bar[sprite["shape_sig"]]
        transformed = np.array(sprite["content"], dtype=int).copy()
        for rr in range(int(sprite["h"])):
            for cc in range(int(sprite["w"])):
                v = int(transformed[rr, cc])
                if v == int(bg):
                    continue
                if v == int(sprite["outer_color"]):
                    transformed[rr, cc] = int(bar["border"])
                else:
                    transformed[rr, cc] = int(bar["fill"])

        place_r = int(bar["r_max"]) - int(sprite["h"]) + 1
        paste(out, transformed, place_r, int(sprite["c0"]), transparent=int(bg))

    return out


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    _mask(["111", "110", "010"]),
    _mask(["011", "111", "010"]),
    _mask(["110", "011", "001"]),
    _mask(["1111", "0011", "0010"]),
    _mask(["1100", "1111", "0010"]),
    _mask(["0110", "1111", "0100"]),
    _mask(["111", "101", "111"]),
    _mask(["1110", "0100", "0111"]),
]


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        top_h = GRID_H - sum(heights) - (n_bars - 1)
        if top_h < MIN_TOP_AREA:
            continue
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {"fill": int(fills[i]), "border": int(borders[i]), "h": h, "r_min": r0, "r_max": r1}
        r = r0 - 1
    return grid, bars


def _choose_shapes(bars: list[dict]) -> list[np.ndarray] | None:
    choices = SHAPE_LIBRARY[:]
    random.shuffle(choices)
    picked: list[np.ndarray] = []
    used: set[tuple[tuple[int, int], ...]] = set()
    for bar in bars:
        found = None
        for mask in choices:
            if int(mask.shape[0]) <= int(bar["h"]):
                sig = _shape_signature(mask)
                if sig not in used:
                    found = mask
                    used.add(sig)
                    choices.remove(mask)
                    break
        if found is None:
            return None
        picked.append(found)
    return picked


def _inner_patch(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    interior = [(r, c) for r, c in coords if 0 < r < mask.shape[0] - 1 and 0 < c < mask.shape[1] - 1]
    if interior:
        return [random.choice(interior)]
    if len(coords) >= 5:
        return [coords[len(coords) // 2]]
    return []


def _make_sprite(mask: np.ndarray, bg: int, outer_color: int, inner_color: int | None) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(outer_color)
    if inner_color is not None:
        for rr, cc in _inner_patch(mask):
            patch[int(rr), int(cc)] = int(inner_color)
    return patch


def _stack_rows(heights: list[int], top_h: int) -> list[int] | None:
    need = sum(int(h) for h in heights) + SPRITE_GAP * (len(heights) - 1)
    if need > top_h:
        return None
    start = random.randint(0, top_h - need)
    rows = []
    cur = int(start)
    for h in heights:
        rows.append(int(cur))
        cur += int(h) + SPRITE_GAP
    return rows


def _carve_notch(grid: np.ndarray, bar: dict, mask: np.ndarray, c0: int, bg: int) -> None:
    r0 = int(bar["r_max"]) - int(mask.shape[0]) + 1
    for rr in range(mask.shape[0]):
        for cc in range(mask.shape[1]):
            if bool(mask[rr, cc]):
                grid[r0 + rr, c0 + cc] = int(bg)


def _render(base: np.ndarray, bars: list[dict], sprites: list[dict], bg: int, shared_c0: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for sprite in sprites:
        paste(inp, sprite["patch"], int(sprite["top_r"]), shared_c0, transparent=int(bg))

        bar = bars[int(sprite["target_idx"])]
        transformed = sprite["patch"].copy()
        for rr in range(transformed.shape[0]):
            for cc in range(transformed.shape[1]):
                v = int(transformed[rr, cc])
                if v == int(bg):
                    continue
                if v == int(sprite["outer_color"]):
                    transformed[rr, cc] = int(bar["border"])
                else:
                    transformed[rr, cc] = int(bar["fill"])
        paste(out, transformed, int(bar["r_max"]) - int(transformed.shape[0]) + 1, shared_c0, transparent=int(bg))

    return inp, out


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

    shapes = _choose_shapes(bars)
    if shapes is None:
        return None

    max_w = max(int(mask.shape[1]) for mask in shapes)
    shared_c0 = (GRID_W - max_w) // 2
    if shared_c0 < 1 or shared_c0 + max_w > GRID_W - 1:
        return None

    for bar, mask in zip(bars, shapes):
        _carve_notch(base, bar, mask, shared_c0, bg)

    order = list(range(n_bars))
    random.shuffle(order)
    top_rows = _stack_rows([int(shapes[i].shape[0]) for i in order], top_h)
    if top_rows is None:
        return None

    sprites: list[dict] = []
    for slot, target_idx in enumerate(order):
        outer_color = int(fills[(target_idx + 1) % n_bars])
        if n_bars >= 3:
            inner_color = int(fills[(target_idx + 2) % n_bars])
        else:
            inner_color = int(fills[target_idx]) if random.random() < 0.5 else None

        patch = _make_sprite(shapes[target_idx], bg, outer_color, inner_color)
        sprites.append(
            {
                "target_idx": int(target_idx),
                "patch": patch,
                "top_r": int(top_rows[slot]),
                "outer_color": int(outer_color),
            }
        )

    inp, out = _render(base, bars, sprites, bg, shared_c0)

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
