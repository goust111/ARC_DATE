# Combo variant of 7c66cb00 (index 14)
# combo_spec: D3/4+D11/0+D14/2
# Name: 分离双组件迭代对接生成新精灵
#
# Design analysis
# ===============
# variant_spec:
#   preserved:
#     grid: 30x24, bg = most common color
#     bars:
#       count: 2-3 full-width horizontal bars in the lower area
#       geometry: fill color in cols 1..W-2, border color on cols 0 and W-1
#       heights: 4-8 with exactly one bg row between adjacent bars
#     per_step_transform:
#       - match by outer-component color == bar fill color
#       - bottom-align the whole sprite bbox at its original x-position
#       - recolor outer component -> matched bar border
#       - recolor inner component -> matched bar fill
#       - erase the original top sprites
#   changed:
#     D3/4:
#       - each logical sprite is now two disconnected monochrome blobs
#       - the larger blob is the outer component, the smaller blob is the inner component
#       - grouping is by spatial proximity, not by direct connectivity
#     D11/0:
#       - the inner blob is no longer docked independently into its own bar
#       - every docking step uses the whole grouped sprite only
#     D14/2:
#       - after one docking, the recolored result becomes a new source sprite
#       - matching repeats while the new outer color equals another bar fill
#       - all generated steps remain visible in the final output
#
# behavior_trace:
#   concrete_example:
#     - suppose bars have fills A,B,C and borders B,C,X
#     - top sprite = larger A-blob + smaller C-blob, separated by one blank column
#   original_behavior:
#     - a monochrome-component scan would misread the A-blob and C-blob as two separate sprites
#     - the C-blob could also dock independently because of the old inner-sub-sprite rule
#     - there is only one outer-bar docking pass
#   fused_behavior:
#     - the nearby A/C blobs are grouped into one logical sprite
#     - step 1: match outer A to bar A, recolor to (B outer, A inner), paste into bar A
#     - step 2: treat that whole B/A result as a new sprite, match outer B to bar B,
#       recolor to (C outer, B inner), paste into bar B
#     - step 3: outer C matches bar C, recolor to (X outer, C inner), paste into bar C
#     - stop because X is not a bar fill
#   visible_difference:
#     - output shows one chained trail of the same two-blob layout across multiple bars
#     - no standalone inner-only rectangle appears anywhere
#
# complexity_calibration:
#   - every generated sprite has exactly two disconnected blobs, so D3/4 is always visible
#   - the bar color graph is a short acyclic chain, so multi-step behavior is present but bounded
#   - only 2-3 bars and 2-3 sprites are used, keeping the hypothesis space small
#   - at least one generated sprite must have chain length >= 2, otherwise the sample is rejected
#
# implicit_assumption_audit:
#   original_main_dependencies_that_break:
#     - treating each connected monochrome region as a full sprite no longer works
#     - one-pass placement is insufficient because recolored sprites can match again
#     - the old inner-only docking branch conflicts with the new whole-sprite-only rule
#     - in-place recoloring can cascade incorrectly when a border color equals another bar fill
#   new_main_assumptions:
#     - each proximity group contains exactly one larger blob and one smaller blob of different colors
#     - distinct sprite groups are separated enough that one-cell dilation does not merge them
#     - the chain induced by bar border->fill links is finite; the solver still uses a visited-color
#       guard so accidental cycles terminate safely
#   generator_guarantees:
#     - exactly two blobs per group, with outer area > inner area
#     - grouped union bboxes are well separated in the top area
#     - each sprite height fits every bar on its whole docking chain
#     - per-bar horizontal occupancy is reserved for every step of every sprite chain
#
# placement_legality:
#   - top union bboxes never overlap and keep a margin so grouping stays unambiguous
#   - every docking uses the same original x-position and stays within cols 1..W-2
#   - bars visited by different sprites reserve non-overlapping x-intervals across all iterative steps
#
# visual_observability:
#   - opaque pasting is required in the bars so the blank gap between blobs remains visible
#   - repeated placements of the same disconnected layout across different bars make iteration legible
#   - the only matching cue is visible color structure; no hidden ordering by color id is used
#
# inferability_and_cleanliness:
#   - color changes are tied to explicit bar structure: fill chooses the target, border becomes the new outer color
#   - the chain is short and repeated across samples, so humans can infer "new outer color drives the next step"
#   - outputs avoid extra overlays beyond the chained sprite copies, keeping the bars readable

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (3, 9)
GEN_RETRIES = 500
TOP_GROUP_MARGIN = 3
BAR_PLACEMENT_MARGIN = 1


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    bars: list[dict] = []
    current_bar = None
    for r in range(H):
        row = g[r, :]
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
            elif current_bar:
                bars.append(current_bar)
                current_bar = None
        elif current_bar:
            bars.append(current_bar)
            current_bar = None
    if current_bar:
        bars.append(current_bar)
    return bars


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - 1)
        r1 = min(H, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(W, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _collect_top_components(top: np.ndarray, bg: int) -> list[dict]:
    components: list[dict] = []
    for color in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(color)
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _group_components(top: np.ndarray, bg: int) -> list[dict]:
    components = _collect_top_components(top, bg)
    if not components:
        return []

    expanded = _dilate_once(top != int(bg))
    group_labels, _ = label_mask(expanded, connectivity=8)
    grouped: dict[int, list[dict]] = {}
    for comp in components:
        first_r, first_c = map(int, comp["coords"][0])
        group_id = int(group_labels[first_r, first_c])
        grouped.setdefault(group_id, []).append(comp)

    patterns: list[dict] = []
    for comps in grouped.values():
        ordered = sorted(comps, key=lambda comp: (-int(comp["area"]), int(comp["bbox"][1]), int(comp["bbox"][0])))
        if len(ordered) < 2:
            continue
        outer = ordered[0]
        inner = None
        for comp in ordered[1:]:
            if int(comp["color"]) != int(outer["color"]):
                inner = comp
                break
        if inner is None:
            continue

        all_coords = np.vstack([outer["coords"], inner["coords"]])
        r0, c0, r1, c1 = bbox_coords(all_coords)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        content = crop_bbox(top, bbox)
        patterns.append(
            {
                "outer_color": int(outer["color"]),
                "inner_color": int(inner["color"]),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(content.shape[0]),
                "w": int(content.shape[1]),
                "content": content,
            }
        )

    return patterns


def _recolor_patch(
    patch: np.ndarray,
    outer_color: int,
    inner_color: int | None,
    new_outer: int,
    new_inner: int | None,
) -> np.ndarray:
    transformed = np.array(patch, dtype=int).copy()
    outer_mask = patch == int(outer_color)
    transformed[outer_mask] = int(new_outer)
    if inner_color is not None and new_inner is not None:
        inner_mask = patch == int(inner_color)
        transformed[inner_mask] = int(new_inner)
    return transformed


def _trace_chain(
    patch: np.ndarray,
    outer_color: int,
    inner_color: int | None,
    sprite_h: int,
    fill_to_bar: dict[int, dict],
) -> list[dict]:
    placements: list[dict] = []
    current_patch = np.array(patch, dtype=int)
    current_outer = int(outer_color)
    current_inner = int(inner_color) if inner_color is not None else None
    visited_outer_colors: set[int] = set()

    while current_outer in fill_to_bar and current_outer not in visited_outer_colors:
        bar = fill_to_bar[int(current_outer)]
        bar_h = int(bar["r_max"]) - int(bar["r_min"]) + 1
        if int(sprite_h) > int(bar_h):
            break

        transformed = _recolor_patch(
            current_patch,
            int(current_outer),
            current_inner,
            int(bar["border"]),
            int(bar["fill"]) if current_inner is not None else None,
        )
        placements.append({"bar": bar, "patch": transformed})
        visited_outer_colors.add(int(current_outer))
        current_patch = transformed
        current_outer = int(bar["border"])
        current_inner = int(bar["fill"]) if current_inner is not None else None

    return placements


def _extract_case_structure(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict], list[dict], dict[int, dict], int]:
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    bars = _find_bars(g, bg)
    assert bars, "No bars found"
    pattern_area_end = min(int(bar["r_min"]) for bar in bars)
    top = g[:pattern_area_end, :]
    patterns = _group_components(top, bg)
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}
    return g, int(bg), bars, patterns, fill_to_bar, int(pattern_area_end)


def _single_step_output(grid: np.ndarray) -> np.ndarray:
    g, bg, _bars, patterns, fill_to_bar, pattern_area_end = _extract_case_structure(grid)
    out = g.copy()
    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for pat in patterns:
        outer_bar = fill_to_bar.get(int(pat["outer_color"]))
        if outer_bar is None:
            continue
        bar_h = int(outer_bar["r_max"]) - int(outer_bar["r_min"]) + 1
        if int(pat["h"]) > int(bar_h):
            continue
        transformed = _recolor_patch(
            pat["content"],
            int(pat["outer_color"]),
            int(pat["inner_color"]),
            int(outer_bar["border"]),
            int(outer_bar["fill"]),
        )
        place_r = int(outer_bar["r_max"]) - int(pat["h"]) + 1
        paste(out, transformed, place_r, int(pat["c0"]))
    return out


def _case_chain_lengths(grid: np.ndarray) -> list[int]:
    _g, _bg, _bars, patterns, fill_to_bar, _pattern_area_end = _extract_case_structure(grid)
    return [
        len(_trace_chain(pat["content"], int(pat["outer_color"]), int(pat["inner_color"]), int(pat["h"]), fill_to_bar))
        for pat in patterns
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly dock grouped two-blob sprites through a color-link chain of horizontal bars.

    Input structure:
    - bg is the most common color.
    - The top area contains logical sprites made of two disconnected nearby blobs; the larger blob is the outer component and the smaller blob is the inner component.
    - The lower area contains 2-3 full-width bars with fill colors inside and border colors on both edges.

    Rule:
    - Group each nearby blob pair into one sprite, match its outer color to the bar whose fill has that color, and dock the whole grouped bbox at the same x-position, bottom-aligned.
    - Recolor the whole sprite on each docking step: outer component -> matched bar border, inner component -> matched bar fill.
    - Treat the recolored result as a new whole sprite and repeat while its new outer color matches another bar fill; do not dock the inner blob separately.
    - Erase the original top sprites from the input area.

    Key insight:
    - The disconnected blobs still behave as one object, and each step's newly created outer color determines whether the same whole sprite can continue into another bar.
    """
    g, bg, _bars, patterns, fill_to_bar, pattern_area_end = _extract_case_structure(grid)
    out = g.copy()
    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for pat in patterns:
        placements = _trace_chain(
            pat["content"],
            int(pat["outer_color"]),
            int(pat["inner_color"]),
            int(pat["h"]),
            fill_to_bar,
        )
        for step in placements:
            bar = step["bar"]
            place_r = int(bar["r_max"]) - int(pat["h"]) + 1
            paste(out, step["patch"], place_r, int(pat["c0"]))

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
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = heights[i]
        r0, r1 = r - h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = fills[i]
        grid[r0 : r1 + 1, 0] = borders[i]
        grid[r0 : r1 + 1, GRID_W - 1] = borders[i]
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = r0 - 1
    return grid, bars


def _choose_bar_colors(n_bars: int) -> tuple[int, list[int], list[int], list[int]]:
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    spare = [int(v) for v in palette[1 + n_bars :]]

    order = list(range(n_bars))
    random.shuffle(order)
    borders = [-1] * n_bars

    if n_bars == 2:
        borders[order[0]] = int(fills[order[1]])
        borders[order[1]] = int(spare[0])
    else:
        if random.random() < 0.65:
            borders[order[0]] = int(fills[order[1]])
            borders[order[1]] = int(fills[order[2]])
            borders[order[2]] = int(spare[0])
        else:
            borders[order[0]] = int(fills[order[1]])
            borders[order[1]] = int(spare[0])
            borders[order[2]] = int(spare[1])

    next_idx = []
    fill_to_idx = {int(fill): i for i, fill in enumerate(fills)}
    for border in borders:
        next_idx.append(fill_to_idx.get(int(border), -1))

    return bg, fills, [int(v) for v in borders], next_idx


def _path_for_start(start_idx: int, next_idx: list[int]) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    cur = int(start_idx)
    while cur >= 0 and cur not in seen:
        path.append(int(cur))
        seen.add(int(cur))
        cur = int(next_idx[cur])
    return path


def _make_group_patch(bg: int, outer_color: int, inner_color: int, max_h: int) -> dict | None:
    for _ in range(250):
        outer_h = random.randint(1, min(3, max_h, SPRITE_H_RANGE[1]))
        outer_w = random.randint(2, 5)
        inner_h = random.randint(1, min(2, max_h, SPRITE_H_RANGE[1]))
        inner_w = random.randint(1, 3)
        if outer_h * outer_w <= inner_h * inner_w:
            continue

        if random.random() < 0.5:
            row_delta = random.randint(-1, 1)
            outer_top = max(0, -row_delta)
            inner_top = max(0, row_delta)
            h = max(outer_top + outer_h, inner_top + inner_h)
            w = outer_w + 1 + inner_w
            if not (SPRITE_H_RANGE[0] <= h <= min(SPRITE_H_RANGE[1], max_h) and SPRITE_W_RANGE[0] <= w <= SPRITE_W_RANGE[1]):
                continue
            outer_left = 0
            inner_left = outer_w + 1
            if random.random() < 0.5:
                outer_left, inner_left = inner_left, 0
        else:
            col_delta = random.randint(-1, 1)
            outer_left = max(0, -col_delta)
            inner_left = max(0, col_delta)
            h = outer_h + 1 + inner_h
            w = max(outer_left + outer_w, inner_left + inner_w)
            if not (SPRITE_H_RANGE[0] <= h <= min(SPRITE_H_RANGE[1], max_h) and SPRITE_W_RANGE[0] <= w <= SPRITE_W_RANGE[1]):
                continue
            outer_top = 0
            inner_top = outer_h + 1
            if random.random() < 0.5:
                outer_top, inner_top = inner_top, 0

        patch = np.full((h, w), bg, dtype=int)
        patch[outer_top : outer_top + outer_h, outer_left : outer_left + outer_w] = int(outer_color)
        patch[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = int(inner_color)
        if np.any(patch == int(outer_color)) and np.any(patch == int(inner_color)):
            return {"patch": patch, "h": int(h), "w": int(w)}
    return None


def _interval_free(occ: np.ndarray, start: int, length: int, margin: int = BAR_PLACEMENT_MARGIN) -> bool:
    left = max(0, int(start) - int(margin))
    right = min(len(occ), int(start) + int(length) + int(margin))
    return not occ[left:right].any()


def _rect_separated(
    rects: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    margin: int = TOP_GROUP_MARGIN,
) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - margin or rr1 < r0 - margin or c1 < cc0 - margin or cc1 < c0 - margin):
            return False
    return True


def _make_sprite(start_idx: int, bars: list[dict], bg: int, next_idx: list[int]) -> dict | None:
    path = _path_for_start(int(start_idx), next_idx)
    max_h = min(int(bars[idx]["h"]) for idx in path)
    inner_choices = [i for i in range(len(bars)) if i != int(start_idx)]
    inner_idx = random.choice(inner_choices)
    shape = _make_group_patch(bg, int(bars[start_idx]["fill"]), int(bars[inner_idx]["fill"]), int(max_h))
    if shape is None:
        return None
    return {
        "outer_idx": int(start_idx),
        "inner_idx": int(inner_idx),
        "path": [int(idx) for idx in path],
        "patch": shape["patch"],
        "h": int(shape["h"]),
        "w": int(shape["w"]),
    }


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        path = [int(idx) for idx in sp["path"]]
        candidates = []
        for c0 in range(1, GRID_W - w):
            if any(not _interval_free(bar_occ[idx], c0, w) for idx in path):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        for idx in path:
            left = max(0, int(c0) - BAR_PLACEMENT_MARGIN)
            right = min(GRID_W, int(c0) + int(w) + BAR_PLACEMENT_MARGIN)
            bar_occ[idx][left:right] = True
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        paste(inp, patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

        placements = _trace_chain(
            patch,
            int(bars[int(sp["outer_idx"])]["fill"]),
            int(bars[int(sp["inner_idx"])]["fill"]),
            int(sp["h"]),
            fill_to_bar,
        )
        for step in placements:
            bar = step["bar"]
            place_r = int(bar["r_max"]) - int(sp["h"]) + 1
            paste(out, step["patch"], place_r, int(sp["c0"]))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    bg, fills, borders, next_idx = _choose_bar_colors(n_bars)
    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    long_chain_starts = [idx for idx in range(n_bars) if len(_path_for_start(idx, next_idx)) >= 2]
    if not long_chain_starts:
        return None

    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)

    forced_start = random.choice(long_chain_starts)
    outer_slots.remove(int(forced_start))
    starts = [int(forced_start)]
    while len(starts) < n_sprites:
        starts.append(int(outer_slots.pop()))

    sprites: list[dict] = []
    for start_idx in starts:
        sp = _make_sprite(int(start_idx), bars, bg, next_idx)
        if sp is None:
            return None
        sprites.append(sp)

    sprites.sort(key=lambda sp: (-len(sp["path"]), -int(sp["w"]), -int(sp["h"])))
    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)
    if get_bg(inp) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None

    chain_lengths = _case_chain_lengths(inp)
    if not chain_lengths or max(chain_lengths) < 2:
        return None
    if np.array_equal(out, _single_step_output(inp)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
