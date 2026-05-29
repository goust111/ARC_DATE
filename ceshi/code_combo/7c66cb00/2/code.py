# Combo variant of 7c66cb00 (index 2)
# combo_spec: D1/0+D3/4+D11/1
# Name: 分离双组件形状匹配与双组件独立对接

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


"""
design_analysis:
  variant_goal:
    preserve:
      - 30x24 canvas, bottom stacked bars with left/right borders, top objects erased in the output
      - bottom alignment at each component's original x-position
      - bar holes are visually filled by the matched top component
    change:
      - a logical sprite is no longer one connected two-tone blob
      - each top sprite group contains exactly two detached monochrome components that are near each other
      - the bar for component A is chosen by notch-shape matching, not by A's color
      - both components dock independently: A goes to the notch-matched bar, B goes to the bar whose fill equals B's color

  a_concretization_and_trace:
    small_example:
      - input top group: a hook-shaped orange piece A and, two columns to its right, a small teal rectangle B
      - lower bars: bar X has a hook-shaped bg notch; bar Y has fill teal
      - original task would read the dominant color and send the whole sprite to the orange bar
      - fused task instead groups the detached pair by proximity, recognizes that only A fits bar X's notch, docks A into X,
        then docks B by itself into bar Y
      - difference:
        - grouping is spatial, not connectivity-based
        - A's destination is geometric, not color-based
        - the two pieces separate in the output instead of moving as one combined patch

  b_complexity_calibration:
    - every sample uses exactly 3 bars and 3 top groups so the new cues repeat enough for humans to generalize
    - each bar has one unique notch shape and each group has exactly one notch-matching component, preventing ambiguous pairing
    - A colors are rotated away from their target bars; B colors still identify their own target bars
    - all A components share the same x column family, so x-position cannot replace shape reasoning

  c_implicit_assumption_audit:
    original_breakage:
      - per-color component scans would split a logical sprite into two unrelated objects
      - outer_color == bar.fill no longer identifies A's bar
      - bar detection based on nearly full rows must tolerate a carved bg notch
    solver_fixes:
      - top components are found as non-bg 4-components, then regrouped by one-step dilation proximity
      - bars are recognized by stable side borders plus a dominant interior fill
      - each bar exposes one bg component inside its bbox; that hole shape is the match signature
      - inside each group, the unique component whose shape matches some notch is treated as A

  d_placement_legality:
    - A notch height is always <= its bar height
    - B pieces are narrow rectangles placed far enough from the notch columns to avoid overlap inside bars
    - top groups are stacked with at least 3 blank rows between union bboxes, so dilation merges within-group pieces but not across groups
    - output placements are bottom-aligned and fully in-bounds

  e_visual_observability:
    - detached pairs are visibly different from the original connected sprites
    - each bar shows a missing silhouette that only one top component can fill
    - the second component does not fill a hole; it reappears in a different bar indicated by its color, so both routing rules are visible

  f_visual_cleanliness:
    - one notch per bar, one A and one B placement per bar
    - top groups use consistent left/right local layout and large vertical spacing
    - no output overlap occurs inside any bar

  g_change_inferability:
    - palette size stays modest: bg + 3 fills + 3 borders
    - the bar hole gives a structural explanation for A's destination
    - repeated cyclic color assignment lets humans see that B still follows color while A does not

  solver_assumptions_to_be_guaranteed_by_generator:
    - exactly 3 bars exist
    - each bar has exactly one unique notch component
    - each top proximity group contains exactly 2 monochrome components
    - exactly one component per group matches a notch signature
    - the other component's color matches exactly one bar fill and targets a different bar
"""


GRID_H = 30
GRID_W = 24
BAR_COUNT = 3
BAR_HEIGHT_RANGE = (4, 5)
MAX_BAR_TOTAL = 13
MIN_TOP_AREA = 15
TOP_GROUP_GAP = 3
GEN_RETRIES = 500
SHARED_A_C0 = 8
B_GAP = 2


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
                    candidate = {
                        "fill": int(fill),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                    }

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


def _largest_bg_component_signature(g: np.ndarray, bar: dict, bg: int) -> tuple[tuple[int, int], ...]:
    crop = g[int(bar["r_min"]) : int(bar["r_max"]) + 1, :]
    lab, n = label_mask(crop == int(bg), connectivity=4)
    assert n >= 1, "Each bar must contain a bg notch"

    components = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size:
            components.append(coords)

    coords = max(components, key=lambda arr: int(arr.shape[0]))
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in coords:
        mask[int(rr) - int(r0), int(cc) - int(c0)] = True
    return _shape_signature(mask)


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
    lab, n = label_mask(top != int(bg), connectivity=4)
    components: list[dict] = []

    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = crop_bbox(top, (int(r0), int(c0), int(r1), int(c1)))
        colors = sorted(set(int(v) for v in patch.flatten() if int(v) != int(bg)))
        assert len(colors) == 1, "Each top component must be monochrome"
        mask = patch != int(bg)
        components.append(
            {
                "color": int(colors[0]),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "patch": patch,
                "shape_sig": _shape_signature(mask),
            }
        )

    components.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    return components


def _group_top_components(top: np.ndarray, bg: int) -> list[list[dict]]:
    components = _collect_top_components(top, bg)
    if not components:
        return []

    expanded = _dilate_once(top != int(bg))
    lab, _ = label_mask(expanded, connectivity=8)
    grouped: dict[int, list[dict]] = {}
    for comp in components:
        rr, cc = map(int, comp["coords"][0])
        group_id = int(lab[rr, cc])
        grouped.setdefault(group_id, []).append(comp)

    groups = []
    for comps in grouped.values():
        comps.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
        groups.append(comps)
    groups.sort(key=lambda comps: min(int(comp["r0"]) for comp in comps))
    return groups


def _border_patch(patch: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = np.full_like(np.array(patch, dtype=int), int(bg))
    out[np.array(patch, dtype=int) != int(bg)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each detached two-piece sprite group, route one piece by notch shape and the other by color, then dock both pieces separately.

    Input structure:
    - Background is the most common color.
    - The top area contains nearby pairs of detached monochrome components; each pair is one logical sprite group.
    - The bottom area contains three horizontal bars with side borders, and each bar has one bg notch cut into its interior.

    Rule:
    - Group nearby top components by proximity.
    - In each group, the unique component whose silhouette matches a bar notch is component A; dock it into that matched bar at its own x-position and recolor it to that bar's border color.
    - The other component B docks independently into the bar whose fill color matches B's color, again at B's own x-position and recolored to that bar's border color.
    - Erase the original top components.

    Key insight:
    - Solving requires chaining three cues: proximity groups the detached pieces, notch geometry selects A's destination, and B still follows the bar-fill color system on its own.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _detect_bars(g, bg)
    assert len(bars) == BAR_COUNT, "Expected exactly three bars"

    top_end = min(int(bar["r_min"]) for bar in bars)
    groups = _group_top_components(g[:top_end, :], bg)

    notch_to_bar: dict[tuple[tuple[int, int], ...], dict] = {}
    fill_to_bar: dict[int, dict] = {}
    for bar in bars:
        sig = _largest_bg_component_signature(g, bar, bg)
        assert sig not in notch_to_bar, "Notch signatures must be unique"
        notch_to_bar[sig] = bar
        fill_to_bar[int(bar["fill"])] = bar

    out[:top_end, :] = int(bg)

    for comps in groups:
        assert len(comps) == 2, "Each group must contain exactly two components"
        matching = [comp for comp in comps if comp["shape_sig"] in notch_to_bar]
        assert len(matching) == 1, "Each group must contain exactly one notch-matching component"

        comp_a = matching[0]
        comp_b = comps[0] if comps[1] is comp_a else comps[1]

        bar_a = notch_to_bar[comp_a["shape_sig"]]
        bar_b = fill_to_bar.get(int(comp_b["color"]))
        assert bar_b is not None, "Component B must use a bar fill color"

        patch_a = _border_patch(np.array(comp_a["patch"], dtype=int), bg, int(bar_a["border"]))
        place_a_r = int(bar_a["r_max"]) - int(comp_a["h"]) + 1
        paste(out, patch_a, place_a_r, int(comp_a["c0"]), transparent=int(bg))

        patch_b = _border_patch(np.array(comp_b["patch"], dtype=int), bg, int(bar_b["border"]))
        place_b_r = int(bar_b["r_max"]) - int(comp_b["h"]) + 1
        paste(out, patch_b, place_b_r, int(comp_b["c0"]), transparent=int(bg))

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


B_SHAPES = [
    _mask(["11"]),
    _mask(["111"]),
    _mask(["1", "1"]),
    _mask(["11", "11"]),
]


def _sample_bar_heights() -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(BAR_COUNT)]
        if sum(heights) > MAX_BAR_TOTAL:
            continue
        top_h = GRID_H - sum(heights) - (BAR_COUNT - 1)
        if top_h < MIN_TOP_AREA:
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
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = r0 - 1
    return grid, bars


def _choose_a_shapes(bars: list[dict]) -> list[np.ndarray] | None:
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


def _carve_notch(grid: np.ndarray, bar: dict, mask: np.ndarray, c0: int, bg: int) -> None:
    r0 = int(bar["r_max"]) - int(mask.shape[0]) + 1
    for rr in range(mask.shape[0]):
        for cc in range(mask.shape[1]):
            if bool(mask[rr, cc]):
                grid[r0 + rr, c0 + cc] = int(bg)


def _render_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _stack_rows(group_heights: list[int], top_h: int) -> list[int] | None:
    need = sum(int(h) for h in group_heights) + TOP_GROUP_GAP * (len(group_heights) - 1)
    if need > top_h:
        return None
    start = random.randint(0, top_h - need)
    rows = []
    cur = int(start)
    for h in group_heights:
        rows.append(int(cur))
        cur += int(h) + TOP_GROUP_GAP
    return rows


def _top_group_height(a_mask: np.ndarray, b_mask: np.ndarray, b_top: int) -> int:
    return max(int(a_mask.shape[0]), int(b_top) + int(b_mask.shape[0]))


def _build_groups(bars: list[dict], a_masks: list[np.ndarray]) -> list[dict] | None:
    specs: list[dict] = []
    top_heights: list[int] = []

    for target_idx in range(BAR_COUNT):
        a_mask = a_masks[target_idx]
        b_target_idx = (target_idx + 1) % BAR_COUNT
        a_color_idx = (target_idx + 2) % BAR_COUNT
        b_mask = random.choice(B_SHAPES)
        b_top = random.randint(0, max(0, int(a_mask.shape[0]) - int(b_mask.shape[0])))
        group_h = _top_group_height(a_mask, b_mask, b_top)
        a_c0 = SHARED_A_C0
        b_c0 = SHARED_A_C0 + int(a_mask.shape[1]) + B_GAP
        if b_c0 + int(b_mask.shape[1]) > GRID_W - 1:
            return None

        specs.append(
            {
                "target_idx": int(target_idx),
                "b_target_idx": int(b_target_idx),
                "a_color": int(bars[a_color_idx]["fill"]),
                "b_color": int(bars[b_target_idx]["fill"]),
                "a_mask": a_mask,
                "b_mask": b_mask,
                "a_c0": int(a_c0),
                "b_c0": int(b_c0),
                "b_top": int(b_top),
                "group_h": int(group_h),
            }
        )
        top_heights.append(int(group_h))

    order = list(range(BAR_COUNT))
    random.shuffle(order)
    top_h = int(bars[0]["r_min"])
    rows = _stack_rows([top_heights[idx] for idx in order], top_h)
    if rows is None:
        return None

    placed: list[dict] = []
    for row_slot, spec_idx in enumerate(order):
        spec = specs[spec_idx]
        placed.append({**spec, "top_r": int(rows[row_slot])})
    return placed


def _render(base: np.ndarray, groups: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for group in groups:
        a_patch = _render_mask(np.array(group["a_mask"], dtype=bool), int(group["a_color"]), bg)
        b_patch = _render_mask(np.array(group["b_mask"], dtype=bool), int(group["b_color"]), bg)

        paste(inp, a_patch, int(group["top_r"]), int(group["a_c0"]), transparent=int(bg))
        paste(inp, b_patch, int(group["top_r"]) + int(group["b_top"]), int(group["b_c0"]), transparent=int(bg))

        bar_a = bars[int(group["target_idx"])]
        out_a = _render_mask(np.array(group["a_mask"], dtype=bool), int(bar_a["border"]), bg)
        paste(out, out_a, int(bar_a["r_max"]) - out_a.shape[0] + 1, int(group["a_c0"]), transparent=int(bg))

        bar_b = bars[int(group["b_target_idx"])]
        out_b = _render_mask(np.array(group["b_mask"], dtype=bool), int(bar_b["border"]), bg)
        paste(out, out_b, int(bar_b["r_max"]) - out_b.shape[0] + 1, int(group["b_c0"]), transparent=int(bg))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + BAR_COUNT]]
    borders = [int(v) for v in palette[1 + BAR_COUNT : 1 + 2 * BAR_COUNT]]

    heights = _sample_bar_heights()
    base, bars = _build_bars(bg, fills, borders, heights)

    a_masks = _choose_a_shapes(bars)
    if a_masks is None:
        return None

    max_a_w = max(int(mask.shape[1]) for mask in a_masks)
    if SHARED_A_C0 < 1 or SHARED_A_C0 + max_a_w > GRID_W - 1:
        return None

    for bar, mask in zip(bars, a_masks):
        _carve_notch(base, bar, mask, SHARED_A_C0, bg)

    groups = _build_groups(bars, a_masks)
    if groups is None:
        return None

    inp, out = _render(base, groups, bars, bg)

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
