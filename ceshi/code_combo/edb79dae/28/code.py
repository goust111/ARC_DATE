# Combo variant of edb79dae (index 28)
# combo_spec: D10/2+D15/0
# Name: 冲突检查步骤+颜色混合重叠

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Original task: each inside-frame placeholder directly becomes the matching outside sprite,
#    then the legend recolors the placed sprite. No overlap reasoning is needed because the
#    placeholders are solid NxN blocks that stay disjoint.
#    Fused task here: the inside-frame placeholders are 1-cell anchors. A sprite is placed with
#    its bbox top-left aligned to that anchor, so neighboring anchors in the same row create a
#    1-column shape overlap. Example with a large-area blue sprite and a smaller red sprite:
#    after placement both would hit the same column; the new conflict step says the larger sprite
#    keeps its full shape, while the smaller sprite is suppressed except on the shared cells.
#    During recoloring, those shared cells use the smaller sprite's legend color, so the final
#    output shows a large silhouette with a narrow smaller-color trace inside the overlap.
# b) Complexity calibration:
#    Overlaps are restricted to horizontal neighbors only, so the effect is visible but not messy.
#    Every color's sprite has a unique filled area, which makes the priority rule observable and
#    avoids tie cases. The smallest-area color is repeated in both conflicted and isolated slots,
#    so a solver can compare "full placement" versus "trace-only" behavior inside one example.
# c) Broken assumptions from the original main():
#    - Placeholder bbox size no longer equals sprite bbox size, so size-matching must be removed.
#    - "Paint all sprites directly" is no longer valid; a conflict-resolution pass is required.
#    - Simple overwrite order is no longer the rule on overlaps; overlap color comes from the
#      smaller-area sprite, while survival is determined by the larger-area sprite.
#    - Overlap must be checked on occupied sprite cells, not just on placeholder blocks.
# d) Placement legality guaranteed by the generator:
#    - All anchors stay inside the frame with one bg cell of margin around the eventual sprite.
#    - Vertical anchor spacing is n+1, so different rows never overlap.
#    - Horizontal spacing is n-1, so only adjacent filled slots in the same row overlap, and only
#      by one column of bbox width; with a full middle row in each sprite, this always yields
#      at least one actual shared pixel.
#    - Outside-frame legends and sprite templates are placed with bg padding so they never touch.
# e) Visual observability:
#    The priority signal is human-visible because it depends on sprite filled area, which can be
#    read directly from the outside templates. The overlap signal is human-visible because the
#    trace color reappears only on the shared strip. No hidden numeric ordering on color ids is used.
# f) Visual neatness:
#    The generator forbids vertical overlaps and triple-stacked clutter. Each row uses one overlap
#    pair plus one isolated anchor, so the frame remains interpretable.
# g) Inferability of recoloring:
#    Legend blocks keep the original 3x4 mapping cue, and target colors are distinct from all
#    source colors. This keeps the recoloring rule separable from the new overlap rule.


FRAME_COL = 5


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    return int(candidates[0][0]), candidates[0][1]


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _collect_outside_components(
    g: np.ndarray, outside: np.ndarray, bg: int, frame_col: int
) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts_tuples),
                }
            )
            cid += 1
    return comps


def _extract_legend_map(
    g: np.ndarray, comps: list[dict[str, object]], placeholder_colors: set[int], bg: int, frame_col: int
) -> tuple[dict[int, int], set[int]]:
    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            if ci["col"] not in sub_colors or cj["col"] not in sub_colors:
                continue

            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]

            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[int(src)] = int(tgt)
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )
    return color_map, legend_ids


def _extract_sprites(
    comps: list[dict[str, object]], placeholder_colors: set[int], legend_ids: set[int]
) -> dict[int, dict[str, object]]:
    sprite_specs: dict[int, dict[str, object]] = {}
    for src_col in sorted(placeholder_colors):
        candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert candidates, f"no non-legend sprite found for color {src_col}"
        pts_set = set()
        for comp in candidates:
            pts_set.update(comp["pts"])
        coords_arr = np.array(sorted(pts_set), dtype=int)
        sr0, sc0, sr1, sc1 = bbox_coords(coords_arr)
        coords = tuple(sorted((int(r - sr0), int(c - sc0)) for r, c in pts_set))
        sprite_specs[src_col] = {
            "coords": coords,
            "h": int(sr1 - sr0 + 1),
            "w": int(sc1 - sc0 + 1),
            "area": len(coords),
        }
    areas = [int(spec["area"]) for spec in sprite_specs.values()]
    assert len(set(areas)) == len(areas), "sprite areas must be unique to define conflict priority"
    return sprite_specs


def _resolve_variant(
    placements: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[tuple[int, int], list[dict[str, object]]]]:
    resolved: list[dict[str, object]] = []
    pixel_to_sources: dict[tuple[int, int], list[dict[str, object]]] = {}
    for idx, placement in enumerate(placements):
        overlap_with_higher: set[tuple[int, int]] = set()
        for jdx, other in enumerate(placements):
            if idx == jdx:
                continue
            if placement["area"] == other["area"]:
                shared = placement["cells"] & other["cells"]
                assert not shared, "equal-area overlapping sprites make the priority rule ambiguous"
            if other["area"] > placement["area"]:
                overlap_with_higher |= placement["cells"] & other["cells"]
        visible = set(placement["cells"]) if not overlap_with_higher else overlap_with_higher
        resolved_item = dict(placement)
        resolved_item["visible"] = visible
        resolved_item["suppressed"] = bool(overlap_with_higher)
        resolved.append(resolved_item)
        for cell in visible:
            pixel_to_sources.setdefault(cell, []).append(resolved_item)
    return resolved, pixel_to_sources


def _render_variant_crop(
    frame_h: int, frame_w: int, bg: int, placements: list[dict[str, object]]
) -> tuple[np.ndarray, dict[str, int]]:
    out = _make_bordered_frame(frame_h, frame_w, bg)
    resolved, pixel_to_sources = _resolve_variant(placements)
    mixed_pixels = 0
    for (rr, cc), contenders in pixel_to_sources.items():
        if len(contenders) > 1:
            mixed_pixels += 1
        out[rr, cc] = min(contenders, key=lambda item: item["area"])["tgt_col"]
    stats = {
        "suppressed_count": sum(1 for item in resolved if item["suppressed"]),
        "mixed_pixels": mixed_pixels,
        "overlap_pairs": sum(
            1
            for i in range(len(placements))
            for j in range(i + 1, len(placements))
            if placements[i]["cells"] & placements[j]["cells"]
        ),
    }
    return out, stats


def _render_baseline_crop(frame_h: int, frame_w: int, bg: int, placements: list[dict[str, object]]) -> np.ndarray:
    out = _make_bordered_frame(frame_h, frame_w, bg)
    for placement in placements:
        for rr, cc in placement["cells"]:
            out[rr, cc] = placement["tgt_col"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place legend-recolored sprites into a framed region using single-cell anchors, then resolve overlaps.

    Input structure:
    - A single gray rectangular frame surrounds sparse colored anchor pixels on a bg interior.
    - Outside the frame, each anchor color has one same-color sprite template and one compact two-color legend block.

    Rule:
    - Match each inside-frame anchor color to the outside sprite of the same color and to its legend-defined target color.
    - Stamp the sprite with its bbox top-left aligned to the anchor cell.
    - Before final recoloring, run a conflict step: if a placed sprite overlaps any larger-area sprite, it loses all
      non-overlap cells and survives only on the shared cells; those shared cells are then colored by the smallest-area
      sprite participating at that pixel.

    Key insight:
    - The outside templates encode two different pieces of information: sprite filled area gives the overlap priority,
      while the legend still gives the final color. Large sprites control survival; small sprites donate the overlap color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder anchors found inside frame"

    placeholders: list[tuple[int, int, int]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            assert (r0, c0) == (r1, c1), "this fused variant expects 1-cell anchors inside the frame"
            placeholders.append((col, int(r0), int(c0)))

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _collect_outside_components(g, outside, bg, frame_col)
    color_map, legend_ids = _extract_legend_map(g, comps, placeholder_colors, bg, frame_col)
    sprite_specs = _extract_sprites(comps, placeholder_colors, legend_ids)

    placements: list[dict[str, object]] = []
    for src_col, anchor_r, anchor_c in sorted(placeholders, key=lambda item: (item[1], item[2], item[0])):
        sprite = sprite_specs[src_col]
        cells = set()
        for dr, dc in sprite["coords"]:
            rr = anchor_r + dr
            cc = anchor_c + dc
            assert 1 <= rr < h - 1 and 1 <= cc < w - 1, "placed sprite spills onto or beyond the frame border"
            cells.add((rr, cc))
        placements.append(
            {
                "src_col": int(src_col),
                "tgt_col": int(color_map[src_col]),
                "area": int(sprite["area"]),
                "cells": cells,
            }
        )

    out, _ = _render_variant_crop(h, w, bg, placements)
    return out


# --------------- generator ---------------

SPRITE_N_RANGE = (4, 6)            # 4 or 5
GRID_ROWS_RANGE = (2, 4)           # 2 or 3
GRID_COLS = 4
NUM_COLORS_RANGE = (2, 3)          # inclusive
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
PATCH_PADDING = 1
FRAME_POS_JITTER_PROB = 0.3
GEN_RETRIES = 400

TOP_PAD = 1
BOTTOM_PAD = 1
LEFT_PAD = 1
RIGHT_PAD = 1
ROW_PATTERNS = (
    (1, 1, 0, 1),
    (1, 0, 1, 1),
)


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [
            [src, src, tgt, tgt],
            [src, src, src, tgt],
            [src, src, tgt, tgt],
        ],
        dtype=int,
    )


def _sprite_mask(n: int, target_area: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mid = n // 2
    mask[mid, :] = True
    while int(mask.sum()) < target_area:
        frontier = np.zeros_like(mask, dtype=bool)
        frontier[1:, :] |= mask[:-1, :]
        frontier[:-1, :] |= mask[1:, :]
        frontier[:, 1:] |= mask[:, :-1]
        frontier[:, :-1] |= mask[:, 1:]
        frontier &= ~mask
        choices = list(zip(*np.where(frontier)))
        r, c = random.choice(choices)
        mask[r, c] = True
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _choose_target_areas(n: int, k: int) -> list[int] | None:
    lo = max(n + 3, int(np.ceil(0.45 * n * n)))
    hi = min(n * n - 2, int(np.floor(0.8 * n * n)))
    candidates = list(range(lo, hi + 1))
    if len(candidates) < k:
        return None
    for _ in range(50):
        picked = sorted(random.sample(candidates, k))
        if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _row_assignment(pattern: tuple[int, int, int, int], pair_colors: tuple[int, int], iso_color: int) -> dict[int, int]:
    filled_cols = [idx for idx, v in enumerate(pattern) if v]
    if filled_cols == [0, 1, 3]:
        return {0: pair_colors[0], 1: pair_colors[1], 3: iso_color}
    assert filled_cols == [0, 2, 3], f"unexpected row pattern {pattern}"
    return {0: iso_color, 2: pair_colors[0], 3: pair_colors[1]}


def _build_slot_colors(rows: int, src_cols: list[int], area_map: dict[int, int]) -> dict[tuple[int, int], int]:
    by_priority = sorted(src_cols, key=lambda col: area_map[col], reverse=True)
    high = by_priority[0]
    low = by_priority[-1]
    mid = by_priority[1] if len(by_priority) == 3 else high

    if len(by_priority) == 2:
        schemes = [
            ((high, low), low),
            ((low, high), high),
            ((high, low), low),
        ]
    else:
        schemes = [
            ((high, low), low),
            ((mid, low), high),
            ((high, mid), low),
        ]

    slot_colors: dict[tuple[int, int], int] = {}
    for rr in range(rows):
        pattern = random.choice(ROW_PATTERNS)
        pair_colors, iso_color = schemes[rr % len(schemes)]
        for cc, col in _row_assignment(pattern, pair_colors, iso_color).items():
            slot_colors[(rr, cc)] = int(col)
    return slot_colors


def _gen_attempt() -> np.ndarray | None:
    n = random.randrange(*SPRITE_N_RANGE)
    rows = random.randrange(*GRID_ROWS_RANGE)
    cols = GRID_COLS
    step_y = n + 1
    step_x = n - 1

    frame_h = 2 + TOP_PAD + (rows - 1) * step_y + n + BOTTOM_PAD
    frame_w = 2 + LEFT_PAD + (cols - 1) * step_x + n + RIGHT_PAD

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (FRAME_COL, bg)]
    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_pool = [c for c in usable if c not in src_cols]
    tgt_cols = random.sample(tgt_pool, k)
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    target_areas = _choose_target_areas(n, k)
    if target_areas is None:
        return None
    random.shuffle(target_areas)
    masks = {}
    area_map = {}
    for src, area in zip(src_cols, target_areas):
        masks[src] = _sprite_mask(n, area)
        area_map[src] = int(masks[src].sum())
    if len(set(area_map.values())) != k:
        return None

    slot_colors = _build_slot_colors(rows, src_cols, area_map)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    placements: list[dict[str, object]] = []
    for (rr, cc), src in sorted(slot_colors.items()):
        anchor_r = 1 + TOP_PAD + rr * step_y
        anchor_c = 1 + LEFT_PAD + cc * step_x
        input_frame[anchor_r, anchor_c] = src
        coords = tuple(sorted(zip(*np.where(masks[src]))))
        cells = {(anchor_r + dr, anchor_c + dc) for dr, dc in coords}
        placements.append(
            {
                "src_col": int(src),
                "tgt_col": int(color_to_tgt[src]),
                "area": int(area_map[src]),
                "cells": cells,
            }
        )

    expected_out, stats = _render_variant_crop(frame_h, frame_w, bg, placements)
    baseline_out = _render_baseline_crop(frame_h, frame_w, bg, placements)
    if stats["suppressed_count"] == 0 or stats["mixed_pixels"] == 0 or np.array_equal(expected_out, baseline_out):
        return None

    ext_area = k * (n * n + 12 + 8)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src, tgt in zip(src_cols, tgt_cols):
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, tgt), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid fused edb79dae input")
