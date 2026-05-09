# Combo variant of 5dbc8537 (index 13)
# combo_spec: D3/2+D4/0+D15/0
# Name: 洞内标记驱动颜色重映射（图例辅助）

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


# -----------------------------------------------------------------------------
# Design analysis for the combo variant D3/2 + D4/0 + D15/0
#
# a) Concrete behavior and flow impact
#    - The large base-colored strip is still the destination template, and the
#      outside objects are still full rectangular sprites.
#    - Two new information channels are added and must work together:
#      1. A corner legend is a small vertical 2-column table. Each row is a
#         source-color -> target-color pair.
#      2. Every hole inside the template contains 1-2 colored guide cells on
#         its boundary. Those guide cells show colors after recoloring, not the
#         raw source sprite colors.
#    - Example:
#        source sprite: 3x4 with blue main color and red accent
#        legend rows:   [blue green], [red yellow]
#        mapped sprite: green main color and yellow accent
#        hole A guides: top-left green, right-edge yellow
#      Original task: match 3x4 by size and paste as-is.
#      This combo: first recolor the sprite through the legend, then require the
#      recolored patch to agree with the hole guides at the same offsets, then
#      paste the recolored patch.
#    - So the old pipeline "size match -> paste" becomes
#      "read legend -> recolor candidate sprite -> marker-constrained placement
#      -> paste recolored patch".
#
# b) Complexity calibration
#    - If every hole size were unique, the guides would be decorative. The
#      generator therefore enforces repeated (h,w) groups.
#    - If guide colors were equal to the raw sprite colors, the legend would be
#      decorative. The generator uses disjoint source and target palettes, so
#      guide cells only match after recoloring.
#    - The legend uses only 3 mappings per sample, and those mappings repeat
#      across many sprites, which keeps the color rule visible and learnable.
#
# c) Implicit-assumption audit against the original main()
#    - Original `hole_mask = dst != base_col` still works, but its meaning is
#      now "hole area including bg plus embedded guide cells".
#    - Original outside-component extraction would incorrectly treat the legend
#      as a sprite. The combo solver must explicitly detect and remove the
#      corner legend before building sprite candidates.
#    - Original candidate search only checked same-sized all-hole rectangles.
#      That is no longer sufficient: a candidate must also satisfy guide colors
#      after applying the legend mapping to the sprite.
#    - Original paste preserved source colors. The combo must paste the mapped
#      patch instead.
#
# d) Placement legality
#    - Hole rectangles remain non-overlapping axis-aligned rectangles whose union
#      is a single corridor-like connected region inside the strip.
#    - The legend is reserved in one free grid corner and sprites are prevented
#      from touching grid corners/edges, so the legend is structurally unique.
#    - Every hole gets at least one guide cell, and same-size holes receive
#      distinct guide signatures after recoloring.
#
# e) Visual observability
#    - Humans can read the corner table as a literal palette legend because its
#      left-column colors appear on source sprites while its right-column colors
#      reappear in outputs and in hole guides.
#    - Repeated hole sizes make the guide-matching behavior visible.
#    - Recolored outputs keep each sprite's internal pattern, so the task looks
#      like a genuine recolor rather than a uniform paint fill.
#
# f) Visual cleanliness
#    - The legend occupies a compact corner block and never overlaps sprites or
#      the template.
#    - Hole guides are sparse boundary pixels only; they do not clutter the hole
#      interiors or obscure the rectangular cavity structure.
#
# g) Color inference
#    - The recolor rule is not based on hidden color ids. It is shown explicitly
#      by the corner legend and reinforced by repeated source/target reuse.
#    - Guides use mapped colors only, so the solver must infer a visible
#      "apply legend first, then compare" rule rather than relying on arbitrary
#      ordering or invisible metadata.
#
# New solver assumptions introduced here and guaranteed by the generator:
#    1. There is exactly one non-template outside component whose bbox touches a
#       grid corner; that component is the legend and has width 2.
#    2. Every sprite uses only colors from the legend source column.
#    3. Every real slot contains at least one guide cell.
#    4. Among slots with the same size, recolored guide signatures are unique.
# -----------------------------------------------------------------------------


def _apply_color_map(patch: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = np.array(patch, copy=True)
    for src, dst in color_map.items():
        out[patch == int(src)] = int(dst)
    return out


def _guide_positions(h: int, w: int) -> list[tuple[int, int]]:
    pts = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (0, w // 2),
        (h - 1, w // 2),
        (h // 2, 0),
        (h // 2, w - 1),
    ]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for p in pts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _component_is_full_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return int(comp["coords"].shape[0]) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _touches_corner(bbox: tuple[int, int, int, int], H: int, W: int) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == W - 1)
        or (r1 == H - 1 and c0 == 0)
        or (r1 == H - 1 and c1 == W - 1)
    )


def _extract_legend_and_sprites(
    g: np.ndarray,
    bg: int,
    dst_bbox: tuple[int, int, int, int],
) -> tuple[dict[int, int], list[dict]]:
    H, W = map(int, g.shape)
    r0, c0, r1, c1 = map(int, dst_bbox)
    masked = np.array(g, copy=True)
    masked[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    legend_candidates = [
        comp
        for comp in comps
        if _component_is_full_rectangle(comp)
        and _touches_corner(tuple(map(int, comp["bbox"])), H, W)
        and crop_bbox(g, tuple(map(int, comp["bbox"]))).shape[1] == 2
    ]
    assert len(legend_candidates) == 1
    legend_comp = legend_candidates[0]
    legend_bbox = tuple(map(int, legend_comp["bbox"]))
    legend_patch = crop_bbox(g, legend_bbox)
    lh, lw = map(int, legend_patch.shape)
    assert lw == 2 and lh >= 2

    color_map: dict[int, int] = {}
    for rr in range(lh):
        src = int(legend_patch[rr, 0])
        dst = int(legend_patch[rr, 1])
        assert src != int(bg) and dst != int(bg) and src != dst
        assert src not in color_map
        color_map[src] = dst

    sprites: list[dict] = []
    for comp in comps:
        if tuple(map(int, comp["bbox"])) == legend_bbox:
            continue
        b = tuple(map(int, comp["bbox"]))
        pr0, pc0, _pr1, _pc1 = b
        patch = crop_bbox(g, b)
        ph, pw = map(int, patch.shape)
        coords = comp["coords"]
        mask = np.zeros((ph, pw), dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        assert bool(mask.all())
        mapped_patch = _apply_color_map(patch, color_map)
        sprites.append(
            {
                "h": int(ph),
                "w": int(pw),
                "mask": mask,
                "patch": patch,
                "mapped_patch": mapped_patch,
            }
        )
    return color_map, sprites


def _solve_instance(grid: np.ndarray, return_debug: bool = False):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))

    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)
    guide_mask = hole_mask & (dst != int(bg))

    color_map, sprites = _extract_legend_and_sprites(g, bg, dst_bbox)
    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["mask"].sum())))

    out = np.full((dh, dw), int(base_col), dtype=int)

    candidates: list[list[tuple[int, int]]] = []
    areas = [int(s["h"] * s["w"]) for s in sprites]
    assert sum(areas) == int(hole_mask.sum())
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr0 in range(int(dh - h + 1)):
                for cc0 in range(int(dw - w + 1)):
                    window_holes = hole_mask[rr0 : rr0 + h, cc0 : cc0 + w]
                    if not bool(window_holes.all()):
                        continue
                    window_guides = guide_mask[rr0 : rr0 + h, cc0 : cc0 + w]
                    if not bool(window_guides.any()):
                        continue
                    window_vals = dst[rr0 : rr0 + h, cc0 : cc0 + w]
                    if not np.array_equal(sp["mapped_patch"][window_guides], window_vals[window_guides]):
                        continue
                    pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    assert all(candidates)
    used = ~hole_mask
    placement: list[tuple[int, int] | None] = [None] * len(sprites)

    def dfs(i: int) -> bool:
        if i == len(sprites):
            return True

        h, w = int(sprites[i]["h"]), int(sprites[i]["w"])
        for rr0, cc0 in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(i + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    assert dfs(0)

    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        sp = sprites[i]
        paste(out, sp["mapped_patch"], rr0, cc0, mask=sp["mask"], overwrite=True)

    if return_debug:
        return out, {
            "bg": bg,
            "base_col": base_col,
            "dst": dst,
            "hole_mask": hole_mask,
            "guide_mask": guide_mask,
            "color_map": color_map,
            "sprites": sprites,
            "placement": placement,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the template strip by recoloring scattered sprites through a corner legend and matching them to guide-marked holes.

    Input structure:
    - The most common color is background; one dominant non-background color forms a large strip-shaped template.
    - Outside the strip are full rectangular sprites plus one small corner legend whose rows show source-color -> target-color pairs.
    - Inside the strip, each rectangular hole is mostly background but contains 1-2 colored guide cells on its boundary.

    Rule:
    - Detect the base strip from the largest non-background connected component and crop to its bbox.
    - Read the corner legend to build a global color remapping for the outside sprites.
    - For each sprite, apply that remapping first, then place it only into a same-sized hole whose visible guide cells match the recolored sprite at the same offsets.
    - Paste the recolored sprites into a non-overlapping exact cover and return the cropped filled template.

    Key insight:
    - Size alone is no longer enough, and raw sprite colors are no longer enough: the legend explains how colors change, and the hole markers verify where each recolored sprite belongs.
    """
    return _solve_instance(grid, return_debug=False)


def _audit_metrics(grid: np.ndarray) -> dict[str, int]:
    _out, debug = _solve_instance(grid, return_debug=True)
    sprites = debug["sprites"]
    placement = debug["placement"]
    dst = debug["dst"]
    guide_mask = debug["guide_mask"]

    shape_counts: dict[tuple[int, int], int] = {}
    for sp in sprites:
        key = (int(sp["h"]), int(sp["w"]))
        shape_counts[key] = shape_counts.get(key, 0) + 1

    guide_cells = 0
    raw_mismatch = 0
    mapped_match = 0
    recolored_cells = 0
    for sp, pos in zip(sprites, placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        h, w = int(sp["h"]), int(sp["w"])
        window_guides = guide_mask[rr0 : rr0 + h, cc0 : cc0 + w]
        target_vals = dst[rr0 : rr0 + h, cc0 : cc0 + w]
        guide_cells += int(window_guides.sum())
        raw_mismatch += int(np.count_nonzero(sp["patch"][window_guides] != target_vals[window_guides]))
        mapped_match += int(np.count_nonzero(sp["mapped_patch"][window_guides] == target_vals[window_guides]))
        recolored_cells += int(np.count_nonzero(sp["patch"] != sp["mapped_patch"]))

    return {
        "sprite_count": len(sprites),
        "dup_groups": sum(1 for v in shape_counts.values() if v >= 2),
        "guide_cells": guide_cells,
        "guide_mismatch_before_mapping": raw_mismatch,
        "guide_match_after_mapping": mapped_match,
        "recolored_cells": recolored_cells,
        "legend_pairs": len(debug["color_map"]),
    }


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 12
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.34, 0.46
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_SHAPE_GROUPS = 3
MIN_DUP_GROUPS = 2
LEGEND_COLOR_COUNT = 3
BICHROMATIC_FRAC = 0.45
WIDTH_TRIES = 220
MAX_RETRIES = 500


def _split_lengths(total: int, count: int) -> list[int] | None:
    if count <= 0 or count > total or count * MAX_RECT_LEN < total:
        return None
    lengths = [1] * count
    remaining = total - count
    while remaining:
        choices = [i for i, v in enumerate(lengths) if v < MAX_RECT_LEN]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(remaining, MAX_RECT_LEN - lengths[i]))
        lengths[i] += add
        remaining -= add
    random.shuffle(lengths)
    return lengths


def _shape_stats(shapes: list[tuple[int, int]]) -> tuple[int, int]:
    counts: dict[tuple[int, int], int] = {}
    for key in shapes:
        counts[key] = counts.get(key, 0) + 1
    return len(counts), sum(1 for v in counts.values() if v >= 2)


def _choose_widths(lengths: list[int], thick: int, full_area: int) -> list[int] | None:
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 2:
        return None
    lo = math.ceil(HOLE_FRAC_LO * full_area)
    hi = math.floor(HOLE_FRAC_HI * full_area)
    width_choices = list(range(2, max_w + 1))
    for _ in range(WIDTH_TRIES):
        palette_size = random.randint(2, min(3, len(width_choices)))
        width_palette = random.sample(width_choices, palette_size)
        widths = [random.choice(width_palette) for _ in lengths]
        area = sum(int(h) * int(w) for h, w in zip(lengths, widths))
        if not (lo <= area <= hi):
            continue
        distinct, dup_groups = _shape_stats(list(zip(lengths, widths)))
        if distinct < MIN_SHAPE_GROUPS or dup_groups < MIN_DUP_GROUPS:
            continue
        return widths
    return None


def _corridor_rects(long: int, thick: int, count: int) -> list[tuple[int, int, int, int]] | None:
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    for _ in range(120):
        hole_len = random.randint(min_len, long - 1)
        lengths = _split_lengths(hole_len, count)
        if lengths is None:
            continue
        widths = _choose_widths(lengths, thick, long * thick)
        if widths is None:
            continue

        start = random.randint(0, long - hole_len)
        r = start
        prev_c = prev_w = None
        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for h, w in zip(lengths, widths):
            c_hi = thick - w - 1
            if c_hi < 1:
                ok = False
                break
            if prev_c is None:
                c = random.randint(1, c_hi)
            else:
                c_lo = max(1, prev_c - w + 1)
                c_hi = min(c_hi, prev_c + prev_w - 1)
                if c_lo > c_hi:
                    ok = False
                    break
                c = random.randint(c_lo, c_hi)
            rects.append((r, c, h, w))
            r += h
            prev_c, prev_w = c, w
        if ok:
            return rects
    return None


def _make_source_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), int(cols[0]), dtype=int)
    if len(cols) == 1:
        return out

    accent = int(cols[1])
    rr, cc = np.indices((h, w))
    pattern = random.choice(["checker", "sparse", "split", "corner"])
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = accent
    elif pattern == "sparse":
        coords = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(coords)
        use = max(1, min(len(coords) - 1, max(1, (h * w) // 3)))
        for r, c in coords[:use]:
            out[r, c] = accent
    elif pattern == "split":
        if h > 1 and (w == 1 or random.random() < 0.5):
            out[random.randint(1, h - 1) :, :] = accent
        else:
            out[:, random.randint(1, w - 1) :] = accent
    else:
        bh = random.randint(1, max(1, h // 2))
        bw = random.randint(1, max(1, w // 2))
        out[h - bh :, w - bw :] = accent
    if np.all(out == int(cols[0])) or np.all(out == accent):
        out[0, 0] = int(cols[0])
        out[h - 1, w - 1] = accent
    return out


def _make_pieces_and_guides(
    rects: list[tuple[int, int, int, int]],
    bg: int,
    base: int,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[tuple[int, int, int], ...]], dict[int, int]] | None:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_r, _c, h, w) in enumerate(rects):
        groups.setdefault((h, w), []).append(i)
    distinct, dup_groups = _shape_stats(list(groups.keys()) + [k for k, ids in groups.items() for _ in ids[1:]])
    if distinct < MIN_SHAPE_GROUPS or dup_groups < MIN_DUP_GROUPS:
        return None

    available = [c for c in range(1, 10) if c not in (bg, base)]
    if len(available) < 2 * LEGEND_COLOR_COUNT:
        return None
    source_palette = random.sample(available, LEGEND_COLOR_COUNT)
    target_choices = [c for c in available if c not in source_palette]
    target_palette = random.sample(target_choices, LEGEND_COLOR_COUNT)
    color_map = {int(src): int(dst) for src, dst in zip(source_palette, target_palette)}

    pieces: dict[int, np.ndarray] = {}
    guides: dict[int, tuple[tuple[int, int, int], ...]] = {}
    source_use = {int(c): 0 for c in source_palette}
    target_use = {int(c): 0 for c in target_palette}

    for key, ids in groups.items():
        h, w = map(int, key)
        anchors = _guide_positions(h, w)
        used_signatures: set[tuple[tuple[int, int, int], ...]] = set()
        for idx in ids:
            for _ in range(400):
                primary = int(random.choice(source_palette))
                cols = [primary]
                if h * w >= 3 and random.random() < BICHROMATIC_FRAC:
                    secondary = int(random.choice([c for c in source_palette if c != primary]))
                    cols.append(secondary)
                patch = _make_source_patch(h, w, cols)
                mapped_patch = _apply_color_map(patch, color_map)
                marker_count = 2 if len(ids) > 1 and len(anchors) >= 2 else random.choice([1, 2])
                marker_count = min(marker_count, len(anchors))
                chosen = tuple(sorted(random.sample(anchors, marker_count)))
                signature = tuple(sorted((int(r), int(c), int(mapped_patch[r, c])) for r, c in chosen))
                if signature in used_signatures:
                    continue
                pieces[idx] = patch
                guides[idx] = signature
                used_signatures.add(signature)
                for col in np.unique(patch):
                    source_use[int(col)] += int(np.count_nonzero(patch == int(col)))
                for _r, _c, col in signature:
                    target_use[int(col)] += 1
                break
            else:
                return None

    if not all(v > 0 for v in source_use.values()):
        return None
    if not all(v > 0 for v in target_use.values()):
        return None
    return pieces, guides, color_map


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
    color_map: dict[int, int],
) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for i, (r, c, _h, _w) in enumerate(rects):
        paste(out, _apply_color_map(pieces[i], color_map), r, c)
    return out


def _legend_origin(
    H: int,
    W: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    legend_h: int,
    legend_w: int,
) -> tuple[int, int]:
    if dh == H:
        c0 = W - legend_w if tc0 == 0 else 0
        r0 = 0 if random.random() < 0.5 else H - legend_h
    else:
        r0 = H - legend_h if tr0 == 0 else 0
        c0 = 0 if random.random() < 0.5 else W - legend_w
    return int(r0), int(c0)


def _paint_legend(canvas: np.ndarray, color_map: dict[int, int], r0: int, c0: int) -> None:
    rows = list(color_map.items())
    random.shuffle(rows)
    for i, (src, dst) in enumerate(rows):
        canvas[r0 + i, c0] = int(src)
        canvas[r0 + i, c0 + 1] = int(dst)


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
    guides: dict[int, tuple[tuple[int, int, int], ...]],
    color_map: dict[int, int],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for i, (r, c, h, w) in enumerate(rects):
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)
        for gr, gc, col in guides[i]:
            inp[tr0 + r + gr, tc0 + c + gc] = int(col)

    legend_h, legend_w = len(color_map), 2
    lr0, lc0 = _legend_origin(H, W, tr0, tc0, dh, dw, legend_h, legend_w)
    _paint_legend(inp, color_map, lr0, lc0)

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = inp[tr0 : tr0 + dh, tc0 : tc0 + dw]
    placement[lr0 : lr0 + legend_h, lc0 : lc0 + legend_w] = inp[lr0 : lr0 + legend_h, lc0 : lc0 + legend_w]
    placement[0, :] = int(base)
    placement[H - 1, :] = int(base)
    placement[:, 0] = int(base)
    placement[:, W - 1] = int(base)

    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c)
        paste(placement, patch, r, c)
    return inp


def _variant_visible(
    pieces: dict[int, np.ndarray],
    guides: dict[int, tuple[tuple[int, int, int], ...]],
    color_map: dict[int, int],
    rects: list[tuple[int, int, int, int]],
) -> bool:
    changed = 0
    mismatched_guides = 0
    used_targets: set[int] = set()
    shape_counts: dict[tuple[int, int], int] = {}
    for _r, _c, h, w in rects:
        key = (int(h), int(w))
        shape_counts[key] = shape_counts.get(key, 0) + 1
    for i, patch in pieces.items():
        mapped = _apply_color_map(patch, color_map)
        changed += int(np.count_nonzero(mapped != patch))
        for gr, gc, col in guides[i]:
            used_targets.add(int(col))
            if int(patch[gr, gc]) != int(col):
                mismatched_guides += 1
    dup_groups = sum(1 for v in shape_counts.values() if v >= 2)
    return changed >= 8 and mismatched_guides >= max(4, len(rects) // 2) and len(used_targets) == len(color_map) and dup_groups >= MIN_DUP_GROUPS


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0
    long = dh if vertical else dw

    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    canon = _corridor_rects(long, thick, total)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h) for r, c, h, w in canon]

    bg, base = random.sample(range(1, 10), 2)
    made = _make_pieces_and_guides(rects, bg, base)
    if made is None:
        return None
    pieces, guides, color_map = made
    if not _variant_visible(pieces, guides, color_map, rects):
        return None

    out = _render_output(dh, dw, base, rects, pieces, color_map)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces, guides, color_map)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
