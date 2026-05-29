# Combo variant of a251c730 (index 25)
# combo_spec: D7/2+D14/1
# Name: 排除式映射+多类型

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D7/2 + D14/1 combo
# variant:
#   concretization_and_trace:
#     - The scene still has two bordered rectangles on striped wallpaper.
#     - The source rectangle now always shows 3-4 distinct sprite types. Each type has
#       one singleton anchor color and may appear in several identical copies.
#     - The destination rectangle shows only singleton markers, but only for a strict
#       subset of those anchor colors. The remaining source types are distractors.
#     - Concrete trace:
#       before: source anchors {red, teal}, destination markers {red, teal, red}
#       original solve: recover both sprite templates, then stamp red / teal / red.
#       combo: source anchors {red, teal, yellow, maroon}, destination markers
#       {red, teal, red}. The stamping step is still red / teal / red, but now the
#       solver must first notice that yellow and maroon never occur in the target and
#       therefore must be excluded.
#   complexity_calibration:
#     - Total source type count is forced to 3 or 4 so the source visibly exceeds the
#       original 1-2 type regime.
#     - Active destination colors are 2 or 3, while total marker count is 3-5, so the
#       target still has repeated evidence and a larger matching workload than the base task.
#     - Source copy count is 4-6 so each sample contains repeated templates without
#       overcrowding the source box.
#   implicit_assumption_review:
#     - Rectangle detection still assumes one rectangle contains multi-cell components
#       and the other contains only singletons; the generator preserves that split.
#     - Anchor extraction can no longer rely on "pick the least frequent color, break
#       ties by color id". The combo solver instead requires a unique least-frequent
#       color with count 1, matching the generator's singleton-anchor guarantee.
#     - Patch holes are stored as -1, not 0, so visible color 0 would not be mistaken
#       for empty space if it ever appeared.
#     - Template lookup keyed by anchor color remains valid because all copies of one
#       type are still identical within a sample.
#   placement_legality:
#     - Destination construction is output-first: full sprites are placed into a blank
#       destination interior with padding, then reduced to singleton markers for input.
#       This guarantees every later stamp fits and prevents overlaps.
#     - Source sprites are placed with 1-cell border and padding, so 3-4 types plus
#       repeated copies remain separated and inside the source frame.
#     - Rectangle placement on the 30x30 wallpaper remains non-overlapping.
#   visual_observability:
#     - Every sample contains at least one unused source anchor color, and often two
#       when four source types are present. A human can compare source anchor colors
#       against destination markers and spot which types are decoys.
#     - The source visibly contains 3-4 template families, which makes the D14 change
#       legible even when only 2-3 of those families are active in the target.
#   cleanliness:
#     - Marker spacing is enforced through padded full-sprite placement, so input
#       singletons stay isolated and output stamps do not merge into noisy blobs.
#     - Source copies are spaced apart, which keeps distractors readable instead of
#       letting them blend into an ambiguous cluster.
#   inference_of_change:
#     - No rule depends on numeric color ordering or invisible metadata.
#     - The structural clue is fully visible: "copy only the source sprite families
#       whose singleton anchor colors reappear as destination markers".
#   new_main_assumptions_checked_against_generator:
#     - Each source component has exactly one singleton anchor color: guaranteed by the
#       fixed sprite templates.
#     - Every destination marker color appears in the source at least once: guaranteed
#       by choosing active types from the source type set.
#     - Some source anchor colors are unused: guaranteed by choosing fewer active types
#       than total source types and checked again by variant_metrics().


def _interior_bg(g: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(g: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != ibg, connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        patch[~mask] = -1
        vals = [int(inner[int(r), int(c)]) for r, c in coords]
        comps.append(
            {
                "size": int(coords.shape[0]),
                "colors": dict(Counter(vals)),
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
            }
        )

    comps.sort(key=lambda d: -int(d["size"]))
    return int(ibg), comps


def _find_src_dst(g: np.ndarray) -> tuple[tuple, tuple]:
    H, W = g.shape
    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[r0, c0])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if int(g[r0, c1]) != bc or int(g[r1, c0]) != bc or int(g[r1, c1]) != bc:
                        continue
                    if not bool((g[r0, c0 : c1 + 1] == bc).all()):
                        continue
                    if not bool((g[r1, c0 : c1 + 1] == bc).all()):
                        continue
                    if not bool((g[r0 : r1 + 1, c0] == bc).all()):
                        continue
                    if not bool((g[r0 : r1 + 1, c1] == bc).all()):
                        continue
                    ibg = _interior_bg(g, (r0, c0, r1, c1))
                    if ibg == bc:
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect)
        max_sz = max((int(c["size"]) for c in comps), default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and comps and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source and destination rectangles"
    return src, dst


def _extract_anchor(comp: dict) -> tuple[int, tuple[int, int]] | None:
    cols = {int(k): int(v) for k, v in comp["colors"].items()}
    if not cols:
        return None
    least = min(cols.values())
    anchor_colors = [color for color, count in cols.items() if int(count) == int(least)]
    if least != 1 or len(anchor_colors) != 1:
        return None
    anchor_col = int(anchor_colors[0])
    anchors = np.argwhere(comp["mask"] & (comp["patch"] == anchor_col))
    if anchors.shape[0] != 1:
        return None
    ar, ac = map(int, anchors[0])
    return int(anchor_col), (int(ar), int(ac))


def _marker_points(dst_comps: list[dict]) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for comp in dst_comps:
        if int(comp["size"]) != 1:
            continue
        colors = {int(k): int(v) for k, v in comp["colors"].items()}
        if len(colors) != 1:
            continue
        color, count = next(iter(colors.items()))
        if int(count) != 1:
            continue
        rr0, cc0, _rr1, _cc1 = comp["bbox"]
        markers.append((int(rr0), int(cc0), int(color)))
    markers.sort()
    return markers


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    _src_rect, _src_bc, _src_bg, src_comps = src
    _dst_rect, _dst_bc, _dst_bg, dst_comps = dst
    source_anchors = set()
    for comp in src_comps:
        anchor = _extract_anchor(comp)
        if anchor is not None:
            source_anchors.add(int(anchor[0]))
    markers = _marker_points(dst_comps)
    marker_colors = {int(color) for _rr, _cc, color in markers}
    return {
        "source_anchor_count": int(len(source_anchors)),
        "marker_anchor_count": int(len(marker_colors)),
        "unused_source_anchor_count": int(len(source_anchors - marker_colors)),
        "total_marker_count": int(len(markers)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by stamping only the source sprite families that the target actually requests.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a source rectangle with 3-4 multi-cell sprite types and a destination rectangle with singleton anchor-colored markers.
    - Every sprite has one singleton anchor color, but some source anchor colors never appear in the destination and therefore belong to distractor types.

    Rule:
    - Detect source sprites, extract each unique singleton anchor color, and keep only the templates whose anchor color is present among destination markers.
    - For every destination marker, align the matching source sprite so that its anchor lands on the marker, then stamp into a cleared destination frame.
    - Ignore all source sprite types whose anchor color has no marker in the destination.

    Key insight: Anchor color still defines the stamp template, but the destination first tells you which of the 3-4 source families are relevant and which are decoys.
    """
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, dst_comps = dst

    markers = _marker_points(dst_comps)
    marker_colors = {int(color) for _rr, _cc, color in markers}

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        anchor = _extract_anchor(comp)
        if anchor is None:
            continue
        anchor_col, anchor_rc = anchor
        if anchor_col not in marker_colors:
            continue
        patch = comp["patch"]
        mask = comp["mask"]
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col][1].sum()):
            templates[anchor_col] = (patch.copy(), mask.copy(), anchor_rc)

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    for rr, cc, mcol in markers:
        if mcol not in templates:
            continue
        patch, mask, (ar, ac) = templates[mcol]
        top = int(1 + rr - ar)
        left = int(1 + cc - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
GAP_RANGE = (1, 3)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (13, 19)
SRC_W_RANGE = (12, 19)
DST_H_RANGE = (10, 18)
DST_W_RANGE = (10, 19)
DST_W_SIDE_MAX = 18

# === CFG: Type counts ===
TOTAL_TYPE_CHOICES = (3, 4)
FOUR_TYPE_ACTIVE_CHOICES = (2, 3, 3)
MARKER_COUNT_CAP = 5
SOURCE_COPY_EXTRA_RANGE = (1, 2)

# === CFG: Color/layout ===
WALLPAPER_COLOR_CHOICES = (4, 5)
MARKER_PLACE_RETRIES = 35
SOURCE_SPRITE_PADDING = 1
DEST_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 600

SPRITE_SHAPES = [
    (("AB.", "BBB"), (0, 0)),
    (("ABB", "BBB"), (0, 0)),
    ((".A.", "BBB", ".B."), (0, 1)),
    (("AB.", "BBB", ".BB"), (0, 0)),
    (("ABB", "BBB", "..B"), (0, 0)),
    ((".AB", "BBB", "BB."), (0, 1)),
    (("B.B", "BBB", ".A."), (2, 1)),
    ((".B.", "BAB", ".B."), (1, 1)),
    (("BBB", "BAB"), (1, 1)),
    (("AB.", "BBB", ".B."), (0, 0)),
    (("BAB", "BBB"), (0, 1)),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    char_map = {"A": int(anchor), "B": int(body)}
    rows = [[char_map.get(ch, int(bg)) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), int(bg), dtype=int)
    rect[0, :] = rect[-1, :] = int(border)
    rect[:, 0] = rect[:, -1] = int(border)
    return rect


def _choose_counts(n_total: int, n_bins: int) -> list[int]:
    counts = [1] * int(n_bins)
    for _ in range(int(n_total) - int(n_bins)):
        counts[random.randrange(n_bins)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < DST_W_RANGE[0]:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < DST_H_RANGE[0]:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return src_h, src_w, dst_h, dst_w


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        slack = GRID_SIZE - src_w - dst_w - gap
        left_margin = random.randint(0, slack)
        src_r = random.randint(0, GRID_SIZE - src_h)
        dst_r = random.randint(0, GRID_SIZE - dst_h)
        if random.random() < 0.5:
            src_c, dst_c = left_margin, left_margin + src_w + gap
        else:
            dst_c, src_c = left_margin, left_margin + dst_w + gap
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        slack = GRID_SIZE - src_h - dst_h - gap
        top_margin = random.randint(0, slack)
        src_c = random.randint(0, GRID_SIZE - src_w)
        dst_c = random.randint(0, GRID_SIZE - dst_w)
        if random.random() < 0.5:
            src_r, dst_r = top_margin, top_margin + src_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (int(src_r), int(src_c)), (int(dst_r), int(dst_c))


def _assign_palette(total_types: int) -> tuple[list[int], int, int, int, int, list[int], list[int]] | None:
    wallpaper_n = 4 if total_types == 4 else random.choice(list(WALLPAPER_COLOR_CHOICES))
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    if len(remaining) < 1 + total_types:
        return None

    src_border = int(remaining.pop())
    anchors = [int(remaining.pop()) for _ in range(total_types)]

    src_bg = int(random.choice(wallpaper))

    extra_dst_choices = [int(c) for c in remaining if int(c) != int(src_border)]
    if extra_dst_choices and random.random() < 0.5:
        dst_border = int(random.choice(extra_dst_choices))
        remaining.remove(dst_border)
    else:
        wallpaper_border_choices = [int(c) for c in wallpaper if int(c) != int(src_border)]
        if not wallpaper_border_choices:
            return None
        dst_border = int(random.choice(wallpaper_border_choices))

    dst_bg_choices = [int(c) for c in wallpaper if int(c) != int(dst_border)]
    if not dst_bg_choices:
        return None
    dst_bg = int(random.choice(dst_bg_choices))

    body_pool = [int(c) for c in wallpaper if int(c) != int(src_bg) and int(c) != int(dst_bg)]
    if remaining:
        body_pool.append(int(random.choice(remaining)))
    if not body_pool:
        return None

    return wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, body_pool


def _choose_bodies(body_pool: list[int], total_types: int) -> list[int]:
    bodies: list[int] = []
    shuffled_pool = body_pool[:]
    random.shuffle(shuffled_pool)
    for i in range(total_types):
        if i < len(shuffled_pool):
            bodies.append(int(shuffled_pool[i]))
        else:
            bodies.append(int(random.choice(body_pool)))
    return bodies


def _ordered_patch_items(sprite_types: list[dict], type_ids: list[int], bg: int) -> list[tuple[int, np.ndarray]]:
    items = []
    for type_id in type_ids:
        sprite = sprite_types[int(type_id)]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], bg)
        area = int(np.count_nonzero(patch != int(bg)))
        items.append((int(type_id), patch, area))
    items.sort(key=lambda item: (-int(item[2]), random.random()))
    return [(int(type_id), patch) for type_id, patch, _area in items]


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), int(dst_bg), dtype=int)
    marker_points: list[tuple[int, int, int]] = []

    for _ in range(MARKER_PLACE_RETRIES):
        inner.fill(int(dst_bg))
        marker_points = []
        ok = True
        for type_id, patch in _ordered_patch_items(sprite_types, marker_type_ids, dst_bg):
            sprite = sprite_types[int(type_id)]
            try:
                r0, c0 = random_free_location(
                    inner,
                    patch,
                    transparent=int(dst_bg),
                    padding=DEST_SPRITE_PADDING,
                )
            except ValueError:
                ok = False
                break
            paste(inner, patch, r0, c0, transparent=int(dst_bg))
            mr = int(r0 + sprite["anchor_rc"][0])
            mc = int(c0 + sprite["anchor_rc"][1])
            marker_points.append((int(mr), int(mc), int(type_id)))
        if ok:
            return inner.copy(), marker_points[:]
    return None


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), int(src_bg), dtype=int)
    for _ in range(MARKER_PLACE_RETRIES):
        inner.fill(int(src_bg))
        ok = True
        for type_id, patch in _ordered_patch_items(sprite_types, source_type_ids, src_bg):
            try:
                r0, c0 = random_free_location(
                    inner,
                    patch,
                    transparent=int(src_bg),
                    border=SOURCE_SPRITE_PADDING,
                    padding=SOURCE_SPRITE_PADDING,
                )
            except ValueError:
                ok = False
                break
            paste(inner, patch, r0, c0, transparent=int(src_bg))
        if ok:
            return inner.copy()
    return None


def _gen_attempt() -> np.ndarray | None:
    total_types = int(random.choice(TOTAL_TYPE_CHOICES))
    if total_types == 4:
        active_types = int(random.choice(FOUR_TYPE_ACTIVE_CHOICES))
    else:
        active_types = 2
    distractor_types = int(total_types - active_types)
    if distractor_types < 1:
        return None

    palette_result = _assign_palette(total_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, body_pool = palette_result

    sprite_specs = random.sample(SPRITE_SHAPES, total_types)
    bodies = _choose_bodies(body_pool, total_types)
    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": int(body),
            "anchor": int(anchor),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors)
    ]

    active_type_ids = list(range(active_types))
    n_markers_min = max(3, active_types + 1)
    n_markers_max = min(MARKER_COUNT_CAP, active_types + 2)
    if n_markers_min > n_markers_max:
        return None
    n_markers = int(random.randint(n_markers_min, n_markers_max))
    active_counts = _choose_counts(n_markers, active_types)
    marker_type_ids = [active_type_ids[i] for i, ct in enumerate(active_counts) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = int(total_types + random.randint(*SOURCE_COPY_EXTRA_RANGE))
    source_counts = [1] * total_types
    extra_slots = n_source - total_types
    extra_bias = list(range(active_types)) * 3 + list(range(active_types, total_types))
    for _ in range(extra_slots):
        source_counts[random.choice(extra_bias)] += 1
    source_type_ids = [i for i, ct in enumerate(source_counts) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if int(src_h * src_w) <= int(dst_h * dst_w):
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id in marker_points:
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[int(type_id)]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    metrics = variant_metrics(grid)
    if metrics["source_anchor_count"] not in TOTAL_TYPE_CHOICES:
        return None
    if metrics["unused_source_anchor_count"] < 1:
        return None
    if metrics["marker_anchor_count"] < 2:
        return None
    if metrics["total_marker_count"] < 3:
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(
        f"failed to generate a251c730 combo D7/2+D14/1 input after {MAX_GEN_ATTEMPTS} attempts"
    )
