# Variant of a251c730
# Dimension: D7 — 源-目标关系：基数映射 (meta)
# Change: 每种精灵类型在目标中恰好对应一个标记（一对一映射），源中可有同类型的多个副本但只有一个被使用... → 排除式映射

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D7 exclusion-mapping variant
# variant:
#   source_task: none
#   concreteization:
#     - The input still contains two bordered rectangles on striped wallpaper.
#     - The destination rectangle still contains only singleton anchor-color markers.
#     - The source rectangle now contains all usable sprite types plus exactly one extra
#       distractor type whose singleton anchor color never appears among the destination
#       markers.
#     - Output stays the cropped destination frame with its interior reset to the
#       destination background before stamping only the marker-matched sprites.
#   behavior_trace:
#     - Example: source anchors = {red, teal, yellow}, destination markers = {red, teal}.
#     - Original data model had no yellow distractor, so every source type was relevant.
#     - In this variant, yellow must be ignored because no yellow marker exists.
#     - Solver flow therefore becomes: read marker colors first -> keep only source
#       sprites whose unique singleton anchor color is in that set -> stamp at markers.
#   complexity_calibration:
#     - Active sprite types remain 1-2, matching the original task's small hypothesis
#       space.
#     - A single distractor type is always added, so the source shows 2-3 distinct
#       anchor colors while the destination shows only 1-2 of them.
#     - Marker count stays 2-3 and source sprite count stays 3-5, so examples remain
#       readable and repeated enough for humans to notice the exclusion rule.
#   implicit_assumption_review:
#     - Rectangle detection still assumes one rectangle has only singleton components
#       and the other has multi-cell components; the generator preserves that.
#     - Anchor extraction can no longer rely on color-id tie breaking. It now requires
#       a unique least-frequent color with count 1, which the generator guarantees.
#     - Template selection now assumes every destination marker color appears in at
#       least one source sprite and that unused source anchor colors also exist.
#     - Duplicate copies of the same source type remain harmless because copies are
#       identical; the solver keeps the largest matching template for safety.
#   placement_legality:
#     - Destination markers are placed by first laying down the full output sprites in
#       an empty destination interior, so every later stamp fits in-bounds.
#     - Source sprites use random_free_location with border and padding, so sprites do
#       not overlap each other or the source frame.
#     - Rectangle placement on the 30x30 wallpaper remains non-overlapping.
#   visual_observability:
#     - Every generated instance contains at least one unused source anchor color.
#     - Because anchor colors are the only singleton colors inside sprites, a human can
#       compare source anchor colors against destination markers and spot the missing one.
#   cleanliness:
#     - Only one extra distractor type is introduced, avoiding clutter while still
#       making the source visibly richer than the destination.
#     - Padding around source sprites and non-adjacent marker anchors keep regions legible.
#   color_inference:
#     - No visible rule depends on numeric color ordering.
#     - Anchor colors are unique per type and excluded from wallpaper/background roles,
#       so marker-to-sprite matching is recoverable from a few examples.


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
        patch[~mask] = 0
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
                    rects.append((area, (r0, c0, r1, c1), bc, ibg))

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
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source and destination rectangles"
    return src, dst


def _extract_anchor(comp: dict) -> tuple[int, tuple[int, int]] | None:
    cols = {int(k): int(v) for k, v in comp["colors"].items()}
    if not cols:
        return None
    least = min(cols.values())
    anchor_colors = [col for col, cnt in cols.items() if int(cnt) == int(least)]
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
        if count != 1:
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
    source_anchors = []
    for comp in src_comps:
        anchor = _extract_anchor(comp)
        if anchor is not None:
            source_anchors.append(int(anchor[0]))
    marker_colors = [int(color) for _rr, _cc, color in _marker_points(dst_comps)]
    source_set = set(source_anchors)
    marker_set = set(marker_colors)
    return {
        "source_anchor_count": int(len(source_set)),
        "marker_anchor_count": int(len(marker_set)),
        "unused_source_anchor_count": int(len(source_set - marker_set)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp only the source sprites that are actually requested by destination markers.

    Input structure:
    - Two bordered rectangles appear on a striped wallpaper: a source box full of
      multi-cell sprites and a destination box holding singleton color markers.
    - Each sprite still has one singleton anchor color, but the source can now
      include extra sprite types whose anchor color never appears in the destination.

    Rule:
    - Read the destination marker colors first, then keep only source sprites whose
      unique anchor color is present among those markers.
    - For each destination marker, translate its matching source sprite so the
      sprite's anchor lands on the marker, and stamp into a cleared destination frame.
    - Ignore any source sprite type whose anchor color has no destination marker.

    Key insight: The destination tells you which anchor colors matter; unmatched
    source anchors are decoys and must be excluded before stamping.
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


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (8, 9)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
ACTIVE_TYPE_CHOICES = (1, 2)
NOISE_TYPE_COUNT = 1
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (3, 5)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 400

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
    char_map = {"B": body, "A": anchor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_bins: int) -> list[int]:
    counts = [1] * n_bins
    for _ in range(n_total - n_bins):
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
    return (src_r, src_c), (dst_r, dst_c)


def _assign_palette(total_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    wallpaper_n = random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + total_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(total_types)]

    dst_border_extra = extras_needed >= 2 or (extras_needed == 1 and random.random() < 0.5)
    body_extra = extras_needed >= 2 or (extras_needed >= 1 and not dst_border_extra)

    if dst_border_extra:
        if not remaining:
            return None
        dst_border = remaining.pop()
    else:
        border_choices = [c for c in wallpaper if c != src_border]
        if not border_choices:
            return None
        dst_border = random.choice(border_choices)

    body_pool = wallpaper[:]
    forced_body: int | None = None
    if body_extra:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr = int(r0 + sprite["anchor_rc"][0])
            mc = int(c0 + sprite["anchor_rc"][1])
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((mr, mc, type_id))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], src_bg)
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=src_bg,
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, r0, c0, transparent=src_bg)
    return inner


def _gen_attempt() -> np.ndarray | None:
    n_active_types = random.choice(list(ACTIVE_TYPE_CHOICES))
    total_types = int(n_active_types + NOISE_TYPE_COUNT)

    palette_result = _assign_palette(total_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, total_types)
    bodies: list[int] = []
    for i in range(total_types):
        choices = [c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(forced_body)
        else:
            bodies.append(random.choice(choices))

    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": body,
            "anchor": anchor,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors)
    ]

    active_type_ids = list(range(n_active_types))
    n_markers = random.randint(max(2, n_active_types), MARKER_COUNT_RANGE[1])
    active_counts = _choose_counts(n_markers, n_active_types)
    marker_type_ids = [active_type_ids[i] for i, ct in enumerate(active_counts) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    min_source = max(total_types + 1, SOURCE_SPRITE_RANGE[0])
    if min_source > SOURCE_SPRITE_RANGE[1]:
        return None
    n_source = random.randint(min_source, SOURCE_SPRITE_RANGE[1])
    source_counts = _choose_counts(n_source, total_types)
    source_type_ids = [i for i, ct in enumerate(source_counts) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id in marker_points:
        dest_rect[1 + mr, 1 + mc] = sprite_types[type_id]["anchor"]

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    metrics = variant_metrics(grid)
    if metrics["unused_source_anchor_count"] < 1:
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D7 input after {MAX_GEN_ATTEMPTS} attempts")
