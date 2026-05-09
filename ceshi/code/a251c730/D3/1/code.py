# Variant of a251c730
# Dimension: D3 — 目标结构：标记类型与放置 (dst)
# Change: 目标矩形内部放置2-3个单像素标记，每个标记颜色对应一种精灵的锚点颜色，标记周围有足够空间容纳完整精灵印刻... → 标记在边框上而非内部

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D3 variant
# ----------------------------------
# spec:
#   original_core:
#     - source rectangle still contains 2-4 copies of 1-2 multicolor sprites
#     - destination rectangle is still the output crop and still uses anchor-color matching
#     - stamping still translates a source sprite so its anchor pixel lands on a marker pixel
#   changed_dst_structure:
#     - destination markers move from the interior onto the rectangle border
#     - markers are non-corner border singletons; the rest of the border keeps one dominant border color
#     - output is the full destination rectangle after stamping from those border anchors
#   concretized_border_behavior:
#     - each sprite type is only allowed to use sides where border alignment clips away at least one
#       colored cell while still leaving a visible inward remnant
#     - after alignment, part of the sprite may still touch the border near the marker, but the main
#       visible mass extends inward and some colored cells are clipped outside the frame
#     - this makes the side-to-inward relation visible without adding rotation or reflection
#   traced_difference:
#     - original: a marker at interior (4,6) produced a full sprite entirely inside the destination interior
#     - variant: a top-edge marker at border (0,6) aligns the same anchor to row 0, so cells above the
#       anchor are clipped away and only the lower part of the sprite remains inside the frame
#     - the changed step is therefore "where markers are read from" and "how much of the translated patch
#       survives clipping", not the color-matching logic
#   complexity_calibration:
#     - keep 1-2 sprite types and 2-3 markers so color matching remains repeated and unambiguous
#     - allow all four sides across samples, but never corners, avoiding an extra corner-specific rule
#     - require every placed marker to clip at least one colored sprite cell, so the variant does not
#       collapse back into the original full-inside stamping behavior
#   implicit_assumption_audit:
#     - original destination detection assumed a strict uniform border and singleton interior components;
#       that fails here because the border is intentionally interrupted and the interior is blank
#     - solver is updated to detect source rectangles with strict borders, and destination rectangles with
#       the same-color corners, 2-3 non-corner border deviations, and a uniform interior background
#     - original marker scan only checked destination interior; that must move to perimeter scanning
#     - original unconditional border redraw remains safe because destination input only contains border
#       markers, while output stamping is applied afterward and can overwrite those border pixels
#   placement_legality:
#     - generator enumerates side-specific border positions and accepts only non-overlapping visible stamp
#       footprints, so partial imprints stay readable and solver clipping matches the intended output
#     - source placement stays unchanged: sprites are separated from each other and from the source border
#   visual_observability:
#     - marker colors are still the only matching key, but the changed position is obvious because every
#       marker sits on an edge and every stamp loses at least one colored cell outside the frame
#     - no color-id ordering is used for generation choices; color ids only serve as raw ARC palette labels
#   cleanliness:
#     - visible stamp footprints are forced not to overlap, keeping the output easy to parse
#     - only 2-3 border markers are used, so even when a stamp colors multiple edge cells, the frame stays
#       mostly intact and the local edge-to-inward direction remains readable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via color-matched border markers.

    Input structure:
    - Two bordered rectangles appear over striped wallpaper: a source box with multi-cell sprites and a
      destination box whose border contains 2-3 singleton marker pixels.
    - Each source sprite has a unique minority-color anchor pixel; destination markers reuse those colors.

    Rule:
    - Match each destination border marker to the source sprite with the same anchor color.
    - Translate the sprite so its anchor lands exactly on that border marker, then clip anything that falls
      outside the destination frame.
    - Output is the cropped destination rectangle after clearing it and applying all such border-aligned stamps.

    Key insight: The color still selects the sprite, but the marker's edge location determines which part of
    that sprite survives inside the frame after clipping.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
        comps = []

        for coords in components_mask(inner != int(ibg), connectivity=4):
            if coords.size == 0:
                continue
            rr0, cc0, rr1, cc1 = bbox_coords(coords)
            mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
            mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
            patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
            patch[~mask] = 0

            vals = [int(inner[int(r), int(c)]) for r, c in coords]
            cols = Counter(vals)
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(cols),
                    "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                    "mask": mask,
                    "patch": patch,
                }
            )

        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

    def border_markers(rect: tuple[int, int, int, int], border_color: int) -> list[tuple[int, int, int]]:
        r0, c0, r1, c1 = map(int, rect)
        pts: list[tuple[int, int, int]] = []
        for c in range(c0 + 1, c1):
            if int(g[r0, c]) != int(border_color):
                pts.append((int(r0), int(c), int(g[r0, c])))
            if int(g[r1, c]) != int(border_color):
                pts.append((int(r1), int(c), int(g[r1, c])))
        for r in range(r0 + 1, r1):
            if int(g[r, c0]) != int(border_color):
                pts.append((int(r), int(c0), int(g[r, c0])))
            if int(g[r, c1]) != int(border_color):
                pts.append((int(r), int(c1), int(g[r, c1])))
        return pts

    strict_rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    loose_rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(H):
        for c0 in range(W):
            bc = int(g[r0, c0])
            for r1 in range(r0 + min_h - 1, H):
                for c1 in range(c0 + min_w - 1, W):
                    if int(g[r0, c1]) != int(bc) or int(g[r1, c0]) != int(bc) or int(g[r1, c1]) != int(bc):
                        continue
                    inner = g[r0 + 1 : r1, c0 + 1 : c1]
                    if inner.size == 0:
                        continue
                    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
                    if int(ibg) == int(bc):
                        continue

                    top = g[r0, c0 : c1 + 1]
                    bottom = g[r1, c0 : c1 + 1]
                    left = g[r0 : r1 + 1, c0]
                    right = g[r0 : r1 + 1, c1]
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))

                    if bool((top == int(bc)).all()) and bool((bottom == int(bc)).all()) and bool(
                        (left == int(bc)).all()
                    ) and bool((right == int(bc)).all()):
                        strict_rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))
                        continue

                    mismatch_count = int(np.count_nonzero(top[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(bottom[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(left[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(right[1:-1] != int(bc)))

                    if 2 <= mismatch_count <= 3 and bool((inner == int(ibg)).all()):
                        loose_rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert strict_rects and loose_rects

    strict_rects.sort(reverse=True, key=lambda x: int(x[0]))
    loose_rects.sort(reverse=True, key=lambda x: int(x[0]))

    src = None
    for _area, rect, bc, _ibg in strict_rects[:8]:
        ibg2, comps = rect_components(rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1:
            src = (rect, bc, ibg2, comps)
            break

    assert src is not None, "must find the source rectangle"
    (sr, sc, sr1, sc1), _src_bc, src_bg, src_comps = src

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {k: int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if not cols:
            continue
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    dst = None
    for _area, rect, bc, ibg in loose_rects[:10]:
        markers = border_markers(rect, int(bc))
        if len(markers) < 2 or len(markers) > 3:
            continue
        if all(int(col) in templates for _r, _c, col in markers):
            dst = (rect, bc, ibg, markers)
            break

    assert dst is not None, "must find the destination rectangle with border markers"
    (dr, dc, dr1, dc1), dst_bc, dst_bg, dst_markers = dst

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    for mr, mc, mcol in dst_markers:
        if int(mcol) not in templates:
            continue
        patch, mask, (ar, ac) = templates[int(mcol)]
        top = int((mr - dr) - int(ar))
        left = int((mc - dc) - int(ac))
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 300

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
    """Render a sprite template into a colored patch. 'A'=anchor, 'B'=body, '.'=bg."""
    char_map = {"B": body, "A": anchor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    """Create a 30x30 striped wallpaper (vertical or horizontal)."""
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    """Create an h x w rectangle with a 1-pixel border and interior filled with bg."""
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    """Distribute n_total items across n_types bins, each bin >= 1."""
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    """Pick (src_h, src_w, dst_h, dst_w) that fit the 30x30 canvas for the given layout."""
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
    """Compute non-overlapping (row, col) origins for src and dst rectangles."""
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    """Allocate wallpaper, border, anchor, and body colors."""
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + n_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]

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


def _compatible_sides(shape_rows: tuple[str, ...], anchor_rc: tuple[int, int]) -> list[str]:
    """Return border sides where anchoring on that side necessarily clips colored cells."""
    h = len(shape_rows)
    w = len(shape_rows[0])
    ar, ac = map(int, anchor_rc)

    non_bg_above = any(shape_rows[r][c] != "." for r in range(ar) for c in range(w))
    non_bg_below = any(shape_rows[r][c] != "." for r in range(ar + 1, h) for c in range(w))
    non_bg_left = any(shape_rows[r][c] != "." for r in range(h) for c in range(ac))
    non_bg_right = any(shape_rows[r][c] != "." for r in range(h) for c in range(ac + 1, w))

    sides: list[str] = []
    if ar > 0 and non_bg_above:
        sides.append("top")
    if ar < h - 1 and non_bg_below:
        sides.append("bottom")
    if ac > 0 and non_bg_left:
        sides.append("left")
    if ac < w - 1 and non_bg_right:
        sides.append("right")
    return sides


def _visible_stamp_stats(
    patch: np.ndarray,
    anchor_rc: tuple[int, int],
    marker_rc: tuple[int, int],
    frame_h: int,
    frame_w: int,
    bg: int,
) -> tuple[set[tuple[int, int]], int]:
    """Visible cells and clipped colored-cell count after anchoring patch at marker_rc."""
    ar, ac = map(int, anchor_rc)
    mr, mc = map(int, marker_rc)
    top = int(mr - ar)
    left = int(mc - ac)

    visible: set[tuple[int, int]] = set()
    total_colored = 0
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            if int(patch[r, c]) == int(bg):
                continue
            total_colored += 1
            rr = int(top + r)
            cc = int(left + c)
            if 0 <= rr < frame_h and 0 <= cc < frame_w:
                visible.add((int(rr), int(cc)))
    return visible, int(total_colored - len(visible))


def _place_border_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int, dst_border: int
) -> tuple[np.ndarray, list[tuple[int, int, int, str, int]]] | None:
    """Place border markers and build the stamped output frame they imply."""
    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    occupied: set[tuple[int, int]] = set()
    marker_points: list[tuple[int, int, int, str, int]] = []

    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        candidates: list[tuple[int, int, str, set[tuple[int, int]], int]] = []

        for side in sprite["compatible_sides"]:
            if side == "top":
                for mc in range(1, dst_w - 1):
                    visible, clipped = _visible_stamp_stats(patch, sprite["anchor_rc"], (0, mc), dst_h, dst_w, dst_bg)
                    candidates.append((0, mc, side, visible, clipped))
            elif side == "bottom":
                for mc in range(1, dst_w - 1):
                    visible, clipped = _visible_stamp_stats(
                        patch, sprite["anchor_rc"], (dst_h - 1, mc), dst_h, dst_w, dst_bg
                    )
                    candidates.append((dst_h - 1, mc, side, visible, clipped))
            elif side == "left":
                for mr in range(1, dst_h - 1):
                    visible, clipped = _visible_stamp_stats(patch, sprite["anchor_rc"], (mr, 0), dst_h, dst_w, dst_bg)
                    candidates.append((mr, 0, side, visible, clipped))
            else:
                for mr in range(1, dst_h - 1):
                    visible, clipped = _visible_stamp_stats(
                        patch, sprite["anchor_rc"], (mr, dst_w - 1), dst_h, dst_w, dst_bg
                    )
                    candidates.append((mr, dst_w - 1, side, visible, clipped))

        random.shuffle(candidates)
        placed = False
        for mr, mc, side, visible, clipped in candidates:
            if clipped <= 0:
                continue
            if not any(0 < r < dst_h - 1 and 0 < c < dst_w - 1 for r, c in visible):
                continue
            if visible & occupied:
                continue
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _tid, _side, _clip in marker_points):
                continue

            paste(output, patch, mr - sprite["anchor_rc"][0], mc - sprite["anchor_rc"][1], transparent=dst_bg, clip=True)
            occupied |= visible
            marker_points.append((int(mr), int(mc), int(type_id), side, int(clipped)))
            placed = True
            break

        if not placed:
            return None

    return output, marker_points


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    """Place sprites into the source interior, returning stamped_inner or None."""
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], src_bg)
        try:
            r0, c0 = random_free_location(
                inner, patch, transparent=src_bg, border=SOURCE_SPRITE_PADDING, padding=SOURCE_SPRITE_PADDING
            )
        except ValueError:
            return None
        paste(inner, patch, r0, c0, transparent=src_bg)
    return inner


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    bodies: list[int] = []
    for i in range(n_types):
        choices = [c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(forced_body)
        else:
            bodies.append(random.choice(choices))

    sprite_types = []
    for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors):
        compatible_sides = _compatible_sides(shape_rows, anchor_rc)
        if not compatible_sides:
            return None
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": anchor_rc,
                "body": body,
                "anchor": anchor,
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
                "compatible_sides": compatible_sides,
            }
        )

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_border_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg, dst_border)
    if marker_result is None:
        return None
    output, marker_points = marker_result

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id, _side, _clipped in marker_points:
        dest_rect[mr, mc] = sprite_types[type_id]["anchor"]

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 D3 input after %d attempts" % MAX_GEN_ATTEMPTS)
