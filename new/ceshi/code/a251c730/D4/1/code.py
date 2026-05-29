# Variant of a251c730
# Dimension: D4 — 变换：锚点对齐方式 (transform)
# Change: 将精灵平移使其锚点像素与目标标记像素完全重合，精灵保持原始朝向不变... → 镜像翻转后对齐

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# ===============
# design:
#   reference_task:
#     source_task: b10624e5
#     status: solver file not present in the provided workspace; only the JSON examples are available.
#     fallback: keep the original a251c730 object-discovery pipeline and realize the requested
#       "reflection" idea with an explicit quadrant-to-flip rule that is readable from the grids.
#   concrete_behavior:
#     input:
#       - Same global structure as the source task: wallpaper + one source frame + one destination frame.
#       - Source frame still contains 1-2 sprite types, each keyed by a singleton anchor color.
#       - Destination frame still contains only singleton anchor-color markers.
#     output:
#       - Start from a clean destination frame.
#       - For each marker, fetch the source sprite with the same anchor color.
#       - Decide flips from the marker's position inside the destination interior:
#         * left half  -> horizontal mirror (`np.fliplr`)
#         * upper half -> vertical mirror (`np.flipud`)
#         * top-left quadrant applies both flips; bottom-right applies none.
#       - After flipping, align the transformed anchor pixel to the marker and stamp the sprite.
#   behavior_trace:
#     example:
#       source_sprite:
#         shape_rows: ["AB.", "BBB", ".BB"]
#         anchor: A at local (0,0)
#       original_logic:
#         marker at bottom-left -> place unchanged sprite with A on the marker.
#       new_logic:
#         marker at bottom-left -> marker is in the left half but not the upper half.
#         apply `fliplr`, anchor moves to local (0,2), then align that new anchor to the marker.
#         visually, the heavier body shifts to the opposite side of the marker.
#       difference:
#         orientation now depends on destination position, not just color matching.
#   complexity_calibration:
#     - The flip rule uses only two binary tests, so it stays learnable.
#     - The generator forces at least two distinct transform states per instance and at least one
#       non-identity state, preventing degeneration back to the original "translate only" task.
#     - Sprite templates are restricted to shapes that are asymmetric under both LR and UD flips,
#       otherwise a mirror could become visually indistinguishable.
#   implicit_assumption_audit:
#     - Rectangle finding is unchanged; the variant does not alter frame detection.
#     - Template extraction still assumes each sprite has exactly one least-frequent anchor color.
#       The generator preserves that by keeping anchor count = 1 and body count > 1.
#     - The original stamping code assumed the anchor coordinates were invariant. That is no longer
#       true after reflection, so the solver must recompute anchor coordinates after each flip.
#     - The original generator placed markers by stamping the unmodified sprite. That becomes invalid
#       because a marker near the left/top must now reserve space for a reflected footprint instead.
#   placement_legality:
#     - Marker placement is output-first: for each desired transform state, enumerate marker cells
#       whose quadrant implies that state, transform the sprite, and only accept placements where
#       the full transformed footprint stays inside the destination interior and does not overlap
#       earlier sprites.
#     - Source placement remains non-overlapping with a one-cell buffer, matching the old task.
#   visual_observability:
#     - The generator always includes an LR-triggered marker and a UD-triggered marker.
#       With three markers it adds either an identity or double-flip example.
#     - Because the source sprites are asymmetric, the output visibly differs from the original
#       orientation; the rule is inferable from marker position relative to the destination center.
#   visual_cleanliness:
#     - Destination output sprites are kept non-overlapping, and markers keep at least Manhattan
#       distance 2 in the input so singleton cues remain separable.
#   color_inference:
#     - No color-id ordering is used for visible reasoning. Colors only serve the original
#       source-to-marker matching role, and the transform is driven purely by position.


TRANSFORM_STATES: dict[str, tuple[bool, bool]] = {
    "id": (False, False),
    "h": (True, False),
    "v": (False, True),
    "hv": (True, True),
}


def _marker_flip_flags(rr: int, cc: int, inner_h: int, inner_w: int) -> tuple[bool, bool]:
    """Return (flip_lr, flip_ud) induced by a marker at interior coord (rr, cc)."""
    flip_lr = int(cc) < (int(inner_w) / 2.0)
    flip_ud = int(rr) < (int(inner_h) / 2.0)
    return bool(flip_lr), bool(flip_ud)


def _state_name(rr: int, cc: int, inner_h: int, inner_w: int) -> str:
    """Compact name for the marker-induced transform."""
    return {v: k for k, v in TRANSFORM_STATES.items()}[_marker_flip_flags(rr, cc, inner_h, inner_w)]


def _transform_template(
    patch: np.ndarray, mask: np.ndarray, anchor_rc: tuple[int, int], flip_lr: bool, flip_ud: bool
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Apply the requested mirror flips and update the anchor coordinate accordingly."""
    out_patch = patch.copy()
    out_mask = mask.copy()
    ar, ac = map(int, anchor_rc)
    h, w = map(int, out_patch.shape)
    if bool(flip_ud):
        out_patch = np.flipud(out_patch).copy()
        out_mask = np.flipud(out_mask).copy()
        ar = int(h - 1 - ar)
    if bool(flip_lr):
        out_patch = np.fliplr(out_patch).copy()
        out_mask = np.fliplr(out_mask).copy()
        ac = int(w - 1 - ac)
    return out_patch, out_mask, (int(ar), int(ac))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by stamping source sprites onto anchor-colored markers.

    Input structure:
    - Two bordered rectangles sit on top of a striped wallpaper: a source frame with full sprites
      and a destination frame with only singleton anchor markers.
    - Each source sprite still carries a unique singleton anchor color that matches one destination marker color.

    Rule:
    - Match each destination marker to the source sprite that shares its anchor color.
    - Before stamping, mirror the sprite according to the marker's position in the destination interior:
      markers in the left half trigger a left-right flip, and markers in the upper half trigger an up-down flip.
    - Align the transformed anchor pixel to the marker and output the cleaned destination rectangle.

    Key insight: Anchor colors identify which sprite to copy, while the marker's position relative to the
    destination center determines which mirror symmetry to apply before alignment.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
        comps = []

        for coords in components_mask(inner != int(ibg), connectivity=4):
            if coords.size == 0:
                continue
            rr0, cc0, rr1, cc1 = bbox_coords(coords)
            mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
            mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
            patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
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

    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if int(g[int(r0), int(c1)]) != int(bc) or int(g[int(r1), int(c0)]) != int(bc) or int(
                        g[int(r1), int(c1)]
                    ) != int(bc):
                        continue
                    if not bool((g[int(r0), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r1), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c0)] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c1)] == int(bc)).all()):
                        continue
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

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

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    inner_h, inner_w = map(int, inner.shape)
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mr = int(dr + 1 + int(rr))
        mc = int(dc + 1 + int(cc))
        mcol = int(g[mr, mc])
        if int(mcol) not in templates:
            continue
        patch, mask, (ar, ac) = templates[int(mcol)]
        flip_lr, flip_ud = _marker_flip_flags(int(rr), int(cc), inner_h, inner_w)
        tpatch, tmask, (tar, tac) = _transform_template(patch, mask, (ar, ac), flip_lr, flip_ud)
        top = int((mr - dr) - int(tar))
        left = int((mc - dc) - int(tac))
        paste(out, tpatch, int(top), int(left), mask=tmask, overwrite=True, clip=True)

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
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 500

SPRITE_SHAPES = [
    (("AB.", "BBB"), (0, 0)),
    (("ABB", "BBB"), (0, 0)),
    (("AB.", "BBB", ".BB"), (0, 0)),
    (("ABB", "BBB", "..B"), (0, 0)),
    ((".AB", "BBB", "BB."), (0, 1)),
    (("AB.", "BBB", ".B."), (0, 0)),
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


def _desired_states(n_markers: int) -> list[str]:
    """Force at least one horizontal-triggered and one vertical-triggered example."""
    states = ["h", "v"]
    if int(n_markers) == 3:
        states.append(random.choice(["id", "hv"]))
    random.shuffle(states)
    return states


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, str]]] | None:
    """Place transformed sprite outputs into the dest interior, returning (inner, marker_points)."""
    inner_h = int(dst_h - 2)
    inner_w = int(dst_w - 2)
    inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    marker_points: list[tuple[int, int, int, str]] = []

    desired_states = _desired_states(len(marker_type_ids))
    marker_plan = list(zip(marker_type_ids, desired_states))
    marker_plan.sort(key=lambda item: {"hv": 0, "h": 1, "v": 1, "id": 2}[item[1]])

    for type_id, desired_state in marker_plan:
        sprite = sprite_types[int(type_id)]
        base_patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        base_mask = base_patch != int(dst_bg)
        flip_lr, flip_ud = TRANSFORM_STATES[desired_state]
        patch, mask, (ar, ac) = _transform_template(base_patch, base_mask, sprite["anchor_rc"], flip_lr, flip_ud)
        ph, pw = map(int, patch.shape)

        candidates: list[tuple[int, int, int, int]] = []
        for mr in range(inner_h):
            for mc in range(inner_w):
                if _state_name(mr, mc, inner_h, inner_w) != desired_state:
                    continue
                if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc, _, _ in marker_points):
                    continue
                top = int(mr - ar)
                left = int(mc - ac)
                if top < 0 or left < 0 or top + ph > inner_h or left + pw > inner_w:
                    continue
                if bool((occupied[top : top + ph, left : left + pw] & mask).any()):
                    continue
                candidates.append((int(mr), int(mc), int(top), int(left)))

        if not candidates:
            return None

        mr, mc, top, left = random.choice(candidates)
        paste(inner, patch, int(top), int(left), mask=mask, overwrite=True, clip=False)
        occupied[top : top + ph, left : left + pw] |= mask
        marker_points.append((int(mr), int(mc), int(type_id), desired_state))

    return inner, marker_points


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    """Place sprites into the source interior, returning stamped_inner or None."""
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
        choices = [c for c in body_pool if c not in anchors and c not in {src_bg, dst_bg, dst_border}]
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
    for mr, mc, type_id, _state in marker_points:
        dest_rect[1 + mr, 1 + mc] = sprite_types[type_id]["anchor"]

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 D4 input after %d attempts" % MAX_GEN_ATTEMPTS)
