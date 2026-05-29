# Variant of a251c730
# Dimension: D15 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码——精灵和标记的对应关系直接由颜色相等表达，不需要额外图例解读... → 精灵内嵌方向指示

from __future__ import annotations

# Design analysis for the D15 variant
# -----------------------------------
# variant_spec:
#   core_change:
#     old: marker color selected a sprite, and the sprite was stamped exactly as seen in the source frame
#     new: marker color still selects the sprite type, but the source sprite's anchor corner now encodes a
#       transform that must be applied before stamping:
#       top-left -> identity
#       top-right -> horizontal flip
#       bottom-left -> vertical flip
#       bottom-right -> rotate 180 degrees
#   concrete_pixels:
#     - The source rectangle still contains 2-4 disconnected multicolor sprites.
#     - Each sprite still has one singleton anchor color and a larger body color.
#     - The anchor is forced to lie on one bbox corner, never in the middle.
#     - The destination rectangle still shows only singleton markers in the input.
#     - The output is still the destination rectangle with full sprites stamped at marker positions.
#   behavior_trace:
#     example:
#       source_sprite: canonical output shape is a 3x3 asymmetric motif with anchor at its upper-left corner.
#       if the source shows the horizontally flipped version, its anchor moves to the upper-right corner.
#       old_logic: stamp the flipped source directly, so the output keeps the flipped orientation.
#       new_logic: detect "anchor at upper-right", flip horizontally first, then align the transformed anchor
#         to the marker; the output becomes the canonical unflipped motif.
#     difference:
#       the matching key is still color equality, but orientation is no longer read from color or marker
#       position. It is encoded locally inside the source sprite by anchor corner placement.
#   complexity_calibration:
#     - Sprite count, rectangle finding, wallpaper, and color-matching remain unchanged.
#     - To make D15 visible, source sprites are generated from asymmetric canonical motifs, so flips/rotation
#       are visually different.
#     - Every generated puzzle contains at least one non-identity direction, preventing degeneration back to
#       the original task.
#     - When two sprite types are used, their directions are sampled without replacement so a single example
#       often shows more than one branch.
#   implicit_assumptions_review:
#     - Original template extraction stored the raw source patch. That is no longer valid because stamping now
#       depends on anchor corner. The solver must normalize each template before saving it.
#     - Original anchor detection had a color-id tie break. Here the generator guarantees a unique singleton
#       anchor, so the solver can require a unique least-frequent color instead of relying on color ordering.
#     - The solver now assumes the anchor is on one of the four bbox corners. The generator enforces this by
#       building every source sprite from a canonical top-left-anchor motif and applying only self-inverse
#       transforms (horizontal flip, vertical flip, 180-degree rotation).
#   placement_legality:
#     - Destination markers are still placed output-first by reserving the full stamped sprite footprint, so
#       markers always have enough margin for the transformed stamp.
#     - Source sprites are placed with the original padding rules, using the transformed source view.
#     - Source and destination rectangles remain non-overlapping and keep the original size constraints.
#   visual_observability:
#     - The only new symbol system is anchor corner position; no invisible ordering of color ids is used.
#     - Canonical motifs are asymmetric and span their full bbox, so each corner code produces a visibly
#       different source sprite but a clean, stable output orientation.
#   visual_tidiness:
#     - The destination input remains sparse single-pixel markers.
#     - Output overlays do not overlap because placement still reserves full footprints before markers are
#       reduced to singletons.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


CORNER_TO_TRANSFORM = {
    "top_left": "identity",
    "top_right": "hflip",
    "bottom_left": "vflip",
    "bottom_right": "rot180",
}


def _apply_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return arr.copy()
    if mode == "hflip":
        return np.fliplr(arr).copy()
    if mode == "vflip":
        return np.flipud(arr).copy()
    if mode == "rot180":
        return np.rot90(arr, 2).copy()
    raise ValueError(f"unknown transform mode: {mode}")


def _normalize_template(
    patch: np.ndarray, mask: np.ndarray, anchor_col: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str] | None:
    """Return the normalized patch/mask whose anchor-corner code has been applied."""
    anchors = np.argwhere((patch == int(anchor_col)) & mask)
    if anchors.shape[0] != 1:
        return None
    ar, ac = map(int, anchors[0])
    h, w = map(int, mask.shape)

    if (ar, ac) == (0, 0):
        mode = "identity"
    elif (ar, ac) == (0, w - 1):
        mode = "hflip"
    elif (ar, ac) == (h - 1, 0):
        mode = "vflip"
    elif (ar, ac) == (h - 1, w - 1):
        mode = "rot180"
    else:
        return None

    norm_patch = _apply_transform(patch, mode)
    norm_mask = _apply_transform(mask.astype(int), mode).astype(bool)
    norm_anchors = np.argwhere((norm_patch == int(anchor_col)) & norm_mask)
    if norm_anchors.shape[0] != 1:
        return None
    nar, nac = map(int, norm_anchors[0])
    return norm_patch, norm_mask, (int(nar), int(nac)), mode


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via color-matched singleton markers.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper: the source holds multi-cell sprites, the destination
      holds only singleton markers.
    - Each source sprite has one singleton anchor color, and that anchor lies on one corner of the sprite bbox.

    Rule:
    - Match each destination marker to the source sprite with the same anchor color.
    - Read the anchor corner as a direction code, normalize the sprite with the corresponding flip/rotation,
      then translate the transformed anchor onto the marker position.
    - Output is the destination rectangle alone, cleared to its background before stamping.

    Key insight: The rare anchor color still identifies which sprite to use, but its corner position inside the
    sprite now serves as a local legend telling you how to orient that sprite before imprinting it.
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
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if not cols:
            continue
        min_count = min(cols.values())
        anchor_candidates = [int(col) for col, cnt in cols.items() if int(cnt) == int(min_count)]
        if len(anchor_candidates) != 1:
            continue
        anchor_col = int(anchor_candidates[0])
        normalized = _normalize_template(comp["patch"], comp["mask"], anchor_col)
        if normalized is None:
            continue
        norm_patch, norm_mask, anchor_rc, _mode = normalized
        if int(anchor_col) not in templates or int(norm_mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (norm_patch.copy(), norm_mask.copy(), anchor_rc)

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mr = int(dr + 1 + int(rr))
        mc = int(dc + 1 + int(cc))
        mcol = int(g[mr, mc])
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
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 300

# Canonical output motifs: anchor is always the singleton at the top-left.
CANONICAL_SHAPES = [
    ("AB.", "B..", "BBB"),
    ("AB.", "BBB", "..B"),
    ("ABB", "B.B", "..B"),
    ("AB.", "B.B", "BBB"),
    ("ABB", "..B", ".BB"),
    ("A..", "BBB", "B.B"),
]

DIRECTION_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")


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


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    """
    Allocate wallpaper, border, anchor, and body colors.

    Returns (wallpaper, src_border, dst_border, anchors, body_pool, forced_body)
    or None if color constraints can't be satisfied.
    """
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + n_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    if not remaining:
        return None
    src_border = remaining.pop()

    anchor_pool = [c for c in remaining if c != 0]
    if len(anchor_pool) < n_types:
        return None
    anchors = random.sample(anchor_pool, n_types)
    remaining = [c for c in remaining if c not in anchors]

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

    body_pool = [c for c in wallpaper if c != 0]
    forced_body: int | None = None
    if body_extra:
        nonzero_remaining = [c for c in remaining if c != 0]
        if not nonzero_remaining:
            return None
        forced_body = random.choice(nonzero_remaining)
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _choose_directions(n_types: int) -> list[str]:
    if n_types == 1:
        return [random.choice(list(DIRECTION_ORDER[1:]))]
    return random.sample(list(DIRECTION_ORDER), n_types)


def _render_stamp_patch(sprite: dict, bg: int) -> np.ndarray:
    return _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], bg)


def _render_source_patch(sprite: dict, bg: int) -> np.ndarray:
    base = _render_stamp_patch(sprite, bg)
    return _apply_transform(base, CORNER_TO_TRANSFORM[sprite["direction"]])


def _anchor_rc(patch: np.ndarray, anchor: int) -> tuple[int, int]:
    coords = np.argwhere(patch == int(anchor))
    if coords.shape[0] != 1:
        raise ValueError("sprite anchor must appear exactly once")
    r, c = map(int, coords[0])
    return int(r), int(c)


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place canonical stamped sprites into the dest interior, returning (stamped_inner, marker_points)."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_stamp_patch(sprite, dst_bg)
        ar, ac = _anchor_rc(patch, sprite["anchor"])
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr, mc = r0 + ar, c0 + ac
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
    """Place transformed source sprites into the source interior, returning stamped_inner or None."""
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_source_patch(sprite, src_bg)
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

    sprite_specs = random.sample(CANONICAL_SHAPES, n_types)
    directions = _choose_directions(n_types)

    bodies: list[int] = []
    for i in range(n_types):
        choices = [
            c
            for c in body_pool
            if c not in anchors and c not in {src_bg, dst_bg, src_border, dst_border}
        ]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(forced_body)
        else:
            bodies.append(random.choice(choices))

    sprite_types = [
        {
            "shape_rows": tuple(shape_rows),
            "direction": direction,
            "body": body,
            "anchor": anchor,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for shape_rows, direction, body, anchor in zip(sprite_specs, directions, bodies, anchors)
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
    for mr, mc, type_id in marker_points:
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
    raise RuntimeError("failed to generate a251c730 input after %d attempts" % MAX_GEN_ATTEMPTS)
