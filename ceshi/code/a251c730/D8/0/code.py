# Variant of a251c730
# Dimension: D8 — 精灵锚点定义 (meta)
# Change: 锚点是精灵中出现次数最少的颜色的像素（单一像素），作为匹配键和定位参考点... → 锚点为几何中心像素

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D8 variant
# ----------------------------------
# analysis:
#   source_task: none
#   concrete_behavior:
#     - The input still contains two bordered rectangles on striped wallpaper.
#     - The source rectangle still holds 2-color sprites. Each sprite still has a
#       singleton accent color, but that color is no longer the operational anchor.
#     - The destination rectangle still contains only singleton markers in the input.
#       In the variant, each marker color equals the color of the sprite's geometric
#       center cell, and the full sprite is stamped so that this center cell lands on
#       the marker.
#     - Geometric center is computed from the sprite bbox, not from occupied-mask
#       mass. In local bbox coordinates this is:
#         center_rc = (floor((h - 1) / 2), floor((w - 1) / 2))
#       so a 2x3 sprite uses the top-middle cell and a 3x3 sprite uses the middle.
#   behavior_trace:
#     - Old rule example: shape ('AB.', 'BBB') uses the singleton 'A' at (0,0).
#       A destination marker with color A means "place the sprite so (0,0) sits here".
#     - New rule on the same shape: the bbox center is (0,1), whose color is 'B'.
#       A destination marker with color B now means "place the sprite so (0,1) sits here".
#     - Difference: both the lookup color and the alignment offset move from the rare
#       pixel to the bbox center pixel, so the same marker location produces a visibly
#       shifted stamp and usually uses a different marker color.
#   complexity_calibration:
#     - If center colors were allowed to collide across sprite types, marker matching
#       would become ambiguous. The generator therefore keeps center colors unique per
#       type.
#     - If some sprites had center == old singleton anchor, the variant could regress
#       to the original behavior. The generator therefore only uses shapes whose center
#       cell is occupied and whose singleton accent is off-center.
#   implicit_assumption_audit:
#     - Rectangle detection does not depend on the anchor rule and stays unchanged.
#     - Source component extraction still assumes sprites are connected 4-neighbor
#       components inside the source frame; generator preserves that.
#     - Template extraction previously assumed the least-frequent color existed exactly
#       once. That assumption is removed. The new solver instead assumes the bbox
#       center cell belongs to the component and has a usable color. The generator
#       enforces this by choosing only center-filled shapes.
#     - Destination scanning still assumes singleton colored markers on a blank
#       interior. Generator preserves that and colors those markers with center colors.
#   placement_legality:
#     - Output-first generation still places the full sprite patch first, then records
#       the marker at the center cell of that placement. Because the patch is placed
#       wholly inside the destination interior, replaying the same offset in `main()`
#       is guaranteed to fit.
#     - Source sprites remain padded from borders and from each other so component
#       extraction is stable.
#   observability_and_cleanliness:
#     - Most selected shapes are asymmetric and put the singleton accent away from the
#       center. Markers therefore use a body color rather than the rare accent color,
#       making the D8 change visible in a few examples.
#     - The destination input stays visually clean: only singleton markers appear in
#       the blank destination frame, and the output is the same frame with full stamps.


def _center_rc_from_shape(h: int, w: int) -> tuple[int, int]:
    """Return floor(center) in bbox-local coordinates."""
    return (int((h - 1) // 2), int((w - 1) // 2))


def _center_rc_from_patch(patch: np.ndarray) -> tuple[int, int]:
    """Return floor(center) for a cropped component patch."""
    h, w = map(int, patch.shape)
    return _center_rc_from_shape(h, w)


def _shape_singleton_rc(shape_rows: tuple[str, ...], ch: str) -> tuple[int, int]:
    """Locate the unique template character used for diagnostics/generation."""
    hits = [(r, c) for r, row in enumerate(shape_rows) for c, val in enumerate(row) if val == ch]
    if len(hits) != 1:
        raise ValueError(f"shape must contain exactly one {ch!r}")
    return hits[0]


def _shape_center_char(shape_rows: tuple[str, ...]) -> str:
    """Read the character at the geometric center cell of a template."""
    cr, cc = _center_rc_from_shape(len(shape_rows), len(shape_rows[0]))
    return shape_rows[cr][cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by stamping source sprites at center-colored markers.

    Input structure:
    - Two bordered rectangles sit on top of a striped wallpaper: a source frame with
      multi-cell sprites and a destination frame with singleton markers.
    - Each source sprite is keyed by the color found at the geometric center of its
      own bounding box, not by its rare accent pixel.

    Rule:
    - Find the source sprite whose center-cell color matches each destination marker.
    - Translate that sprite so the sprite's bbox-center cell lands on the marker.
    - Return the destination rectangle alone, cleared to its background before
      stamping the matched sprites.

    Key insight: the lookup key and placement anchor are both the sprite bbox center,
    so the singleton accent color becomes decorative rather than operational.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # --- find all uniform-border rectangles (brute force, small grid) ---
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

    # --- build templates keyed by center-cell color ---
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        patch = comp["patch"]
        mask = comp["mask"]
        ar, ac = _center_rc_from_patch(patch)
        if not bool(mask[int(ar), int(ac)]):
            continue
        anchor_col = int(patch[int(ar), int(ac)])
        if int(anchor_col) == 0:
            continue
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

# Every shape below has:
# - exactly one 'A' singleton accent
# - an occupied geometric center cell
# - center cell different from the singleton anchor location
SPRITE_SHAPES = [
    ("AB.", "BBB"),
    ("ABB", "BBB"),
    (".A.", "BBB", ".B."),
    ("AB.", "BBB", ".BB"),
    ("ABB", "BBB", "..B"),
    (".AB", "BBB", "BB."),
    ("B.B", "BBB", ".A."),
    ("BBB", "BAB"),
    ("AB.", "BBB", ".B."),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, rare: int, bg: int) -> np.ndarray:
    """Render a sprite template. 'A' is the singleton accent, 'B' the body."""
    char_map = {"B": body, "A": rare}
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
    """Allocate wallpaper, border, singleton-accent, and body colors."""
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
    rare_colors = [remaining.pop() for _ in range(n_types)]

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

    return wallpaper, src_border, dst_border, rare_colors, body_pool, forced_body


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place full sprite stamps into the output interior and record center-marker positions."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["rare"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr, mc = r0 + sprite["center_rc"][0], c0 + sprite["center_rc"][1]
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
    """Place sprites into the source interior, returning stamped_inner or None."""
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["rare"], src_bg)
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
    wallpaper, src_border, dst_border, rare_colors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)

    body_choices = [c for c in body_pool if c not in rare_colors and c != src_bg and c != dst_bg]
    if len(set(body_choices)) < n_types:
        return None
    if forced_body is not None and forced_body in body_choices:
        other_choices = [c for c in body_choices if c != forced_body]
        if n_types == 1:
            bodies = [forced_body]
        else:
            if len(set(other_choices)) < n_types - 1:
                return None
            bodies = [forced_body] + random.sample(list(dict.fromkeys(other_choices)), n_types - 1)
            random.shuffle(bodies)
    else:
        bodies = random.sample(list(dict.fromkeys(body_choices)), n_types)

    sprite_types = []
    for shape_rows, body, rare in zip(sprite_specs, bodies, rare_colors):
        center_rc = _center_rc_from_shape(len(shape_rows), len(shape_rows[0]))
        rare_rc = _shape_singleton_rc(shape_rows, "A")
        center_char = _shape_center_char(shape_rows)
        if center_char not in {"A", "B"}:
            return None
        center_color = int(rare if center_char == "A" else body)
        if center_rc == rare_rc or center_color == int(rare):
            return None
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "center_rc": center_rc,
                "rare_rc": rare_rc,
                "body": int(body),
                "rare": int(rare),
                "center_color": center_color,
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
            }
        )

    if len({int(sprite["center_color"]) for sprite in sprite_types}) != n_types:
        return None

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
        dest_rect[1 + mr, 1 + mc] = sprite_types[type_id]["center_color"]

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
    raise RuntimeError("failed to generate a251c730 D8 input after %d attempts" % MAX_GEN_ATTEMPTS)
