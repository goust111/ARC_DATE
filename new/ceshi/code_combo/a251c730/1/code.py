# Combo variant of a251c730 (index 1)
# combo_spec: D1/1+D8/0
# Name: 主体色匹配+几何中心锚点
#
# Design analysis
# ---------------
# analysis:
#   concretization_and_trace:
#     - The fused rule splits the original "anchor" into two independent jobs.
#       Matching now uses the sprite's dominant body color, while placement uses
#       the geometric center of the sprite bbox.
#     - Concrete type U: shape ("AB.", "BBB"), body=7, rare=2.
#       Original task: marker color 2, align local cell (0,0).
#       Fused task: marker color 7, align local center cell (0,1).
#       Same marker location now produces a horizontally shifted stamp, and the
#       cue color is no longer the rare pixel.
#     - Concrete type V: shape (".B.", "BAB", ".B."), body=4, rare=9.
#       Original task: marker color 9, align local rare pixel (1,1).
#       Fused task: marker color 4, but placement still aligns local center
#       (1,1), because the center happens to be the rare pixel.
#       Here the geometry matches the original anchor location, but the cue color
#       flips from rare to dominant.
#     - Using both U and V in the same sample makes the fusion observable:
#       one type exhibits the geometry change, the other exhibits the color-key
#       change, so neither D1-only nor D8-only explains the whole grid.
#   complexity_calibration:
#     - Generation is tightened to exactly 2 sprite types.
#     - One type is forced from a "center is body color" pool and one from a
#       "center is rare color" pool.
#     - Both types must appear at least once in source and destination.
#       This prevents collapse into a single-type sample or into a sample that
#       is behaviorally identical to a single reference variant.
#     - Body colors are unique across the 2 types, so destination marker lookup
#       is unambiguous and repeated evidence of the body-color rule appears.
#   implicit_assumption_review:
#     - Rectangle detection is unchanged; it still depends only on bordered
#       frames with wallpaper backgrounds.
#     - Source components are still assumed to be 4-connected non-background
#       sprites. Generator preserves that.
#     - Original template extraction assumed the least-frequent color was both
#       key and anchor. That no longer holds. Solver must infer:
#         1. dominant color = lookup key
#         2. bbox center = placement anchor
#       and these two facts are intentionally decoupled.
#     - The center cell must belong to the sprite mask. Generator enforces that
#       by only choosing shapes whose geometric center is occupied.
#     - The marker loop assumes each cue color identifies one template. Generator
#       guarantees distinct dominant body colors.
#   placement_legality:
#     - Output-first construction still places full sprite patches inside the
#       destination interior, then records the marker at the center cell of that
#       placement. Replaying the same center offset in `main()` therefore fits.
#     - Source sprites are placed with border/padding so connected-component
#       extraction remains stable.
#   visual_observability:
#     - Markers always reuse a color that appears multiple times inside the
#       matched source sprite, which supports the "majority color is the key"
#       inference.
#     - Because one forced type has center=body and the other has center=rare,
#       examples jointly show both parts of the fusion: color-based lookup and
#       geometry-based placement are visibly independent.
#   visual_cleanliness:
#     - The destination input still contains only singleton markers on a blank
#       interior, so cue reading stays clean.
#     - Full output sprites do not overlap because marker positions are derived
#       from legal full-stamp placements.
#   inferability_guard:
#     - No invisible ordering on numeric color ids is used.
#     - The human-visible evidence is: repeated body color in the source chooses
#       the marker match, while the landed sprite position is explained by bbox
#       center rather than by the rare pixel for at least one type per sample.
#   generator_guarantees_for_main:
#     - Exactly one dominant color and one rare color per sprite.
#     - Exactly two sprite types: one center-body, one center-rare.
#     - Both center cells are occupied.
#     - Both types appear in source and among destination markers.
#     - Dominant body colors are distinct and differ from src/dst backgrounds.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _dominant_and_rare(comp: dict) -> tuple[int, int] | None:
    """Return (body_color, rare_color) when the component has unique max/min counts."""
    items = [(int(col), int(cnt)) for col, cnt in comp["colors"].items()]
    if not items:
        return None
    max_count = max(cnt for _, cnt in items)
    min_count = min(cnt for _, cnt in items)
    if max_count == min_count:
        return None
    body_cols = [col for col, cnt in items if cnt == max_count]
    rare_cols = [col for col, cnt in items if cnt == min_count]
    if len(body_cols) != 1 or len(rare_cols) != 1:
        return None
    return int(body_cols[0]), int(rare_cols[0])


def _center_rc_from_shape(h: int, w: int) -> tuple[int, int]:
    """Return floor(center) in bbox-local coordinates."""
    return (int((h - 1) // 2), int((w - 1) // 2))


def _center_rc_from_patch(patch: np.ndarray) -> tuple[int, int]:
    """Return floor(center) for a cropped component patch."""
    h, w = map(int, patch.shape)
    return _center_rc_from_shape(h, w)


def _shape_center_char(shape_rows: tuple[str, ...]) -> str:
    """Return the template character at geometric center."""
    cr, cc = _center_rc_from_shape(len(shape_rows), len(shape_rows[0]))
    return str(shape_rows[cr][cc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by matching source sprites on body color and aligning on bbox center.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a source frame with small 2-color sprites and a destination frame with isolated one-pixel markers.
    - Each sprite still has a repeated body color and a singleton rare color, but the operational lookup color and the operational anchor are no longer the same feature.

    Rule:
    - For each source sprite, use its dominant color as the template key.
    - For each destination marker, pick the source sprite with the same dominant color, then shift that sprite so the geometric center of its bounding box lands on the marker cell.
    - Return the destination rectangle alone, rebuilt from its border/background before stamping the matched sprites.

    Key insight: color matching uses the sprite's majority color, but placement uses pure geometry, so the cue color and the alignment cell must be inferred separately.
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
        colors = _dominant_and_rare(comp)
        if colors is None:
            continue
        body_col, _rare_col = colors
        patch = comp["patch"]
        mask = comp["mask"]
        ar, ac = _center_rc_from_patch(patch)
        if not bool(mask[int(ar), int(ac)]):
            continue
        if int(body_col) not in templates or int(mask.sum()) > int(templates[int(body_col)][1].sum()):
            templates[int(body_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

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


GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (2,)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 300

CENTER_BODY_SHAPES = [
    ("AB.", "BBB"),
    ("ABB", "BBB"),
    (".A.", "BBB", ".B."),
    ("AB.", "BBB", ".BB"),
    ("ABB", "BBB", "..B"),
    (".AB", "BBB", "BB."),
    ("B.B", "BBB", ".A."),
    ("AB.", "BBB", ".B."),
]

CENTER_RARE_SHAPES = [
    (".B.", "BAB", ".B."),
    ("BBB", "BAB"),
    ("BAB", "BBB"),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, rare: int, bg: int) -> np.ndarray:
    """Render a sprite template. 'A' is the singleton rare pixel, 'B' the dominant body."""
    char_map = {"B": body, "A": rare}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    """Create a 30x30 striped wallpaper."""
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    """Create a bordered rectangle with uniform interior background."""
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
    """Pick rectangle sizes that fit the 30x30 canvas."""
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
    """Compute non-overlapping rectangle origins."""
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
    """Allocate wallpaper, border, rare, and body candidate colors."""
    wallpaper_n = random.choice(list(WALLPAPER_COLORS))
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
    """Place output sprites first, then record center-aligned marker positions."""
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
            mr = int(r0 + sprite["center_rc"][0])
            mc = int(c0 + sprite["center_rc"][1])
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
    """Place source sprites with padding so components stay separated."""
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


def _build_sprite_types(src_bg: int, dst_bg: int, rare_colors: list[int], body_pool: list[int], forced_body: int | None) -> list[dict] | None:
    """Construct exactly one center-body type and one center-rare type."""
    shape_specs = [
        random.choice(CENTER_BODY_SHAPES),
        random.choice(CENTER_RARE_SHAPES),
    ]

    body_choices = [c for c in body_pool if c not in rare_colors and c != src_bg and c != dst_bg]
    unique_body_choices = list(dict.fromkeys(body_choices))
    if len(unique_body_choices) < 2:
        return None

    if forced_body is not None and forced_body in unique_body_choices:
        other_choices = [c for c in unique_body_choices if c != forced_body]
        if not other_choices:
            return None
        bodies = [forced_body, random.choice(other_choices)]
        random.shuffle(bodies)
    else:
        bodies = random.sample(unique_body_choices, 2)

    sprite_types = []
    center_body_count = 0
    center_rare_count = 0
    for shape_rows, body, rare in zip(shape_specs, bodies, rare_colors):
        center_char = _shape_center_char(shape_rows)
        center_rc = _center_rc_from_shape(len(shape_rows), len(shape_rows[0]))
        if center_char == ".":
            return None
        center_color = int(body if center_char == "B" else rare)
        sprite = {
            "shape_rows": shape_rows,
            "center_rc": center_rc,
            "body": int(body),
            "rare": int(rare),
            "center_color": int(center_color),
            "center_is_body": bool(center_char == "B"),
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        center_body_count += int(sprite["center_is_body"])
        center_rare_count += int(not sprite["center_is_body"])
        sprite_types.append(sprite)

    if center_body_count != 1 or center_rare_count != 1:
        return None
    if len({int(sprite["body"]) for sprite in sprite_types}) != 2:
        return None
    return sprite_types


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid fused-combo input grid."""
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

    sprite_types = _build_sprite_types(src_bg, dst_bg, rare_colors, body_pool, forced_body)
    if sprite_types is None:
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
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[type_id]["body"])

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
    raise RuntimeError("failed to generate a251c730 combo input after %d attempts" % MAX_GEN_ATTEMPTS)
