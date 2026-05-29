# Combo variant of a251c730 (index 2)
# combo_spec: D1/1+D8/1
# Name: 主体色匹配+左上角锚点
#
# Design analysis
# ===============
# fusion_rule:
#   - Matching key changes from the original singleton anchor color to the sprite's dominant body color.
#   - Placement anchor changes from the original singleton anchor pixel to the top-left occupied pixel of the
#     sprite bbox, which the generator guarantees is always patch[0, 0].
#   - Therefore the same source sprite now exposes two independent cues:
#       1. color cue for lookup = repeated body color
#       2. geometric cue for placement = bbox top-left corner
#
# concrete_behavior_trace:
#   - Suppose a source sprite patch is:
#       C C B
#       B A B
#       B B .
#     with body color 7, corner color 3, and rare accent color 2.
#   - Original task:
#       marker color = 2, and the patch is shifted so the internal A lands on the marker.
#   - This combo:
#       marker color = 7, but the patch is shifted so its top-left C lands on the marker.
#   - Visible consequence:
#       the marker matches the common color seen across the sprite, yet after stamping the cue location becomes
#       a differently colored corner pixel rather than the rare accent position.
#
# complexity_calibration:
#   - The generator fixes n_types = 2 so the "majority color is the key" rule is supported by repeated evidence
#     instead of collapsing into a one-type trivial copy task.
#   - Body colors are unique across types and disjoint from all accent colors, so body-match and original
#     anchor-match cannot accidentally agree.
#   - Shapes are restricted to templates whose bbox top-left is occupied by a dedicated corner color C, while a
#     different singleton accent A remains elsewhere. This prevents collapse to either single variant.
#
# implicit_assumption_audit_of_original_main:
#   - Original `anchor_col = min(cols.items(), key=(count, color))[0]` assumed the rarest color is both key and
#     anchor. That fails twice here: matching uses the dominant color, and placement no longer depends on color.
#   - Original `anchors = np.argwhere((patch == anchor_col) & mask)` assumed anchor location must be recovered by
#     searching for a special color. Here the anchor is geometric and fixed at (0, 0), so that search is invalid.
#   - Original `templates` keyed by anchor color would miss every marker, because destination markers now carry
#     body colors. The solver must key by dominant color instead.
#   - Original marker stamping offset used `(mr-dr)-ar, (mc-dc)-ac`. With a fixed top-left anchor, the correct
#     offset is simply `(mr-dr, mc-dc)` into the cropped destination rectangle.
#
# placement_legality:
#   - Destination examples are still built output-first by first placing full patches into the destination interior.
#     Marker positions are then read back from each placed patch's top-left corner, so every cue has legal room.
#   - Source sprites are placed with padding to avoid component mergers.
#   - Rectangle sizes and non-overlap logic remain within the original 30x30 spec envelope.
#
# visual_observability_and_cleanliness:
#   - Each destination input contains only singleton markers, keeping the cue layer sparse.
#   - Each source sprite still contains a singleton accent pixel, but that pixel is visibly not where alignment
#     happens: markers sit at the stamped sprite's top-left corner.
#   - Because marker colors equal the repeated body colors, humans can infer lookup by the common color while also
#     inferring that placement follows the corner, not the rare accent.
#
# new_main_assumptions_and_generator_guarantees:
#   - Every source component has exactly one dominant body color, a non-dominant corner color present at patch[0, 0],
#     and a singleton accent color elsewhere.
#   - Every source component has mask[0, 0] == True and patch[0, 0] different from the dominant body color.
#   - Different sprite types have distinct dominant colors, and destination markers use only those colors.
#   - Both sprite types appear at least once in source and at least once among destination markers.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _dominant_color(comp: dict) -> int | None:
    """Return the unique dominant color of a component, or None if not well-defined."""
    items = [(int(col), int(cnt)) for col, cnt in comp["colors"].items()]
    if not items:
        return None
    max_count = max(cnt for _, cnt in items)
    dominant = [col for col, cnt in items if cnt == max_count]
    if len(dominant) != 1:
        return None
    return int(dominant[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by matching body-colored markers to corner-anchored sprites.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper: the source holds small multicolor sprites and the destination holds singleton cue pixels.
    - In each source sprite, the repeated body color is the lookup cue, while the bbox top-left pixel is the placement anchor even when that corner uses another color.

    Rule:
    - Extract each source sprite as a connected component, identify its dominant body color, and store the full patch under that color.
    - For every destination singleton marker, choose the sprite whose dominant body color matches the marker color.
    - Paste that sprite into the cleared destination rectangle so the sprite's bbox top-left corner lands exactly on the marker.

    Key insight: color matching and geometric alignment come from different clues: the common color selects the sprite, but the fixed top-left corner determines where it lands.
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
                    if (
                        int(g[int(r0), int(c1)]) != int(bc)
                        or int(g[int(r1), int(c0)]) != int(bc)
                        or int(g[int(r1), int(c1)]) != int(bc)
                    ):
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
        if max_sz == 1 and comps and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for comp in src_comps:
        patch = comp["patch"]
        mask = comp["mask"]
        if not bool(mask[0, 0]):
            continue
        body_col = _dominant_color(comp)
        if body_col is None or int(patch[0, 0]) == int(body_col):
            continue
        if int(body_col) not in templates or int(mask.sum()) > int(templates[int(body_col)][1].sum()):
            templates[int(body_col)] = (patch.copy(), mask.copy())

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
        patch, mask = templates[int(mcol)]
        top = int(mr - dr)
        left = int(mc - dc)
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4,)
TOTAL_COLORS_TARGET = (9, 10)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (2,)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 300

# All shapes have an occupied top-left corner colored by C, not by the dominant body color B.
# A is a singleton accent kept away from the top-left so corner-anchor and rare-accent anchor remain distinct.
SPRITE_SHAPES = [
    ("CCB", "BAB", "BB."),
    ("CCB", "BBB", ".A."),
    ("CCB", ".AB", ".BB"),
    ("CCB", "B.A", "BB."),
    ("CCB", "ABB", "B.."),
    ("CCB", "B..", "ABB"),
    ("CCB", ".BA", ".BB"),
    ("CCB", "B.B", ".A."),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, corner: int, accent: int, bg: int) -> np.ndarray:
    """Render a sprite patch where B is dominant, C is the top-left anchor color, and A is a singleton accent."""
    char_map = {"B": body, "C": corner, "A": accent}
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, int, int, list[int], int | None] | None:
    """Allocate wallpaper, borders, shared corner/accent colors, and distinct body-key colors."""
    wallpaper_n = WALLPAPER_COLORS[0]
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 3 + n_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    nonzero_remaining = [c for c in remaining if c != 0]
    if len(nonzero_remaining) < 2:
        return None
    corner = nonzero_remaining.pop()
    accent = nonzero_remaining.pop()
    remaining = [c for c in remaining if c not in {corner, accent}]

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

    body_pool = [c for c in wallpaper if c not in {corner, accent}]
    forced_body: int | None = None
    if body_extra:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, corner, accent, body_pool, forced_body


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place output sprites in the dest interior; record each sprite's top-left marker location."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["corner"], sprite["accent"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            if any(abs(r0 - pr) + abs(c0 - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((int(r0), int(c0), int(type_id)))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    """Place sprites into the source interior, returning stamped_inner or None."""
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["corner"], sprite["accent"], src_bg)
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
    wallpaper, src_border, dst_border, corner, accent, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    body_choices = [c for c in body_pool if c not in {corner, accent, src_bg, dst_bg, 0}]
    unique_body_choices = list(dict.fromkeys(body_choices))
    if len(unique_body_choices) < n_types:
        return None
    if forced_body is not None and forced_body in unique_body_choices:
        remaining_bodies = [c for c in unique_body_choices if c != forced_body]
        if len(remaining_bodies) < n_types - 1:
            return None
        bodies = [forced_body] + random.sample(remaining_bodies, n_types - 1)
    else:
        bodies = random.sample(unique_body_choices, n_types)
    random.shuffle(bodies)

    if len(set(bodies)) != n_types or set(bodies) & {corner, accent}:
        return None

    sprite_types = [
        {
            "shape_rows": shape_rows,
            "body": body,
            "corner": corner,
            "accent": accent,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for shape_rows, body in zip(sprite_specs, bodies)
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
