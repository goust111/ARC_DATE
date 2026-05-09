# Combo variant of a251c730 (index 4)
# combo_spec: D1/1+D7/0
# Name: 主体色匹配+一对多印刻

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task:
#      source sprites are keyed by their singleton anchor color, and the
#      destination shows exactly one marker per type using that same color.
#      Example: sprite T has body=7 and anchor=2, so a destination marker 2
#      selects T and T is stamped once with its 2-pixel aligned to the marker.
#    - Combo behavior:
#      the destination marker now uses the sprite's majority body color instead
#      of its anchor color, and one body color appears on multiple markers.
#      Example: sprite T still has body=7 and anchor=2, but destination may
#      contain two singleton markers colored 7. The solver must infer that 7 is
#      the lookup key, while 2 remains the alignment cell inside the sprite, and
#      stamp the same T template at both 7-marked positions.
#    - Visible difference:
#      compared with the original task, the repeated color in the source sprite
#      now matches the target cue, and the output shows repeated copies of one
#      asymmetric sprite shape.
#
# b) Complexity calibration
#    - The generator is tightened to exactly 2 sprite types and exactly 3 target
#      markers. One type repeats twice and the other appears once.
#    - This keeps the rule small enough to learn while forcing both parts of the
#      combo to appear in every sample:
#        1. body-color matching is needed because cue colors equal body colors;
#        2. one-to-many application is needed because one cue color repeats.
#    - Body colors are unique across the 2 types, so the lookup key is never
#      ambiguous.
#
# c) Hidden assumptions review
#    - The original solver keyed templates by least-frequent color. That no
#      longer works; the combo solver must instead key by the unique majority
#      color while still locating the unique minority color for alignment.
#    - The original solver happened to iterate every marker cell already, but it
#      treated each in isolation. Here the intended semantics are explicit:
#      gather all marker positions by body color, then stamp one template at all
#      of those positions.
#    - Any tie in color frequencies inside a sprite would break the inference of
#      "body" vs "anchor", so the generator guarantees each sprite patch has one
#      unique most-frequent color and one unique least-frequent color.
#    - Because the destination markers no longer use anchor colors, the combo
#      generator guarantees body colors are distinct from all anchor colors, so
#      anchor-based matching cannot accidentally succeed.
#
# d) Placement legality
#    - Output-first generation is preserved. Full sprite patches are first
#      placed into a clean destination interior with random_free_location, so
#      every future stamp is in-bounds and non-overlapping.
#    - The input destination rectangle keeps only the validated anchor
#      locations, recolored to the sprite body color.
#    - Source sprites are placed with edge padding and mutual spacing, keeping
#      connected components separable for template extraction.
#
# e) Visual observability
#    - Every sample contains two sprite types, two distinct body colors, and a
#      repeated body-color marker. A human can see that the destination cue
#      matches the large repeated color inside the source sprite, not the rare
#      anchor color.
#    - The repeated target cue produces two identical copies in the output,
#      making the one-to-many relation directly observable from a single pair.
#
# f) Visual cleanliness
#    - Output stamps never overlap because they are placed as full sprites before
#      being reduced to singleton markers in the input.
#    - Markers are kept non-adjacent, and sprite shapes remain small asymmetric
#      2x3 or 3x3 patterns, so repeated copies stay easy to parse.
#
# g) Rule inferability
#    - No invisible ordering on color ids is used. Matching depends only on the
#      visible majority-vs-minority frequency structure inside each sprite and
#      on repeated reuse of the same cue color.
#    - The palette remains close to the original task, avoiding excess color
#      noise that would hide the combo change.
#
# Main() assumptions guaranteed by generate_input()
#    - There are exactly two bordered rectangles with non-border interiors.
#    - The source rectangle contains exactly 2 sprite types, each with one
#      unique dominant body color and one unique rare anchor color.
#    - The destination rectangle contains exactly 3 singleton markers colored by
#      body color, with marker multiplicities {2, 1}.
#    - Body colors are unique across types and disjoint from anchor colors, so
#      body-based lookup is well-defined and differs from the original rule.


def _body_and_anchor_colors(comp: dict) -> tuple[int, int] | None:
    """Return (body_color, anchor_color) when counts have unique max/min colors."""
    items = [(int(col), int(cnt)) for col, cnt in comp["colors"].items()]
    if not items:
        return None
    max_count = max(cnt for _, cnt in items)
    min_count = min(cnt for _, cnt in items)
    if max_count == min_count:
        return None
    body_colors = [col for col, cnt in items if cnt == max_count]
    anchor_colors = [col for col, cnt in items if cnt == min_count]
    if len(body_colors) != 1 or len(anchor_colors) != 1:
        return None
    return int(body_colors[0]), int(anchor_colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via repeated body-colored markers.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper: the source rectangle
      contains multi-cell two-color sprites, while the destination rectangle
      contains isolated one-pixel markers.
    - Each source sprite has a repeated body color and a singleton anchor color,
      but the destination markers copy the body color and one body color may
      appear on multiple markers.

    Rule:
    - Extract one template per source sprite type, using the sprite's unique
      dominant color as the lookup key and its unique rare color as the anchor
      position inside the template.
    - Group destination markers by color, match each color to the source sprite
      with the same dominant color, and stamp that template at every marker in
      the group so the template anchor lands on the marker.
    - Rebuild and return only the destination rectangle after all repeated
      stamps have been applied.

    Key insight: The common body color selects the sprite type, while the rare
    anchor pixel still determines where every copied sprite is aligned.
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

        for coords in components_mask(inner != ibg, connectivity=4):
            if coords.size == 0:
                continue
            rr0, cc0, rr1, cc1 = bbox_coords(coords)
            mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
            mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
            patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
            patch[~mask] = 0
            vals = [int(inner[r, c]) for r, c in coords]
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(Counter(vals)),
                    "mask": mask,
                    "patch": patch,
                }
            )

        comps.sort(key=lambda d: -int(d["size"]))
        return ibg, comps

    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(H):
        for c0 in range(W):
            bc = int(g[r0, c0])
            for r1 in range(r0 + min_h - 1, H):
                for c1 in range(c0 + min_w - 1, W):
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
                    ibg = interior_bg((r0, c0, r1, c1))
                    if ibg == bc:
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    rects.append((int(area), (r0, c0, r1, c1), bc, ibg))

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

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        key_colors = _body_and_anchor_colors(comp)
        if key_colors is None:
            continue
        body_col, anchor_col = key_colors
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == anchor_col) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if body_col not in templates or int(mask.sum()) > int(templates[body_col][1].sum()):
            templates[body_col] = (patch.copy(), mask.copy(), (ar, ac))

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[dr + 1 : dr1, dc + 1 : dc1]
    marker_positions: dict[int, list[tuple[int, int]]] = {}
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        body_col = int(inner[int(rr), int(cc)])
        marker_positions.setdefault(body_col, []).append((int(rr) + 1, int(cc) + 1))

    for body_col, positions in marker_positions.items():
        if body_col not in templates:
            continue
        patch, mask, (ar, ac) = templates[body_col]
        for mr, mc in positions:
            top = int(mr - ar)
            left = int(mc - ac)
            paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

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
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1
OUTPUT_SPRITE_PADDING = 1

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
    wallpaper_n = random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + n_types)
    if extras_needed < 1 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]

    dst_border_extra = extras_needed == 2 and random.random() < 0.5
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
    if extras_needed >= 1:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _choose_body_colors(
    body_pool: list[int],
    anchors: list[int],
    src_bg: int,
    dst_bg: int,
    src_border: int,
    dst_border: int,
    forced_body: int | None,
) -> list[int] | None:
    choices = [
        c
        for c in dict.fromkeys(body_pool)
        if c not in anchors and c not in {src_bg, dst_bg, src_border, dst_border}
    ]
    if len(choices) < 2:
        return None

    if forced_body is not None and forced_body in choices:
        other_choices = [c for c in choices if c != forced_body]
        if not other_choices:
            return None
        bodies = [forced_body, random.choice(other_choices)]
        random.shuffle(bodies)
        return bodies

    return random.sample(choices, 2)


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
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg, padding=OUTPUT_SPRITE_PADDING)
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


def _choose_marker_type_ids(n_types: int) -> list[int]:
    if n_types != 2:
        raise ValueError("combo generator expects exactly 2 sprite types")
    repeated_type = random.randrange(n_types)
    single_type = 1 - repeated_type
    marker_type_ids = [repeated_type, repeated_type, single_type]
    random.shuffle(marker_type_ids)
    return marker_type_ids


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

    bodies = _choose_body_colors(body_pool, anchors, src_bg, dst_bg, src_border, dst_border, forced_body)
    if bodies is None or len(set(bodies)) != 2 or set(bodies) & set(anchors):
        return None

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
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

    marker_type_ids = _choose_marker_type_ids(n_types)
    if Counter(marker_type_ids).most_common(1)[0][1] < 2:
        return None

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
        dest_rect[1 + mr, 1 + mc] = sprite_types[type_id]["body"]

    marker_counts = Counter(type_id for _mr, _mc, type_id in marker_points)
    if sorted(marker_counts.values()) != [1, 2]:
        return None

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
    raise RuntimeError(f"failed to generate a251c730 combo input after {MAX_GEN_ATTEMPTS} attempts")
