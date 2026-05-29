# Combo variant of a251c730 (index 29)
# combo_spec: D1/1+D2/0+D4/2
# Name: 主体色匹配+三色精灵+颜色互换

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# a) Concrete behavior and trace:
#    - Layout stays the same: striped wallpaper, one source rectangle with sprites, one destination rectangle
#      with singleton markers.
#    - Each source sprite is now tri-color with visible frequencies:
#        anchor = 1 pixel, decoration = 2 pixels, body = 3 or 4 pixels.
#    - Matching no longer uses the rare anchor color. If a source sprite has
#        anchor=2, decor=4, body=7,
#      then the destination marker is color 7.
#    - Placement still aligns the singleton anchor position to that marker cell, but the stamped sprite swaps
#      body and anchor colors before drawing:
#        source patch  : A B C      output patch : B A C
#                        B B C                     A A C
#      So the lookup color is the source majority color, yet after stamping that color only occupies the anchor
#      slot while the old anchor color floods the former body pixels.
# b) Complexity calibration:
#    - Generation is fixed to exactly 2 sprite types so body-color matching is not trivial.
#    - Body colors are unique across types; anchor colors are also unique and disjoint from body colors.
#    - Both types appear in the source and among destination markers, which gives repeated evidence for the
#      "majority color selects type" rule and for the subsequent color swap.
#    - Decoration uses one shared color across types to keep the palette readable; it stays unchanged after
#      stamping, which makes the swap visually local to the body/anchor roles.
# c) Implicit assumptions changed from the original main():
#    - The original code keyed templates by least-frequent color; this is invalid here and must be replaced by
#      the unique most-frequent color.
#    - The least-frequent color is still needed, but only to locate the alignment cell inside the sprite.
#    - The original solver could treat all non-anchor cells as one color. That fails for tri-color sprites, so the
#      solver now requires an exact visible count pattern (1,2,3/4) and preserves the decoration pixels.
#    - Any color id, including 0, may be a real sprite color. Therefore swaps and anchor searches must always be
#      masked; patch padding set to 0 cannot be treated as "not a real color".
# d) Placement legality:
#    - Destination legality is enforced output-first by placing the fully swapped sprite footprints into a blank
#      destination interior, then reducing those placements to singleton body-colored markers for the input.
#    - Source sprites are placed with border/padding >= 1 so 4-connected component extraction recovers each sprite
#      exactly once.
# e) Visual observability:
#    - The destination cue color visibly matches the repeated color in the source sprite, not the singleton pixel.
#    - The output visibly reverses the body/anchor color roles while leaving the decoration untouched, so the
#      fusion behavior differs from both the original task and the single D1 variant.
# f) Visual cleanliness:
#    - Markers remain isolated singletons and are spaced so stamped sprites do not collide.
#    - Tri-color sprites use only one extra decoration color and small 2x3/3x3 footprints, keeping rectangles easy
#      to parse by eye.
# g) Generator guarantees required by the new solver:
#    - Every source component has exactly 3 colors with counts (1,2,3) or (1,2,4).
#    - Every destination marker color equals exactly one sprite body's color.
#    - Repeated copies of a type are identical, so any copy can be used as the template.


def _classify_tricolor_sprite(colors: dict[int, int]) -> tuple[int, int, int] | None:
    """Return (body_color, anchor_color, decor_color) for a strict 1/2/3-or-4 tri-color sprite."""
    items = [(int(col), int(cnt)) for col, cnt in colors.items()]
    if len(items) != 3:
        return None
    ordered = sorted(items, key=lambda kv: kv[1])
    counts = [cnt for _col, cnt in ordered]
    if counts[0] != 1 or counts[1] != 2 or counts[2] not in (3, 4):
        return None
    if not (counts[0] < counts[1] < counts[2]):
        return None
    anchor_col = int(ordered[0][0])
    decor_col = int(ordered[1][0])
    body_col = int(ordered[2][0])
    return int(body_col), int(anchor_col), int(decor_col)


def _swap_patch_colors(patch: np.ndarray, mask: np.ndarray, a: int, b: int) -> np.ndarray:
    """Swap two colors only on visible sprite cells; masked-out padding is preserved."""
    out = patch.copy()
    a_mask = (patch == int(a)) & mask
    b_mask = (patch == int(b)) & mask
    out[a_mask] = int(b)
    out[b_mask] = int(a)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp swapped tri-color sprites into the destination frame via body-colored singleton markers.

    Input structure:
    - Two bordered rectangles lie on top of a striped wallpaper.
    - The source rectangle contains disconnected three-color sprites whose visible roles are body, decoration, and
      singleton anchor.
    - The destination rectangle contains only one-pixel markers, and each marker color matches a sprite's body color.

    Rule:
    - Parse each source sprite by color frequency: most common = body, unique least common = anchor, middle count =
      decoration.
    - Use the body color to choose which sprite a destination marker refers to, but still align the sprite by its
      anchor cell.
    - Before stamping, swap the sprite's body and anchor colors while keeping the decoration unchanged, then return
      the rebuilt destination rectangle.

    Key insight: The repeated color is the lookup cue, the rare pixel still determines geometry, and the stamped
    sprite inverts those two color roles at the last step.
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
                    "mask": mask,
                    "patch": patch,
                }
            )

        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

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
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if ibg == bc:
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects, "no candidate rectangles found"

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

    assert src is not None and dst is not None, "must find source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        roles = _classify_tricolor_sprite({int(k): int(v) for k, v in comp["colors"].items()})
        if roles is None:
            continue
        body_col, anchor_col, _decor_col = roles
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        swapped = _swap_patch_colors(patch, mask, int(body_col), int(anchor_col))
        if int(body_col) not in templates or int(mask.sum()) > int(templates[int(body_col)][1].sum()):
            templates[int(body_col)] = (swapped, mask.copy(), (int(ar), int(ac)))

    assert templates, "no valid tri-color source templates found"

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[dr + 1 : dr1, dc + 1 : dc1]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mr = int(dr + 1 + int(rr))
        mc = int(dc + 1 + int(cc))
        mcol = int(g[mr, mc])
        if mcol not in templates:
            continue
        patch, mask, (ar, ac) = templates[mcol]
        top = int((mr - dr) - ar)
        left = int((mc - dc) - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

GRID_SIZE = 30
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

SPRITE_SHAPES = [
    ("ABC", "BBC"),
    ("ABB", "CBC"),
    ("ABB", "BCC"),
    ("ABB", "BCC", "..B"),
    (".AB", "BBC", "..C"),
    ("AB.", "BBC", ".C."),
    ("A.B", "BBC", "C.."),
    ("A.B", "BBC", ".BC"),
    ("A..", "BBB", "CC."),
    ("A..", "BBB", "BCC"),
    ("..A", "BBB", ".CC"),
    ("..A", "BBB", "CCB"),
    (".AB", "BCB", "C.."),
    ("AB.", "BCB", "..C"),
    ("AB.", "BCB", ".BC"),
    ("A..", "BCB", ".BC"),
]


def _shape_info(shape_rows: tuple[str, ...]) -> tuple[tuple[int, int], Counter]:
    counts: Counter = Counter()
    anchor_rc: tuple[int, int] | None = None
    for r, row in enumerate(shape_rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            counts[ch] += 1
            if ch == "A":
                anchor_rc = (int(r), int(c))
    assert anchor_rc is not None
    assert counts["A"] == 1
    assert counts["C"] == 2
    assert counts["B"] in (3, 4)
    return anchor_rc, counts


for _shape in SPRITE_SHAPES:
    _anchor_rc, _counts = _shape_info(_shape)
    assert len(_shape) in (2, 3)
    assert len(_shape[0]) == 3


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, decor: int, bg: int) -> np.ndarray:
    """Render a tri-color sprite. A=anchor location, B=body region, C=decoration, .=bg."""
    char_map = {"A": anchor, "B": body, "C": decor}
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


def _assign_palette() -> tuple[list[int], int, int, list[int], list[int], int]:
    """Use all 10 ARC colors in stable roles: 4 wallpaper + dst border on wallpaper + 6 non-wallpaper roles."""
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, 4)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    src_border = int(remaining.pop())
    anchors = [int(remaining.pop()) for _ in range(2)]
    bodies = [int(remaining.pop()) for _ in range(2)]
    decor = int(remaining.pop())
    dst_border = int(random.choice(wallpaper))
    return wallpaper, src_border, dst_border, anchors, bodies, decor


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place the swapped output sprites, then record their anchor cells as future input markers."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["anchor"], sprite["body"], sprite["decor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr, mc = r0 + sprite["anchor_rc"][0], c0 + sprite["anchor_rc"][1]
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((int(mr), int(mc), int(type_id)))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], sprite["decor"], src_bg)
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
    assert n_types == 2

    wallpaper, src_border, dst_border, anchors, bodies, decor = _assign_palette()

    src_bg = int(random.choice(wallpaper))
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = int(random.choice(dst_bg_choices))

    sprite_shapes = random.sample(SPRITE_SHAPES, n_types)
    sprite_types = []
    for idx, (shape_rows, anchor, body) in enumerate(zip(sprite_shapes, anchors, bodies)):
        anchor_rc, counts = _shape_info(shape_rows)
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": anchor_rc,
                "body": int(body),
                "anchor": int(anchor),
                "decor": int(decor),
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
                "area": int(sum(counts.values())),
                "type_id": int(idx),
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

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 combo variant after {MAX_GEN_ATTEMPTS} attempts")
