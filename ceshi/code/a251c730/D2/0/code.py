# Variant of a251c730
# Dimension: D2 — 源对象：精灵结构 (src)
# Change: 源矩形中有2-4个精灵，1-2种类型，每种精灵为2x3或3x3外接矩形、5-7像素、恰好2种颜色（主体色=多数、锚点色=单一像素）→ 三色精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D2 variant
# a) Concrete behavior:
#    - The global layout is unchanged: wallpaper + one source rectangle + one destination rectangle.
#    - The destination still contains only singleton anchor markers.
#    - The source sprites now have exactly three visible colors:
#      body = most frequent, decoration = middle frequency, anchor = unique rarest pixel.
#    - Example trace:
#      old source sprite:     new source sprite:
#      A B B                  A B C
#      B B .                  B B C
#      Old solver could treat "all non-anchor pixels" as one body color.
#      New solver must keep the full 3-color patch, but still align by the unique rarest color A.
# b) Complexity calibration:
#    - Decoration count is fixed to 2 pixels; anchor count is fixed to 1 pixel.
#    - Body count is 3 or 4 pixels, so the frequency order is always visually readable:
#      anchor < decoration < body.
#    - This necessarily realizes the sprite area as 6-7 cells rather than 5-7: a 5-cell tri-color
#      sprite would force a body/decoration tie or a second singleton, which would hide the anchor cue.
#    - We keep 1-2 sprite types and 2-4 total source copies so the repeated evidence remains strong.
# c) Implicit assumptions changed from the original main():
#    - The old code picked the minimum count color using a color-id tiebreaker.
#      That is unsafe here because the task is about a visible unique rarest color, not numeric ids.
#    - The old code also effectively assumed "non-anchor pixels behave as one bucket".
#      That is no longer true; the stored template must preserve both body and decoration colors.
#    - The new solver therefore accepts a source component only if it has exactly 3 colors with
#      counts (1, 2, 3/4) in strictly increasing order, and then uses the unique rarest color as anchor.
# d) Placement legality:
#    - Source sprites are placed with padding=1 so components stay separate for 4-connectivity parsing.
#    - Marker placement still uses the full future sprite footprint on a temporary output canvas; only
#      the anchor pixel survives in the input. This guarantees in-bounds, non-overlapping stamps later.
# e) Visual observability:
#    - The variant is directly visible: source and output sprites are tri-color rather than bi-color.
#    - The marker remains a single pixel, so the intended inference is "find the unique rarest color inside
#      each source sprite, then use that color as the destination key".
# f) Visual cleanliness:
#    - Decoration is only 2 pixels and sprites never overlap, so the rectangles stay readable.
#    - The destination input stays sparse because it shows only singleton markers.
# g) Color inference:
#    - No hidden color ordering or remapping is used. Marker color equals anchor color exactly.
#    - The generator guarantees the visible frequency cue instead of relying on color ids.
# h) Variant metric used after implementation:
#    - For generated samples we measure the multiset of per-sprite color counts inside the source rectangle.
#    - A valid non-degenerate sample must have every source sprite counted as (1,2,3) or (1,2,4);
#      the original generator never produced such 3-color components.
# i) New solver assumptions to satisfy:
#    - Every source sprite component has exactly 3 colors with a unique rarest color.
#    - Every destination marker color appears as an anchor color of some source sprite type.
#    - Repeated copies of a type are identical, so any copy can serve as the template.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by matching tri-color source sprites to singleton anchor markers.

    Input structure:
    - Two bordered rectangles lie on top of a striped wallpaper.
    - The source rectangle contains 2-4 disconnected sprites; each sprite uses body, decoration, and anchor colors.
    - The destination rectangle contains only single-pixel markers colored like sprite anchors.

    Rule:
    - For each source sprite, identify the unique least-frequent color as the anchor.
    - Match each destination marker to the source sprite whose anchor has the same color.
    - Stamp the entire three-color sprite into the destination frame with the anchor aligned to the marker.

    Key insight: The anchor is no longer "the only non-body color"; it is the unique rarest color inside a
    three-color sprite, while the medium-frequency decoration color must be preserved as part of the template.
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
                    "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                    "mask": mask,
                    "patch": patch,
                }
            )

        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

    def classify_tricolor_sprite(colors: dict[int, int]) -> tuple[int, int, int] | None:
        if len(colors) != 3:
            return None
        ordered = sorted(((int(cnt), int(col)) for col, cnt in colors.items()), key=lambda x: x[0])
        counts = [cnt for cnt, _col in ordered]
        if counts[0] != 1:
            return None
        if not (counts[0] < counts[1] < counts[2]):
            return None
        anchor_col = ordered[0][1]
        decor_col = ordered[1][1]
        body_col = ordered[2][1]
        return int(anchor_col), int(decor_col), int(body_col)

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
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

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

    assert src is not None and dst is not None, "must find source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        roles = classify_tricolor_sprite({int(k): int(v) for k, v in comp["colors"].items()})
        if roles is None:
            continue
        anchor_col, _decor_col, _body_col = roles
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == anchor_col) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col][1].sum()):
            templates[anchor_col] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    assert templates, "no tri-color source templates found"

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

# === CFG: Canvas & layout ===
GRID_SIZE = 30
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
    """Render a tri-color sprite. A=anchor, B=body, C=decoration, .=bg."""
    char_map = {"A": anchor, "B": body, "C": decor}
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], int, int] | None:
    """Allocate wallpaper, borders, anchor colors, body color, and decoration color."""
    wallpaper_n = 5 if n_types == 1 else 4
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    needed = 1 + n_types + 2
    if len(remaining) < needed:
        return None

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]
    body = remaining.pop()
    decor = remaining.pop()

    dst_border_choices = [c for c in wallpaper if c != src_border]
    if not dst_border_choices:
        return None
    dst_border = random.choice(dst_border_choices)
    return wallpaper, src_border, dst_border, anchors, body, decor


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place sprite footprints in a temp destination interior; return output interior and marker anchors."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], sprite["decor"], dst_bg)
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
            marker_points.append((mr, mc, type_id))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    """Place tri-color sprites in the source interior."""
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

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body, decor = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)
    if body in (src_bg, dst_bg) or decor in (src_bg, dst_bg):
        return None

    sprite_shapes = random.sample(SPRITE_SHAPES, n_types)
    sprite_types = []
    for shape_rows, anchor in zip(sprite_shapes, anchors):
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
        dest_rect[1 + mr, 1 + mc] = sprite_types[type_id]["anchor"]

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
    raise RuntimeError(f"failed to generate a251c730 D2 variant after {MAX_GEN_ATTEMPTS} attempts")
