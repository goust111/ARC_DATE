# Variant of a251c730
# Dimension: D15 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码——精灵和标记的对应关系直接由颜色相等表达，不需要额外图例解读... → 颜色图例指示匹配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the D15 variant
# ==================================
# a) Concrete behavior and trace
#    - The original task has two bordered rectangles on striped wallpaper:
#      a source rectangle with multi-cell sprites and a destination rectangle
#      with singleton markers. A sprite is chosen by direct color equality
#      between the sprite's anchor color and the destination marker color.
#    - This variant adds a small 2 x N legend at the top-left corner of the
#      whole grid, outside both rectangles. Each column is one mapping pair:
#      row 0 = source anchor color, row 1 = destination marker color.
#    - Example trace:
#        source anchor colors: [2, 8]
#        legend top row:       [8, 2]
#        legend bottom row:    [6, 7]
#        destination marker 6 now means "use the sprite whose anchor is 8".
#      In the original task the solver would look for template key 6 directly;
#      in the variant it must translate 6 -> 8 first.
# b) Complexity calibration
#    - Keep the old source/destination rectangle structure unchanged so the
#      task is still recognizably the same family.
#    - Make 2 sprite types common so the legend is visually meaningful.
#    - Keep the legend tiny (2 x 1 or 2 x 2) so D15 is the main added step,
#      not a new spatial puzzle.
# c) Implicit-assumption audit of the original main()
#    - Rectangle discovery remains valid because the legend is unbordered and
#      never becomes a candidate rectangle.
#    - Template extraction from source sprites still works because sprites are
#      unchanged: the least frequent color remains the singleton anchor.
#    - The old stamping loop assumed marker_color == anchor_color:
#          if mcol not in templates: continue
#      That assumption is false now and must become:
#          anchor = legend_map[mcol]; use templates[anchor]
# d) Placement legality
#    - The generator reserves the top-left corner (legend cells plus one-cell
#      moat) so neither bordered rectangle can overlap or touch the legend.
#    - Destination marker positions are still derived from full sprite
#      placements, so every later stamp fits completely inside the destination
#      interior exactly as in the original output-first generator.
# e) Visual observability
#    - Marker colors are distinct from sprite anchors, bodies, borders, and
#      wallpaper, so direct color matching is visibly wrong.
#    - The paired 2-row legend provides an explicit structure humans can read
#      from 2-3 examples; no color-id ordering is used anywhere.
# f) Visual cleanliness
#    - The legend lives alone in the corner on wallpaper, rather than being
#      mixed into either rectangle, so source sprites, target markers, and the
#      legend stay visually separable.
# g) Color-change inferability
#    - Mapping is one-to-one and column-wise. No hidden arithmetic or sorting
#      on color ids is involved. The same legend colors reappear in the source
#      anchors and destination markers, which gives the needed cross-reference.
# h) New solver assumptions to guarantee in generate_input()
#    - The legend is always in the top-left corner and has width equal to the
#      number of distinct sprite types.
#    - Legend row 0 stores anchor colors; row 1 stores marker colors.
#    - Destination marker colors do not coincide with any source anchor color,
#      otherwise the variant could collapse back to direct matching.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame using a corner color legend.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a source area with multi-cell sprites and a destination area with singleton markers.
    - A small 2-row legend is placed in the top-left corner of the grid; each column pairs a source anchor color (top) with a destination marker color (bottom).

    Rule:
    - Extract one template per source sprite type, keyed by its singleton anchor color.
    - Decode each destination marker color through the legend to recover which anchor color it refers to.
    - Stamp the matching source sprite so its anchor lands on that marker, then return the cropped destination rectangle with a cleared interior.

    Key insight: Destination marker colors are symbolic codes; the legend translates marker colors back into sprite anchor colors before placement.
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
                    rects.append((area, (r0, c0, r1, c1), bc, ibg))

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

    assert src is not None and dst is not None, "must find both source and destination rectangles"

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
        anchors = np.argwhere((patch == anchor_col) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col][1].sum()):
            templates[anchor_col] = (patch.copy(), mask.copy(), (ar, ac))

    legend_w = len(templates)
    assert legend_w >= 1 and W >= legend_w and H >= 2, "legend must fit in the top-left corner"

    marker_to_anchor: dict[int, int] = {}
    for c in range(legend_w):
        anchor_col = int(g[0, c])
        marker_col = int(g[1, c])
        marker_to_anchor[marker_col] = anchor_col

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[dr + 1 : dr1, dc + 1 : dc1]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        marker_col = int(inner[int(rr), int(cc)])
        anchor_col = marker_to_anchor.get(marker_col)
        if anchor_col is None or anchor_col not in templates:
            continue
        patch, mask, (ar, ac) = templates[int(anchor_col)]
        top = int((1 + int(rr)) - ar)
        left = int((1 + int(cc)) - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (D15) ---------------

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
SPRITE_TYPE_CHOICES = (2, 2, 1)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Legend & placement protection ===
LEGEND_HEIGHT = 2
LEGEND_CLEARANCE = 1
RECT_PLACE_RETRIES = 200

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


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int, legend_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    reserved = (0, 0, LEGEND_HEIGHT + LEGEND_CLEARANCE - 1, legend_w + LEGEND_CLEARANCE - 1)
    for _ in range(RECT_PLACE_RETRIES):
        if layout == "side_by_side":
            gap = random.randint(*GAP_RANGE)
            if src_w + dst_w + gap > GRID_SIZE:
                continue
            slack = GRID_SIZE - src_w - dst_w - gap
            left_margin = random.randint(0, slack)
            src_r = random.randint(0, GRID_SIZE - src_h)
            dst_r = random.randint(0, GRID_SIZE - dst_h)
            if random.random() < 0.5:
                src_c, dst_c = left_margin, left_margin + src_w + gap
            else:
                dst_c, src_c = left_margin, left_margin + dst_w + gap
        else:
            gap = random.randint(*GAP_RANGE)
            if src_h + dst_h + gap > GRID_SIZE:
                continue
            slack = GRID_SIZE - src_h - dst_h - gap
            top_margin = random.randint(0, slack)
            src_c = random.randint(0, GRID_SIZE - src_w)
            dst_c = random.randint(0, GRID_SIZE - dst_w)
            if random.random() < 0.5:
                src_r, dst_r = top_margin, top_margin + src_h + gap
            else:
                dst_r, src_r = top_margin, top_margin + dst_h + gap

        src_box = (src_r, src_c, src_r + src_h - 1, src_c + src_w - 1)
        dst_box = (dst_r, dst_c, dst_r + dst_h - 1, dst_c + dst_w - 1)
        if _boxes_overlap(src_box, reserved) or _boxes_overlap(dst_box, reserved):
            continue
        return (src_r, src_c), (dst_r, dst_c)

    raise ValueError("failed to place rectangles away from the legend")


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
            mr = r0 + sprite["anchor_rc"][0]
            mc = c0 + sprite["anchor_rc"][1]
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
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette = list(range(1, 10))
    wallpaper_n = 4 if n_types == 2 else random.choice((4, 5))
    wallpaper = random.sample(palette, wallpaper_n)
    src_bg = random.choice(wallpaper)

    remaining = [c for c in palette if c not in wallpaper]
    random.shuffle(remaining)
    if len(remaining) < 1 + n_types + n_types:
        return None

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]
    markers = [remaining.pop() for _ in range(n_types)]

    if remaining and random.random() < 0.5:
        dst_border = remaining.pop()
    else:
        dst_border_choices = [c for c in wallpaper if c != src_bg]
        if not dst_border_choices:
            return None
        dst_border = random.choice(dst_border_choices)

    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    body_pool = [c for c in wallpaper if c not in {src_bg, dst_bg, dst_border}]
    if remaining and random.random() < 0.65:
        body_pool.append(remaining.pop())
    body_choices = [c for c in body_pool if c not in anchors and c not in markers]
    if not body_choices:
        return None

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    bodies = [random.choice(body_choices) for _ in range(n_types)]
    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": body,
            "anchor": anchor,
            "marker": marker,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor, marker in zip(sprite_specs, bodies, anchors, markers)
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
        dest_rect[1 + mr, 1 + mc] = sprite_types[type_id]["marker"]

    legend_pairs = [(sprite["anchor"], sprite["marker"]) for sprite in sprite_types]
    random.shuffle(legend_pairs)
    legend_w = len(legend_pairs)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w, legend_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])
    for c, (anchor_col, marker_col) in enumerate(legend_pairs):
        grid[0, c] = int(anchor_col)
        grid[1, c] = int(marker_col)

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 D15 input after %d attempts" % MAX_GEN_ATTEMPTS)
