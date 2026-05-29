# Combo variant of a251c730 (index 9)
# combo_spec: D1/1+D15/0
# Name: 主体色匹配+颜色图例

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Baseline task:
#      source sprites are keyed directly by their singleton anchor color, and
#      destination markers use that same anchor color.
#    - D1/1 flips the key:
#      the lookup key becomes the sprite's dominant body color, but placement
#      still aligns the singleton anchor onto the marker cell.
#    - D15/0 adds a corner legend:
#      marker colors no longer equal the lookup key directly; the solver must
#      decode them through a 2-row legend.
#    - This combo fuses both changes instead of stacking them independently:
#      legend row 0 = sprite body color, legend row 1 = destination marker code.
#      Example:
#        sprite A uses body=7, anchor=2
#        sprite B uses body=4, anchor=9
#        legend columns might be [(7, 6), (4, 1)]
#        a destination pixel colored 6 means "choose the sprite whose dominant
#        color is 7", then stamp it so the anchor color 2 lands on that cell.
#      Difference from baseline:
#        baseline marker 2 -> choose anchor-2 sprite directly.
#      Difference from D1-only:
#        D1-only marker 7 -> choose body-7 sprite directly.
#      Difference from D15-only:
#        D15-only legend top row would contain anchors, not bodies.
#
# b) Complexity calibration
#    - The generator is tightened to exactly 2 sprite types, as in the D1/1
#      reference, so "majority color is the identity" is observable and cannot
#      collapse into a one-type trivial case.
#    - Both types are guaranteed to appear in source and destination. This gives
#      repeated evidence for the legend mapping and for body-based lookup.
#    - Marker code colors are forced to differ from both body colors and anchor
#      colors, so neither direct body match nor direct anchor match can solve
#      the task.
#
# c) Implicit-assumption audit of the original main()
#    - Original template extraction assumed "least frequent color is both the
#      key and the anchor". In the combo, the least frequent color remains the
#      anchor only; the key is the unique most frequent color.
#    - Original marker loop assumed marker_color == template_key. In the combo,
#      marker_color must first be translated by legend row1 -> row0.
#    - The D15 single variant assumed legend row0 stores anchors. That is false
#      here: row0 stores dominant body colors instead.
#    - The original code dropped color 0 while reading component colors. That is
#      unnecessary and unsafe, so the combo solver keeps all visible colors.
#
# d) Placement legality
#    - Destination marker positions are still generated output-first by first
#      placing full sprites into the destination interior and recording each
#      sprite's anchor cell. This guarantees every later stamp fits.
#    - The legend occupies the top-left corner and a one-cell moat around it is
#      reserved during rectangle placement, so bordered rectangles never touch
#      or hide the legend.
#    - Source sprites are placed with padding exactly as in the original task,
#      so source components remain separated and easy to recover.
#
# e) Visual observability
#    - The top legend row reuses body colors that also visibly dominate the
#      source sprites, while the bottom row reuses the isolated destination code
#      colors. This provides an explicit cross-reference humans can read.
#    - Because marker colors differ from both anchor and body colors, the input
#      visibly contradicts the original direct-match rule. The legend is needed.
#    - Because the stamped anchor cell still changes to the rare anchor color in
#      the output, the solver must also infer that the body color is only the
#      identifier, not the alignment pixel.
#
# f) Visual cleanliness
#    - The legend is kept outside both rectangles, so source sprites, target
#      markers, and legend cells live in separate visual regions.
#    - Only one symbolic layer is added: a tiny 2x2 legend. Destination inputs
#      stay as clean singleton markers rather than overlapping glyphs.
#
# g) Change inferability and anti-degeneration guarantees
#    - Mapping is purely column-wise in the legend; no hidden ordering on color
#      ids is used.
#    - Generator guarantees:
#      * exactly two sprite types
#      * unique dominant body colors and unique anchor colors
#      * legend row0 = body colors, row1 = marker code colors
#      * each marker code differs from every body and anchor color
#      * both sprite types appear at least once in source and destination
#    - Under these guarantees, running baseline, D1-only, or D15-only logic on
#      the same input will usually fail to stamp the correct sprites, so the
#      combo does not collapse back to a simpler rule family.


def _dominant_and_anchor(comp: dict) -> tuple[int, int] | None:
    """Return (body_color, anchor_color) when the component has unique max/min counts."""
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
    Stamp source sprites into a destination frame using a body-color legend.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: the source contains multi-cell two-color sprites and the destination contains isolated one-pixel codes.
    - A tiny 2-row legend in the top-left corner pairs each sprite's dominant body color (top row) with the destination code color that refers to it (bottom row).

    Rule:
    - Recover one template per source sprite type, identified by its unique most frequent color while still locating its singleton anchor cell.
    - Decode each destination marker through the legend to recover which body color it names.
    - Stamp the matching source sprite so its singleton anchor lands on the marker position, then return the cropped destination rectangle with a cleared interior.

    Key insight: The legend says which marker codes name which body colors, but the rare anchor pixel still controls placement inside the stamped sprite.
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

    assert rects, "must find bordered rectangles"

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
        key_colors = _dominant_and_anchor(comp)
        if key_colors is None:
            continue
        body_col, anchor_col = key_colors
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if int(body_col) not in templates or int(mask.sum()) > int(templates[int(body_col)][1].sum()):
            templates[int(body_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    legend_w = len(templates)
    assert legend_w >= 1 and H >= 2 and W >= legend_w, "legend must fit in the top-left corner"

    marker_to_body: dict[int, int] = {}
    for c in range(legend_w):
        body_col = int(g[0, c])
        marker_col = int(g[1, c])
        marker_to_body[int(marker_col)] = int(body_col)

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
        body_col = marker_to_body.get(int(marker_col))
        if body_col is None or int(body_col) not in templates:
            continue
        patch, mask, (ar, ac) = templates[int(body_col)]
        top = int((1 + int(rr)) - int(ar))
        left = int((1 + int(cc)) - int(ac))
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (combo D1/1 + D15/0) ---------------

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

LEGEND_HEIGHT = 2
LEGEND_CLEARANCE = 1
RECT_PLACE_RETRIES = 200

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


def _assign_palette() -> tuple[list[int], int, int, int, list[int], list[int], list[int]]:
    all_colors = list(range(10))
    # Keep wallpaper to 3 colors so the 2 body colors can stay outside it.
    # This makes the legend's top row visually stand out instead of blending
    # back into the wallpaper stripes.
    wallpaper = random.sample(all_colors, 3)
    random.shuffle(wallpaper)

    shared_bg, dst_border = random.sample(wallpaper, 2)

    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    src_border = remaining.pop()
    body_colors = [remaining.pop() for _ in range(2)]
    anchors = [remaining.pop() for _ in range(2)]
    markers = [remaining.pop() for _ in range(2)]
    return wallpaper, int(src_border), int(dst_border), int(shared_bg), list(body_colors), anchors, markers


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
    if n_types != 2:
        return None

    wallpaper, src_border, dst_border, shared_bg, body_colors, anchors, markers = _assign_palette()
    if len(set(body_colors)) != 2 or len(set(anchors)) != 2 or len(set(markers)) != 2:
        return None
    if set(body_colors) & set(anchors):
        return None
    if set(body_colors) & set(markers):
        return None
    if set(anchors) & set(markers):
        return None

    src_bg = int(shared_bg)
    dst_bg = int(shared_bg)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    random.shuffle(body_colors)
    random.shuffle(anchors)
    random.shuffle(markers)
    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": int(body),
            "anchor": int(anchor),
            "marker": int(marker),
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor, marker in zip(sprite_specs, body_colors, anchors, markers)
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

    legend_pairs = [(sprite["body"], sprite["marker"]) for sprite in sprite_types]
    random.shuffle(legend_pairs)
    legend_w = len(legend_pairs)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w, legend_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])
    for c, (body_col, marker_col) in enumerate(legend_pairs):
        grid[0, c] = int(body_col)
        grid[1, c] = int(marker_col)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 combo input after %d attempts" % MAX_GEN_ATTEMPTS)
