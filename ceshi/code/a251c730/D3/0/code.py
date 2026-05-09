# Variant of a251c730
# Dimension: D3 — 目标结构：标记类型与放置 (dst)
# Change: 目标矩形内部放置2-3个单像素标记，每个标记颜色对应一种精灵的锚点颜色，标记周围有足够空间容纳完整精灵印刻... → 标记为小色块而非单像素

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the D3 variant
# ==================================
# a) Concrete behavior and trace
#    - The source rectangle is unchanged: it contains 2-4 multi-cell two-color sprites.
#      Each sprite type has one singleton anchor color and one body color.
#    - The destination rectangle no longer contains singleton anchor markers. Instead,
#      each marker is a solid monochrome block whose color equals a sprite anchor color.
#      Allowed marker shapes are 2x2 and 3x1. The placement rule is fixed:
#      the sprite anchor must land on the block's top-left cell.
#    - Small trace:
#        original: a red singleton at (5,7) means "paste the red-anchor sprite so its
#        anchor pixel sits exactly on (5,7)".
#        variant: a red 2x2 block with bbox rows 5..6, cols 7..8 means "paste the
#        red-anchor sprite so its anchor pixel sits on the block's top-left corner (5,7)".
#      The color match is unchanged; only the positional cue changes from "the marked
#      pixel itself" to "the top-left corner of the marked block".
# b) Complexity calibration
#    - Keeping only two block shapes preserves a small hypothesis space for humans.
#    - Marker blocks are still keyed by anchor colors, so the main novelty is spatial:
#      the solver must recognize a component as a marker block, then extract its
#      corner as the real anchor point.
#    - Using 2-3 markers and 1-2 sprite types keeps repeated evidence in each sample.
# c) Hidden-assumption audit against the original main()
#    - Original destination detection used `max_sz == 1`, which fails once markers have
#      area 3 or 4. This must be replaced by a structural test for "all components are
#      solid small rectangles of allowed shapes".
#    - Original stamping loop iterated over every non-background destination pixel.
#      In the variant, iterating over all non-background pixels would stamp the same
#      sprite multiple times per block and shift placements incorrectly. The loop must
#      iterate over connected marker components, not pixels.
#    - Original marker position was the pixel coordinate itself. In the variant, the
#      meaningful point is the marker block bbox top-left corner.
#    - Original least-frequent-color logic relied on the source sprite having a unique
#      singleton anchor. The generator must continue to guarantee that uniqueness.
# d) Placement legality
#    - Generator still places output sprites first so they never overlap and always fit.
#    - For each placed sprite, the marker block is attached to the sprite anchor point.
#      The generator rejects placements where the block would cross the destination
#      interior boundary.
#    - Marker blocks are kept at least one cell apart so same-color blocks do not merge
#      into one component and so the destination remains visually clean.
# e) Visual observability
#    - The difference from the original task is visible in every valid sample because
#      destination clues are always area-3/4 blocks, never singleton pixels.
#    - The top-left alignment rule is recoverable from input/output pairs: the output
#      sprite anchor consistently lands on the top-left cell of the colored block.
# f) Visual cleanliness
#    - Destination inputs contain only sparse solid blocks over a plain interior.
#      Source sprites stay separated by padding, and marker blocks stay separated too.
# g) Color inferability
#    - No hidden ordering by color id is used in the task logic. Colors only act as
#      visible equality keys between source anchors and destination marker blocks.
# h) Variant-specific quantitative check to run after implementation
#    - Metric 1: every destination marker component must have area in {3,4}.
#    - Metric 2: compare the new solver against a naive singleton-marker interpretation;
#      they should disagree on generated samples, proving the block-corner rule matters.
# i) New solver assumptions that the generator must enforce
#    - Destination rectangles have exactly 2-3 non-background components.
#    - Every destination component is a filled 2x2 or 3x1 rectangle of a single color.
#    - The block's top-left corner leaves enough room to stamp the whole sprite.
#    - Source sprites keep a unique least-frequent color so anchor extraction is stable.


MARKER_SHAPES = ((2, 2), (3, 1))


def _interior_bg(grid: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(grid: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
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
        cols = Counter(int(inner[r, c]) for r, c in coords)
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


def _component_nonzero_colors(comp: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}


def _is_marker_block(comp: dict) -> bool:
    colors = _component_nonzero_colors(comp)
    if len(colors) != 1:
        return False
    rr0, cc0, rr1, cc1 = comp["bbox"]
    h = int(rr1 - rr0 + 1)
    w = int(cc1 - cc0 + 1)
    if (h, w) not in MARKER_SHAPES:
        return False
    return bool(comp["mask"].all()) and int(comp["size"]) == int(h * w)


def _anchor_color_from_sprite(comp: dict) -> int | None:
    colors = _component_nonzero_colors(comp)
    if not colors:
        return None
    min_count = min(colors.values())
    anchors = [color for color, count in colors.items() if int(count) == int(min_count)]
    if len(anchors) != 1:
        return None
    return int(anchors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via color-matched marker blocks.

    Input structure:
    - Two bordered rectangles sit on the wallpaper: a source frame with multi-cell sprites
      and a destination frame with 2-3 solid colored marker blocks.
    - Each sprite's unique rare color is its anchor, and each destination block uses one
      of those anchor colors.

    Rule:
    - Match each destination block to the source sprite with the same anchor color.
    - Use the block's top-left corner as the true anchor location, then translate the
      sprite so its anchor pixel lands there.
    - Output the destination rectangle alone, cleared to its background before stamping.

    Key insight: The destination clue is no longer the colored pixels themselves but the
    corner of each small monochrome block, while color still identifies which sprite to use.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

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
                    ibg = _interior_bg(g, (int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect)
        if not comps:
            continue
        if src is None:
            if any(int(c["size"]) >= 5 and len(_component_nonzero_colors(c)) >= 2 for c in comps):
                src = (rect, bc, ibg2, comps)
                continue
        if dst is None:
            if 2 <= len(comps) <= 3 and all(_is_marker_block(c) for c in comps):
                dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find source sprites and destination marker blocks"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        anchor_col = _anchor_color_from_sprite(comp)
        if anchor_col is None:
            continue
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.shape[0] != 1:
            continue
        ar, ac = map(int, anchors[0])
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col][1].sum()):
            templates[anchor_col] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    for comp in dst_comps:
        if not _is_marker_block(comp):
            continue
        colors = _component_nonzero_colors(comp)
        marker_color = int(next(iter(colors)))
        if marker_color not in templates:
            continue
        rr0, cc0, _rr1, _cc1 = comp["bbox"]
        marker_r = int(rr0 + 1)
        marker_c = int(cc0 + 1)
        patch, mask, (ar, ac) = templates[marker_color]
        top = int(marker_r - ar)
        left = int(marker_c - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

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
MARKER_PLACE_RETRIES = 60
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
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


def _choose_marker_shape() -> tuple[int, int]:
    return random.choice(MARKER_SHAPES)


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]] | None:
    inner_h = int(dst_h - 2)
    inner_w = int(dst_w - 2)
    out_inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
    marker_reserved = np.zeros((inner_h, inner_w), dtype=bool)
    marker_blocks: list[tuple[int, int, int, int, int]] = []

    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        bh, bw = _choose_marker_shape()
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(out_inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr = int(r0 + sprite["anchor_rc"][0])
            mc = int(c0 + sprite["anchor_rc"][1])
            if mr + bh > inner_h or mc + bw > inner_w:
                continue
            if bool(marker_reserved[mr : mr + bh, mc : mc + bw].any()):
                continue
            paste(out_inner, patch, r0, c0, transparent=dst_bg)
            marker_blocks.append((mr, mc, bh, bw, type_id))
            marker_reserved[max(0, mr - 1) : min(inner_h, mr + bh + 1), max(0, mc - 1) : min(inner_w, mc + bw + 1)] = True
            placed = True
            break
        if not placed:
            return None
    return out_inner, marker_blocks


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

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    bodies: list[int] = []
    for i in range(n_types):
        choices = [c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg]
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
    out_inner, marker_blocks = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, bh, bw, type_id in marker_blocks:
        dest_rect[1 + mr : 1 + mr + bh, 1 + mc : 1 + mc + bw] = int(sprite_types[type_id]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

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
    raise RuntimeError(f"failed to generate a251c730 D3 input after {MAX_GEN_ATTEMPTS} attempts")
