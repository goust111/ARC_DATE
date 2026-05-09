# Combo variant of a251c730 (index 11)
# combo_spec: D2/3+D4/4
# Name: 大稀疏精灵+2倍缩放

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location, scale_sprite

# Design analysis for the combo variant
# - Concrete behavior:
#   - The source box still contains 1-2 sprite types and the destination box still contains
#     singleton anchor-colored markers, but the source sprites are no longer compact 2x3/3x3
#     blobs. Each sprite now occupies a 5x5 or 6x6 bounding box while using only 5-7 cells,
#     so the visible structure is a sparse diagonal/hook with large holes.
#   - On one marker, the old flow was "pick matching sprite -> paste raw patch so anchor
#     coincides with the marker". The fused flow is "pick matching sparse sprite -> scale
#     every source cell to a 2x2 block -> paste the scaled sparse patch so the marker lands
#     on the upper-left cell of the enlarged anchor block".
#   - Example: a raw 5x5 pattern with cells at (0,0),(1,1),(2,2),(3,3),(4,3),(4,4) becomes
#     a 10x10 pattern with the same hole layout, just doubled. The marker must therefore be
#     offset by (2*anchor_r, 2*anchor_c), not by the raw anchor coordinate.
# - Complexity calibration:
#   - The generator keeps only 1-2 sprite types and 2 destination markers, so there is repeated
#     evidence for the anchor-color matching rule while the enlarged sparse silhouettes remain
#     visually salient.
#   - The sparse templates are intentionally asymmetric and come from a short catalog. That
#     keeps them distinguishable without introducing too many cases for the human solver.
# - Old assumptions that must be revised:
#   - The original solver relied on 4-connectivity for source sprites. That breaks here
#     because a 5x5 or 6x6 sprite with only 5-7 cells can span its whole bbox only through
#     diagonal links. The source extraction must therefore use 8-connectivity.
#   - The original generator preferred compact non-overlapping placements. Here the scaled
#     patches are much larger, so the destination must be generated from the sparse occupied
#     cells rather than from raw bbox area alone.
#   - The old area heuristic "source rectangle larger than destination rectangle" is no
#     longer reliable once the destination must hold 2x enlarged sparse stamps, so the combo
#     generator does not depend on it.
# - Placement legality:
#   - Source sprites are placed with padding so separate sparse components do not touch even
#     diagonally; otherwise 8-connectivity would merge them.
#   - Destination placements are tested using the scaled patch itself, so every stamped sprite
#     fits fully inside the destination interior before the corresponding singleton marker is
#     written into the input.
# - Visual readability:
#   - The input destination remains clean because it shows only singleton markers.
#   - The output visibly differs from the base task because every copied sprite becomes a much
#     larger sparse silhouette with doubled holes.
# - New solver assumptions matched by generation:
#   - Each sprite type still uses exactly two colors, with one anchor-colored singleton cell.
#   - Each source sprite is exactly one 8-connected component.
#   - Markers are kept 8-disconnected, so the destination rectangle still decomposes into
#     singleton prompts even under 8-connectivity.

SCALE_FACTOR = 2
COMPONENT_CONNECTIVITY = 8

# --------------- solver ---------------


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recreate enlarged sparse sprites inside the destination frame via anchor-colored prompts.

    Input structure:
    - Two bordered rectangles sit on the striped wallpaper: one source box with sparse two-color sprites and one destination box with singleton marker pixels.
    - Each source sprite spans a 5x5 or 6x6 bbox but uses only a few cells; its unique least-frequent color is the anchor, and the destination markers reuse those anchor colors.

    Rule:
    - Match every destination marker to the source sprite whose anchor color is the same.
    - Scale the matched sparse sprite by 2 in both dimensions, then stamp it so the marker marks the upper-left cell of the scaled 2x2 anchor block.
    - Return the destination rectangle after clearing its interior back to the destination background and placing all enlarged sparse sprites.

    Key insight: The marker still selects a sprite by anchor color, but sparse source components must be read with 8-connectivity and their placement offset must be computed in scaled coordinates.
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

        for coords in components_mask(inner != int(ibg), connectivity=COMPONENT_CONNECTIVITY):
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

    assert rects, "must find candidate bordered rectangles"

    rects.sort(reverse=True, key=lambda x: int(x[0]))

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)
        if src is not None and dst is not None:
            break

    assert src is not None and dst is not None, "must find both source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        if len(cols) != 2:
            continue
        min_freq = min(cols.values())
        anchor_cols = [col for col, freq in cols.items() if int(freq) == int(min_freq)]
        if len(anchor_cols) != 1 or int(min_freq) != 1:
            continue
        anchor_col = int(anchor_cols[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
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
        scaled_patch = scale_sprite(patch, SCALE_FACTOR)
        scaled_mask = _scale_mask(mask, SCALE_FACTOR)
        top = int((mr - dr) - SCALE_FACTOR * int(ar))
        left = int((mc - dc) - SCALE_FACTOR * int(ac))
        paste(out, scaled_patch, int(top), int(left), mask=scaled_mask, overwrite=True, clip=True)

    return out


# --------------- generator ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (12, 18)
SRC_W_RANGE = (12, 18)
DST_H_RANGE = (14, 20)
DST_W_RANGE = (14, 20)
DST_W_SIDE_MAX = 18

SPRITE_TYPE_CHOICES = (1, 2, 2, 2)
MARKER_COUNT_CHOICES = (2,)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 60
SOURCE_PLACE_RETRIES = 40
MARKER_PADDING = 1
SOURCE_SPRITE_BORDER = 1
SOURCE_SPRITE_PADDING = 1
LAYOUT_CHOICES = ("side_by_side", "side_by_side", "stacked")

MAX_GEN_ATTEMPTS = 500

SPRITE_SHAPES = [
    (("A....", ".B...", "..B..", "...B.", "...BB"), (0, 0)),
    (("....A", "...B.", "..B..", ".B...", "BB..."), (0, 4)),
    (("A....", ".BB..", "...B.", "....B", "...B."), (0, 0)),
    (("....A", "..BB.", ".B...", "B....", ".B..."), (0, 4)),
    (("A.....", ".B....", "..B...", "...B..", "....B.", "....BB"), (0, 0)),
    ((".....A", "....B.", "...B..", "..B...", ".B....", "BB...."), (0, 5)),
    (("A.....", ".BB...", "...B..", "....B.", ".....B", "....B."), (0, 0)),
    ((".....A", "...BB.", "..B...", ".B....", "B.....", ".B...."), (0, 5)),
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


def _choose_dimensions(
    layout: str,
    *,
    min_src_inner_h: int,
    min_src_inner_w: int,
    min_dst_inner_h: int,
    min_dst_inner_w: int,
) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    src_h_values = list(range(max(SRC_H_RANGE[0], int(min_src_inner_h) + 2), SRC_H_RANGE[1] + 1))
    src_w_values = list(range(max(SRC_W_RANGE[0], int(min_src_inner_w) + 2), SRC_W_RANGE[1] + 1))
    dst_h_values = list(range(max(DST_H_RANGE[0], int(min_dst_inner_h) + 2), DST_H_RANGE[1] + 1))
    dst_w_values = list(range(max(DST_W_RANGE[0], int(min_dst_inner_w) + 2), DST_W_RANGE[1] + 1))

    if not src_h_values or not src_w_values or not dst_h_values or not dst_w_values:
        raise ValueError("no feasible dimensions after inner-size constraints")

    if layout == "side_by_side":
        width_pairs = [
            (src_w, dst_w)
            for src_w in src_w_values
            for dst_w in dst_w_values
            if int(dst_w) <= int(DST_W_SIDE_MAX) and int(src_w) + int(dst_w) + int(gap) <= int(GRID_SIZE)
        ]
        if not width_pairs:
            raise ValueError("no side-by-side width fit")
        src_w, dst_w = random.choice(width_pairs)
        src_h = random.choice(src_h_values)
        dst_h = random.choice(dst_h_values)
        return int(src_h), int(src_w), int(dst_h), int(dst_w)

    height_pairs = [
        (src_h, dst_h)
        for src_h in src_h_values
        for dst_h in dst_h_values
        if int(src_h) + int(dst_h) + int(gap) <= int(GRID_SIZE)
    ]
    if not height_pairs:
        raise ValueError("no stacked height fit")
    src_h, dst_h = random.choice(height_pairs)
    src_w = random.choice(src_w_values)
    dst_w = random.choice(dst_w_values)
    return int(src_h), int(src_w), int(dst_h), int(dst_w)


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
    return (int(src_r), int(src_c)), (int(dst_r), int(dst_c))


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


def _scaled_sprite_patch(sprite: dict, bg: int) -> np.ndarray:
    patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], bg)
    return scale_sprite(patch, SCALE_FACTOR)


def _place_scaled_markers(
    sprite_types: list[dict],
    marker_type_ids: list[int],
    dst_h: int,
    dst_w: int,
    dst_bg: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []

    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _scaled_sprite_patch(sprite, dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg, padding=MARKER_PADDING)
            except ValueError:
                return None
            mr = int(r0 + SCALE_FACTOR * int(sprite["anchor_rc"][0]))
            mc = int(c0 + SCALE_FACTOR * int(sprite["anchor_rc"][1]))
            if any(max(abs(mr - pr), abs(mc - pc)) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((int(mr), int(mc), int(type_id)))
            placed = True
            break
        if not placed:
            return None

    return inner, marker_points


def _place_sources(
    sprite_types: list[dict],
    source_type_ids: list[int],
    src_h: int,
    src_w: int,
    src_bg: int,
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], src_bg)
        placed = False
        for _ in range(SOURCE_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(
                    inner,
                    patch,
                    transparent=src_bg,
                    border=SOURCE_SPRITE_BORDER,
                    padding=SOURCE_SPRITE_PADDING,
                    padding_connectivity=COMPONENT_CONNECTIVITY,
                )
            except ValueError:
                return None
            paste(inner, patch, r0, c0, transparent=src_bg)
            placed = True
            break
        if not placed:
            return None
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
        choices = [c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg and c != dst_border]
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
            "body": int(body),
            "anchor": int(anchor),
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
            "area": sum(ch != "." for row in shape_rows for ch in row),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors)
    ]

    n_markers = random.choice(list(MARKER_COUNT_CHOICES))
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    max_raw_h = max(int(sprite["h"]) for sprite in sprite_types)
    max_raw_w = max(int(sprite["w"]) for sprite in sprite_types)
    max_scaled_h = max(SCALE_FACTOR * int(sprite["h"]) for sprite in sprite_types)
    max_scaled_w = max(SCALE_FACTOR * int(sprite["w"]) for sprite in sprite_types)

    layout = random.choice(LAYOUT_CHOICES)
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(
            layout,
            min_src_inner_h=max_raw_h + 2 * SOURCE_SPRITE_BORDER,
            min_src_inner_w=max_raw_w + 2 * SOURCE_SPRITE_BORDER,
            min_dst_inner_h=max_scaled_h,
            min_dst_inner_w=max_scaled_w,
        )
    except ValueError:
        return None

    marker_result = _place_scaled_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
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
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[type_id]["anchor"])

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
