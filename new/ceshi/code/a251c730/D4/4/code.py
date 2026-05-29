# Variant of a251c730
# Dimension: D4 — 变换：锚点对齐方式 (transform)
# Change: 将精灵平移使其锚点像素与目标标记像素完全重合，精灵保持原始朝向不变... → 缩放后对齐

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location, scale_sprite

# Design analysis for the D4 variant
# - Concrete behavior:
#   - The input still contains a source rectangle with 2-color sprites and a destination
#     rectangle with singleton marker pixels.
#   - The anchor color still identifies which sprite to copy, but the copied sprite is
#     enlarged 2x2 per source cell before being stamped into the output destination box.
#   - To keep the alignment rule integer-valued on the pixel grid, the singleton marker
#     denotes the upper-left cell of the enlarged 2x2 anchor block.
# - Flow delta on one marker:
#   - Original: top-left = marker - raw_anchor_rc, then paste raw patch.
#   - Variant: top-left = marker - 2 * raw_anchor_rc, then paste the 2x-scaled patch.
#   - The matching step is unchanged; only the stamping coordinates and patch size change.
# - Complexity calibration:
#   - The task still uses only 1-2 sprite types and 2-3 markers, so there is repeated
#     evidence for the scaling rule without introducing many branches.
#   - The sprites remain asymmetric enough for shape identity, while the 2x size jump is
#     visually obvious even when a sprite happens to be mirror-symmetric.
# - Old assumptions that must be revised:
#   - The original solver assumed the pasted patch size equaled the source bbox size.
#     That no longer holds; offsets must be computed in scaled coordinates.
#   - The original template extraction ignored color 0 to avoid masked-out zeros.
#     Here the component color counts come from real component cells, so color 0 is safe
#     and should not be excluded.
# - Placement legality guaranteed by the generator:
#   - Destination placement reserves full enlarged bounding boxes with padding, so enlarged
#     stamps stay in-bounds and do not visually interlock through background holes.
#   - Marker pixels are also kept 4-disconnected so the destination rectangle still looks
#     like a box of singleton prompts rather than multi-cell objects.
# - Visual readability:
#   - The destination input remains sparse because it only shows singleton markers.
#   - The output is clean because enlarged sprites are separated by at least one cell and
#     never clipped by the border.
# - New solver assumptions checked against generation:
#   - Each sprite type has exactly one least-frequent color (the anchor).
#   - The marker indicates the top-left cell of the scaled anchor block.
#   - The destination interior is large enough to contain every enlarged sprite fully.

SCALE_FACTOR = 2


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _random_box_location(
    occupied: np.ndarray,
    patch_h: int,
    patch_w: int,
    *,
    border: int = 0,
    padding: int = 0,
) -> tuple[int, int]:
    """Pick a free top-left location while reserving the full bounding box."""
    H, W = occupied.shape
    border = int(border)
    padding = int(padding)
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, H - border - patch_h + 1):
        for c0 in range(border, W - border - patch_w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(H, r0 + patch_h + padding)
            cc1 = min(W, c0 + patch_w + padding)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        raise ValueError("no free enlarged bounding-box location")
    return random.choice(candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project source sprites into a destination frame via anchor-colored singleton markers.

    Input structure:
    - Two bordered rectangles sit on the wallpaper: a source box with multi-cell two-color sprites and a destination box with one-pixel markers.
    - Within each source sprite, the unique least-frequent color is the anchor; destination markers reuse those anchor colors.

    Rule:
    - Match each destination marker to the source sprite with the same anchor color.
    - Enlarge the matched sprite by a factor of 2 in both dimensions, then place it so the marker lands on the upper-left cell of the scaled anchor block.
    - Return the destination rectangle after resetting its interior to the destination background and stamping all enlarged sprites.

    Key insight: The anchor color still selects the correct template, but the final offset must be computed in scaled coordinates rather than raw sprite coordinates.
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
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        if not cols:
            continue
        min_freq = min(cols.values())
        anchor_cols = [col for col, freq in cols.items() if int(freq) == int(min_freq)]
        if len(anchor_cols) != 1:
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


# --------------- generator (O2) ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 16)
SRC_W_RANGE = (8, 15)
DST_H_RANGE = (12, 20)
DST_W_RANGE = (12, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 40
MARKER_BOX_PADDING = 1
SOURCE_SPRITE_PADDING = 1

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


def _choose_dimensions(
    layout: str,
    *,
    min_dst_inner_h: int,
    min_dst_inner_w: int,
) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    dst_h_min = max(DST_H_RANGE[0], int(min_dst_inner_h) + 2)
    dst_w_min = max(DST_W_RANGE[0], int(min_dst_inner_w) + 2)

    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w, DST_W_RANGE[1])
        if dst_w_max < dst_w_min:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(dst_w_min, dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        if DST_H_RANGE[1] < dst_h_min:
            raise ValueError("no side-by-side height fit")
        dst_h = random.randint(dst_h_min, DST_H_RANGE[1])
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < dst_h_min:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(dst_h_min, dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        if DST_W_RANGE[1] < dst_w_min:
            raise ValueError("no stacked width fit")
        dst_w = random.randint(dst_w_min, DST_W_RANGE[1])

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
    occupied = np.zeros(inner.shape, dtype=bool)
    marker_points: list[tuple[int, int, int]] = []

    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _scaled_sprite_patch(sprite, dst_bg)
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = _random_box_location(occupied, ph, pw, padding=MARKER_BOX_PADDING)
            except ValueError:
                return None
            mr = int(r0 + SCALE_FACTOR * int(sprite["anchor_rc"][0]))
            mc = int(c0 + SCALE_FACTOR * int(sprite["anchor_rc"][1]))
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            marker_points.append((mr, mc, type_id))
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

    max_scaled_h = max(SCALE_FACTOR * int(sprite["h"]) for sprite in sprite_types)
    max_scaled_w = max(SCALE_FACTOR * int(sprite["w"]) for sprite in sprite_types)
    min_dst_inner_h = min(DST_H_RANGE[1] - 2, max_scaled_h + (n_markers - 1) * 2)
    min_dst_inner_w = min(DST_W_RANGE[1] - 2, max_scaled_w + (n_markers - 1) * 2)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(
            layout,
            min_dst_inner_h=min_dst_inner_h,
            min_dst_inner_w=min_dst_inner_w,
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
    raise RuntimeError(f"failed to generate a251c730 D4 input after {MAX_GEN_ATTEMPTS} attempts")
