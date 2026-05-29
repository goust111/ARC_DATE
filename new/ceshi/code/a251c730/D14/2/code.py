# Variant of a251c730
# Dimension: D14 — 精灵类型多样性 (meta)
# Change: 1-2种不同的精灵类型，从11种预定义形状模板中选取，同类型精灵形状完全相同... → 精灵形状运行时随机生成

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D14 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     before: source sprites were drawn from a fixed catalog of 11 small templates
#     after: each task constructs 1-2 sprite types at runtime by growing a fresh 4-connected shape
#     invariant: copies of the same type remain identical within one task, and destination markers still encode types by anchor color
#   pixel_level_behavior:
#     source_rect:
#       still contains 2-4 multi-cell sprites
#       each sprite still uses exactly 2 colors:
#         - one singleton anchor pixel
#         - one body color on all remaining occupied cells
#       difference:
#         - occupied mask is no longer chosen from a hard-coded list
#         - the tight bbox can vary across both axes in the range 2..4 after cropping
#         - every generated type is rejected if it matches any of the 11 original template signatures
#     dest_rect:
#       still contains only singleton anchor-colored markers in the input
#       output is still the cleared destination frame with full sprites stamped so that anchor aligns to each marker
#   flow_trace:
#     original_local_step:
#       generator picked template like ('AB.','BBB') and reused it verbatim
#       solver isolated source components, found the unique rare color, then stamped the stored patch
#     variant_local_step:
#       generator grows a fresh connected occupied set, picks one occupied cell as the unique anchor,
#       colors all other occupied cells with the body color, and then reuses that exact patch for source + output construction
#       solver still isolates source components and still uses the unique least-frequent color as anchor,
#       but now the recovered patch can be any connected shape rather than one of 11 memorized masks
#     practical_difference:
#       matching by anchor color survives unchanged
#       memorizing template identities no longer helps; the instance itself is the only source of shape information
#   complexity_calibration:
#     kept:
#       - 1-2 sprite types
#       - 2-3 destination markers
#       - 2-4 source sprites
#       - same wallpaper / two-rectangle scaffold
#     changed carefully:
#       - random shapes are still constrained to small connected masks with 5-8 occupied cells
#       - each generated task repeats each type at least once, so a human sees reusable sprite evidence rather than one-off noise
#       - when there are 2 types, their signatures are forced distinct
#   implicit_assumption_audit:
#     original_main_dependencies:
#       - source components are connected foreground blobs: still true by construction
#       - one color inside each sprite is uniquely rare and therefore the anchor: enforced as singleton anchor pixel
#       - tie-breaking by color id is harmless: no longer relied upon; main now requires a unique least-frequency color instead of using color-id fallback
#       - color 0 is not a sprite color because patch holes were stored as 0: fixed by storing holes as -1 and using the component mask explicitly
#       - source background is the most common interior color: generator now checks this and rejects crowded source layouts
#       - destination markers are singletons and full stamps fit in-frame: generator still guarantees this by placing the full patch first on a blank inner canvas
#   placement_legality:
#     source placement:
#       sprites are placed with 1-cell border and 1-cell padding inside the source interior
#     destination placement:
#       the output-view patch is placed first on a blank destination interior with 1-cell padding between stamps
#       marker coordinates are then read from anchor positions of those placed patches
#     solver compatibility:
#       input destination rectangle shows only singleton markers, so solver can still detect it as the rectangle whose largest component has size 1
#   visual_observability:
#     - random masks are rejected if they match the original template catalog, so the D14 change is visible rather than merely latent
#     - masks are cropped to their tight bbox and allowed to use wider 4-column / 4-row shapes, making per-task variation easy to spot
#     - only anchor color drives matching; no invisible ordering or color-id arithmetic is used
#   neatness:
#     - destination stamps are kept non-overlapping with padding, so outputs remain readable
#     - source sprites are spaced apart, so each exemplar is visually separable
#   color_inference:
#     - colors are unchanged semantically from the original task
#     - anchor color still means "which source sprite to copy", body color remains part of the copied sprite appearance
#     - no extra color remapping rule is introduced
#   new_main_assumptions_checked_against_generator:
#     - each source component must have exactly one least-frequent color: guaranteed by singleton anchor generation
#     - the chosen source rectangle must have background as the dominant interior color: checked before accepting a sample
#     - duplicate copies of a type may exist; choosing the largest recovered patch per anchor remains safe because copies are identical within a task

# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by copying runtime-generated source sprites onto matching markers.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper: one source panel with full sprites, one destination panel with only anchor-colored singleton markers.
    - Each sprite is a connected two-color shape created for this instance; its unique singleton color is the anchor.

    Rule:
    - Recover each source sprite as a template, using its uniquely rare color as both the type key and alignment anchor.
    - For every destination marker, pick the source sprite with the same anchor color and stamp it so the anchor lands on the marker.
    - Return the destination rectangle alone, with its interior reset before stamping.

    Key insight: The source panel already contains the exact per-instance sprite geometry, so the solver only needs to detect each connected shape and align it by its singleton anchor color.
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
            patch[~mask] = -1

            vals = [int(inner[r, c]) for r, c in coords]
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
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        max_sz = max((int(c["size"]) for c in comps), default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and comps and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        if not cols:
            continue
        min_count = min(cols.values())
        rare_cols = [col for col, count in cols.items() if int(count) == int(min_count)]
        if len(rare_cols) != 1:
            continue
        anchor_col = int(rare_cols[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere(mask & (patch == anchor_col))
        if anchors.shape[0] != 1:
            continue
        ar, ac = map(int, anchors[0])
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col][1].sum()):
            templates[anchor_col] = (patch.copy(), mask.copy(), (ar, ac))

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


# --------------- generator ---------------

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
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1
DEST_SPRITE_PADDING = 1
SPRITE_BBOX_H_RANGE = (2, 4)
SPRITE_BBOX_W_RANGE = (3, 4)
SPRITE_AREA_RANGE = (5, 8)
SPRITE_GEN_RETRIES = 120

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 400

ORIGINAL_TEMPLATE_SPECS = [
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


def _mask_signature(mask: np.ndarray, anchor_rc: tuple[int, int]) -> tuple:
    coords = tuple(sorted(tuple(map(int, rc)) for rc in np.argwhere(mask)))
    return (tuple(map(int, mask.shape)), coords, tuple(map(int, anchor_rc)))


def _original_template_signatures() -> set[tuple]:
    sigs: set[tuple] = set()
    for rows, anchor_rc in ORIGINAL_TEMPLATE_SPECS:
        mask = np.array([[ch != "." for ch in row] for row in rows], dtype=bool)
        sigs.add(_mask_signature(mask, anchor_rc))
    return sigs


ORIGINAL_TEMPLATE_SIGNATURES = _original_template_signatures()


def _render_sprite(mask: np.ndarray, anchor_rc: tuple[int, int], body: int, anchor: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(body)
    patch[int(anchor_rc[0]), int(anchor_rc[1])] = int(anchor)
    return patch


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), int(bg), dtype=int)
    rect[0, :] = rect[-1, :] = int(border)
    rect[:, 0] = rect[:, -1] = int(border)
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    counts = [1] * int(n_types)
    for _ in range(int(n_total) - int(n_types)):
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

    return wallpaper, int(src_border), int(dst_border), [int(a) for a in anchors], [int(c) for c in body_pool], forced_body


def _bg_is_dominant(inner: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, inner.flatten()))
    bg_count = int(counts.get(int(bg), 0))
    return bg_count > max((count for color, count in counts.items() if int(color) != int(bg)), default=0)


def _neighbors_in_bbox(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((int(nr), int(nc)))
    return out


def _random_sprite_mask(used_signatures: set[tuple]) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(SPRITE_GEN_RETRIES):
        h = random.randint(*SPRITE_BBOX_H_RANGE)
        w = random.randint(*SPRITE_BBOX_W_RANGE)
        max_area = min(SPRITE_AREA_RANGE[1], h * w)
        if max_area < SPRITE_AREA_RANGE[0]:
            continue
        target_area = random.randint(SPRITE_AREA_RANGE[0], max_area)

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier = {
                nbr
                for cell in cells
                for nbr in _neighbors_in_bbox(cell, h, w)
                if nbr not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        if len(cells) != target_area:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 == r0 or c1 == c0:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if int(mask.sum()) < SPRITE_AREA_RANGE[0]:
            continue

        occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        anchor_rc = random.choice(occupied)
        sig = _mask_signature(mask, anchor_rc)
        if sig in ORIGINAL_TEMPLATE_SIGNATURES or sig in used_signatures:
            continue

        body_degrees = []
        for rc in occupied:
            deg = sum(mask[nr, nc] for nr, nc in _neighbors_in_bbox(rc, mask.shape[0], mask.shape[1]))
            body_degrees.append(int(deg))
        if max(body_degrees) < 2:
            continue

        return mask, (int(anchor_rc[0]), int(anchor_rc[1]))
    return None


def _build_sprite_types(
    n_types: int, anchors: list[int], body_pool: list[int], src_bg: int, dst_bg: int, src_border: int, dst_border: int
) -> list[dict] | None:
    choices = [c for c in body_pool if c not in anchors and c not in {src_bg, dst_bg, src_border, dst_border}]
    if not choices:
        return None

    bodies: list[int] = []
    random.shuffle(choices)
    for _ in range(n_types):
        available = [c for c in choices if c not in bodies]
        pick_from = available or choices
        bodies.append(int(random.choice(pick_from)))

    used_signatures: set[tuple] = set()
    sprite_types: list[dict] = []
    for anchor, body in zip(anchors, bodies):
        sprite = _random_sprite_mask(used_signatures)
        if sprite is None:
            return None
        mask, anchor_rc = sprite
        used_signatures.add(_mask_signature(mask, anchor_rc))
        sprite_types.append(
            {
                "mask": mask,
                "anchor_rc": anchor_rc,
                "body": int(body),
                "anchor": int(anchor),
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "area": int(mask.sum()),
            }
        )
    return sprite_types


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), int(dst_bg), dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["mask"], sprite["anchor_rc"], sprite["body"], sprite["anchor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(
                    inner,
                    patch,
                    transparent=dst_bg,
                    padding=DEST_SPRITE_PADDING,
                )
            except ValueError:
                return None
            mr = int(r0 + sprite["anchor_rc"][0])
            mc = int(c0 + sprite["anchor_rc"][1])
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((mr, mc, int(type_id)))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), int(src_bg), dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["mask"], sprite["anchor_rc"], sprite["body"], sprite["anchor"], src_bg)
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
    wallpaper, src_border, dst_border, anchors, body_pool, _forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_types = _build_sprite_types(n_types, anchors, body_pool, src_bg, dst_bg, src_border, dst_border)
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

    max_src_pixels = sum(sprite_types[type_id]["area"] for type_id in source_type_ids)
    src_inner_area = (src_h - 2) * (src_w - 2)
    if max_src_pixels * 2 >= src_inner_area:
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
    if not _bg_is_dominant(source_inner, src_bg):
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
    raise RuntimeError(f"failed to generate a251c730 D14 input after {MAX_GEN_ATTEMPTS} attempts")
