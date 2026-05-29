# Combo variant of a251c730 (index 12)
# combo_spec: D2/0+D4/0
# Name: 三色精灵+旋转对齐

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for combo D2/0 + D4/0
# --------------------------------------
# a) Concrete behavior trace:
#    - Original task: source sprites are two-color, destination shows singleton anchor markers,
#      and the matched sprite is stamped without changing orientation.
#    - Combo task: each source sprite is tri-color with counts (anchor=1, decor=2, body=3/4).
#      A destination marker still matches by anchor color, but its quadrant inside the destination
#      interior now selects a clockwise rotation before the anchor is aligned.
#    - Example sketch:
#        source type:        marker in UR quadrant:
#          A B C              old rule -> same patch translated
#          B B C              combo    -> patch rotated 90 deg CW, then translated
#      The extra decoration color makes the quarter-turn visually distinguishable.
# b) Complexity calibration:
#    - The generator always uses 3 destination markers spread across 3 distinct quadrants.
#      With at most 2 sprite types, some anchor color must repeat across different quadrants,
#      so rotation cannot be dismissed as an accidental one-off.
#    - Sprite shapes are chosen only from tri-color templates whose 4 rotations are all distinct.
#      This prevents degeneration into orientation-invariant stamps.
# c) Implicit-assumption audit:
#    - The original solver assumed the anchor is merely "the least frequent visible color" and
#      that all remaining pixels can be treated as one body color bucket. That fails here because
#      the decoration color must be preserved and also helps reveal rotation direction.
#    - The original stamping code also assumed the template anchor offset is fixed. After adding
#      rotation, anchor coordinates must be rotated together with the patch and mask.
#    - The combo solver therefore accepts only 3-color components with counts (1,2,3/4), extracts
#      the unique rarest anchor color, then rotates the whole template according to marker quadrant.
# d) Placement legality:
#    - Destination generation is output-first. Rotated full patches are placed in the destination
#      interior first, with bounds and overlap checks, and only their anchor pixels remain in input.
#    - Source sprites still use padding=1 so 4-connected component extraction remains reliable.
# e) Visual observability:
#    - All control variables are human-visible: anchor color, decoration layout, and marker quadrant.
#      No color-id ordering or hidden metadata is used.
#    - Repeated anchor colors across different quadrants show the same sprite type in multiple
#      orientations inside one sample, which exposes the quadrant->rotation rule directly.
# f) Visual cleanliness:
#    - At most one marker is assigned to each quadrant and output-first placement rejects overlap,
#      so the destination stays readable even after rotation.
#    - The source rectangle remains sparse enough that the tri-color role frequencies are visible.
# g) Fusion-specific non-degeneracy target:
#    - For a healthy sample, at least one anchor color must appear in multiple quadrants and the
#      output produced by quadrant-driven rotation must differ from the output of plain translation.
#      The validation script below measures both conditions after generation.
# h) New solver assumptions that the generator guarantees:
#    - Every usable source component is a connected tri-color sprite with one anchor pixel.
#    - Every destination marker lies strictly inside one quadrant, never on the center row/column.
#    - The rotated sprite footprint for each marker fits entirely inside the destination interior.


def _interior_bg(g: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(g: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != int(ibg), connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
        patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
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


def _find_src_dst(g: np.ndarray) -> tuple[tuple, tuple]:
    H, W = g.shape
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
                    ibg = _interior_bg(g, (int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects, "no bordered rectangles found"

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and comps and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source and destination rectangles"
    return src, dst


def _classify_tricolor_sprite(colors: dict[int, int]) -> tuple[int, int, int] | None:
    if len(colors) != 3:
        return None
    ordered = sorted(((int(cnt), int(col)) for col, cnt in colors.items()), key=lambda x: x[0])
    counts = [int(cnt) for cnt, _col in ordered]
    if counts[0] != 1 or counts[1] != 2 or counts[2] not in (3, 4):
        return None
    if not (counts[0] < counts[1] < counts[2]):
        return None
    return int(ordered[0][1]), int(ordered[1][1]), int(ordered[2][1])


def _quarter_turns_cw_for_marker(rr: int, cc: int, inner_h: int, inner_w: int) -> int:
    top_half = int(2 * int(rr) + 1) < int(inner_h)
    left_half = int(2 * int(cc) + 1) < int(inner_w)
    if top_half and left_half:
        return 0
    if top_half and not left_half:
        return 1
    if not top_half and not left_half:
        return 2
    return 3


def _rotate_template(
    patch: np.ndarray, mask: np.ndarray, anchor_rc: tuple[int, int], quarter_turns_cw: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    k_ccw = (-int(quarter_turns_cw)) % 4
    anchor_mask = np.zeros_like(mask, dtype=bool)
    anchor_mask[int(anchor_rc[0]), int(anchor_rc[1])] = True
    rot_patch = np.rot90(patch, k=int(k_ccw))
    rot_mask = np.rot90(mask, k=int(k_ccw))
    rot_anchor_mask = np.rot90(anchor_mask, k=int(k_ccw))
    ar, ac = map(int, np.argwhere(rot_anchor_mask)[0])
    return rot_patch, rot_mask, (int(ar), int(ac))


def _solve(grid: np.ndarray, *, rotate_markers: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        roles = _classify_tricolor_sprite({int(k): int(v) for k, v in comp["colors"].items()})
        if roles is None:
            continue
        anchor_col, _decor_col, _body_col = roles
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    assert templates, "no valid tri-color source templates found"

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    inner_h, inner_w = map(int, inner.shape)
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mcol = int(inner[int(rr), int(cc)])
        if int(mcol) not in templates:
            continue
        patch, mask, anchor_rc = templates[int(mcol)]
        if rotate_markers:
            qturns = _quarter_turns_cw_for_marker(int(rr), int(cc), int(inner_h), int(inner_w))
            patch, mask, anchor_rc = _rotate_template(patch, mask, anchor_rc, qturns)
        ar, ac = map(int, anchor_rc)
        top = int(1 + int(rr) - int(ar))
        left = int(1 + int(cc) - int(ac))
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the destination frame by matching tri-color sprites and rotating them by marker quadrant.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper: a source box with disconnected tri-color sprites and a destination box with singleton anchor-color markers.
    - Every source sprite has exactly three visible colors: a unique anchor pixel, a two-pixel decoration color, and a larger body color.

    Rule:
    - Identify each source sprite's anchor as its unique least-frequent color and use that color to match destination markers.
    - For each marker, determine which quadrant of the destination interior it occupies, rotate the matched sprite by 0/90/180/270 degrees clockwise accordingly, and then align the rotated anchor onto the marker.
    - Clear the destination interior to its background first and output only that bordered destination rectangle after all rotated stamps are applied.

    Key insight: The rare anchor color still provides the sprite identity, but the marker position contributes a second control signal; the decoration color breaks rotational symmetry so the quarter-turn can be inferred from the visible color layout.
    """
    return _solve(grid, rotate_markers=True)


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
MARKER_COUNT_RANGE = (3, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 32
MARKER_LAYOUT_RETRIES = 48
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


def _shape_has_four_distinct_rotations(shape_rows: tuple[str, ...]) -> bool:
    arr = np.array([list(row) for row in shape_rows])
    rotations = {tuple(map(tuple, np.rot90(arr, k=int(k)))) for k in range(4)}
    return len(rotations) == 4


for _shape in SPRITE_SHAPES:
    _anchor_rc, _counts = _shape_info(_shape)
    assert len(_shape) in (2, 3)
    assert len(_shape[0]) == 3
    assert _shape_has_four_distinct_rotations(_shape)


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, decor: int, bg: int) -> np.ndarray:
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], int, int] | None:
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
    return wallpaper, src_border, dst_border, anchors, int(body), int(decor)


def _quadrant_ranges(inner_h: int, inner_w: int, quadrant: int) -> tuple[list[int], list[int]]:
    top_rows = list(range(0, int(inner_h // 2)))
    bottom_rows = list(range(int((inner_h + 1) // 2), int(inner_h)))
    left_cols = list(range(0, int(inner_w // 2)))
    right_cols = list(range(int((inner_w + 1) // 2), int(inner_w)))
    if quadrant == 0:
        return top_rows, left_cols
    if quadrant == 1:
        return top_rows, right_cols
    if quadrant == 2:
        return bottom_rows, right_cols
    return bottom_rows, left_cols


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    inner_h, inner_w = int(dst_h - 2), int(dst_w - 2)
    if inner_h < 4 or inner_w < 4:
        return None

    for _ in range(MARKER_LAYOUT_RETRIES):
        inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
        marker_points: list[tuple[int, int, int, int]] = []
        quadrants = random.sample([0, 1, 2, 3], len(marker_type_ids))
        placement_order = list(zip(marker_type_ids, quadrants))
        random.shuffle(placement_order)

        placed_all = True
        for type_id, quadrant in placement_order:
            sprite = sprite_types[int(type_id)]
            base_patch = _render_sprite(
                sprite["shape_rows"], sprite["body"], sprite["anchor"], sprite["decor"], dst_bg
            )
            base_mask = base_patch != int(dst_bg)
            patch, mask, (ar, ac) = _rotate_template(base_patch, base_mask, sprite["anchor_rc"], int(quadrant))

            rows, cols = _quadrant_ranges(inner_h, inner_w, int(quadrant))
            if not rows or not cols:
                placed_all = False
                break
            candidates = [(int(r), int(c)) for r in rows for c in cols]
            random.shuffle(candidates)

            placed = False
            for mr, mc in candidates[: max(1, MARKER_PLACE_RETRIES * 4)]:
                top = int(mr - ar)
                left = int(mc - ac)
                if top < 0 or left < 0 or top + patch.shape[0] > inner_h or left + patch.shape[1] > inner_w:
                    continue
                if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc, _q, _t in marker_points):
                    continue
                window = inner[int(top) : int(top + patch.shape[0]), int(left) : int(left + patch.shape[1])]
                if bool((mask & (window != int(dst_bg))).any()):
                    continue
                paste(inner, patch, int(top), int(left), mask=mask, overwrite=True)
                marker_points.append((int(mr), int(mc), int(quadrant), int(type_id)))
                placed = True
                break

            if not placed:
                placed_all = False
                break

        if placed_all:
            colors_by_quadrant = {(int(q), int(sprite_types[int(t)]["anchor"])) for _r, _c, q, t in marker_points}
            if len(colors_by_quadrant) != len(marker_points):
                continue
            marker_points.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[3])))
            return inner, marker_points

    return None


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
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
        paste(inner, patch, int(r0), int(c0), transparent=src_bg)
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
    for mr, mc, _quadrant, type_id in marker_points:
        dest_rect[1 + int(mr), 1 + int(mc)] = int(sprite_types[int(type_id)]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    try:
        solved = main(grid)
        baseline = _solve(grid, rotate_markers=False)
    except AssertionError:
        return None

    if not np.array_equal(solved, output):
        return None
    if np.array_equal(solved, baseline):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate combo a251c730/12 after {MAX_GEN_ATTEMPTS} attempts")
