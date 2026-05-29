# Combo variant of a251c730 (index 14)
# combo_spec: D2/3+D8/0
# Name: 大稀疏精灵+几何中心锚点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

COMPONENT_CONNECTIVITY = 8

# Design analysis for the combo variant
# -------------------------------------
# analysis:
#   source_task: a251c730
#   concrete_behavior:
#     - The scene still has two bordered rectangles on striped wallpaper: a source
#       frame with reusable sprites and a destination frame with singleton markers.
#     - The source sprites are no longer compact 2x3 / 3x3 blobs. Each type now
#       occupies a 5x5 or 6x6 bounding box and is visibly sparse, with only 5-7
#       filled cells and a hollow geometric center. The stamped output must preserve
#       these holes.
#     - The operational placement anchor is the geometric center of the 5x5 bbox,
#       even though that cell is usually empty. A destination marker therefore marks
#       the center of an implied box, not a painted source pixel.
#     - Because an empty geometric center cannot supply a visible per-type marker
#       color, the fused generator keeps the singleton accent color as the lookup key
#       while moving only the placement anchor to the geometric center. This resolves
#       the D2+D8 conflict without losing the intended "marker lands in a hole"
#       behavior.
#   behavior_trace:
#     - Old task example: a compact sprite with singleton color A at local (0,0)
#       stamps so that A lands exactly on the destination marker.
#     - Fused task example: a sparse 5x5 sprite with singleton color A near one end
#       and an empty center at (2,2) uses A only to identify the template. The
#       destination marker says "place this sprite so the empty center sits here".
#     - Resulting difference: the input marker disappears in the output and the
#       surrounding sparse body appears around that now-empty center. The old solver
#       would instead move the singleton accent onto the marker, producing a visibly
#       shifted stamp.
#   complexity_calibration:
#     - To make the requested 5-7 pixel sparse boxes stable while still spanning a
#       full 5x5/6x6 bbox, source extraction uses 8-connectivity. This allows thin
#       diagonal chains that remain visually sparse without collapsing into multiple
#       objects.
#     - The center cell is forced empty for every template, and the singleton accent
#       is forced off-center, so the center-anchor behavior cannot collapse back to
#       the original rare-pixel anchor.
#     - Marker colors remain unique per type via the singleton accent color, avoiding
#       ambiguity that would arise if empty centers all keyed to the same background.
#   implicit_assumption_audit:
#     - Rectangle detection is unchanged and still only assumes bordered rectangles.
#     - Source component extraction uses 8-connected components so very sparse
#       diagonal templates can still be parsed as one object.
#     - Template extraction no longer assumes the anchor location is itself occupied.
#       `main()` now computes the anchor from bbox geometry alone and uses the rare
#       singleton color only for matching.
#     - Destination scanning still assumes singleton markers on a blank interior. The
#       generator preserves that and ensures each marker color maps to exactly one
#       source template.
#     - Replaying the stamp assumes the full 5x5 patch fits around the marker. The
#       generator guarantees this by placing output stamps first and deriving markers
#       from those placements.
#   placement_legality:
#     - Both source and destination placements reserve full 5x5 or 6x6 boxes, not
#       only the occupied pixels. This prevents sparse templates from overlapping or
#       creating accidental component mergers through touching boundary pixels.
#     - Source placement also keeps a one-cell gap from borders and between reserved
#       boxes so component extraction remains stable.
#   observability_and_cleanliness:
#     - The destination input remains clean: only singleton markers appear inside an
#       otherwise empty destination frame.
#     - In the output, each marker location becomes background, surrounded by the
#       sparse sprite. This center-hole effect is strong enough to infer from a few
#       examples and is the main visual cue for the fused rule.


def _center_rc_from_shape(h: int, w: int) -> tuple[int, int]:
    """Return the floor geometric center of a bbox."""
    return (int((h - 1) // 2), int((w - 1) // 2))


def _center_rc_from_patch(patch: np.ndarray) -> tuple[int, int]:
    """Return the floor geometric center of a cropped component patch."""
    h, w = map(int, patch.shape)
    return _center_rc_from_shape(h, w)


def _shape_singleton_rc(shape_rows: tuple[str, ...], ch: str) -> tuple[int, int]:
    """Locate the unique template character used as the singleton accent."""
    hits = [(r, c) for r, row in enumerate(shape_rows) for c, val in enumerate(row) if val == ch]
    if len(hits) != 1:
        raise ValueError(f"shape must contain exactly one {ch!r}")
    return hits[0]


def _render_sprite(shape_rows: tuple[str, ...], body: int, rare: int, bg: int) -> np.ndarray:
    """Render a sparse sprite template. 'A' is the singleton accent, 'B' the body."""
    char_map = {"B": body, "A": rare}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _validate_shape(shape_rows: tuple[str, ...]) -> None:
    """Sanity-check that a template matches the fused variant constraints."""
    patch = _render_sprite(shape_rows, 1, 2, 0)
    mask = patch != 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("shape must be non-empty")
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    bbox_hw = (int(rr1 - rr0 + 1), int(cc1 - cc0 + 1))
    if bbox_hw not in {(5, 5), (6, 6)}:
        raise ValueError("shape must have exact 5x5 or 6x6 bbox")
    if not 5 <= int(mask.sum()) <= 7:
        raise ValueError("shape must use 5-7 occupied cells")
    cr, cc = _center_rc_from_shape(*bbox_hw)
    if bool(mask[cr, cc]):
        raise ValueError("shape center must be empty")
    if _shape_singleton_rc(shape_rows, "A") == (cr, cc):
        raise ValueError("singleton accent must be off-center")
    comps = components_mask(mask, connectivity=COMPONENT_CONNECTIVITY)
    if len(comps) != 1:
        raise ValueError("shape must be 8-connected")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by centering sparse source sprites on singleton markers.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a source frame with sparse
      5x5/6x6 sprites and a destination frame with singleton color markers.
    - Each source sprite still has a unique singleton accent color for identification,
      but its operational anchor is the geometric center of the full 5x5 bbox.

    Rule:
    - Match each destination marker to the source sprite whose singleton accent color
      is the same.
    - Stamp that sprite so the bbox center, not the singleton accent pixel, lands on
      the marker position.
    - Return only the destination rectangle, cleared to its background before the
      sparse stamps are applied.

    Key insight: the marker chooses a sprite by its rare accent color, but the actual
    translation is controlled by an empty geometric center, so the marker vanishes
    into the sprite's central hole in the output.
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

        for coords in components_mask(inner != int(ibg), connectivity=COMPONENT_CONNECTIVITY):
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
    min_h = 8
    min_w = 8
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if int(g[int(r0), int(c1)]) != int(bc) or int(g[int(r1), int(c0)]) != int(bc) or int(
                        g[int(r1), int(c1)]
                    ) != int(bc):
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

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if not cols:
            continue
        rare_col, rare_count = min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
        if int(rare_count) != 1:
            continue
        patch = comp["patch"]
        mask = comp["mask"]
        ar, ac = _center_rc_from_patch(patch)
        if int(rare_col) not in templates or int(mask.sum()) > int(templates[int(rare_col)][1].sum()):
            templates[int(rare_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

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
        top = int((mr - dr) - int(ar))
        left = int((mc - dc) - int(ac))
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (combo) ---------------

GRID_SIZE = 30
COLORS = tuple(range(1, 10))
WALLPAPER_COLORS = (4, 5)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (15, 19)
SRC_W_RANGE = (15, 18)
DST_H_RANGE = (14, 19)
DST_W_RANGE = (14, 20)
DST_W_SIDE_MAX = 15

SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
MAX_GEN_ATTEMPTS = 400

SOURCE_BOX_BORDER = 1
SOURCE_BOX_PADDING = 1
DEST_BOX_BORDER = 0
DEST_BOX_PADDING = 1

SPRITE_SHAPES_BY_SIZE = {
    5: [
        ("A....", ".BB..", "...B.", "....B", "....B"),
        ("....A", "..BB.", ".B...", "B....", "B...."),
        ("....A", "...B.", "....B", "...B.", "BBB.."),
        ("A....", ".B...", "B....", ".B...", "..BBB"),
    ],
    6: [
        ("A.....", ".BB...", "...B..", "....B.", ".....B", ".....B"),
        (".....A", "....B.", "....B.", "...B..", "..B...", "BB...."),
        ("A.....", "B.....", ".B....", "..B...", "...B..", "....BB"),
        (".....A", ".....B", "....B.", "...B..", "..B...", "BB...."),
    ],
}

for _shape_list in SPRITE_SHAPES_BY_SIZE.values():
    for _shape in _shape_list:
        _validate_shape(_shape)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    """Create a 30x30 striped wallpaper."""
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    """Create a bordered rectangle with interior background."""
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    """Distribute n_total items across n_types bins, each at least 1."""
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _box_capacity(inner_h: int, inner_w: int, box_h: int, box_w: int, border: int, padding: int) -> int:
    """Upper bound on how many full boxes can be packed on an axis-aligned grid."""
    eff_h = int(inner_h) - 2 * int(border)
    eff_w = int(inner_w) - 2 * int(border)
    if eff_h < int(box_h) or eff_w < int(box_w):
        return 0
    rows = int((eff_h + int(padding)) // (int(box_h) + int(padding)))
    cols = int((eff_w + int(padding)) // (int(box_w) + int(padding)))
    return int(rows * cols)


def _choose_dimensions(
    layout: str, src_need: int, dst_need: int, box_h: int, box_w: int
) -> tuple[int, int, int, int]:
    """Pick rectangle sizes that can pack the required number of reserved 5x5 boxes."""
    for _ in range(200):
        gap = random.randint(*GAP_RANGE)
        if layout == "side_by_side":
            src_w = random.randint(*SRC_W_RANGE)
            dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
            if dst_w_max < DST_W_RANGE[0]:
                continue
            dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
            src_h = random.randint(*SRC_H_RANGE)
            dst_h = random.randint(*DST_H_RANGE)
        else:
            src_h = random.randint(*SRC_H_RANGE)
            dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
            if dst_h_max < DST_H_RANGE[0]:
                continue
            dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
            src_w = random.randint(*SRC_W_RANGE)
            dst_w = random.randint(*DST_W_RANGE)

        src_inner_h = int(src_h - 2)
        src_inner_w = int(src_w - 2)
        dst_inner_h = int(dst_h - 2)
        dst_inner_w = int(dst_w - 2)

        if _box_capacity(src_inner_h, src_inner_w, box_h, box_w, SOURCE_BOX_BORDER, SOURCE_BOX_PADDING) < int(src_need):
            continue
        if _box_capacity(dst_inner_h, dst_inner_w, box_h, box_w, DEST_BOX_BORDER, DEST_BOX_PADDING) < int(dst_need):
            continue
        if int(src_h * src_w) <= int(dst_h * dst_w):
            continue
        return int(src_h), int(src_w), int(dst_h), int(dst_w)

    raise ValueError("no dimensions satisfy box-packing constraints")


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute non-overlapping rectangle origins."""
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int]] | None:
    """Allocate wallpaper, borders, and singleton-accent colors from visible colors."""
    wallpaper_n = random.choice(list(WALLPAPER_COLORS))
    if wallpaper_n + 2 + n_types > len(COLORS):
        return None

    wallpaper = random.sample(list(COLORS), wallpaper_n)
    remaining = [c for c in COLORS if c not in wallpaper]
    random.shuffle(remaining)

    if len(remaining) < 2 + n_types:
        return None

    src_border = int(remaining.pop())
    dst_border = int(remaining.pop())
    rare_colors = [int(remaining.pop()) for _ in range(n_types)]
    return wallpaper, src_border, dst_border, rare_colors


def _reserve_and_stamp(
    reserve: np.ndarray,
    visual: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    border: int,
    padding: int,
) -> tuple[int, int] | None:
    """Reserve a full bbox slot, then stamp the sparse visual patch into it."""
    box_h, box_w = map(int, patch.shape)
    box_patch = np.ones((box_h, box_w), dtype=int)
    for _ in range(MARKER_PLACE_RETRIES):
        try:
            r0, c0 = random_free_location(
                reserve,
                box_patch,
                transparent=0,
                border=border,
                padding=padding,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(reserve, box_patch, int(r0), int(c0), transparent=0)
        paste(visual, patch, int(r0), int(c0), mask=mask)
        return int(r0), int(c0)
    return None


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place sparse stamps into the output interior and record center-marker positions."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    reserve = np.zeros_like(inner, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        placed = _reserve_and_stamp(
            reserve,
            inner,
            sprite["patch"],
            sprite["mask"],
            DEST_BOX_BORDER,
            DEST_BOX_PADDING,
        )
        if placed is None:
            return None
        r0, c0 = placed
        mr = int(r0 + sprite["center_rc"][0])
        mc = int(c0 + sprite["center_rc"][1])
        marker_points.append((mr, mc, type_id))
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    """Place sparse source sprites with reserved full 5x5 footprints."""
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    reserve = np.zeros_like(inner, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        placed = _reserve_and_stamp(
            reserve,
            inner,
            sprite["patch"],
            sprite["mask"],
            SOURCE_BOX_BORDER,
            SOURCE_BOX_PADDING,
        )
        if placed is None:
            return None
    return inner


def _gen_attempt(sprite_box_size: int) -> np.ndarray | None:
    """Single attempt to generate a valid combo input grid."""
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, rare_colors = palette_result

    src_bg = int(random.choice(wallpaper))
    dst_bg = int(random.choice(wallpaper))

    body_choices = [int(c) for c in wallpaper if int(c) not in {int(src_bg), int(dst_bg)}]
    if len(set(body_choices)) < n_types:
        return None
    bodies = random.sample(list(dict.fromkeys(body_choices)), n_types)

    sprite_specs = random.sample(SPRITE_SHAPES_BY_SIZE[sprite_box_size], n_types)
    sprite_types = []
    for shape_rows, body, rare in zip(sprite_specs, bodies, rare_colors):
        patch = _render_sprite(shape_rows, int(body), int(rare), 0)
        center_rc = _center_rc_from_shape(*patch.shape)
        mask = patch != 0
        if int(patch[center_rc]) != 0:
            return None
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "patch": patch,
                "mask": mask,
                "center_rc": center_rc,
                "rare_rc": _shape_singleton_rc(shape_rows, "A"),
                "body": int(body),
                "rare": int(rare),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
            }
        )

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    box_h = int(sprite_types[0]["h"])
    box_w = int(sprite_types[0]["w"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout, n_source, n_markers, box_h, box_w)
    except ValueError:
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
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[type_id]["rare"])

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
    preferred_size = random.choice((5, 6))
    fallback_size = 11 - preferred_size

    for sprite_box_size in (preferred_size, fallback_size):
        for _ in range(MAX_GEN_ATTEMPTS // 2):
            grid = _gen_attempt(sprite_box_size)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a251c730 combo input after %d attempts" % MAX_GEN_ATTEMPTS)
