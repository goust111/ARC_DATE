# Combo variant of a251c730 (index 21)
# combo_spec: D4/0+D10/0
# Name: 旋转+有序印刻

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) concretization / behavior trace
#    - Base task: find the source rectangle, extract one sprite template per anchor color,
#      then stamp each matched sprite into the destination rectangle so the anchor pixel
#      lands on the singleton marker.
#    - D4 adds a visible spatial control: the marker's quadrant inside the destination
#      interior determines a clockwise quarter-turn before alignment.
#    - D10 adds a temporal control: stamps are no longer conceptually simultaneous.
#      They are applied one color-group at a time, ordered by anchor-color value, so
#      later groups overwrite earlier groups where rotated footprints overlap.
#    - Small trace:
#         source has anchors 1 and 8. A marker of color 1 sits near the upper-right
#         quadrant, so its sprite rotates 90 degrees clockwise. A nearby marker of color 8
#         sits in the lower-right quadrant, so that sprite rotates 180 degrees. If their
#         rotated masks overlap, the color-8 stamp is applied second and covers the same
#         cells that color-1 had filled first.
# b) complexity calibration
#    - The generator fixes exactly two sprite types and three markers. One anchor color is
#      repeated in two different quadrants, so rotation is supported by within-example
#      repetition instead of being a one-off coincidence.
#    - Overlap is required between different anchor colors, but capped to a small number
#      of cells so the output stays readable rather than dissolving into clutter.
#    - Anchor colors are chosen from one fixed visible ordered pair across samples, making
#      the color-priority relation stable enough to observe from examples instead of being
#      a hidden per-instance permutation.
# c) implicit-assumption audit for main()
#    - Template extraction still relies on the anchor being the rarest color in a source
#      component; the generator keeps each sprite two-colored with exactly one anchor cell.
#    - The original direct-stamp assumption breaks twice: rotation changes the anchor's
#      relative coordinates, and overlapping stamps make iteration order semantically
#      meaningful. The solver therefore materializes marker entries first, sorts them by
#      anchor color, then pastes in that order.
#    - Marker cells must stay singleton components inside the destination input rectangle.
#      The generator keeps markers non-adjacent even though the eventual rotated stamps may
#      overlap in the output.
# d) placement legality
#    - Each marker is chosen from a quadrant-specific set of anchor cells where the rotated
#      patch fits entirely inside the destination interior.
#    - Source sprites remain non-overlapping and padded from edges. Destination markers are
#      also pairwise non-adjacent in Manhattan distance.
# e) visual observability
#    - Rotation is driven only by visible position, not by hidden metadata.
#    - Ordered overwrite is forced to matter: the generator rejects any sample where the
#      anchor-sorted result matches a row-major stamping baseline.
# f) visual cleanliness
#    - Only one overlap relation between different colors is needed; large pileups are
#      rejected. This keeps the final figure legible while still demonstrating coverage.
# g) change inferability
#    - The generator also rejects any sample where removing rotation yields the same output.
#      That prevents the combo from collapsing into a pure ordering variant.
# h) post-change assumptions now guaranteed by generate_input()
#    - There are exactly two anchor colors, exactly three markers, three distinct marker
#      quadrants, at least one repeated anchor color across quadrants, nonzero cross-color
#      overlap after rotation, and a measurable difference versus both row-major stamping
#      and no-rotation stamping.


ANCHOR_COLORS = (1, 8)
GRID_SIZE = 30
WALLPAPER_COLOR_COUNT = 4
GAP_RANGE = (1, 4)
SRC_H_RANGE = (10, 19)
SRC_W_RANGE = (10, 18)
DST_H_RANGE = (9, 15)
DST_W_RANGE = (9, 15)
DST_W_SIDE_MAX = 15
SOURCE_SPRITE_RANGE = (2, 4)
SOURCE_SPRITE_PADDING = 1
MARKER_COUNT = 3
MARKER_LAYOUT_RETRIES = 80
MARKER_POSITION_RETRIES = 120
MARKER_CANDIDATE_POOL = 12
MAX_CROSS_COLOR_OVERLAP = 5
MAX_GEN_ATTEMPTS = 400

SPRITE_SHAPES = [
    (("A..", "BBB", ".B."), (0, 0)),
    (("..A", "BBB", "B.."), (0, 2)),
    ((".A.", "BBB", "..B"), (0, 1)),
    (("A.B", "BBB", "..."), (0, 0)),
    (("AB.", "B..", "BBB"), (0, 0)),
    ((".AB", "..B", "BBB"), (0, 1)),
]


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

    assert rects
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


def _extract_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if not cols:
            continue
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))
    return templates


def _prepare_marker_entries(
    inner: np.ndarray,
    dst_bg: int,
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    *,
    rotate: bool,
) -> list[dict]:
    inner_h, inner_w = map(int, inner.shape)
    entries: list[dict] = []
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mcol = int(inner[int(rr), int(cc)])
        if int(mcol) not in templates:
            continue
        patch, mask, anchor_rc = templates[int(mcol)]
        if rotate:
            qturns = _quarter_turns_cw_for_marker(int(rr), int(cc), int(inner_h), int(inner_w))
            patch, mask, anchor_rc = _rotate_template(patch, mask, anchor_rc, qturns)
        ar, ac = map(int, anchor_rc)
        entries.append(
            {
                "rr": int(rr),
                "cc": int(cc),
                "color": int(mcol),
                "patch": patch,
                "mask": mask,
                "anchor_rc": (int(ar), int(ac)),
            }
        )
    return entries


def _sorted_entries(entries: list[dict], sort_mode: str) -> list[dict]:
    if sort_mode == "anchor":
        return sorted(entries, key=lambda e: (int(e["color"]), int(e["rr"]), int(e["cc"])))
    if sort_mode == "reverse_anchor":
        return sorted(entries, key=lambda e: (-int(e["color"]), int(e["rr"]), int(e["cc"])))
    if sort_mode == "rowmajor":
        return sorted(entries, key=lambda e: (int(e["rr"]), int(e["cc"]), int(e["color"])))
    raise ValueError(f"unsupported sort_mode: {sort_mode}")


def _stamp_entries(out: np.ndarray, entries: list[dict]) -> np.ndarray:
    for entry in entries:
        ar, ac = map(int, entry["anchor_rc"])
        top = int(1 + int(entry["rr"]) - int(ar))
        left = int(1 + int(entry["cc"]) - int(ac))
        paste(out, entry["patch"], int(top), int(left), mask=entry["mask"], overwrite=True, clip=True)
    return out


def _destination_from_grid(
    grid: np.ndarray, *, rotate: bool, sort_mode: str, return_entries: bool = False
) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst
    templates = _extract_templates(src_comps)

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    entries = _prepare_marker_entries(inner, int(dst_bg), templates, rotate=rotate)
    entries = _sorted_entries(entries, sort_mode)
    _stamp_entries(out, entries)
    if return_entries:
        return out, entries
    return out


def analyze_behavior(grid: np.ndarray) -> dict[str, int]:
    rotated_sorted, entries = _destination_from_grid(grid, rotate=True, sort_mode="anchor", return_entries=True)
    rotated_rowmajor = _destination_from_grid(grid, rotate=True, sort_mode="rowmajor")
    unrotated_sorted = _destination_from_grid(grid, rotate=False, sort_mode="anchor")

    entry_cells: list[tuple[int, set[tuple[int, int]]]] = []
    for entry in entries:
        ar, ac = map(int, entry["anchor_rc"])
        top = int(1 + int(entry["rr"]) - int(ar))
        left = int(1 + int(entry["cc"]) - int(ac))
        cells = {
            (int(top) + int(r), int(left) + int(c))
            for r, c in np.argwhere(entry["mask"])
        }
        entry_cells.append((int(entry["color"]), cells))

    cross_overlap = 0
    max_pair_overlap = 0
    for i in range(len(entry_cells)):
        col_i, cells_i = entry_cells[i]
        for j in range(i + 1, len(entry_cells)):
            col_j, cells_j = entry_cells[j]
            if int(col_i) == int(col_j):
                continue
            overlap = int(len(cells_i & cells_j))
            cross_overlap += overlap
            max_pair_overlap = max(max_pair_overlap, overlap)

    return {
        "ordered_vs_rowmajor_diff": int(np.count_nonzero(rotated_sorted != rotated_rowmajor)),
        "rotation_vs_no_rotation_diff": int(np.count_nonzero(rotated_sorted != unrotated_sorted)),
        "cross_color_overlap": int(cross_overlap),
        "max_pair_overlap": int(max_pair_overlap),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into the destination frame after rotating and color-ordering them.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a source box containing multi-cell two-color sprites and a destination box containing singleton anchor-color markers.
    - Each source sprite has one least-frequent anchor color, and destination markers reuse those same anchor colors.

    Rule:
    - Match each marker to the source sprite with the same anchor color.
    - Rotate the sprite according to the marker's quadrant inside the destination interior, then align the rotated anchor pixel to the marker cell.
    - Clear the destination interior first, stamp markers in ascending anchor-color order, and let later stamps overwrite earlier ones where rotated footprints overlap.

    Key insight: The marker color still selects the sprite, but the marker position controls orientation and the anchor-color ordering controls which rotated stamp stays visible in overlap zones.
    """
    return _destination_from_grid(grid, rotate=True, sort_mode="anchor")


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    char_map = {"B": body, "A": anchor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    stripe = np.array(palette * ((GRID_SIZE + len(palette) - 1) // len(palette)), dtype=int)[:GRID_SIZE]
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


def _assign_palette() -> tuple[list[int], int, int, int, int, list[int], list[int]] | None:
    anchors = list(ANCHOR_COLORS)
    non_anchor = [c for c in range(10) if c not in anchors]
    wallpaper = random.sample(non_anchor, WALLPAPER_COLOR_COUNT)
    remaining = [c for c in non_anchor if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    if random.random() < 0.6:
        dst_border = remaining.pop()
    else:
        dst_border = random.choice(wallpaper)

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    body_choices = [c for c in wallpaper + remaining if c not in {src_bg, dst_bg, src_border, dst_border}]
    if len(body_choices) < 2:
        return None
    bodies = random.sample(body_choices, 2)
    return wallpaper, int(src_border), int(dst_border), int(src_bg), int(dst_bg), anchors, bodies


def _quadrant_ranges(inner_h: int, inner_w: int, quadrant: int) -> tuple[list[int], list[int]]:
    top_rows = list(range(0, int(inner_h // 2)))
    bot_rows = list(range(int((inner_h + 1) // 2), int(inner_h)))
    left_cols = list(range(0, int(inner_w // 2)))
    right_cols = list(range(int((inner_w + 1) // 2), int(inner_w)))
    if quadrant == 0:
        return top_rows, left_cols
    if quadrant == 1:
        return top_rows, right_cols
    if quadrant == 2:
        return bot_rows, right_cols
    return bot_rows, left_cols


def _candidate_marker_positions(
    inner_h: int, inner_w: int, quadrant: int, patch: np.ndarray, anchor_rc: tuple[int, int]
) -> list[tuple[int, int]]:
    rows, cols = _quadrant_ranges(inner_h, inner_w, quadrant)
    h, w = map(int, patch.shape)
    ar, ac = map(int, anchor_rc)
    candidates: list[tuple[int, int]] = []
    for mr in rows:
        top = int(mr - ar)
        if top < 0 or top + h > inner_h:
            continue
        for mc in cols:
            left = int(mc - ac)
            if left < 0 or left + w > inner_w:
                continue
            candidates.append((int(mr), int(mc)))
    random.shuffle(candidates)
    center_r = inner_h / 2.0
    center_c = inner_w / 2.0
    candidates.sort(
        key=lambda rc: (
            abs((float(rc[0]) + 0.5) - center_r) + abs((float(rc[1]) + 0.5) - center_c),
            rc[0],
            rc[1],
        )
    )
    return candidates


def _marker_cells(mask: np.ndarray, marker_rc: tuple[int, int], anchor_rc: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = map(int, anchor_rc)
    top = int(marker_rc[0] - ar)
    left = int(marker_rc[1] - ac)
    return {(int(top) + int(r), int(left) + int(c)) for r, c in np.argwhere(mask)}


def _placements_to_inner(
    placements: list[dict], inner_h: int, inner_w: int, dst_bg: int, *, sort_mode: str, rotate: bool
) -> np.ndarray:
    entries: list[dict] = []
    for p in placements:
        if rotate:
            patch = p["rot_patch"]
            mask = p["rot_mask"]
            anchor_rc = p["rot_anchor_rc"]
        else:
            patch = p["base_patch"]
            mask = p["base_mask"]
            anchor_rc = p["base_anchor_rc"]
        entries.append(
            {
                "rr": int(p["mr"]),
                "cc": int(p["mc"]),
                "color": int(p["anchor_color"]),
                "patch": patch,
                "mask": mask,
                "anchor_rc": anchor_rc,
            }
        )
    entries = _sorted_entries(entries, sort_mode)
    out = np.full((inner_h + 2, inner_w + 2), int(dst_bg), dtype=int)
    _stamp_entries(out, entries)
    return out[1:-1, 1:-1]


def _marker_layout_metrics(placements: list[dict]) -> tuple[int, int]:
    cross_overlap = 0
    max_pair_overlap = 0
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            if int(placements[i]["anchor_color"]) == int(placements[j]["anchor_color"]):
                continue
            overlap = int(len(placements[i]["cells"] & placements[j]["cells"]))
            cross_overlap += overlap
            max_pair_overlap = max(max_pair_overlap, overlap)
    return int(cross_overlap), int(max_pair_overlap)


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    inner_h, inner_w = int(dst_h - 2), int(dst_w - 2)
    if inner_h < 5 or inner_w < 5:
        return None

    for _ in range(MARKER_LAYOUT_RETRIES):
        quadrants = random.sample([0, 1, 2, 3], MARKER_COUNT)
        type_ids = marker_type_ids[:]
        random.shuffle(type_ids)

        specs: list[dict] = []
        feasible = True
        for type_id, quadrant in zip(type_ids, quadrants):
            sprite = sprite_types[int(type_id)]
            base_patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
            base_mask = base_patch != int(dst_bg)
            rot_patch, rot_mask, rot_anchor_rc = _rotate_template(base_patch, base_mask, sprite["anchor_rc"], quadrant)
            candidates = _candidate_marker_positions(inner_h, inner_w, quadrant, rot_patch, rot_anchor_rc)
            if not candidates:
                feasible = False
                break
            specs.append(
                {
                    "type_id": int(type_id),
                    "quadrant": int(quadrant),
                    "anchor_color": int(sprite["anchor"]),
                    "base_patch": base_patch,
                    "base_mask": base_mask,
                    "base_anchor_rc": tuple(map(int, sprite["anchor_rc"])),
                    "rot_patch": rot_patch,
                    "rot_mask": rot_mask,
                    "rot_anchor_rc": rot_anchor_rc,
                    "candidates": candidates,
                }
            )
        if not feasible:
            continue

        spec_order = sorted(range(len(specs)), key=lambda i: len(specs[i]["candidates"]))
        for _ in range(MARKER_POSITION_RETRIES):
            chosen: dict[int, tuple[int, int]] = {}
            used_markers: list[tuple[int, int]] = []
            ok = True
            for idx in spec_order:
                candidates = specs[idx]["candidates"][: min(MARKER_CANDIDATE_POOL, len(specs[idx]["candidates"]))]
                feasible_pts = [
                    (int(mr), int(mc))
                    for mr, mc in candidates
                    if all(abs(int(mr) - pr) + abs(int(mc) - pc) > 1 for pr, pc in used_markers)
                ]
                if not feasible_pts:
                    ok = False
                    break
                mr, mc = random.choice(feasible_pts)
                chosen[idx] = (int(mr), int(mc))
                used_markers.append((int(mr), int(mc)))
            if not ok:
                continue

            placements: list[dict] = []
            for idx, spec in enumerate(specs):
                mr, mc = chosen[idx]
                cells = _marker_cells(spec["rot_mask"], (int(mr), int(mc)), spec["rot_anchor_rc"])
                placements.append(
                    {
                        **spec,
                        "mr": int(mr),
                        "mc": int(mc),
                        "cells": cells,
                    }
                )

            cross_overlap, max_pair_overlap = _marker_layout_metrics(placements)
            if cross_overlap <= 0 or max_pair_overlap > MAX_CROSS_COLOR_OVERLAP:
                continue

            anchor_sorted = _placements_to_inner(placements, inner_h, inner_w, dst_bg, sort_mode="anchor", rotate=True)
            rowmajor = _placements_to_inner(placements, inner_h, inner_w, dst_bg, sort_mode="rowmajor", rotate=True)
            no_rotation = _placements_to_inner(
                placements, inner_h, inner_w, dst_bg, sort_mode="anchor", rotate=False
            )
            if np.array_equal(anchor_sorted, rowmajor):
                continue
            if np.array_equal(anchor_sorted, no_rotation):
                continue

            marker_points = [
                (int(p["mr"]), int(p["mc"]), int(p["quadrant"]), int(p["type_id"]))
                for p in placements
            ]
            marker_points.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3])))
            return anchor_sorted, marker_points

    return None


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
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
    palette = _assign_palette()
    if palette is None:
        return None
    wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, bodies = palette

    sprite_specs = random.sample(SPRITE_SHAPES, 2)
    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": int(body),
            "anchor": int(anchor),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors)
    ]

    repeated_type = random.randrange(2)
    single_type = 1 - repeated_type
    marker_type_ids = [int(repeated_type), int(repeated_type), int(single_type)]
    n_source = random.randint(max(2, len(sprite_types)), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, len(sprite_types))) for _ in range(ct)]
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

    if not np.array_equal(main(grid), output):
        return None

    metrics = analyze_behavior(grid)
    if metrics["ordered_vs_rowmajor_diff"] <= 0:
        return None
    if metrics["rotation_vs_no_rotation_diff"] <= 0:
        return None
    if metrics["cross_color_overlap"] <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo a251c730 input after %d attempts" % MAX_GEN_ATTEMPTS)
