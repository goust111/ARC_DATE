# Combo variant of a251c730 (index 33)
# combo_spec: D4/0+D10/0+D14/1
# Name: 旋转+有序印刻+多类型

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) concretization / behavior trace
#    - Original task: find the source sprite whose singleton anchor color matches each
#      destination marker, then translate that unrotated sprite so the anchor lands on
#      the marker.
#    - This combo changes three linked steps at once:
#      1. rotation: the marker quadrant inside the destination interior selects 0/90/180/270
#         degree clockwise rotation before alignment;
#      2. ordered stamping: markers are not independent anymore, because all stamps are
#         applied in ascending anchor-color order and later ones overwrite earlier ones;
#      3. more types: there are now 3-4 distinct source sprite types and 4-5 markers, so
#         the destination commonly contains several overlapping rotated footprints.
#    - Concrete trace: suppose anchor colors are {2, 6, 8}, with 8 repeated once. A top-right
#      marker of color 2 receives sprite(2) rotated 90° CW and is stamped first. Then a
#      bottom-left marker of color 6 receives sprite(6) rotated 270° CW and may cover part of
#      sprite(2). Finally two color-8 markers are stamped, each rotated by its own quadrant,
#      and both can overwrite earlier color-2/color-6 pixels. The final visible pattern is
#      therefore not derivable from pure color matching alone: location controls rotation and
#      color rank controls depth.
# b) complexity calibration
#    - The generator uses only clearly asymmetric 3x3-type templates, so quarter-turns stay
#      visually distinct.
#    - It uses exactly 3 or 4 types, and 1 extra repeated marker, giving 4 or 5 total markers.
#      That is enough to create overlap pressure without turning the destination into noise.
#    - At least one repeated anchor color is placed in a different quadrant from its first copy,
#      so one example already shows “same sprite, different rotation”.
# c) implicit-assumption audit for main()
#    - The original stamping loop assumed source templates can be pasted as-is. That fails once
#      rotation is introduced because the anchor coordinate changes after rotation.
#    - The original loop also assumed stamp order is irrelevant. That fails once overlaps are
#      allowed, so iteration order must be explicit and deterministic.
#    - The original source-template extraction by rarest color still holds, because every source
#      sprite remains two-colored with a singleton anchor cell.
#    - Destination detection still relies on singleton components. The generator therefore keeps
#      markers pairwise non-4-adjacent even when their stamped outputs overlap.
# d) placement legality
#    - Output is built first. For each marker occurrence, the generator rotates the patch, checks
#      interior fit, enforces singleton-marker spacing, and scores candidate anchor cells by how
#      much useful overlap they create.
#    - The repeat marker is forced into a different quadrant from its first occurrence.
#    - Source sprites are still placed non-overlapping with padding in the source rectangle.
# e) visual observability
#    - Rotation depends on a visible spatial cue only: destination quadrant.
#    - Ordered overwriting is made observable by rejecting samples where color-sorted stamping
#      equals simple row-major stamping, or where rotated stamping equals unrotated stamping.
#    - The repeated type across different quadrants gives local evidence that color chooses the
#      template while position chooses the orientation.
# f) visual cleanliness
#    - Overlap is required, but the candidate search prefers moderate center-facing overlap rather
#      than total burial, so several sprite fragments remain legible.
#    - Marker count is capped at 5 and sprites stay small, preventing the destination from becoming
#      a solid block of overwritten pixels.
# g) color inferability
#    - Anchor colors still serve as the only source→destination matching key.
#    - No hidden recoloring is added; overlap only changes which already-visible sprite color ends
#      up on top. The generator favors examples where the ordered result differs from other plausible
#      stamping orders, so the color-rank rule has visible consequences.
# h) post-change assumptions main() now makes
#    - Each destination marker lies strictly in one quadrant, not on an ambiguous centerline.
#    - Rotated sprites fit fully inside the destination interior.
#    - Markers remain singleton 4-connected components.
#    - If multiple markers share an anchor color, they all use the same recovered source template.


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


def _build_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
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


def _collect_marker_records(inner: np.ndarray, dst_bg: int) -> list[dict]:
    records = []
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        records.append({"rr": int(rr), "cc": int(cc), "color": int(inner[int(rr), int(cc)])})
    return records


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _stamp_destination(
    marker_records: list[dict],
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    dst_h: int,
    dst_w: int,
    dst_bc: int,
    dst_bg: int,
    *,
    rotate_markers: bool,
    sort_by_anchor: bool,
    return_stats: bool,
) -> np.ndarray | tuple[np.ndarray, dict]:
    out = _build_frame(int(dst_h), int(dst_w), int(dst_bc), int(dst_bg))
    inner_h = int(dst_h - 2)
    inner_w = int(dst_w - 2)

    if sort_by_anchor:
        order = sorted(marker_records, key=lambda m: (int(m["color"]), int(m["rr"]), int(m["cc"])))
    else:
        order = sorted(marker_records, key=lambda m: (int(m["rr"]), int(m["cc"]), int(m["color"])))

    stats = {
        "events": [],
        "overwrite_cells": 0,
        "overlap_cells": 0,
        "n_markers": len(order),
        "distinct_quadrants": set(),
    }
    for rec in order:
        rr = int(rec["rr"])
        cc = int(rec["cc"])
        mcol = int(rec["color"])
        if int(mcol) not in templates:
            continue
        patch, mask, anchor_rc = templates[int(mcol)]
        qturns = _quarter_turns_cw_for_marker(int(rr), int(cc), int(inner_h), int(inner_w))
        if rotate_markers:
            patch, mask, anchor_rc = _rotate_template(patch, mask, anchor_rc, qturns)
        ar, ac = map(int, anchor_rc)
        top = int(1 + int(rr) - int(ar))
        left = int(1 + int(cc) - int(ac))

        overlap_cells = 0
        overwrite_cells = 0
        if return_stats and 0 <= top and 0 <= left and top + patch.shape[0] <= dst_h and left + patch.shape[1] <= dst_w:
            window = out[int(top) : int(top + patch.shape[0]), int(left) : int(left + patch.shape[1])]
            overlap_mask = mask & (window != int(dst_bg))
            overwrite_mask = overlap_mask & (window != patch)
            overlap_cells = int(overlap_mask.sum())
            overwrite_cells = int(overwrite_mask.sum())

        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

        if return_stats:
            stats["events"].append(
                {
                    "color": int(mcol),
                    "rr": int(rr),
                    "cc": int(cc),
                    "qturns": int(qturns),
                    "overlap_cells": int(overlap_cells),
                    "overwrite_cells": int(overwrite_cells),
                }
            )
            stats["overwrite_cells"] += int(overwrite_cells)
            stats["overlap_cells"] += int(overlap_cells)
            stats["distinct_quadrants"].add(int(qturns))

    if return_stats:
        stats["distinct_quadrants"] = len(stats["distinct_quadrants"])
        return out, stats
    return out


def _solve(
    grid: np.ndarray,
    *,
    rotate_markers: bool,
    sort_by_anchor: bool,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates = _build_templates(src_comps)
    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    markers = _collect_marker_records(inner, int(dst_bg))
    return _stamp_destination(
        markers,
        templates,
        int(dr1 - dr + 1),
        int(dc1 - dc + 1),
        int(dst_bc),
        int(dst_bg),
        rotate_markers=bool(rotate_markers),
        sort_by_anchor=bool(sort_by_anchor),
        return_stats=bool(return_stats),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate and ordered-stamp several source sprite types into the destination frame.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a source box with several multi-cell two-color sprites, and a destination box with singleton anchor-colored markers.
    - Each source sprite has one singleton anchor color; the destination reuses those colors as markers, with 3-4 distinct sprite types overall and one type possibly repeated.

    Rule:
    - Recover one template per anchor color from the source box, then match each destination marker to the template with the same anchor color.
    - Determine the marker's quadrant inside the destination interior, rotate the matched sprite by that quarter-turn, align the rotated anchor onto the marker, and stamp it.
    - Apply all stamps in ascending anchor-color order so later colors overwrite earlier overlaps; clear the destination interior first and output only that bordered destination rectangle.

    Key insight: Marker color chooses which sprite to copy, marker position chooses its rotation, and the final overlap pattern shows that anchor-color rank determines which rotated sprite ends up on top.
    """
    return _solve(grid, rotate_markers=True, sort_by_anchor=True)


# --------------- generator (output-first) ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (9, 10)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (12, 18)
SRC_W_RANGE = (12, 18)
DST_H_RANGE = (12, 17)
DST_W_RANGE = (12, 17)
DST_W_SIDE_MAX = 17

SPRITE_TYPE_CHOICES = (3, 4)
SOURCE_SPRITE_EXTRA = (1, 3)
MARKER_PLACE_RETRIES = 48
SOURCE_SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 700

SPRITE_SHAPES = [
    (("A..", "BBB", ".B."), (0, 0)),
    (("..A", "BBB", "B.."), (0, 2)),
    ((".A.", "BBB", "..B"), (0, 1)),
    (("A.B", "BBB", "..."), (0, 0)),
    (("AB.", "B..", "BBB"), (0, 0)),
    ((".AB", "..B", "BBB"), (0, 1)),
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


def _assign_palette(n_types: int) -> tuple[list[int], int, list[int], list[int]] | None:
    wallpaper_n = 4 if n_types == 4 else random.choice(list(WALLPAPER_COLORS))
    total_target = random.choice(list(TOTAL_COLORS_TARGET))
    if int(wallpaper_n + 1 + n_types) > int(total_target):
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, int(wallpaper_n))
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    if len(remaining) < n_types + 1:
        return None
    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]
    extras = remaining[:]
    return wallpaper, int(src_border), [int(c) for c in anchors], [int(c) for c in extras]


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


def _candidate_marker_positions(inner_h: int, inner_w: int, quadrant: int) -> list[tuple[int, int]]:
    rows, cols = _quadrant_ranges(int(inner_h), int(inner_w), int(quadrant))
    if not rows or not cols:
        return []

    row_center = (int(inner_h) - 1) / 2.0
    col_center = (int(inner_w) - 1) / 2.0
    near_rows = sorted(rows, key=lambda r: abs(float(r) - row_center))
    near_cols = sorted(cols, key=lambda c: abs(float(c) - col_center))

    near_rows = near_rows[: min(4, len(near_rows))]
    near_cols = near_cols[: min(4, len(near_cols))]

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for r in near_rows:
        for c in near_cols:
            if (int(r), int(c)) not in seen:
                seen.add((int(r), int(c)))
                out.append((int(r), int(c)))

    fallback = [(int(r), int(c)) for r in rows for c in cols]
    fallback.sort(key=lambda rc: (abs(float(rc[0]) - row_center) + abs(float(rc[1]) - col_center), rc[0], rc[1]))
    for rc in fallback:
        if rc not in seen:
            seen.add(rc)
            out.append(rc)

    return out


def _place_markers(
    sprite_types: list[dict], marker_occurrences: list[tuple[int, int]], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]], dict] | None:
    inner_h = int(dst_h - 2)
    inner_w = int(dst_w - 2)
    if inner_h < 6 or inner_w < 6:
        return None

    inner = np.full((inner_h, inner_w), int(dst_bg), dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    stats = {"overwrite_cells": 0, "overlap_cells": 0, "distinct_quadrants": set()}

    ordered = sorted(
        list(enumerate(marker_occurrences)),
        key=lambda item: (int(sprite_types[int(item[1][0])]["anchor"]), int(item[0])),
    )

    for _occ_id, (type_id, quadrant) in ordered:
        sprite = sprite_types[int(type_id)]
        patch, mask, anchor_rc = _rotate_template(
            sprite["patch"], sprite["mask"], sprite["anchor_rc"], int(quadrant)
        )
        ar, ac = map(int, anchor_rc)
        candidates = _candidate_marker_positions(int(inner_h), int(inner_w), int(quadrant))
        if not candidates:
            return None

        best = None
        for mr, mc in candidates[: max(8, MARKER_PLACE_RETRIES * 2)]:
            if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            top = int(mr - ar)
            left = int(mc - ac)
            if top < 0 or left < 0 or top + patch.shape[0] > inner_h or left + patch.shape[1] > inner_w:
                continue
            window = inner[int(top) : int(top + patch.shape[0]), int(left) : int(left + patch.shape[1])]
            overlap_mask = mask & (window != int(dst_bg))
            overwrite_mask = overlap_mask & (window != patch)
            overlap_cells = int(overlap_mask.sum())
            overwrite_cells = int(overwrite_mask.sum())
            new_cells = int((mask & (window == int(dst_bg))).sum())
            if int(new_cells + overwrite_cells) == 0:
                continue

            score = int(overwrite_cells * 40 + overlap_cells * 8 + min(new_cells, 3))
            if marker_points:
                score += 6 if overlap_cells > 0 else -6
            if overwrite_cells > 0:
                score += 12
            if best is None or int(score) > int(best[0]):
                best = (int(score), int(mr), int(mc), int(top), int(left), int(overlap_cells), int(overwrite_cells))

        if best is None:
            return None

        _score, mr, mc, top, left, overlap_cells, overwrite_cells = best
        paste(inner, patch, int(top), int(left), mask=mask, overwrite=True)
        marker_points.append((int(mr), int(mc), int(type_id)))
        stats["overwrite_cells"] += int(overwrite_cells)
        stats["overlap_cells"] += int(overlap_cells)
        stats["distinct_quadrants"].add(int(quadrant))

    stats["distinct_quadrants"] = len(stats["distinct_quadrants"])
    if int(stats["overwrite_cells"]) <= 0:
        return None
    if int(stats["distinct_quadrants"]) < 3:
        return None
    marker_points.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2])))
    return inner, marker_points, stats


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), int(src_bg), dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        try:
            r0, c0 = random_free_location(
                inner,
                sprite["patch_src"],
                transparent=int(src_bg),
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, sprite["patch_src"], int(r0), int(c0), transparent=int(src_bg))
    return inner


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(int(n_types))
    if palette_result is None:
        return None
    wallpaper, src_border, anchors, extras = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg = random.choice(wallpaper)

    if len(set(wallpaper)) < 2:
        return None
    dst_border_choices = [c for c in wallpaper if c != src_bg]
    if not dst_border_choices:
        return None
    dst_border = random.choice(dst_border_choices)
    if int(dst_bg) == int(dst_border):
        dst_bg_choices = [c for c in wallpaper if c != dst_border]
        if not dst_bg_choices:
            return None
        dst_bg = random.choice(dst_bg_choices)

    body_pool = [c for c in wallpaper if c not in {int(src_bg), int(dst_bg), int(dst_border)}]
    body_pool.extend([c for c in extras if c not in {int(src_border), *map(int, anchors), int(dst_border)}])
    body_pool = [int(c) for c in body_pool if int(c) not in set(map(int, anchors))]
    if not body_pool:
        return None

    sprite_specs = random.sample(SPRITE_SHAPES, int(n_types))
    bodies = [random.choice(body_pool) for _ in range(int(n_types))]
    sprite_types = []
    for type_id, ((shape_rows, anchor_rc), body, anchor) in enumerate(zip(sprite_specs, bodies, anchors)):
        patch_src = _render_sprite(shape_rows, int(body), int(anchor), int(src_bg))
        sprite_types.append(
            {
                "type_id": int(type_id),
                "shape_rows": shape_rows,
                "anchor_rc": (int(anchor_rc[0]), int(anchor_rc[1])),
                "body": int(body),
                "anchor": int(anchor),
                "patch_src": patch_src,
                "patch": _render_sprite(shape_rows, int(body), int(anchor), int(dst_bg)),
                "mask": _render_sprite(shape_rows, 1, 2, 0) != 0,
            }
        )

    sorted_type_ids = sorted(range(int(n_types)), key=lambda tid: int(sprite_types[int(tid)]["anchor"]))
    repeat_type = int(sorted_type_ids[-1] if random.random() < 0.7 else random.choice(sorted_type_ids[1:]))

    base_quadrants = random.sample([0, 1, 2, 3], int(n_types))
    type_quadrants = {int(tid): int(q) for tid, q in zip(range(int(n_types)), base_quadrants)}
    repeat_quadrant_choices = [q for q in [0, 1, 2, 3] if q != int(type_quadrants[int(repeat_type)])]
    repeat_quadrant = random.choice(repeat_quadrant_choices)
    marker_occurrences = [(int(tid), int(type_quadrants[int(tid)])) for tid in range(int(n_types))]
    marker_occurrences.append((int(repeat_type), int(repeat_quadrant)))

    n_source = int(n_types + random.randint(*SOURCE_SPRITE_EXTRA))
    source_type_ids = [i for i, ct in enumerate(_choose_counts(int(n_source), int(n_types))) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if int(src_h * src_w) <= int(dst_h * dst_w):
        return None

    placed = _place_markers(sprite_types, marker_occurrences, int(dst_h), int(dst_w), int(dst_bg))
    if placed is None:
        return None
    out_inner, marker_points, marker_stats = placed

    output = _build_frame(int(dst_h), int(dst_w), int(dst_border), int(dst_bg))
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, int(src_h), int(src_w), int(src_bg))
    if source_inner is None:
        return None

    source_rect = _build_frame(int(src_h), int(src_w), int(src_border), int(src_bg))
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(int(dst_h), int(dst_w), int(dst_border), int(dst_bg))
    for mr, mc, type_id in marker_points:
        dest_rect[1 + int(mr), 1 + int(mc)] = int(sprite_types[int(type_id)]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, int(src_h), int(src_w), int(dst_h), int(dst_w))
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    actual, stats = _solve(grid, rotate_markers=True, sort_by_anchor=True, return_stats=True)
    if not np.array_equal(actual, output):
        return None

    no_order = _solve(grid, rotate_markers=True, sort_by_anchor=False)
    no_rotation = _solve(grid, rotate_markers=False, sort_by_anchor=True)
    if np.array_equal(actual, no_order):
        return None
    if np.array_equal(actual, no_rotation):
        return None
    if int(stats["overwrite_cells"]) <= 0 or int(stats["distinct_quadrants"]) < 3:
        return None
    if int(marker_stats["overwrite_cells"]) <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 combo input after %d attempts" % MAX_GEN_ATTEMPTS)
