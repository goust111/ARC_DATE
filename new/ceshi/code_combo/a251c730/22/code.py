# Combo variant of a251c730 (index 22)
# combo_spec: D4/0+D11/1
# Name: 旋转+移动轨迹

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) concretization / behavior trace
#    - Original task:
#      1. detect the source rectangle with multi-cell sprites and the destination
#         rectangle with singleton markers;
#      2. match by anchor color;
#      3. translate the source template so its anchor lands on the marker;
#      4. output only the cropped destination rectangle.
#    - New combo behavior:
#      1. keep the same source/destination detection and color matching;
#      2. before stamping, rotate the sprite by marker quadrant inside the
#         destination interior: UL=0, UR=90 cw, LR=180, LL=270 cw;
#      3. treat each source sprite copy as moving to one same-color marker;
#      4. erase the moved sprite from the source rectangle, stamp the rotated
#         sprite in the destination rectangle, and paint a straight body-colored
#         trail across the wallpaper gap between the two rectangles;
#      5. output the full 30x30 grid, not a crop.
#    - Example trace: if a source sprite with anchor color 8 sits in the source
#      box and its marker 8 is in the destination's lower-left quadrant, the
#      output removes that source copy, rotates the sprite 270 cw, stamps it at
#      the marker, and draws a body-colored line through the wallpaper corridor
#      linking the two rectangles.
# b) complexity calibration
#    - The generator fixes 3 markers in 3 distinct quadrants. This guarantees at
#      least two non-zero rotations per sample and makes the rotation rule
#      visually obvious.
#    - With 1-2 sprite types, one type necessarily repeats. Repeated anchor
#      colors appearing at different quadrants show that color matching alone is
#      insufficient; position also matters.
#    - Trails are restricted to one horizontal or vertical corridor family,
#      keeping the movement rule visible without introducing many path choices.
# c) implicit-assumption audit for main()
#    - The original solver assumed one template per anchor color was enough and
#      no source instance identity was needed. That breaks once multiple copies
#      of the same color can move to multiple same-color markers. The new solver
#      therefore keeps per-instance source anchors and pairs copies/markers by
#      ordered row or column within each color.
#    - The original output canvas was the cleared destination box. That no longer
#      holds; the new solver must preserve the whole wallpapered grid while
#      modifying both rectangles and the corridor between them.
#    - The original template could be pasted unchanged. Now the anchor location
#      changes after rotation, so rotation must happen before alignment.
# d) placement legality
#    - Destination markers are placed output-first by checking the rotated patch
#      bounds and occupancy, so rotated stamps never overlap or leave the box.
#    - Source copies are placed with a one-cell margin from the source interior
#      edge and a one-cell bbox gap from each other.
#    - Rectangles are aligned side-by-side or stacked so each moving sprite has a
#      clean horizontal or vertical wallpaper corridor.
# e) visual observability
#    - Rotation depends only on visible marker quadrant, never on hidden ids.
#    - Movement is encoded by visible disappearance from source, appearance in
#      destination, and a body-colored trail across wallpaper.
#    - Pairing repeated colors is human-readable because matched copies share the
#      same row corridor (side-by-side) or column corridor (stacked).
# f) visual cleanliness
#    - Trails use unique rows or columns globally, so they do not overlap.
#    - Only one marker per quadrant is used, and destination stamps are checked
#      for non-overlap, preventing ambiguous merges.
# g) color inferability
#    - Anchor colors keep the original lookup role.
#    - Trail color is the sprite's majority/body color, which is already visible
#      inside the source sprite, so no extra hidden color remapping is added.
# h) post-change assumptions main() now makes
#    - Rectangles are separated purely horizontally or purely vertically; the
#      generator enforces this by aligned placement.
#    - For each anchor color, source copy count equals marker count.
#    - Same-color copies are paired by sorted row (horizontal motion) or sorted
#      column (vertical motion); the generator guarantees those coordinates are
#      unique and matched.


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


def _extract_templates_and_instances(
    src_rect: tuple,
) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    (sr, sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src_rect
    templates: dict[int, dict] = {}
    instances: dict[int, list[dict]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        if not cols:
            continue
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        body_col = int(max(((k, v) for k, v in cols.items() if int(k) != int(anchor_col)), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        rr0, cc0, _rr1, _cc1 = map(int, comp["bbox"])
        instance = {
            "patch": patch.copy(),
            "mask": mask.copy(),
            "anchor_rc": (int(ar), int(ac)),
            "anchor_color": int(anchor_col),
            "body_color": int(body_col),
            "global_anchor": (int(sr + 1 + rr0 + ar), int(sc + 1 + cc0 + ac)),
            "coords": [(int(sr + 1 + rr0 + r), int(sc + 1 + cc0 + c)) for r, c in np.argwhere(mask)],
        }
        instances.setdefault(int(anchor_col), []).append(instance)
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)]["mask"].sum()):
            templates[int(anchor_col)] = {
                "patch": patch.copy(),
                "mask": mask.copy(),
                "anchor_rc": (int(ar), int(ac)),
                "body_color": int(body_col),
            }
    return templates, instances


def _movement_axis(src_rect: tuple[int, int, int, int], dst_rect: tuple[int, int, int, int]) -> str:
    sr0, sc0, sr1, sc1 = map(int, src_rect)
    dr0, dc0, dr1, dc1 = map(int, dst_rect)
    if int(sc1) < int(dc0) or int(dc1) < int(sc0):
        return "horizontal"
    if int(sr1) < int(dr0) or int(dr1) < int(sr0):
        return "vertical"
    raise AssertionError("generator should separate rectangles horizontally or vertically")


def _pair_sources_and_markers(
    instances: dict[int, list[dict]],
    markers: dict[int, list[tuple[int, int]]],
    axis: str,
) -> list[tuple[dict, tuple[int, int]]]:
    pairs: list[tuple[dict, tuple[int, int]]] = []
    for acol, src_list in instances.items():
        dst_list = markers.get(int(acol), [])
        key = (lambda item: (int(item["global_anchor"][0]), int(item["global_anchor"][1]))) if axis == "horizontal" else (
            lambda item: (int(item["global_anchor"][1]), int(item["global_anchor"][0]))
        )
        key_marker = (lambda rc: (int(rc[0]), int(rc[1]))) if axis == "horizontal" else (lambda rc: (int(rc[1]), int(rc[0])))
        src_sorted = sorted(src_list, key=key)
        dst_sorted = sorted(dst_list, key=key_marker)
        assert len(src_sorted) == len(dst_sorted)
        for src_inst, marker_rc in zip(src_sorted, dst_sorted):
            pairs.append((src_inst, (int(marker_rc[0]), int(marker_rc[1]))))
    return pairs


def _draw_trail(
    out: np.ndarray,
    *,
    axis: str,
    src_anchor: tuple[int, int],
    dst_anchor: tuple[int, int],
    src_rect: tuple[int, int, int, int],
    dst_rect: tuple[int, int, int, int],
    color: int,
) -> None:
    sr0, sc0, sr1, sc1 = map(int, src_rect)
    dr0, dc0, dr1, dc1 = map(int, dst_rect)
    srow, scol = map(int, src_anchor)
    drow, dcol = map(int, dst_anchor)
    if axis == "horizontal":
        assert int(srow) == int(drow)
        if int(sc1) < int(dc0):
            c_start, c_end = int(sc1 + 1), int(dc0 - 1)
        else:
            c_start, c_end = int(dc1 + 1), int(sc0 - 1)
        if c_start <= c_end:
            out[int(srow), int(c_start) : int(c_end) + 1] = int(color)
    else:
        assert int(scol) == int(dcol)
        if int(sr1) < int(dr0):
            r_start, r_end = int(sr1 + 1), int(dr0 - 1)
        else:
            r_start, r_end = int(dr1 + 1), int(sr0 - 1)
        if r_start <= r_end:
            out[int(r_start) : int(r_end) + 1, int(scol)] = int(color)


def _solve(grid: np.ndarray, *, rotate_markers: bool, draw_trails: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    src_rect, _src_bc, src_bg, _src_comps = src
    dst_rect, _dst_bc, dst_bg, _dst_comps = dst
    axis = _movement_axis(src_rect, dst_rect)

    templates, instances = _extract_templates_and_instances(src)
    marker_lists: dict[int, list[tuple[int, int]]] = {}
    dr, dc, dr1, dc1 = map(int, dst_rect)
    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    inner_h, inner_w = map(int, inner.shape)
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        acol = int(inner[int(rr), int(cc)])
        marker_lists.setdefault(int(acol), []).append((int(dr + 1 + rr), int(dc + 1 + cc)))

    out = g.copy()
    for src_list in instances.values():
        for inst in src_list:
            for r, c in inst["coords"]:
                out[int(r), int(c)] = int(src_bg)
    out[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)] = int(dst_bg)

    pairs = _pair_sources_and_markers(instances, marker_lists, axis)
    for src_inst, (mr, mc) in pairs:
        acol = int(src_inst["anchor_color"])
        template = templates[int(acol)]
        patch = template["patch"].copy()
        mask = template["mask"].copy()
        anchor_rc = template["anchor_rc"]
        if rotate_markers:
            qturns = _quarter_turns_cw_for_marker(int(mr - dr - 1), int(mc - dc - 1), int(inner_h), int(inner_w))
            patch, mask, anchor_rc = _rotate_template(patch, mask, anchor_rc, qturns)
        ar, ac = map(int, anchor_rc)
        paste(out, patch, int(mr - ar), int(mc - ac), mask=mask, overwrite=True, clip=False)
        if draw_trails:
            _draw_trail(
                out,
                axis=axis,
                src_anchor=src_inst["global_anchor"],
                dst_anchor=(int(mr), int(mc)),
                src_rect=src_rect,
                dst_rect=dst_rect,
                color=int(template["body_color"]),
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move source sprites into a destination frame with quadrant-driven rotations and wallpaper trails.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper: the source box contains several two-color sprites, and the destination box contains singleton anchor-color markers.
    - Copies of a sprite share one rare anchor color and one majority body color; repeated anchor colors may appear multiple times in both boxes.

    Rule:
    - Match each destination marker to one source sprite copy of the same anchor color, pairing repeated copies by shared row order or shared column order depending on whether the rectangles are side-by-side or stacked.
    - Rotate the matched sprite according to the marker's quadrant inside the destination interior, stamp the rotated anchor onto the marker, erase the source copy, and draw a straight trail in the sprite's body color across the wallpaper gap between the rectangles.
    - Preserve the full grid: both frames remain in place, the source interior is cleared where sprites moved away, and the destination interior shows the rotated arrivals.

    Key insight: The anchor color still identifies which sprite family to use, but the final output also depends on two spatial signals: marker quadrant decides orientation, and rectangle alignment decides the movement corridor that receives the body-colored trail.
    """
    return _solve(grid, rotate_markers=True, draw_trails=True)


# --------------- generator ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (10, 19)
SRC_W_RANGE = (10, 18)
DST_H_RANGE = (10, 19)
DST_W_RANGE = (10, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT = 3
MAX_GEN_ATTEMPTS = 400
MARKER_LAYOUT_RETRIES = 80
SOURCE_LAYOUT_RETRIES = 80

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


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


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


def _place_rectangles_aligned(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        top = random.randint(0, GRID_SIZE - max(src_h, dst_h))
        left_margin = random.randint(0, GRID_SIZE - src_w - dst_w - gap)
        if random.random() < 0.5:
            src_pos = (int(top), int(left_margin))
            dst_pos = (int(top), int(left_margin + src_w + gap))
        else:
            dst_pos = (int(top), int(left_margin))
            src_pos = (int(top), int(left_margin + dst_w + gap))
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        left = random.randint(0, GRID_SIZE - max(src_w, dst_w))
        top_margin = random.randint(0, GRID_SIZE - src_h - dst_h - gap)
        if random.random() < 0.5:
            src_pos = (int(top_margin), int(left))
            dst_pos = (int(top_margin + src_h + gap), int(left))
        else:
            dst_pos = (int(top_margin), int(left))
            src_pos = (int(top_margin + dst_h + gap), int(left))
    return src_pos, dst_pos


def _assign_palette(n_types: int) -> tuple[list[int], int, int, int, int, list[int], list[int]] | None:
    wallpaper_n = 4 if n_types == 2 else random.choice(list(WALLPAPER_COLORS))
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    if len(remaining) < 2 + n_types + n_types:
        return None
    src_border = remaining.pop()
    dst_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]
    bodies = [remaining.pop() for _ in range(n_types)]
    dst_bg = random.choice(wallpaper)
    src_bg_choices = [c for c in wallpaper if c != dst_bg]
    if not src_bg_choices:
        src_bg_choices = wallpaper[:]
    src_bg = random.choice(src_bg_choices)
    return wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, bodies


def _marker_type_ids(n_types: int) -> list[int]:
    if n_types == 1:
        return [0, 0, 0]
    repeated = random.randrange(n_types)
    ids = [0, 1]
    ids.append(int(repeated))
    random.shuffle(ids)
    return ids


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


def _source_axis_ok(
    sprite: dict,
    coord: int,
    *,
    layout: str,
    src_inner_h: int,
    src_inner_w: int,
) -> bool:
    patch_h, patch_w = sprite["patch"].shape
    ar, ac = map(int, sprite["anchor_rc"])
    if layout == "side_by_side":
        top = int(coord - ar)
        return 1 <= top and top + patch_h <= int(src_inner_h - 1)
    left = int(coord - ac)
    return 1 <= left and left + patch_w <= int(src_inner_w - 1)


def _place_markers(
    sprite_types: list[dict],
    marker_type_ids: list[int],
    dst_h: int,
    dst_w: int,
    dst_bg: int,
    layout: str,
    shared_limit: int,
    src_inner_h: int,
    src_inner_w: int,
) -> tuple[np.ndarray, list[dict]] | None:
    inner_h, inner_w = int(dst_h - 2), int(dst_w - 2)
    if inner_h < 4 or inner_w < 4:
        return None

    for _ in range(MARKER_LAYOUT_RETRIES):
        inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
        points: list[dict] = []
        used_shared: set[int] = set()
        used_markers: list[tuple[int, int]] = []
        quadrants = random.sample([0, 1, 2, 3], len(marker_type_ids))
        order = list(zip(marker_type_ids, quadrants))
        random.shuffle(order)

        placed_all = True
        for type_id, quadrant in order:
            sprite = sprite_types[int(type_id)]
            patch, mask, (ar, ac) = _rotate_template(sprite["patch"], sprite["mask"], sprite["anchor_rc"], quadrant)
            rows, cols = _quadrant_ranges(inner_h, inner_w, quadrant)
            if not rows or not cols:
                placed_all = False
                break
            candidates = [(int(r), int(c)) for r in rows for c in cols]
            random.shuffle(candidates)
            placed = False
            for mr, mc in candidates:
                shared = int(mr) if layout == "side_by_side" else int(mc)
                if shared >= int(shared_limit) or shared in used_shared:
                    continue
                if not _source_axis_ok(
                    sprite,
                    shared,
                    layout=layout,
                    src_inner_h=src_inner_h,
                    src_inner_w=src_inner_w,
                ):
                    continue
                top = int(mr - ar)
                left = int(mc - ac)
                if top < 0 or left < 0 or top + patch.shape[0] > inner_h or left + patch.shape[1] > inner_w:
                    continue
                if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc in used_markers):
                    continue
                window = inner[int(top) : int(top + patch.shape[0]), int(left) : int(left + patch.shape[1])]
                if bool((mask & (window != int(dst_bg))).any()):
                    continue
                paste(inner, patch, int(top), int(left), mask=mask, overwrite=True)
                used_shared.add(int(shared))
                used_markers.append((int(mr), int(mc)))
                points.append(
                    {
                        "mr": int(mr),
                        "mc": int(mc),
                        "quadrant": int(quadrant),
                        "type_id": int(type_id),
                    }
                )
                placed = True
                break
            if not placed:
                placed_all = False
                break

        if placed_all and len(points) == len(marker_type_ids):
            points.sort(key=lambda d: (int(d["mr"]), int(d["mc"]), int(d["type_id"])))
            return inner, points
    return None


def _boxes_separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, box_a)
    br0, bc0, br1, bc1 = map(int, box_b)
    return (
        int(ar1 + gap) < int(br0)
        or int(br1 + gap) < int(ar0)
        or int(ac1 + gap) < int(bc0)
        or int(bc1 + gap) < int(ac0)
    )


def _place_sources(
    sprite_types: list[dict], marker_points: list[dict], src_h: int, src_w: int, src_bg: int, layout: str
) -> tuple[np.ndarray, list[dict]] | None:
    inner_h, inner_w = int(src_h - 2), int(src_w - 2)
    inner = np.full((inner_h, inner_w), src_bg, dtype=int)
    placements: list[dict] = []
    used_boxes: list[tuple[int, int, int, int]] = []

    for _ in range(SOURCE_LAYOUT_RETRIES):
        inner[:, :] = int(src_bg)
        placements.clear()
        used_boxes.clear()
        ok = True
        order = marker_points[:]
        random.shuffle(order)
        for mp in order:
            sprite = sprite_types[int(mp["type_id"])]
            patch = sprite["patch"]
            patch_h, patch_w = patch.shape
            ar, ac = map(int, sprite["anchor_rc"])
            if layout == "side_by_side":
                top = int(mp["mr"] - ar)
                left_candidates = list(range(1, int(inner_w - patch_w)))
                random.shuffle(left_candidates)
                placed = False
                for left in left_candidates:
                    box = (int(top), int(left), int(top + patch_h - 1), int(left + patch_w - 1))
                    if top < 1 or top + patch_h > int(inner_h - 1):
                        continue
                    if any(not _boxes_separated(box, prev, gap=1) for prev in used_boxes):
                        continue
                    paste(inner, patch, int(top), int(left), transparent=src_bg, overwrite=True)
                    used_boxes.append(box)
                    placements.append(
                        {
                            "type_id": int(mp["type_id"]),
                            "anchor_inner": (int(mp["mr"]), int(left + ac)),
                        }
                    )
                    placed = True
                    break
            else:
                left = int(mp["mc"] - ac)
                top_candidates = list(range(1, int(inner_h - patch_h)))
                random.shuffle(top_candidates)
                placed = False
                for top in top_candidates:
                    box = (int(top), int(left), int(top + patch_h - 1), int(left + patch_w - 1))
                    if left < 1 or left + patch_w > int(inner_w - 1):
                        continue
                    if any(not _boxes_separated(box, prev, gap=1) for prev in used_boxes):
                        continue
                    paste(inner, patch, int(top), int(left), transparent=src_bg, overwrite=True)
                    used_boxes.append(box)
                    placements.append(
                        {
                            "type_id": int(mp["type_id"]),
                            "anchor_inner": (int(top + ar), int(mp["mc"])),
                        }
                    )
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and len(placements) == len(marker_points):
            return inner.copy(), placements[:]
    return None


def _expected_output(
    wallpaper: np.ndarray,
    src_rect: np.ndarray,
    dst_rect: np.ndarray,
    src_pos: tuple[int, int],
    dst_pos: tuple[int, int],
    src_bg: int,
    dst_bg: int,
    sprite_types: list[dict],
    marker_points: list[dict],
    source_placements: list[dict],
    layout: str,
) -> np.ndarray:
    out = np.array(wallpaper, copy=True)
    blank_src = src_rect.copy()
    blank_src[1:-1, 1:-1] = int(src_bg)
    blank_dst = dst_rect.copy()
    blank_dst[1:-1, 1:-1] = int(dst_bg)
    paste(out, blank_src, int(src_pos[0]), int(src_pos[1]))
    paste(out, blank_dst, int(dst_pos[0]), int(dst_pos[1]))

    grouped_markers: dict[int, list[dict]] = {}
    for mp in marker_points:
        grouped_markers.setdefault(int(mp["type_id"]), []).append(mp)
    grouped_sources: dict[int, list[dict]] = {}
    for sp in source_placements:
        grouped_sources.setdefault(int(sp["type_id"]), []).append(sp)

    for type_id in grouped_markers:
        mlist = grouped_markers[int(type_id)]
        slist = grouped_sources[int(type_id)]
        if layout == "side_by_side":
            mlist = sorted(mlist, key=lambda d: (int(d["mr"]), int(d["mc"])))
            slist = sorted(slist, key=lambda d: (int(d["anchor_inner"][0]), int(d["anchor_inner"][1])))
        else:
            mlist = sorted(mlist, key=lambda d: (int(d["mc"]), int(d["mr"])))
            slist = sorted(slist, key=lambda d: (int(d["anchor_inner"][1]), int(d["anchor_inner"][0])))
        sprite = sprite_types[int(type_id)]
        for mp, sp in zip(mlist, slist):
            patch, mask, (ar, ac) = _rotate_template(
                sprite["patch"], sprite["mask"], sprite["anchor_rc"], int(mp["quadrant"])
            )
            mg_r = int(dst_pos[0] + 1 + mp["mr"])
            mg_c = int(dst_pos[1] + 1 + mp["mc"])
            paste(out, patch, int(mg_r - ar), int(mg_c - ac), mask=mask, overwrite=True)

            body_col = int(sprite["body"])
            if layout == "side_by_side":
                sg_r = int(src_pos[0] + 1 + sp["anchor_inner"][0])
                if int(src_pos[1]) < int(dst_pos[1]):
                    c0, c1 = int(src_pos[1] + src_rect.shape[1]), int(dst_pos[1] - 1)
                else:
                    c0, c1 = int(dst_pos[1] + dst_rect.shape[1]), int(src_pos[1] - 1)
                if c0 <= c1:
                    out[int(sg_r), int(c0) : int(c1) + 1] = int(body_col)
            else:
                sg_c = int(src_pos[1] + 1 + sp["anchor_inner"][1])
                if int(src_pos[0]) < int(dst_pos[0]):
                    r0, r1 = int(src_pos[0] + src_rect.shape[0]), int(dst_pos[0] - 1)
                else:
                    r0, r1 = int(dst_pos[0] + dst_rect.shape[0]), int(src_pos[0] - 1)
                if r0 <= r1:
                    out[int(r0) : int(r1) + 1, int(sg_c)] = int(body_col)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))
    palette = _assign_palette(n_types)
    if palette is None:
        return None
    wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, bodies = palette

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    sprite_types = []
    for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors):
        patch = _render_sprite(shape_rows, body, anchor, src_bg)
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": tuple(map(int, anchor_rc)),
                "body": int(body),
                "anchor": int(anchor),
                "patch": patch.copy(),
                "mask": patch != int(src_bg),
            }
        )

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    try:
        src_pos, dst_pos = _place_rectangles_aligned(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    marker_type_ids = _marker_type_ids(n_types)
    shared_limit = min(int(src_h - 2), int(dst_h - 2)) if layout == "side_by_side" else min(int(src_w - 2), int(dst_w - 2))
    marker_result = _place_markers(
        sprite_types,
        marker_type_ids,
        dst_h,
        dst_w,
        dst_bg,
        layout,
        shared_limit,
        int(src_h - 2),
        int(src_w - 2),
    )
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    source_result = _place_sources(sprite_types, marker_points, src_h, src_w, src_bg, layout)
    if source_result is None:
        return None
    source_inner, source_placements = source_result

    wallpaper_grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_input = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mp in marker_points:
        dest_input[1 + int(mp["mr"]), 1 + int(mp["mc"])] = int(sprite_types[int(mp["type_id"])]["anchor"])

    dest_output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(dest_output[1:-1, 1:-1], out_inner, 0, 0)

    grid = np.array(wallpaper_grid, copy=True)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_input, int(dst_pos[0]), int(dst_pos[1]))

    expected = _expected_output(
        wallpaper_grid,
        source_rect,
        dest_output,
        src_pos,
        dst_pos,
        src_bg,
        dst_bg,
        sprite_types,
        marker_points,
        source_placements,
        layout,
    )

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 combo input after %d attempts" % MAX_GEN_ATTEMPTS)
