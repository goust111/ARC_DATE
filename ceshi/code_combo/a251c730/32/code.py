# Combo variant of a251c730 (index 32)
# combo_spec: D3/1+D4/0+D5/0
# Name: 边框标记+旋转+去边框输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) concretization / behavior trace
#    - The source side stays close to the original task: one bordered source rectangle contains
#      2-4 copies of 1-2 multicolor sprites. Each sprite has exactly one rare anchor pixel whose
#      color keys the matching process.
#    - The destination side changes in two coupled ways. First, the anchor-color markers move from
#      the interior to the border, so the input rectangle no longer shows any interior singleton
#      clues. Second, the marker side now controls rotation before alignment:
#        top -> 0 turns, right -> 1 clockwise turn, bottom -> 2, left -> 3.
#    - Example with a 3x3 sprite whose anchor sits in the middle row:
#          .B.
#          BA.
#          BBB
#      Original task: an interior marker would receive this exact orientation, translated so A lands
#      on the marker. D3-only border placement: a top-edge marker would clip the cells above A but
#      keep orientation unchanged. This combo: a right-edge marker first rotates the sprite 90°
#      clockwise, then aligns its rotated anchor to the right border marker, so the inward remnant
#      points in a different direction; after stamping, the outer frame is removed from the output.
#    - The output therefore hides the very border that carried the control information. The solver
#      must read side information from the input border, simulate the rotated stamp on the full
#      destination frame, and only then crop away the border.
# b) complexity calibration
#    - The generator uses exactly 3 markers and places them on 3 distinct sides. With only 1-2
#      sprite types, at least one anchor color must repeat across different sides, so rotation can be
#      inferred as a side-driven rule instead of a color-specific lookup.
#    - Sprites are restricted to asymmetric 3x3 templates with a non-edge anchor. This keeps all
#      four side rotations legal while making the rotated interiors visibly different.
#    - Marker count stays small and the destination interior is otherwise blank, so the task remains
#      parseable despite the extra hidden-step crop.
# c) implicit-assumption audit for main()
#    - Original rectangle detection assumed the destination border was fully uniform and the markers
#      lived in the interior. That fails here because the border deliberately contains 3 deviations
#      and the interior is completely empty. The solver must therefore separate strict-border source
#      candidates from loose-border destination candidates.
#    - Original marker scan iterated over `inner != dst_bg`; that would now find nothing. The scan
#      must move to the non-corner perimeter cells that differ from the dominant border color.
#    - Original stamping used the template orientation directly. That assumption fails because the
#      marker side now changes both the patch shape and the anchor location via rotation.
#    - Original return value included the destination border. That assumption also fails: the combo
#      requires stamping on the full frame first, then returning `out[1:-1, 1:-1]`.
# d) placement legality
#    - The generator is output-first. It samples 3 distinct border sides, rotates each chosen sprite
#      according to the side, and only accepts placements with:
#        * a non-corner border marker,
#        * at least one clipped colored cell outside the frame,
#        * at least two visible cells inside the destination interior after cropping,
#        * no overlap among visible interior stamp footprints.
#    - Source placement keeps the original separation constraints via `random_free_location`.
# e) visual observability
#    - Humans can see, in the input, which anchor color sits on which side of the destination frame.
#      In the output those border cues disappear, but the resulting interior sprite fragments differ
#      by orientation. Because one color repeats on multiple sides in the same instance, side-driven
#      rotation has direct within-example evidence.
#    - No hidden ordering or color-id arithmetic is used; colors only identify sprite type.
# f) visual cleanliness
#    - Destination interior footprints are forced disjoint, so the cropped output reads as 2-3 clean
#      rotated imprints on a plain background.
#    - Distinct sides prevent border markers from clustering into a single unreadable edge segment.
# g) post-change assumptions that the generator must guarantee
#    - Every chosen destination marker color exists among the extracted source templates.
#    - Every chosen side has a legal rotated placement for the selected sprite within the destination
#      dimensions.
#    - The final cropped output differs from the D3-only "border marker but no rotation" behavior in
#      normal samples; the verification script below measures that explicitly.


SIDE_TO_TURNS_CW = {"top": 0, "right": 1, "bottom": 2, "left": 3}
SIDES = ("top", "right", "bottom", "left")


def _interior_bg(grid: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(grid: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
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


def _marker_side(marker_r: int, marker_c: int, rect: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = map(int, rect)
    if int(marker_r) == int(r0) and c0 < int(marker_c) < c1:
        return "top"
    if int(marker_r) == int(r1) and c0 < int(marker_c) < c1:
        return "bottom"
    if int(marker_c) == int(c0) and r0 < int(marker_r) < r1:
        return "left"
    if int(marker_c) == int(c1) and r0 < int(marker_r) < r1:
        return "right"
    raise ValueError("marker is not on a non-corner border cell")


def _border_markers(grid: np.ndarray, rect: tuple[int, int, int, int], border_color: int) -> list[tuple[int, int, int, str]]:
    r0, c0, r1, c1 = map(int, rect)
    pts: list[tuple[int, int, int, str]] = []
    for c in range(int(c0) + 1, int(c1)):
        if int(grid[int(r0), int(c)]) != int(border_color):
            pts.append((int(r0), int(c), int(grid[int(r0), int(c)]), "top"))
        if int(grid[int(r1), int(c)]) != int(border_color):
            pts.append((int(r1), int(c), int(grid[int(r1), int(c)]), "bottom"))
    for r in range(int(r0) + 1, int(r1)):
        if int(grid[int(r), int(c0)]) != int(border_color):
            pts.append((int(r), int(c0), int(grid[int(r), int(c0)]), "left"))
        if int(grid[int(r), int(c1)]) != int(border_color):
            pts.append((int(r), int(c1), int(grid[int(r), int(c1)]), "right"))
    return pts


def _find_src_dst(grid: np.ndarray) -> tuple[tuple, tuple]:
    H, W = grid.shape
    strict_rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    loose_rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6

    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(grid[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if (
                        int(grid[int(r0), int(c1)]) != int(bc)
                        or int(grid[int(r1), int(c0)]) != int(bc)
                        or int(grid[int(r1), int(c1)]) != int(bc)
                    ):
                        continue

                    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
                    if inner.size == 0:
                        continue
                    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
                    if int(ibg) == int(bc):
                        continue

                    top = grid[int(r0), int(c0) : int(c1) + 1]
                    bottom = grid[int(r1), int(c0) : int(c1) + 1]
                    left = grid[int(r0) : int(r1) + 1, int(c0)]
                    right = grid[int(r0) : int(r1) + 1, int(c1)]
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))

                    if (
                        bool((top == int(bc)).all())
                        and bool((bottom == int(bc)).all())
                        and bool((left == int(bc)).all())
                        and bool((right == int(bc)).all())
                    ):
                        strict_rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))
                        continue

                    mismatch_count = int(np.count_nonzero(top[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(bottom[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(left[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(right[1:-1] != int(bc)))
                    if 2 <= int(mismatch_count) <= 3 and bool((inner == int(ibg)).all()):
                        loose_rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert strict_rects and loose_rects
    strict_rects.sort(reverse=True, key=lambda x: int(x[0]))
    loose_rects.sort(reverse=True, key=lambda x: int(x[0]))

    src = None
    for _area, rect, bc, _ibg in strict_rects[:8]:
        ibg2, comps = _rect_components(grid, rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if int(max_sz) > 1:
            src = (rect, bc, ibg2, comps)
            break

    assert src is not None, "must find the source rectangle"
    _src_rect, _src_bc, _src_bg, src_comps = src

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    dst = None
    for _area, rect, bc, ibg in loose_rects[:10]:
        markers = _border_markers(grid, rect, int(bc))
        if len(markers) != 3:
            continue
        sides = {side for _r, _c, _col, side in markers}
        if len(sides) != 3:
            continue
        if all(int(col) in templates for _r, _c, col, _side in markers):
            dst = (rect, bc, ibg, markers)
            break

    assert dst is not None, "must find the destination rectangle with border markers"
    return src, dst


def _solve(grid: np.ndarray, *, rotate_markers: bool, crop_border: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, dst_markers = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
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

    rect = (int(dr), int(dc), int(dr1), int(dc1))
    for mr, mc, mcol, side in dst_markers:
        patch, mask, anchor_rc = templates[int(mcol)]
        if rotate_markers:
            qturns = int(SIDE_TO_TURNS_CW[_marker_side(int(mr), int(mc), rect)])
            patch, mask, anchor_rc = _rotate_template(patch, mask, anchor_rc, qturns)
        ar, ac = map(int, anchor_rc)
        top = int((int(mr) - int(dr)) - int(ar))
        left = int((int(mc) - int(dc)) - int(ac))
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out[1:-1, 1:-1] if crop_border else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites from border markers into a destination frame, then drop the frame.

    Input structure:
    - Two bordered rectangles lie on striped wallpaper: a strict-border source rectangle with 2-color
      sprites, and a destination rectangle whose interior is blank but whose border contains 3 anchor-colored markers.
    - Each source sprite has a singleton anchor color, and the same colors appear as the destination's border markers.

    Rule:
    - Match every border marker to the source sprite with the same anchor color.
    - Use the marker's side to choose a clockwise rotation (top/right/bottom/left = 0/1/2/3 turns),
      rotate the sprite, and align the rotated anchor pixel onto that marker while clipping anything outside the frame.
    - After stamping all sprites onto a cleared destination rectangle, remove the 1-pixel border and output only the interior.

    Key insight: The marker color still selects which sprite to copy, but the border side secretly carries the rotation,
    and that side information disappears because the final answer is the destination interior without its frame.
    """
    return _solve(grid, rotate_markers=True, crop_border=True)


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
MARKER_COUNT_RANGE = (3, 3)
SOURCE_SPRITE_RANGE = (2, 4)
SOURCE_SPRITE_PADDING = 1
MIN_VISIBLE_INNER_CELLS = 2
MAX_MARKER_CANDIDATES = 60

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 400

SPRITE_SHAPES = [
    ((".B.", "BA.", "BBB"), (1, 1)),
    (("BB.", ".AB", ".B."), (1, 1)),
    ((".BB", "BA.", ".B."), (1, 1)),
    (("..B", "BAB", "BB."), (1, 1)),
    ((".B.", ".AB", "BBB"), (1, 1)),
    (("B..", ".AB", "BBB"), (1, 1)),
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


def _visible_stamp_stats(
    patch: np.ndarray,
    anchor_rc: tuple[int, int],
    marker_rc: tuple[int, int],
    frame_h: int,
    frame_w: int,
    bg: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int]:
    ar, ac = map(int, anchor_rc)
    mr, mc = map(int, marker_rc)
    top = int(mr - ar)
    left = int(mc - ac)

    visible_full: set[tuple[int, int]] = set()
    visible_inner: set[tuple[int, int]] = set()
    total_colored = 0
    for r in range(int(patch.shape[0])):
        for c in range(int(patch.shape[1])):
            if int(patch[int(r), int(c)]) == int(bg):
                continue
            total_colored += 1
            rr = int(top + int(r))
            cc = int(left + int(c))
            if 0 <= rr < int(frame_h) and 0 <= cc < int(frame_w):
                visible_full.add((int(rr), int(cc)))
                if 0 < rr < int(frame_h) - 1 and 0 < cc < int(frame_w) - 1:
                    visible_inner.add((int(rr), int(cc)))
    return visible_full, visible_inner, int(total_colored - len(visible_full))


def _side_candidates(side: str, frame_h: int, frame_w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, int(c)) for c in range(1, int(frame_w) - 1)]
    if side == "bottom":
        return [(int(frame_h) - 1, int(c)) for c in range(1, int(frame_w) - 1)]
    if side == "left":
        return [(int(r), 0) for r in range(1, int(frame_h) - 1)]
    return [(int(r), int(frame_w) - 1) for r in range(1, int(frame_h) - 1)]


def _place_border_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int, dst_border: int
) -> tuple[np.ndarray, list[tuple[int, int, int, str, int]]] | None:
    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    occupied_inner: set[tuple[int, int]] = set()
    marker_points: list[tuple[int, int, int, str, int]] = []
    chosen_sides = random.sample(list(SIDES), len(marker_type_ids))
    assignments = list(zip(marker_type_ids, chosen_sides))
    random.shuffle(assignments)

    for type_id, side in assignments:
        sprite = sprite_types[int(type_id)]
        base_patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        base_mask = base_patch != int(dst_bg)
        qturns = int(SIDE_TO_TURNS_CW[side])
        patch, mask, anchor_rc = _rotate_template(base_patch, base_mask, sprite["anchor_rc"], qturns)

        candidates = _side_candidates(side, dst_h, dst_w)
        random.shuffle(candidates)

        placed = False
        for mr, mc in candidates[:MAX_MARKER_CANDIDATES]:
            if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc, _tid, _side, _turns in marker_points):
                continue
            visible_full, visible_inner, clipped = _visible_stamp_stats(
                patch, anchor_rc, (int(mr), int(mc)), dst_h, dst_w, dst_bg
            )
            if int(clipped) <= 0:
                continue
            if len(visible_inner) < MIN_VISIBLE_INNER_CELLS:
                continue
            if visible_inner & occupied_inner:
                continue

            ar, ac = map(int, anchor_rc)
            paste(output, patch, int(mr) - int(ar), int(mc) - int(ac), mask=mask, overwrite=True, clip=True)
            occupied_inner |= visible_inner
            marker_points.append((int(mr), int(mc), int(type_id), side, int(qturns)))
            placed = True
            break

        if not placed:
            return None

    marker_points.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2])))
    return output, marker_points


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
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
        paste(inner, patch, int(r0), int(c0), transparent=src_bg)
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

    marker_result = _place_border_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg, dst_border)
    if marker_result is None:
        return None
    output_frame, marker_points = marker_result
    output = output_frame[1:-1, 1:-1].copy()

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id, _side, _qturns in marker_points:
        dest_rect[int(mr), int(mc)] = int(sprite_types[int(type_id)]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, output):
        return None

    # Guard against degeneration into the no-rotation border-marker variant.
    baseline = _solve(grid, rotate_markers=False, crop_border=True)
    if np.array_equal(predicted, baseline):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo a251c730 input after %d attempts" % MAX_GEN_ATTEMPTS)
