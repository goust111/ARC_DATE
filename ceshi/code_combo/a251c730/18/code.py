# Combo variant of a251c730 (index 18)
# combo_spec: D3/3+D4/0
# Name: 方向标记+旋转对齐

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# ----------------
# a) concretization / behavior trace
#    - Original task: the destination rectangle contained 2-3 singleton anchor-colored
#      markers. A red marker simply meant "find the red-anchor sprite in the source and
#      translate it unchanged so that its red pixel lands here".
#    - Fused variant: each destination cue is now a 2-cell domino:
#        * one cell is the anchor color and still chooses the sprite type
#        * the neighboring cell is a direction pointer, colored with the destination border
#      The pointer direction directly encodes the quarter-turn:
#        up    -> 0 degrees
#        right -> 90 degrees clockwise
#        down  -> 180 degrees
#        left  -> 270 degrees clockwise
#      Example: if the source sprite is
#          A.B
#          BBB
#      then an anchor+pointer pair with the pointer on the right means "rotate this
#      sprite 90 degrees clockwise, then place it so the rotated A still lands on the
#      anchor-colored marker cell". This is the D4-style rotate-then-align step, but the
#      angle is supplied by the D3-style pointer instead of by quadrant.
# b) complexity calibration
#    - The control variable is limited to four axial directions, so the extra freedom is
#      visible but still easy to infer from 2-4 examples.
#    - The generator always uses 3 markers and forces at least 2 distinct directions.
#      With only 1-2 sprite types, at least one anchor color repeats, so solvers can see
#      the same sprite appear in multiple orientations inside one example.
#    - Source templates are taken from a library of clearly asymmetric shapes whose anchor
#      sits on the top edge, making quarter-turns visually obvious.
# c) implicit-assumption audit for main()
#    - The original destination detector assumed every destination component was a
#      singleton; this breaks because each marker is now a 2-cell component.
#    - The original interior scan over every non-background pixel would double-handle a
#      directional marker, so the fused solver must iterate connected components instead.
#    - Direct pasting of the stored source template is no longer valid; the anchor
#      position changes after rotation, so the solver must rotate both shape and anchor
#      mask before computing the paste offset.
#    - Using numeric color tie-breaks for anchor extraction would be unsafe. The solver
#      now expects the anchor color to be the unique least-frequent color inside a source
#      component, and the generator guarantees that.
# d) placement legality
#    - The generator stays output-first: it first stamps rotated sprites into the
#      destination output canvas, then projects each stamp back to an input marker domino.
#    - Every rotated sprite must fit inside the destination interior with no overlap.
#    - Every marker domino must stay in-bounds and 4-disconnected from every other marker
#      domino, so the destination input remains easy to segment.
# e) visual observability
#    - The anchor color still tells the viewer which source sprite to copy.
#    - The only new clue is the pointer side, and the same anchor color appearing with
#      different pointer directions demonstrates that the pointer, not the color id, sets
#      the orientation.
#    - No hidden ordering by color number, component index, or quadrant is used.
# f) visual cleanliness
#    - Pointer pixels all reuse the destination border color instead of adding a new role
#      color, so the destination input stays compact and legible.
#    - Rotated stamps are checked for overlap, and source sprites keep padding from one
#      another and from the frame, avoiding clutter in both rectangles.
# g) change inferability / non-degeneracy
#    - Degeneration back to the original task would happen if all markers pointed up, or
#      if a chosen sprite were rotationally symmetric. The generator prevents this by
#      forcing at least two distinct directions and drawing only asymmetric templates.
#    - The intended counterfactual is "ignore the pointer and always stamp the canonical
#      source orientation"; generated samples are constructed so that this counterfactual
#      usually disagrees with the fused output on visibly many cells.
# h) post-change assumptions main() now makes, and how generate_input() guarantees them
#    - Every source sprite component has exactly one least-frequent anchor color cell.
#    - Every destination marker component has size exactly 2 and contains exactly one
#      anchor-colored cell that matches some source template.
#    - Pointer offsets are always one of the four axial unit vectors in DIR_TO_CW.
#    - Rotated patches fit completely inside the destination frame, because placement is
#      validated before the input grid is finalized.


DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_CW = {
    (-1, 0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (0, -1): 3,
}


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
                "mask": mask,
                "patch": patch,
            }
        )

    comps.sort(key=lambda d: -int(d["size"]))
    return int(ibg), comps


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], int, int]]:
    g = np.array(grid, dtype=int)
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

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    return rects[:6]


def _anchor_color_from_component(comp: dict) -> int | None:
    counts = {int(col): int(ct) for col, ct in comp["colors"].items()}
    if not counts:
        return None
    min_ct = min(counts.values())
    anchor_cols = [int(col) for col, ct in counts.items() if int(ct) == int(min_ct)]
    if len(anchor_cols) != 1:
        return None
    return int(anchor_cols[0])


def _rotate_template(
    patch: np.ndarray,
    mask: np.ndarray,
    anchor_rc: tuple[int, int],
    quarter_turns_cw: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    k_ccw = (-int(quarter_turns_cw)) % 4
    anchor_mask = np.zeros_like(mask, dtype=bool)
    anchor_mask[int(anchor_rc[0]), int(anchor_rc[1])] = True
    rot_patch = np.rot90(np.array(patch, dtype=int), k=int(k_ccw))
    rot_mask = np.rot90(np.array(mask, dtype=bool), k=int(k_ccw))
    rot_anchor = np.rot90(anchor_mask, k=int(k_ccw))
    anchors = np.argwhere(rot_anchor)
    if anchors.shape[0] != 1:
        raise ValueError("rotated template must preserve a unique anchor")
    ar, ac = map(int, anchors[0])
    return rot_patch.copy(), rot_mask.copy(), (int(ar), int(ac))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via directional marker dominoes.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper: the source holds full two-color sprites, while the destination holds 2-cell markers made of an anchor-colored pixel plus a neighboring pointer pixel.
    - In each source sprite, the unique least-frequent color is the anchor; destination anchor pixels reuse those colors, and the adjacent pointer cell shows one of the four cardinal directions.

    Rule:
    - Match each destination marker to the source sprite with the same anchor color.
    - Rotate that sprite by the quarter-turn encoded by the pointer direction, then translate the rotated sprite so its anchor pixel lands on the marker's anchor pixel.
    - Clear the destination interior to its background first and return only the bordered destination rectangle after all rotated stamps are applied.

    Key insight: The anchor color still chooses which template to copy, but the attached pointer pixel now supplies the rotation angle, so solving requires fusing color matching with rotate-then-align placement.
    """
    g = np.array(grid, dtype=int)

    rects = _find_rectangles(g)
    assert rects, "must find bordered rectangles"

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect)
        sizes = [int(comp["size"]) for comp in comps]
        if not sizes:
            continue
        if max(sizes) > 2 and src is None:
            src = (rect, bc, ibg2, comps)
        if min(sizes) == 2 and max(sizes) == 2 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source sprites and directional destination markers"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        anchor_col = _anchor_color_from_component(comp)
        if anchor_col is None:
            continue
        patch = np.array(comp["patch"], dtype=int)
        mask = np.array(comp["mask"], dtype=bool)
        anchors = np.argwhere(mask & (patch == int(anchor_col)))
        if anchors.shape[0] != 1:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    assert templates, "must recover at least one source template"

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    for coords in components_mask(inner != int(dst_bg), connectivity=4):
        if int(coords.shape[0]) != 2:
            continue
        cells = [(int(r), int(c), int(inner[int(r), int(c)])) for r, c in coords]
        anchor_cells = [(r, c, col) for r, c, col in cells if int(col) in templates]
        if len(anchor_cells) != 1:
            continue
        ar0, ac0, anchor_col = anchor_cells[0]
        pr0, pc0, _pointer_col = next((r, c, col) for r, c, col in cells if (r, c) != (ar0, ac0))
        direction = (int(pr0 - ar0), int(pc0 - ac0))
        if tuple(direction) not in DIR_TO_CW:
            continue
        patch, mask, anchor_rc = templates[int(anchor_col)]
        rot_patch, rot_mask, rot_anchor = _rotate_template(patch, mask, anchor_rc, DIR_TO_CW[tuple(direction)])
        top = int(1 + ar0 - rot_anchor[0])
        left = int(1 + ac0 - rot_anchor[1])
        paste(out, rot_patch, int(top), int(left), mask=rot_mask, overwrite=True, clip=True)

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
MARKER_COUNT_RANGE = (3, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 60
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 500

SPRITE_SHAPES = [
    (("A.B", "BBB"), (0, 0)),
    ((".AB", "BBB"), (0, 1)),
    (("AB.", "BBB", "..B"), (0, 0)),
    ((".A.", "BBB", ".BB"), (0, 1)),
    (("..A", "BBB", "BB."), (0, 2)),
    (("A..", "BBB", "B.B"), (0, 0)),
    ((".A.", "BBB", "B.."), (0, 1)),
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


def _choose_marker_directions(n_markers: int) -> list[tuple[int, int]]:
    distinct_n = random.randint(2, min(len(DIRECTIONS), n_markers))
    chosen = random.sample(DIRECTIONS, distinct_n)
    while len(chosen) < n_markers:
        chosen.append(random.choice(chosen))
    random.shuffle(chosen)
    return [tuple(d) for d in chosen]


def _marker_cells_are_clear(
    occupied: list[tuple[int, int]],
    new_cells: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    H, W = map(int, shape)
    if len(set(new_cells)) != len(new_cells):
        return False
    for r, c in new_cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        for pr, pc in occupied:
            if abs(int(r) - int(pr)) + abs(int(c) - int(pc)) <= 1:
                return False
    return True


def _stamp_cells_are_clear(
    occupied: list[tuple[int, int]],
    new_cells: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    H, W = map(int, shape)
    for r, c in new_cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        for pr, pc in occupied:
            if abs(int(r) - int(pr)) + abs(int(c) - int(pc)) <= 1:
                return False
    return True


def _place_markers(
    sprite_types: list[dict],
    marker_type_ids: list[int],
    dst_h: int,
    dst_w: int,
    dst_bg: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, tuple[int, int]]]] | None:
    out_inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int, tuple[int, int]]] = []
    occupied_marker_cells: list[tuple[int, int]] = []
    occupied_stamp_cells: list[tuple[int, int]] = []
    directions = _choose_marker_directions(len(marker_type_ids))

    repeated_direction_pair = False

    for type_id, direction in zip(marker_type_ids, directions):
        sprite = sprite_types[type_id]
        patch, mask, anchor_rc = _rotate_template(
            sprite["dst_patch"], sprite["dst_mask"], sprite["anchor_rc"], DIR_TO_CW[tuple(direction)]
        )
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(out_inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            stamp_cells = [
                (int(r0 + rr), int(c0 + cc))
                for rr, cc in np.argwhere(mask)
            ]
            mr = int(r0 + anchor_rc[0])
            mc = int(c0 + anchor_rc[1])
            pointer = (int(mr + direction[0]), int(mc + direction[1]))
            new_cells = [(int(mr), int(mc)), pointer]
            if not _marker_cells_are_clear(occupied_marker_cells, new_cells, out_inner.shape):
                continue
            if not _stamp_cells_are_clear(occupied_stamp_cells, stamp_cells, out_inner.shape):
                continue
            paste(out_inner, patch, int(r0), int(c0), mask=mask, overwrite=True)
            for _pr, _pc, prev_type, prev_dir in marker_points:
                if int(prev_type) == int(type_id) and tuple(prev_dir) != tuple(direction):
                    repeated_direction_pair = True
                    break
            marker_points.append((int(mr), int(mc), int(type_id), tuple(direction)))
            occupied_marker_cells.extend(new_cells)
            occupied_stamp_cells.extend(stamp_cells)
            placed = True
            break
        if not placed:
            return None

    if len(set(directions)) < 2:
        return None
    if len(set(marker_type_ids)) < len(marker_type_ids) and not repeated_direction_pair:
        return None
    return out_inner, marker_points


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
        patch = sprite["src_patch"]
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
        choices = [
            c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg and c != dst_border
        ]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(forced_body)
        else:
            bodies.append(random.choice(choices))

    sprite_types = []
    for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors):
        src_patch = _render_sprite(shape_rows, body, anchor, src_bg)
        dst_patch = _render_sprite(shape_rows, body, anchor, dst_bg)
        src_mask = src_patch != int(src_bg)
        dst_mask = dst_patch != int(dst_bg)
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": tuple(map(int, anchor_rc)),
                "body": int(body),
                "anchor": int(anchor),
                "src_patch": src_patch,
                "dst_patch": dst_patch,
                "src_mask": src_mask,
                "dst_mask": dst_mask,
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
    for mr, mc, type_id, direction in marker_points:
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[int(type_id)]["anchor"])
        dest_rect[1 + mr + direction[0], 1 + mc + direction[1]] = int(dst_border)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    try:
        if not np.array_equal(main(grid), output):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 combo input after {MAX_GEN_ATTEMPTS} attempts")
