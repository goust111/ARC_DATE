# Combo variant of a251c730 (index 31)
# combo_spec: D3/3+D4/0+D7/0
# Name: 方向标记+旋转+一对多

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# ---------------
# a) Concrete behavior and trace
#    - Original task:
#      - source rectangle: 1-2 sprite types, each with a unique rare anchor color
#      - destination rectangle: 2-3 singleton anchor-color markers
#      - output: copy each matched sprite once at each marker, without rotation
#    - Fused variant:
#      - each destination cue is now a 2-cell domino:
#        one cell is the anchor color, one neighboring cell is a pointer drawn in the destination border color
#      - the pointer direction chooses a base quarter-turn before stamping:
#        up=0 deg, right=90 deg clockwise, down=180 deg, left=270 deg clockwise
#      - the marker's quadrant inside the destination contributes an additional quarter-turn:
#        top-left=0, top-right=90 deg clockwise, bottom-right=180 deg, bottom-left=270 deg clockwise
#      - the final rotation is the sum of these two visible cues modulo 360 deg
#      - at least one anchor color appears on multiple domino markers, and those repeated markers use different
#        pointer directions and different quadrants, so the same source sprite is printed multiple times with
#        different rotations
#    - Small trace:
#      - source contains one red-anchor asymmetric sprite in canonical orientation
#      - destination contains two red markers:
#        one in the top-left quadrant as (red + border-above), one in the bottom-right quadrant as (red + border-right)
#      - old logic would stamp the same upright red sprite twice
#      - pointer-only logic would miss the positional offset and stamp 0 deg and 90 deg
#      - fused logic stamps 0 deg at the first marker and 270 deg at the second because 90 deg (pointer) + 180 deg
#        (quadrant) = 270 deg, both aligned by the red anchor cell
#
# b) Complexity calibration
#    - Only four direction states exist, so the new freedom stays bounded.
#    - Sprite library is restricted to asymmetric top-anchored shapes, making rotation visually obvious.
#    - Marker multiplicity is forced rather than random: each sample carries exactly three markers; with one type all
#      three repeat, and with two types exactly one type repeats so the one-to-many mapping is always visible.
#
# c) Implicit-assumption audit against the original main()
#    - "Destination components are singletons" is false now; destination components are size-2 dominoes.
#    - "Every non-background destination pixel is an independent marker" is false; iterating raw foreground cells would
#      double-process each domino.
#    - "Matched template can be pasted directly" is false; anchor coordinates change after rotation.
#    - "Least-frequent color may be chosen by numeric tie-break" is unsafe; the fused solver now expects a unique rare
#      anchor color per source sprite.
#    - D7 adds a second failure mode for hidden assumptions: repeated anchor colors are now intentional, so solver code
#      must not assume one marker per anchor color.
#
# d) Placement legality
#    - Generator stays output-first: place fully rotated sprite stamps into a clean destination interior first, then
#      derive the input markers from the validated anchor locations.
#    - Each rotated stamp must fit in-bounds and not overlap previous stamps.
#    - Each input marker domino must also stay in-bounds and 4-disconnected from every other marker domino, so each cue
#      is parsed as exactly one size-2 component.
#
# e) Visual observability
#    - The repeated anchor color gives the human solver a direct cue that one sprite type is reused.
#    - Different pointer sides attached to the same anchor color show that repetition is not pure copying.
#    - Quadrant diversity ensures the position inside the destination also matters, so the task does not collapse to a
#      pure direction-only rotation rule.
#    - No rule depends on hidden color ordering, component scan order, or counts outside the visible grid.
#
# f) Visual cleanliness
#    - Using the destination border color as the pointer avoids adding a new semantic color role.
#    - Output stamps are non-overlapping and input markers are non-touching, so source sprites, target cues, and final
#      stamps remain easy to separate visually.
#
# g) New solver assumptions checked against the generator
#    - Every source sprite has exactly one uniquely least-frequent anchor color cell.
#    - Every destination foreground component has exactly two cells and exactly one anchor-colored cell.
#    - Repeated markers exist for at least one anchor color, and the repeated set uses at least two directions and at
#      least two quadrants.
#    - The canonical source template is asymmetric enough that at least one rotated repeated copy differs from the
#      unrotated copy, preventing degeneration back to the original task.


DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_TURNS_CW = {
    (-1, 0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (0, -1): 3,
}
QUADRANT_TO_TURNS_CW = {
    0: 0,  # top-left
    1: 1,  # top-right
    2: 2,  # bottom-right
    3: 3,  # bottom-left
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
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
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
    return rects


def _anchor_color_from_component(comp: dict) -> int | None:
    counts = {int(col): int(ct) for col, ct in comp["colors"].items() if int(col) != 0}
    if not counts:
        return None
    min_ct = min(counts.values())
    rare_colors = [int(col) for col, ct in counts.items() if int(ct) == int(min_ct)]
    if len(rare_colors) != 1:
        return None
    return int(rare_colors[0])


def _rotate_template(
    patch: np.ndarray,
    mask: np.ndarray,
    anchor_rc: tuple[int, int],
    turns_cw: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    k_ccw = (-int(turns_cw)) % 4
    anchor_mask = np.zeros_like(mask, dtype=bool)
    anchor_mask[int(anchor_rc[0]), int(anchor_rc[1])] = True
    rot_patch = np.rot90(np.array(patch, dtype=int), k=int(k_ccw))
    rot_mask = np.rot90(np.array(mask, dtype=bool), k=int(k_ccw))
    rot_anchor_mask = np.rot90(anchor_mask, k=int(k_ccw))
    ar, ac = map(int, np.argwhere(rot_anchor_mask)[0])
    return rot_patch.copy(), rot_mask.copy(), (int(ar), int(ac))


def _quadrant_index(r: int, c: int, h: int, w: int) -> int:
    top = int(r) * 2 < int(h)
    left = int(c) * 2 < int(w)
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and not left:
        return 2
    return 3


def _marker_turns(direction: tuple[int, int], r: int, c: int, h: int, w: int) -> int:
    quadrant = _quadrant_index(int(r), int(c), int(h), int(w))
    return (int(DIR_TO_TURNS_CW[tuple(direction)]) + int(QUADRANT_TO_TURNS_CW[int(quadrant)])) % 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via repeated directional markers.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: the source rectangle holds multi-cell two-color sprites, and the destination rectangle holds only 2-cell directional markers.
    - In each destination marker, one cell uses a sprite's unique anchor color and the neighboring border-colored cell indicates a base rotation direction.

    Rule:
    - Extract one template per source sprite type using its uniquely rare anchor color.
    - For every destination marker domino, match the anchor-colored cell to the source sprite of the same anchor color.
    - Rotate that sprite by combining two visible cues: the pointer side gives a base quarter-turn and the marker's quadrant inside the destination gives an additional quarter-turn.
    - Align the rotated anchor onto the anchor-colored marker cell.
    - If the same anchor color appears on multiple markers, stamp that same source sprite at all of them, potentially with different rotations, then return only the cleaned destination rectangle.

    Key insight: The rare anchor color still tells you which sprite to reuse, while the adjacent pointer and the marker's quadrant jointly determine the quarter-turn before anchoring each repeated copy.
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
        if src is None and 2 <= len(comps) <= 4 and min(sizes) >= 5:
            src = (rect, bc, ibg2, comps)
        if dst is None and 2 <= len(comps) <= 3 and min(sizes) == 2 and max(sizes) == 2:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find source sprites and size-2 destination markers"

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

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    inner_h, inner_w = map(int, inner.shape)
    for coords in components_mask(inner != int(dst_bg), connectivity=4):
        if int(coords.shape[0]) != 2:
            continue
        cells = [(int(r), int(c), int(inner[int(r), int(c)])) for r, c in coords]
        anchor_cells = [(r, c, col) for r, c, col in cells if int(col) in templates]
        if len(anchor_cells) != 1:
            continue
        mr, mc, anchor_col = anchor_cells[0]
        pr, pc, pointer_col = next((r, c, col) for r, c, col in cells if (r, c) != (mr, mc))
        if int(pointer_col) != int(dst_bc):
            continue
        direction = (int(pr - mr), int(pc - mc))
        if tuple(direction) not in DIR_TO_TURNS_CW:
            continue
        patch, mask, anchor_rc = templates[int(anchor_col)]
        turns_cw = _marker_turns(tuple(direction), int(mr), int(mc), int(inner_h), int(inner_w))
        rot_patch, rot_mask, rot_anchor = _rotate_template(patch, mask, anchor_rc, turns_cw)
        top = int(1 + int(mr) - int(rot_anchor[0]))
        left = int(1 + int(mc) - int(rot_anchor[1]))
        paste(out, rot_patch, int(top), int(left), mask=rot_mask, overwrite=True, clip=True)

    return out


# --------------- generator ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (1, 2)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 45
SOURCE_SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400

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
    char_map = {"B": int(body), "A": int(anchor)}
    rows = [[char_map.get(ch, int(bg)) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


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
        counts[random.randrange(int(n_types))] += 1
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
    wallpaper_n = 5 if int(n_types) == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = int(used_total_target) - (int(wallpaper_n) + 1 + int(n_types))
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, int(wallpaper_n))
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(int(n_types))]

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

    return wallpaper, int(src_border), int(dst_border), [int(x) for x in anchors], [int(x) for x in body_pool], forced_body


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


def _render_oriented_sprite(
    patch: np.ndarray,
    anchor_rc: tuple[int, int],
    bg: int,
    turns_cw: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    mask = patch != int(bg)
    rot_patch, rot_mask, rot_anchor = _rotate_template(patch, mask, anchor_rc, int(turns_cw))
    out = np.full(rot_patch.shape, int(bg), dtype=int)
    out[rot_mask] = rot_patch[rot_mask]
    return out, rot_anchor


def _choose_marker_specs(n_types: int) -> list[tuple[int, tuple[int, int], int]]:
    quadrants = random.sample(range(4), 3)
    if int(n_types) == 1:
        shared_direction = tuple(random.choice(DIRECTIONS))
        other_direction = tuple(random.choice([d for d in DIRECTIONS if tuple(d) != tuple(shared_direction)]))
        specs = [
            (0, tuple(shared_direction), int(quadrants[0])),
            (0, tuple(shared_direction), int(quadrants[1])),
            (0, tuple(other_direction), int(quadrants[2])),
        ]
        random.shuffle(specs)
        return specs

    repeated_type = random.randrange(int(n_types))
    other_type = 1 - int(repeated_type)
    repeated_dirs = random.sample(DIRECTIONS, 2)
    specs = [
        (int(repeated_type), tuple(repeated_dirs[0]), int(quadrants[0])),
        (int(repeated_type), tuple(repeated_dirs[1]), int(quadrants[1])),
        (int(other_type), tuple(repeated_dirs[0]), int(quadrants[2])),
    ]
    random.shuffle(specs)
    return specs


def _place_markers(
    sprite_types: list[dict],
    marker_specs: list[tuple[int, tuple[int, int], int]],
    dst_h: int,
    dst_w: int,
    dst_bg: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, tuple[int, int]]]] | None:
    inner_h = int(dst_h) - 2
    inner_w = int(dst_w) - 2
    empty = np.full((int(inner_h), int(inner_w)), int(dst_bg), dtype=int)

    def candidate_anchors(direction: tuple[int, int], quadrant: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for mr in range(int(inner_h)):
            for mc in range(int(inner_w)):
                if _quadrant_index(int(mr), int(mc), int(inner_h), int(inner_w)) != int(quadrant):
                    continue
                pr = int(mr + int(direction[0]))
                pc = int(mc + int(direction[1]))
                if 0 <= int(pr) < int(inner_h) and 0 <= int(pc) < int(inner_w):
                    cells.append((int(mr), int(mc)))
        random.shuffle(cells)
        return cells

    def can_place_patch(board: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
        ph, pw = map(int, patch.shape)
        if int(top) < 0 or int(left) < 0 or int(top) + int(ph) > int(inner_h) or int(left) + int(pw) > int(inner_w):
            return False
        patch_mask = patch != int(dst_bg)
        region = board[int(top) : int(top) + int(ph), int(left) : int(left) + int(pw)]
        return not bool(((region != int(dst_bg)) & patch_mask).any())

    def backtrack(
        idx: int,
        board: np.ndarray,
        occupied_marker_cells: list[tuple[int, int]],
        marker_points: list[tuple[int, int, int, tuple[int, int]]],
    ) -> tuple[np.ndarray, list[tuple[int, int, int, tuple[int, int]]]] | None:
        if int(idx) == len(marker_specs):
            return board, marker_points

        type_id, direction, quadrant = marker_specs[int(idx)]
        sprite = sprite_types[int(type_id)]
        attempts = 0
        for mr, mc in candidate_anchors(tuple(direction), int(quadrant)):
            if attempts >= MARKER_PLACE_RETRIES:
                break
            attempts += 1
            pointer = (int(mr + int(direction[0])), int(mc + int(direction[1])))
            new_cells = [(int(mr), int(mc)), (int(pointer[0]), int(pointer[1]))]
            if not _marker_cells_are_clear(occupied_marker_cells, new_cells, (int(inner_h), int(inner_w))):
                continue
            turns_cw = _marker_turns(tuple(direction), int(mr), int(mc), int(inner_h), int(inner_w))
            patch, anchor_rc = _render_oriented_sprite(
                sprite["dst_patch"],
                sprite["anchor_rc"],
                int(dst_bg),
                int(turns_cw),
            )
            top = int(mr - int(anchor_rc[0]))
            left = int(mc - int(anchor_rc[1]))
            if not can_place_patch(board, patch, int(top), int(left)):
                continue
            next_board = board.copy()
            paste(next_board, patch, int(top), int(left), transparent=int(dst_bg))
            next_points = marker_points + [(int(mr), int(mc), int(type_id), tuple(direction))]
            result = backtrack(
                int(idx) + 1,
                next_board,
                occupied_marker_cells + new_cells,
                next_points,
            )
            if result is not None:
                return result
        return None

    placed = backtrack(0, empty, [], [])
    if placed is None:
        return None
    out_inner, marker_points = placed

    repeated_counts = Counter(type_id for _mr, _mc, type_id, _direction in marker_points)
    if max(repeated_counts.values()) < 2:
        return None
    direction_sets: dict[int, set[tuple[int, int]]] = {}
    quadrant_sets: dict[int, set[int]] = {}
    for _mr, _mc, type_id, direction in marker_points:
        direction_sets.setdefault(int(type_id), set()).add(tuple(direction))
        quadrant_sets.setdefault(int(type_id), set()).add(_quadrant_index(int(_mr), int(_mc), int(inner_h), int(inner_w)))
    if not any(repeated_counts[type_id] >= 2 and len(direction_sets[type_id]) >= 2 for type_id in repeated_counts):
        return None
    if not any(repeated_counts[type_id] >= 2 and len(quadrant_sets[type_id]) >= 2 for type_id in repeated_counts):
        return None
    return out_inner, marker_points


def _place_sources(
    sprite_types: list[dict],
    source_type_ids: list[int],
    src_h: int,
    src_w: int,
    src_bg: int,
) -> np.ndarray | None:
    inner = np.full((int(src_h) - 2, int(src_w) - 2), int(src_bg), dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        patch = sprite["src_patch"]
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=int(src_bg),
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, int(r0), int(c0), transparent=int(src_bg))
    return inner


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(int(n_types))
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != int(dst_border)]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, int(n_types))
    bodies: list[int] = []
    for i in range(int(n_types)):
        choices = [
            c
            for c in body_pool
            if c not in anchors and c != int(src_bg) and c != int(dst_bg) and c != int(dst_border)
        ]
        if not choices:
            return None
        if i == 0 and forced_body is not None and int(forced_body) in choices:
            bodies.append(int(forced_body))
        else:
            bodies.append(int(random.choice(choices)))

    sprite_types = []
    for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors):
        src_patch = _render_sprite(shape_rows, int(body), int(anchor), int(src_bg))
        dst_patch = _render_sprite(shape_rows, int(body), int(anchor), int(dst_bg))
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": (int(anchor_rc[0]), int(anchor_rc[1])),
                "body": int(body),
                "anchor": int(anchor),
                "src_patch": src_patch,
                "dst_patch": dst_patch,
            }
        )

    marker_specs = _choose_marker_specs(int(n_types))

    n_source = random.randint(max(2, int(n_types)), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(int(n_source), int(n_types))) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if int(src_h) * int(src_w) <= int(dst_h) * int(dst_w):
        return None

    marker_result = _place_markers(sprite_types, marker_specs, int(dst_h), int(dst_w), int(dst_bg))
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(int(dst_h), int(dst_w), int(dst_border), int(dst_bg))
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, int(src_h), int(src_w), int(src_bg))
    if source_inner is None:
        return None

    source_rect = _build_frame(int(src_h), int(src_w), int(src_border), int(src_bg))
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(int(dst_h), int(dst_w), int(dst_border), int(dst_bg))
    for mr, mc, type_id, direction in marker_points:
        dest_rect[1 + int(mr), 1 + int(mc)] = int(sprite_types[int(type_id)]["anchor"])
        dest_rect[1 + int(mr) + int(direction[0]), 1 + int(mc) + int(direction[1])] = int(dst_border)

    repeated_counts = Counter(type_id for _mr, _mc, type_id, _direction in marker_points)
    if max(repeated_counts.values()) < 2:
        return None
    if not any(
        repeated_counts[type_id] >= 2
        and len({direction for _mr, _mc, t, direction in marker_points if int(t) == int(type_id)}) >= 2
        for type_id in repeated_counts
    ):
        return None
    inner_h = int(dst_h) - 2
    inner_w = int(dst_w) - 2
    if not any(
        repeated_counts[type_id] >= 2
        and len(
            {
                _quadrant_index(int(mr), int(mc), int(inner_h), int(inner_w))
                for mr, mc, t, _direction in marker_points
                if int(t) == int(type_id)
            }
        )
        >= 2
        for type_id in repeated_counts
    ):
        return None

    try:
        src_pos, dst_pos = _place_rectangles(layout, int(src_h), int(src_w), int(dst_h), int(dst_w))
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 combo-31 input after {MAX_GEN_ATTEMPTS} attempts")
