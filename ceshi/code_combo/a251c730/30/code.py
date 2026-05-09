# Combo variant of a251c730 (index 30)
# combo_spec: D2/0+D4/0+D15/2
# Name: 三色精灵+旋转+内嵌方向

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace
#    - The source rectangle still contains disconnected sprites and the destination rectangle still contains
#      singleton anchor-color markers, but each source sprite is now tri-color:
#        anchor = 1 cell, decoration = 2 cells, body = 4 cells.
#    - The decoration pair is not cosmetic noise. Its corner encodes an internal base quarter-turn, and the
#      marker's quadrant inside the destination contributes one more quarter-turn before stamping:
#        decor at upper-left -> 0 turns
#        decor at upper-right -> 1 clockwise turn
#        decor at lower-right -> 2 turns
#        decor at lower-left -> 3 turns
#    - Small example:
#          source sprite:      marker:        stamped output:
#          B C C              anchor color    rotate this tri-color patch 90 deg clockwise,
#          . A .              matches here    then align the rotated A to the marker
#          B B B
#      Original task would copy the source patch as-is. This combo first reads the unique rarest color A as
#      the key, then reads the 2-cell decoration corner as an internal direction code, then adds the
#      destination-quadrant turn before stamping.
# b) Complexity calibration
#    - The three roles are frequency-separable by construction: counts are always (1, 2, 4). This avoids any
#      hidden dependence on color ids when identifying anchor vs decoration.
#    - The generator uses exactly 2 sprite types and 3 markers, so one anchor color repeats across at least
#      two distinct destination quadrants and there are at least two distinct internal direction codes. This
#      gives visible support for both "match by anchor color" and the two-part rotation rule.
#    - The sprite footprint is a 7-cell asymmetric 3x3 motif. After a non-zero quarter-turn, the output is
#      visibly different from the unrotated source, so the fused behavior does not collapse back to the
#      original task.
# c) Implicit-assumption audit of the original main()
#    - Original assumption: the rarest color can be selected by a count+tiebreak on color id.
#      Broken here: the task is about a unique visible singleton anchor, so the solver must require exactly
#      one color with count 1 rather than relying on numeric ids.
#    - Original assumption: after color matching, the stored patch can be pasted directly.
#      Broken here: the decoration cells encode a base rotation and the marker quadrant encodes an extra
#      rotation, so the solver must combine both turns before alignment and rotate the anchor coordinate with
#      the final patch.
#    - Original assumption: extra non-anchor colors are irrelevant.
#      Broken here: the 2-cell decoration color is the control signal, so the full tri-color patch must be
#      preserved and the decoration cells must be located explicitly.
# d) Placement legality
#    - Destination examples are generated output-first: full rotated patches are placed on a temporary clean
#      interior, overlap and bounds are checked there, and only then are those stamps reduced to singleton
#      input markers.
#    - Source sprites are placed with one-cell padding, so 4-connected component parsing continues to recover
#      each sprite cleanly.
#    - Marker anchors are also kept non-adjacent, so the destination input still decomposes into singleton
#      components and rectangle detection remains valid.
# e) Visual observability
#    - All control variables are visible: anchor = unique singleton color, decoration = unique 2-cell color,
#      internal direction = decoration corner, extra rotation = marker quadrant. No ordering by color id or
#      hidden metadata is used.
#    - Because one anchor color repeats across multiple quadrants and the two sprite types use different
#      internal directions, a human can separate "which sprite", "what base turn", and "what extra turn".
# f) Visual cleanliness
#    - The destination input stays sparse because it contains only singleton markers.
#    - The source sprites all use the same compact 3x3 footprint, keeping the rectangle legible even with
#      repeated copies.
#    - Output stamps are prevented from overlapping, so the final target frame does not become cluttered.
# g) Quantitative fusion metric
#    - For a generated sample we can count how many destination markers use a non-zero internal turn, how many
#      use a non-zero quadrant turn, and how many output pixels differ from an "internal-only" baseline.
#    - The generator is accepted only if both turn sources are active and the full fused output differs from
#      the internal-only baseline, proving that the quadrant rule is not inert.
# h) New solver assumptions that the generator must satisfy
#    - Every source component has exactly three colors with counts (1, 2, 4).
#    - The 2-cell decoration cluster lies clearly in one sprite corner; its centroid therefore determines one
#      of the four quarter-turns without ambiguity.
#    - Every destination marker color appears as a source anchor color, repeated anchors span multiple
#      quadrants, and all fully rotated stamps fit within the destination interior.


TURN_TO_NAME = {
    0: "upper_left",
    1: "upper_right",
    2: "lower_right",
    3: "lower_left",
}

NAME_TO_TURN = {name: turn for turn, name in TURN_TO_NAME.items()}

QUADRANT_TO_TURNS_CW = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}

DIRECTION_SHAPES: dict[str, tuple[str, ...]] = {
    "upper_left": ("CCB", ".A.", "BBB"),
    "upper_right": ("BCC", ".A.", "BBB"),
    "lower_right": ("BBB", ".A.", "BCC"),
    "lower_left": ("BBB", ".A.", "CCB"),
}


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
    H, W = map(int, g.shape)
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
        max_sz = max([int(comp["size"]) for comp in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and comps and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source and destination rectangles"
    return src, dst


def _classify_tricolor_roles(colors: dict[int, int]) -> tuple[int, int, int] | None:
    if len(colors) != 3:
        return None
    ordered = sorted((int(cnt), int(col)) for col, cnt in colors.items())
    counts = [cnt for cnt, _col in ordered]
    if counts != [1, 2, 4]:
        return None
    anchor_col = ordered[0][1]
    decor_col = ordered[1][1]
    body_col = ordered[2][1]
    return int(anchor_col), int(decor_col), int(body_col)


def _quarter_turns_from_decor(patch: np.ndarray, mask: np.ndarray, decor_col: int) -> int | None:
    coords = np.argwhere((patch == int(decor_col)) & mask)
    if coords.shape[0] != 2:
        return None
    h, w = map(int, mask.shape)
    mean_r = float(coords[:, 0].mean())
    mean_c = float(coords[:, 1].mean())
    top_half = bool(mean_r < (float(h) - 1.0) / 2.0)
    left_half = bool(mean_c < (float(w) - 1.0) / 2.0)
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


def _quadrant_turns_cw(r: int, c: int, h: int, w: int) -> int:
    return int(QUADRANT_TO_TURNS_CW[int(_quadrant_index(int(r), int(c), int(h), int(w)))])


def _extract_templates(
    src_comps: list[dict],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int], int]], list[int]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int], int]] = {}
    base_turns: list[int] = []
    for comp in src_comps:
        roles = _classify_tricolor_roles({int(k): int(v) for k, v in comp["colors"].items()})
        if roles is None:
            continue
        anchor_col, decor_col, _body_col = roles
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.shape[0] != 1:
            continue
        anchor_rc = tuple(map(int, anchors[0]))
        qturns = _quarter_turns_from_decor(patch, mask, int(decor_col))
        if qturns is None:
            continue
        base_turns.append(int(qturns))
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(anchor_rc[0]), int(anchor_rc[1])), int(qturns))
    return templates, base_turns


def _solve(grid: np.ndarray, *, apply_internal_turn: bool, apply_quadrant_turn: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates, _turns = _extract_templates(src_comps)
    assert templates, "no usable tri-color source templates found"

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
        patch, mask, anchor_rc, base_turn = templates[int(mcol)]
        total_turns = 0
        if apply_internal_turn:
            total_turns += int(base_turn)
        if apply_quadrant_turn:
            total_turns += _quadrant_turns_cw(int(rr), int(cc), int(inner_h), int(inner_w))
        rot_patch, rot_mask, (ar, ac) = _rotate_template(patch, mask, anchor_rc, int(total_turns % 4))
        top = int(1 + int(rr) - int(ar))
        left = int(1 + int(cc) - int(ac))
        paste(out, rot_patch, int(top), int(left), mask=rot_mask, overwrite=True, clip=True)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by matching tri-color sprites whose decoration corner and marker quadrant
    jointly encode rotation.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a source box with disconnected tri-color sprites and a
      destination box with singleton markers.
    - In each source sprite, the unique singleton color is the anchor key, while a 2-cell decoration color
      sits in one corner of the 3x3 sprite footprint.

    Rule:
    - Match each destination marker to the source sprite whose singleton anchor color is the same.
    - Read the decoration corner for the sprite's internal base turn, read the marker's quadrant inside the
      destination for an additional clockwise quarter-turn, add those two visible turns, then align the
      final rotated anchor cell to the marker.
    - Clear the destination interior first and output only that bordered destination rectangle after all
      rotated stamps are applied.

    Key insight: The tri-color sprite carries two independent visible signals at once: the rare singleton color
    tells you which sprite to use, and the final orientation is the sum of the decoration-corner turn and the
    marker-quadrant turn before imprinting.
    """
    return _solve(grid, apply_internal_turn=True, apply_quadrant_turn=True)


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
SPRITE_TYPE_COUNT = 2
MARKER_COUNT = 3
SOURCE_SPRITE_RANGE = (3, 4)
MARKER_PLACE_RETRIES = 40
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 400


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
    for _ in range(int(n_total - n_types)):
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


def _assign_palette() -> tuple[list[int], int, int, list[int], int, int] | None:
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, 4)
    remaining_nonzero = [c for c in all_colors if c not in wallpaper and c != 0]
    if len(remaining_nonzero) < 5:
        return None
    random.shuffle(remaining_nonzero)

    src_border = int(remaining_nonzero.pop())
    anchors = [int(remaining_nonzero.pop()) for _ in range(SPRITE_TYPE_COUNT)]
    body = int(remaining_nonzero.pop())
    decor = int(remaining_nonzero.pop())

    dst_border_choices = [int(c) for c in wallpaper if int(c) != int(src_border)]
    if not dst_border_choices:
        return None
    dst_border = int(random.choice(dst_border_choices))
    return list(map(int, wallpaper)), int(src_border), int(dst_border), anchors, int(body), int(decor)


def _turn_name_from_int(qturns: int) -> str:
    return TURN_TO_NAME[int(qturns)]


def _render_source_patch(sprite: dict, bg: int) -> np.ndarray:
    return _render_sprite(
        DIRECTION_SHAPES[str(sprite["direction"])],
        int(sprite["body"]),
        int(sprite["anchor"]),
        int(sprite["decor"]),
        int(bg),
    )


def _render_output_patch(sprite: dict, bg: int, turns_cw: int) -> tuple[np.ndarray, tuple[int, int]]:
    source_patch = _render_source_patch(sprite, bg)
    source_mask = source_patch != int(bg)
    anchor_coords = np.argwhere(source_patch == int(sprite["anchor"]))
    ar, ac = map(int, anchor_coords[0])
    out_patch, _out_mask, out_anchor_rc = _rotate_template(
        source_patch, source_mask, (int(ar), int(ac)), int(turns_cw)
    )
    return out_patch, out_anchor_rc


def _anchor_rc(patch: np.ndarray, anchor: int) -> tuple[int, int]:
    coords = np.argwhere(patch == int(anchor))
    if coords.shape[0] != 1:
        raise ValueError("anchor must appear exactly once")
    r, c = map(int, coords[0])
    return int(r), int(c)


def _choose_marker_specs(n_types: int) -> list[tuple[int, int]]:
    quadrants = random.sample([0, 1, 2, 3], MARKER_COUNT)
    if int(n_types) == 1:
        return [(0, int(quadrant)) for quadrant in quadrants]
    repeated_type = random.randrange(int(n_types))
    other_type = 1 - int(repeated_type)
    specs = [
        (int(repeated_type), int(quadrants[0])),
        (int(repeated_type), int(quadrants[1])),
        (int(other_type), int(quadrants[2])),
    ]
    random.shuffle(specs)
    return specs


def _place_markers(
    sprite_types: list[dict], marker_specs: list[tuple[int, int]], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    inner_h = int(dst_h) - 2
    inner_w = int(dst_w) - 2
    empty = np.full((inner_h, inner_w), dst_bg, dtype=int)

    def candidate_anchors(quadrant: int) -> list[tuple[int, int]]:
        cells = [
            (int(mr), int(mc))
            for mr in range(inner_h)
            for mc in range(inner_w)
            if _quadrant_index(int(mr), int(mc), int(inner_h), int(inner_w)) == int(quadrant)
        ]
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
        marker_points: list[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
        if int(idx) == len(marker_specs):
            return board, marker_points

        type_id, quadrant = marker_specs[int(idx)]
        sprite = sprite_types[int(type_id)]
        attempts = 0
        for mr, mc in candidate_anchors(int(quadrant)):
            if attempts >= MARKER_PLACE_RETRIES:
                break
            attempts += 1
            if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc, _type_id, _quadrant in marker_points):
                continue
            turns_cw = (int(sprite["quarter_turns"]) + _quadrant_turns_cw(int(mr), int(mc), int(inner_h), int(inner_w))) % 4
            patch, anchor_rc = _render_output_patch(sprite, dst_bg, int(turns_cw))
            top = int(mr - int(anchor_rc[0]))
            left = int(mc - int(anchor_rc[1]))
            if not can_place_patch(board, patch, int(top), int(left)):
                continue
            next_board = board.copy()
            paste(next_board, patch, int(top), int(left), transparent=int(dst_bg))
            result = backtrack(
                int(idx) + 1,
                next_board,
                marker_points + [(int(mr), int(mc), int(type_id), int(quadrant))],
            )
            if result is not None:
                return result
        return None

    placed = backtrack(0, empty, [])
    if placed is None:
        return None
    out_inner, marker_points = placed

    repeated_counts = Counter(type_id for _mr, _mc, type_id, _quadrant in marker_points)
    quadrant_sets: dict[int, set[int]] = {}
    for _mr, _mc, type_id, quadrant in marker_points:
        quadrant_sets.setdefault(int(type_id), set()).add(int(quadrant))
    if not any(repeated_counts[type_id] >= 2 and len(quadrant_sets[type_id]) >= 2 for type_id in repeated_counts):
        return None
    if len({int(quadrant) for _mr, _mc, _type_id, quadrant in marker_points}) < 3:
        return None
    return out_inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        patch = _render_source_patch(sprite, src_bg)
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


def _choose_directions() -> list[int]:
    qturns = random.sample([0, 1, 2, 3], SPRITE_TYPE_COUNT)
    if all(int(q) == 0 for q in qturns):
        qturns[0] = random.choice([1, 2, 3])
    return [int(q) for q in qturns]


def _gen_attempt() -> np.ndarray | None:
    palette_result = _assign_palette()
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body, decor = palette_result

    src_bg = int(random.choice(wallpaper))
    dst_bg_choices = [int(c) for c in wallpaper if int(c) != int(dst_border)]
    if not dst_bg_choices:
        return None
    dst_bg = int(random.choice(dst_bg_choices))
    if int(body) in {int(src_bg), int(dst_bg)} or int(decor) in {int(src_bg), int(dst_bg)}:
        return None

    directions = _choose_directions()
    sprite_types = [
        {
            "direction": _turn_name_from_int(int(qturns)),
            "quarter_turns": int(qturns),
            "body": int(body),
            "anchor": int(anchor),
            "decor": int(decor),
        }
        for qturns, anchor in zip(directions, anchors)
    ]

    marker_specs = _choose_marker_specs(SPRITE_TYPE_COUNT)

    n_source = random.randint(*SOURCE_SPRITE_RANGE)
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, SPRITE_TYPE_COUNT)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if int(src_h * src_w) <= int(dst_h * dst_w):
        return None

    marker_result = _place_markers(sprite_types, marker_specs, dst_h, dst_w, dst_bg)
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
    for mr, mc, type_id, _quadrant in marker_points:
        dest_rect[1 + int(mr), 1 + int(mc)] = int(sprite_types[int(type_id)]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None

    metrics = fusion_metrics(grid)
    if int(metrics["nonzero_internal_turn_markers"]) < 1:
        return None
    if int(metrics["nonzero_quadrant_turn_markers"]) < 1:
        return None
    if int(metrics["distinct_quadrants"]) < 3:
        return None
    if not bool(metrics["repeated_anchor_spans_quadrants"]):
        return None
    if int(metrics["pixels_changed_vs_internal_only"]) < 1:
        return None
    if int(metrics["pixels_changed_vs_no_rotation"]) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate combo variant after {MAX_GEN_ATTEMPTS} attempts")


def fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)
    _src_rect, _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), _dst_bc, dst_bg, _dst_comps = dst

    raw_turns: dict[int, int] = {}
    for comp in src_comps:
        roles = _classify_tricolor_roles({int(k): int(v) for k, v in comp["colors"].items()})
        if roles is None:
            continue
        anchor_col, decor_col, _body_col = roles
        qturns = _quarter_turns_from_decor(comp["patch"], comp["mask"], int(decor_col))
        if qturns is None:
            continue
        raw_turns[int(anchor_col)] = int(qturns)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    inner_h, inner_w = map(int, inner.shape)
    marker_info: list[tuple[int, int, int, int]] = []
    anchor_quadrants: dict[int, set[int]] = {}
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        anchor_col = int(inner[int(rr), int(cc)])
        internal_turn = int(raw_turns[int(anchor_col)])
        quadrant = int(_quadrant_index(int(rr), int(cc), int(inner_h), int(inner_w)))
        quadrant_turn = int(_quadrant_turns_cw(int(rr), int(cc), int(inner_h), int(inner_w)))
        marker_info.append((int(anchor_col), internal_turn, quadrant, quadrant_turn))
        anchor_quadrants.setdefault(int(anchor_col), set()).add(int(quadrant))

    rotated = _solve(g, apply_internal_turn=True, apply_quadrant_turn=True)
    internal_only = _solve(g, apply_internal_turn=True, apply_quadrant_turn=False)
    unrotated = _solve(g, apply_internal_turn=False, apply_quadrant_turn=False)
    return {
        "marker_count": int(len(marker_info)),
        "internal_turns": tuple(info[1] for info in marker_info),
        "quadrants": tuple(info[2] for info in marker_info),
        "quadrant_turns": tuple(info[3] for info in marker_info),
        "nonzero_internal_turn_markers": int(sum(int(info[1]) != 0 for info in marker_info)),
        "nonzero_quadrant_turn_markers": int(sum(int(info[3]) != 0 for info in marker_info)),
        "distinct_quadrants": int(len({int(info[2]) for info in marker_info})),
        "repeated_anchor_spans_quadrants": any(len(quadrants) >= 2 for quadrants in anchor_quadrants.values()),
        "pixels_changed_vs_internal_only": int(np.count_nonzero(rotated != internal_only)),
        "pixels_changed_vs_no_rotation": int(np.count_nonzero(rotated != unrotated)),
    }
