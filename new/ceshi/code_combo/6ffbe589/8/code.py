# Combo variant of 6ffbe589 (index 8)
# combo_spec: D5/0+D17/0
# Name: 非同心层+旋转后覆盖

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The source task cropped one square board whose three color masks were
#      concentric and non-overlapping. After reading same-color outside cues,
#      the solver could rotate each visible color mask in place because every
#      destination cell belonged to at most one layer.
#    - This combo changes both assumptions at once:
#      1) the three logical layers are no longer concentric; their target
#         shapes live in three different board regions (top-left, top-right,
#         bottom-left);
#      2) once each layer is rotated back to its target pose, their central
#         "arms" may claim the same cells, and later redraws must overwrite
#         earlier ones.
#    - Concrete example:
#      old task:
#        color A = outer border, color B = middle ring, color C = center
#        sprite. Rotating A/B/C never creates collisions, so mutating `out`
#        color-by-color is safe.
#      new task:
#        color A's rotated target is a top-left corner sprite with a horizontal
#        arm into the hub, color B's rotated target is a top-right sprite that
#        shares that hub row, and color C's rotated target is a bottom-left
#        sprite that shares the hub column. The visible output must therefore
#        be painted in fixed slot order:
#        top-left first, top-right second, bottom-left last.
#      difference:
#        the solver must reconstruct full layer masks from the input, rotate
#        them independently, identify which target slot each rotated mask
#        belongs to, and then re-render from scratch with overwrite semantics.
# b) Complexity calibration
#    - The board still has exactly three colors. That keeps the correspondence
#      with the original task simple enough for humans to track.
#    - The new geometry uses only three repeated slot types. This is enough to
#      make "non-concentric" visually obvious without introducing arbitrary
#      free-form placement.
#    - Overlap is limited to a small central hub, so the overwrite rule is
#      unmistakable but does not turn the board into unreadable clutter.
# c) Hidden-assumption audit for main()
#    - `crop_components_near_largest(...)` still assumes the board is the
#      unique largest component and cues stay outside the margin-3 expansion.
#      The generator enforces both.
#    - The original "rotate visible colors in whatever order colors_present()
#      returns" is no longer valid. Color ids have no semantic order now.
#    - In-place erasure/repaint also breaks: hidden target pixels can exist
#      underneath another layer after rotation. The solver now renders onto a
#      blank board in slot order instead.
#    - Slot identification needs a new assumption: once rotated to the target
#      pose, one mask touches top+left, one touches top+right, and one touches
#      bottom+left. The generator guarantees those signatures.
# d) Placement legality
#    - Each board sample is still pasted into a 20x20 canvas with cues placed
#      at Chebyshev distance >= 4 from the board and from one another.
#    - The generator rejects any candidate where the inverse-rotated input
#      masks overlap, disconnect the board, or fail to span a square crop.
# e) Visual observability
#    - A human can see three separated corner sprites rather than concentric
#      rings. The same-color cue rule stays unchanged, so the novel part to
#      infer is the relocation/overlap behavior after rotation.
#    - The central hub repeatedly shows the same overwrite hierarchy across
#      examples: the top-right slot covers the top-left on their shared row,
#      and the bottom-left slot covers the top-left on their shared column.
# f) Visual cleanliness
#    - Most pixels remain localized near one corner region, with only a thin
#      arm reaching toward the hub. This preserves a tidy board silhouette.
# g) Change inferability
#    - No hidden sorting uses numeric colors. Region identity is encoded by
#      visible geometry: which edges the rotated mask touches.
#    - The overwrite order is likewise tied to visible slots, not to colors or
#      cue positions.
# h) New solver assumptions and generator guarantees
#    - Assumption: exactly three board colors appear in the crop.
#      Guarantee: generator always uses three non-background colors.
#    - Assumption: every rotating target mask is rotationally asymmetric.
#      Guarantee: generator checks this after overlaps are added.
#    - Assumption: the rotated masks classify uniquely into TL / TR / BL.
#      Guarantee: generator keeps each target mask touching only its intended
#      board edges.
# i) Degeneracy check target
#    - The fusion should not collapse back to the original task. The generator
#      therefore forces non-zero rotations for all three layers and constructs
#      true target overlap in the hub, so the combo solver's output differs
#      from the source solver on the same input.


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Local helper (only used by this solver): crop the cluster near the largest non-bg component.

    Returns:
        (crop, crop_bbox, bg) where crop_bbox is in the original grid coordinates (inclusive).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    mr0, mc0, mr1, mc1 = map(int, largest[0]["bbox"])

    h, w = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(h - 1), int(mr1 + margin))
    ec1 = min(int(w - 1), int(mc1 + margin))

    keep_mask = np.zeros((h, w), dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 < er0 or r0 > er1 or c1 < ec0 or c0 > ec1:
            continue
        coords = comp["coords"]
        keep_mask[coords[:, 0], coords[:, 1]] = True

    keep_coords = np.argwhere(keep_mask)
    assert keep_coords.size > 0, "expected the expanded board crop to retain the largest component"

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def _edge_touches(mask: np.ndarray) -> tuple[bool, bool, bool, bool]:
    n = int(mask.shape[0])
    return (
        bool(mask[0, :].any()),
        bool(mask[:, n - 1].any()),
        bool(mask[n - 1, :].any()),
        bool(mask[:, 0].any()),
    )


def _slot_index(mask: np.ndarray) -> int:
    top, right, bottom, left = _edge_touches(mask)
    if top and left and (not bottom):
        return 0
    if top and right and (not bottom):
        return 1
    if bottom and left and (not top):
        return 2
    raise AssertionError("expected each rotated layer to map uniquely to TL/TR/BL")


def _cue_rotation(outside_grid: np.ndarray, color: int, *, bg: int) -> int:
    comps = find_components(
        outside_grid,
        bg=int(bg),
        connectivity=8,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    assert len(comps) <= 1, f"expected at most one outside cue for color {color}"
    cue_size = int(comps[0]["area"]) if comps else 0
    return (-cue_size) % 4


def _render_layers(shape: tuple[int, int], ordered_layers: list[tuple[int, np.ndarray]], *, bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color, mask in ordered_layers:
        out[mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster, rotate the scattered color layers, and redraw them with slot-based overwrite order.

    Input structure:
    - Background is the dominant color.
    - The largest foreground cluster is a square board containing exactly three non-concentric color layers.
    - Each layer occupies a different corner region of the board and has a same-color monochrome cue outside the board.

    Rule:
    - Crop away the distant cues and isolate the square board.
    - For each board color, read its outside cue area and rotate that color mask by the encoded quarter-turn count.
    - Identify the rotated target slot from geometry (`top-left`, `top-right`, or `bottom-left`).
    - Repaint the rotated masks onto a blank crop in that slot order so later slots cover earlier ones on overlap.

    Key insight:
    Cue size still tells you how far to rotate a same-color layer, but because the layers are scattered and can collide
    after rotation, the final answer must be recomposed in a fixed region order rather than by mutating the visible crop.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    assert crop.shape[0] == crop.shape[1], "expected a square board crop"

    colors = colors_present(crop, exclude={int(bg)})
    assert len(colors) == 3, "expected exactly three board colors"

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    rotated_by_slot: dict[int, tuple[int, np.ndarray]] = {}
    for color in colors:
        mask = crop == int(color)
        rot_k = _cue_rotation(outside_grid, int(color), bg=int(bg))
        rotated = np.rot90(mask, int(rot_k))
        slot = _slot_index(rotated)
        assert slot not in rotated_by_slot, "expected one rotated mask per slot"
        rotated_by_slot[slot] = (int(color), rotated)

    assert set(rotated_by_slot) == {0, 1, 2}, "expected all target slots to be populated"
    ordered_layers = [rotated_by_slot[idx] for idx in (0, 1, 2)]
    return _render_layers(crop.shape, ordered_layers, bg=int(bg))


# --------------- generator ---------------

CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_COLORS = 3

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4

GEN_RETRIES = 700
SHAPE_RETRIES = 120


def _is_rot_asym(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segment endpoints must align horizontally or vertically")


def _transform_coords(coords: list[tuple[int, int]], n: int, slot: int) -> list[tuple[int, int]]:
    if slot == 0:
        return [(int(r), int(c)) for r, c in coords]
    if slot == 1:
        return [(int(r), int(n - 1 - c)) for r, c in coords]
    if slot == 2:
        return [(int(n - 1 - r), int(c)) for r, c in coords]
    raise ValueError("unknown slot")


def _base_corner_blob() -> list[tuple[int, int]]:
    return [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 2)]


def _target_masks(n: int) -> list[np.ndarray]:
    mid = n // 2
    tl_endpoint = (mid - 1, mid - 1)
    tr_endpoint = (mid - 1, mid + 1)
    bl_endpoint = (mid + 1, mid - 1)

    tl_blob = _base_corner_blob()
    starts = {
        0: (3, 2),
        1: (3, n - 3),
        2: (n - 4, 2),
    }
    endpoints = {
        0: tl_endpoint,
        1: tr_endpoint,
        2: bl_endpoint,
    }

    tl_masks: list[np.ndarray] = []
    # Minimal deliberate overlaps: a 3-cell hub row shared by TL/TR and a
    # single hub cell shared by TL/BL. Larger shared segments made the inverse
    # input layout collide too often.
    shared_h = [(mid - 1, c) for c in range(mid, mid + 3)]
    shared_v = [(mid - 1, mid)]

    for slot in range(3):
        mask = np.zeros((n, n), dtype=bool)
        for r, c in _transform_coords(tl_blob, n, slot):
            mask[r, c] = True

        if slot == 0:
            via1 = (3, mid - 2)
            via2 = (mid - 1, mid - 2)
        elif slot == 1:
            via1 = (mid - 2, n - 3)
            via2 = (mid - 2, mid + 1)
        else:
            via1 = (n - 4, mid - 2)
            via2 = (mid + 1, mid - 2)

        route = []
        route.extend(_segment(starts[slot], via1))
        route.extend(_segment(via1, via2))
        route.extend(_segment(via2, endpoints[slot]))
        for r, c in route:
            mask[int(r), int(c)] = True

        tl_masks.append(mask)

    tl_masks[0][tuple(np.array(shared_h).T)] = True
    tl_masks[1][tuple(np.array(shared_h).T)] = True
    tl_masks[0][tuple(np.array(shared_v).T)] = True
    tl_masks[2][tuple(np.array(shared_v).T)] = True

    return tl_masks


def _pick_rotations(n: int) -> list[int]:
    # Fixed non-zero tuples chosen so the backward-rotated input remains
    # disjoint and square-spanning for the matching board parity.
    return [1, 2, 3] if (n % 2) == 0 else [3, 1, 2]


def _cue_patch(color: int, area: int) -> np.ndarray:
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    arr = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        arr[r, c] = int(color)
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = patch.shape
    mask = patch != 0
    candidates = []
    for r in range(CANVAS_SIZE - h + 1):
        for c in range(CANVAS_SIZE - w + 1):
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            candidates.append((r, c, bb))
    if not candidates:
        return False
    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _board_anchor(n: int) -> tuple[int, int]:
    spare = CANVAS_SIZE - n
    side = random.randrange(4)
    if side == 0:
        return spare, random.randint(0, spare)
    if side == 1:
        return 0, random.randint(0, spare)
    if side == 2:
        return random.randint(0, spare), spare
    return random.randint(0, spare), 0


def _compose_input_layers(dst_masks: list[np.ndarray], colors: list[int], rot_ks: list[int]) -> np.ndarray | None:
    n = int(dst_masks[0].shape[0])
    board = np.zeros((n, n), dtype=int)
    for color, dst_mask, rot_k in zip(colors, dst_masks, rot_ks):
        inv_mask = np.rot90(dst_mask, (-int(rot_k)) % 4)
        if np.any((board != 0) & inv_mask):
            return None
        board[inv_mask] = int(color)
    comps = find_components(board, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 1:
        return None
    bbox = tuple(map(int, comps[0]["bbox"]))
    if bbox != (0, 0, n - 1, n - 1):
        return None
    return board


def _overlap_metrics(dst_masks: list[np.ndarray]) -> dict[str, int]:
    cover = sum(mask.astype(int) for mask in dst_masks)
    return {
        "hidden_cells": int(np.sum(cover >= 2)),
        "max_stack": int(cover.max()),
        "pair_01": int(np.sum(dst_masks[0] & dst_masks[1])),
        "pair_02": int(np.sum(dst_masks[0] & dst_masks[2])),
        "pair_12": int(np.sum(dst_masks[1] & dst_masks[2])),
    }


def _target_is_valid(dst_masks: list[np.ndarray]) -> bool:
    for idx, mask in enumerate(dst_masks):
        if not _is_rot_asym(mask):
            return False
        if _slot_index(mask) != idx:
            return False
    metrics = _overlap_metrics(dst_masks)
    return (
        metrics["hidden_cells"] >= 3
        and metrics["max_stack"] >= 2
        and metrics["pair_01"] > 0
        and metrics["pair_02"] > 0
    )


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    colors = random.sample(range(1, 10), NUM_COLORS)
    n = random.randint(BOARD_N_MIN, BOARD_N_MAX)
    rot_ks = _pick_rotations(n)

    dst_masks = None
    in_board = None
    for _ in range(SHAPE_RETRIES):
        candidate_masks = _target_masks(n)
        if not _target_is_valid(candidate_masks):
            continue
        candidate_in = _compose_input_layers(candidate_masks, colors, rot_ks)
        if candidate_in is None:
            continue
        dst_masks = candidate_masks
        in_board = candidate_in
        break
    if dst_masks is None or in_board is None:
        return None

    out_board = _render_layers((n, n), list(zip(colors, dst_masks)), bg=0)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [
        (col, random.choice([1, 5]) if (-k) % 4 == 1 else (-k) % 4)
        for col, k in zip(colors, rot_ks)
    ]
    random.shuffle(cue_items)
    for col, area in cue_items:
        if not _place_cue(canvas, _cue_patch(col, area), board_bbox, cue_bboxes):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None

    solved = main(canvas)
    if not np.array_equal(solved, out_board):
        return None
    return canvas, out_board


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _out = result
            return grid
    raise RuntimeError("failed to generate a valid combo 6ffbe589 input")
