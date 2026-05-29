# Variant of 8b9c3697
# Dimension: D6 — 网格形状：输出框架 (grid_shape)
# Change: 输出与输入网格尺寸完全相同（19-23行 x 22-27列），仅改变部分像素值... → 仅输出结构内部区域

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis
# ===============
# spec:
#   baseline:
#     - Input generation and docking behavior stay the same as the original task:
#       2-3 comb/bracket structures, 0-1 unmatched structures, one matched red rectangle
#       per matched structure, and extra unmatched red distractors.
#     - The reference task a251c730 establishes the target complexity for D6:
#       the solver may keep a large input scene but emit only a cropped/repacked output view.
#   concrete_behavior:
#     - First solve the original task on the full canvas: matched red rectangles slide inward,
#       their travel paths turn black, unmatched reds disappear.
#     - Then, for each structure component found in the input, take the inclusive bbox of that
#       structure and crop that rectangle from the solved full-canvas result.
#     - Sort these cropped rectangles by the visible top-left position of their source structures
#       (row-major order: r0, then c0) and pack them left-to-right into one output strip,
#       top-aligned on a fresh background canvas.
#   behavior_trace:
#     - Original flow: a structure at bbox (3:6, 5:13) receives a red block from below; the
#       full 19x22 output still contains the whole board, including the black trail extending
#       outside the structure bbox.
#     - Variant flow: the same slide is solved first, but only rows 3..6 and cols 5..13 from
#       that solved board are kept for this structure. A second structure is cropped the same
#       way, then the two crops are concatenated horizontally in input scan order.
#     - Difference: the docking logic is unchanged; only the final frame is cropped and repacked.
#   complexity_calibration:
#     - Keep all original spatial ambiguity: 2-3 structures, arbitrary rotations, matched vs
#       unmatched structures, and 2-8 distractor reds remain.
#     - The new rule adds exactly one new inference step: identify structure bboxes and pack
#       their solved interior rectangles. The order is deterministic and visually checkable.
#   implicit_assumptions_review:
#     - The original ray-cast logic assumes sockets are bg cells strictly inside a structure bbox.
#       This remains valid because the input scene is unchanged.
#     - The crop step must happen after full-canvas solving; cropping earlier would erase the
#       outer part of the travel path before it can influence the visible cells inside the bbox.
#     - Structure detection still assumes structures are connected non-bg, non-red components.
#       The generator preserves padding so separate structures never touch.
#     - Packing assumes a stable visible order; row-major order by structure position is human-
#       observable and does not depend on hidden ids or color numbers.
#   placement_legality:
#     - Structure placement is unchanged: non-overlapping patches with padding, so bbox crops are
#       disjoint and well-defined.
#     - Packed width stays manageable because each structure bbox is at most 9 cells wide and
#       there are only 2-3 structures; the solver allocates the exact packed width anyway.
#   visual_observability:
#     - Output height drops from 19-23 down to 4-9 in every sample, so D6 is immediately visible.
#     - The kept content is still rich: structure walls, teeth, the docked red, and any black
#       trail segments that lie inside the bbox survive, while outer clutter disappears.
#   cleanliness:
#     - No extra color mapping is introduced.
#     - Packing into a single strip avoids a noisy large canvas and makes each crop legible.


# --------------- shared helpers ---------------

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _extract_structures(g: np.ndarray, bg: int) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int, int, int]]]:
    """Return structure coordinate sets and inclusive bboxes."""
    struct_mask = (g != int(bg)) & (g != int(Color.RED))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))
    return struct_coords, struct_bbox


def _solve_full_canvas(
    g: np.ndarray,
    bg: int,
    struct_coords: list[set[tuple[int, int]]],
    struct_bbox: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Apply the original docking rule and return the full-canvas solved grid."""
    H, W = g.shape
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)

    assert red_val in set(map(int, np.unique(g)))

    red_comps = components_mask(g == red_val, connectivity=4)
    assert red_comps

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
        for rc in pts:
            red_at[rc] = int(rid)

    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []
    for sid, coords in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best = None

        for dr, dc in _DIRS:
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            for r, c in coords:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                rr, cc = int(nr), int(nc)
                while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)

                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == red_val:
                    rid = int(red_at[(int(rr), int(cc))])
                    rec = hits[rid]
                    rec[0] += 1
                    dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                    if int(dist) < int(rec[1]):
                        rec[1] = int(dist)
                        rec[2] = (int(r), int(c))

            if not hits:
                continue

            rid_best = max(hits.keys(), key=lambda k: (int(hits[k][0]), -int(hits[k][1])))
            cnt, mind, anchor = hits[rid_best]
            cand = (int(cnt), -int(mind), int(rid_best), (int(dr), int(dc)), anchor)
            if best is None or cand > best:
                best = cand

        if best is not None:
            cnt, neg_mind, rid, dirv, anchor = best
            struct_best.append(
                ((int(cnt), int(neg_mind)), int(rid), (int(dirv[0]), int(dirv[1])), anchor, int(sid))
            )

    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for _score, rid, (dr, dc), anchor, _sid in struct_best:
        if int(rid) in used_red or anchor is None:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dr), int(dc)), (int(anchor[0]), int(anchor[1]))))

    out = g.copy()
    out[out == red_val] = int(bg)

    for rid, (dr, dc), (ar, ac) in assignments:
        coords = red_coords[int(rid)]
        r0, c0, r1, c1 = map(int, red_bbox[int(rid)])
        tr, tc = int(ar + dr), int(ac + dc)

        if dr == 0 and dc == 1:
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = black_val

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = red_val

    return out


def _pack_structure_crops(
    solved_full: np.ndarray, struct_bbox: list[tuple[int, int, int, int]], bg: int
) -> np.ndarray:
    """Crop each structure bbox from solved_full and pack them left-to-right in row-major order."""
    ordered = sorted(struct_bbox, key=lambda b: (int(b[0]), int(b[1]), int(b[2]), int(b[3])))
    crops = [solved_full[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy() for r0, c0, r1, c1 in ordered]
    out_h = max(int(crop.shape[0]) for crop in crops)
    out_w = sum(int(crop.shape[1]) for crop in crops)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    cc = 0
    for crop in crops:
        h, w = crop.shape
        out[: int(h), int(cc) : int(cc + w)] = crop
        cc += int(w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide visible red blocks into comb sockets, then emit only the structure crops.

    Input structure:
    - The canvas still contains large non-bg, non-red comb/bracket structures plus red blocks
      elsewhere on the board.
    - Structures can face any cardinal direction, and some red blocks are unrelated distractors.

    Rule:
    - First perform the original docking process: each structure claims the red block seen by
      ray-casting from its internal sockets, paints the travelled path black, and unmatched reds vanish.
    - Then crop the solved board to each structure's inclusive bbox and pack those rectangles
      left-to-right in row-major structure order.

    Key insight: The motion rule is unchanged; the new challenge is to recognize that the final
    answer is not the full solved scene but a repacked strip of structure-local solved regions.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    struct_coords, struct_bbox = _extract_structures(g, bg)
    solved_full = _solve_full_canvas(g, bg, struct_coords, struct_bbox)
    return _pack_structure_crops(solved_full, struct_bbox, bg)


# --------------- generator (codex-att2, D6 variant) ---------------

# ── CFG: canvas & input grid ────────────────────────────────────────
GRID_H_RANGE = (19, 23)          # input grid height
GRID_W_RANGE = (22, 27)          # input grid width
BG_COLORS = (3, 4)               # candidate background colors
STRUCT_COLORS = (1, 4, 8)        # candidate structure colors

# ── CFG: structures ─────────────────────────────────────────────────
STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))  # (back_wall_width, teeth_count)
STRUCT_COUNT_RANGE = (2, 3)      # number of comb/bracket structures
UNMATCHED_PROB = 0.35            # probability of 1 unmatched structure
SLIDE_RANGE = (4, 8)             # red slide distance into socket
STRUCT_PADDING = 4               # min gap between placed structures

# ── CFG: distractors ────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5                # min total red components (matched + distractors)
TOTAL_RED_MAX = 11               # max total red components
DISTRACTOR_MIN = 2               # absolute min distractors
DISTRACTOR_MAX = 8               # absolute max distractors

# ── CFG: retry ──────────────────────────────────────────────────────
MAX_RETRIES = 300                # outer retry loop for generate_input


def _dilate4(mask: np.ndarray) -> np.ndarray:
    """4-connected dilation of a boolean mask."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    """Build a 4-row normalized comb structure (opening facing down)."""
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start : start + teeth] = sc
    return g


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build input/output/structure patches for a matched structure+red pair."""
    red_val = int(Color.RED)
    black_val = int(Color.BLACK)
    red_h = min(teeth, 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    for off in range(1, slide + 1):
        output_patch[2 + off : 2 + off + red_h, start : start + teeth] = black_val
    output_patch[2 : 2 + red_h, start : start + teeth] = red_val
    input_patch[2 + slide : 2 + slide + red_h, start : start + teeth] = red_val

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build patches for an unmatched structure (no red component)."""
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {"structure": structure, "input": structure.copy(), "output": structure.copy()}


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    """Return a boolean mask of cells on line-of-sight from any socket outward."""
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for dr, dc in _DIRS:
            for r, c in item["coords"]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if struct_grid[nr, nc] != bg:
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = nr, nc
                while 0 <= rr < h and 0 <= cc < w and struct_grid[rr, cc] == bg:
                    forbidden[rr, cc] = True
                    rr += dr
                    cc += dc
    return forbidden


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    """Try to place patch on canvas; return (r, c) or None."""
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _unique_matched_visibility(grid: np.ndarray, bg: int, expected_matched: int) -> bool:
    """Check that exactly expected_matched reds are each visible from exactly one structure."""
    h, w = grid.shape
    red_val = int(Color.RED)
    red_comps = components_mask(grid == red_val, connectivity=4)
    if len(red_comps) != expected_matched:
        return False

    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        for r, c in coords:
            red_at[(int(r), int(c))] = rid

    struct_mask = (grid != bg) & (grid != red_val)
    struct_comps = components_mask(struct_mask, connectivity=4)
    red_to_structs: dict[int, set[int]] = {}
    struct_hits = 0

    for sid, coords in enumerate(struct_comps):
        pts = {(int(r), int(c)) for r, c in coords}
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        visible_rids: set[int] = set()
        for dr, dc in _DIRS:
            for r, c in pts:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if grid[nr, nc] != bg:
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = nr, nc
                while 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == bg:
                    rr += dr
                    cc += dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == red_val:
                    visible_rids.add(red_at[(rr, cc)])
        if len(visible_rids) > 1:
            return False
        if visible_rids:
            struct_hits += 1
            rid = next(iter(visible_rids))
            red_to_structs.setdefault(rid, set()).add(sid)

    if struct_hits != expected_matched:
        return False
    return len(red_to_structs) == expected_matched and all(len(sids) == 1 for sids in red_to_structs.values())


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    """Place all structure patches on fresh canvases. Returns (inp, out, struct_grid, placed) or None."""
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] == sc)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append({"coords": coords, "bbox": (min(rows), min(cols), max(rows), max(cols))})

    return inp, out, struct_grid, placed


def _place_distractors(
    inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int
) -> bool:
    """Scatter distractor red rectangles avoiding line-of-sight. Returns False on failure."""
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    red_mask = inp == int(Color.RED)

    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(red_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = sc
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is not None:
                paste(inp, patch, loc[0], loc[1], transparent=bg)
                red_mask[loc[0] : loc[0] + rh, loc[1] : loc[1] + rw] = True
                ok = True
                break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist:
        return None
    distractor_count = random.randint(min_dist, max_dist)

    specs = []
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, teeth, slide, rot_k, bg, sc))
        else:
            specs.append(_unmatched_patches(width, teeth, rot_k, bg, sc))
    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg, sc)
    if result is None:
        return None
    inp, out_full, struct_grid, placed = result

    if not _unique_matched_visibility(inp, bg, matched):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    expected = _pack_structure_crops(out_full, [item["bbox"] for item in placed], bg)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D6 variant")
