# Variant of 8b9c3697
# Dimension: D11 — 干扰项分布约束 (meta)
# Change: 干扰红色色块（2-8个）散布在网格中，必须不在任何结构插槽的视线（line-of-sight）范围内... → 干扰项可在视线范围内但不匹配

from __future__ import annotations

import random
import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D11 variant
# ===================================
# a) Concrete behavior and trace
#    The original task lets a structure choose the first red blob found by casting
#    rays from interior socket cells through the opening, and the generator avoids
#    putting distractors on those rays at all. In this variant, at least one
#    distractor is deliberately placed on a structure's opening line, but with a
#    rectangle size that cannot fit that structure's docking slot. A solver that
#    only checks alignment would produce a false positive; the new solver must also
#    infer the expected rectangle shape from the comb teeth and reject mismatches.
#    Small trace:
#    - Original unmatched structure + aligned 2x1 distractor: old code sees one hit
#      and docks it, even though the slot expects 1x1 or 2x2.
#    - Variant: the same distractor is still visible on the opening ray, but shape
#      filtering rejects it, so the structure stays empty and the red vanishes.
# b) Complexity calibration
#    The change stays focused on D11. Structures, colors, movement, and black trail
#    semantics are unchanged. Only one extra reasoning axis is introduced:
#    "aligned" is necessary but not sufficient; the rectangle dimensions must match
#    the slot implied by the teeth. To keep the effect visible from few examples,
#    every generated grid contains at least one visible-but-invalid distractor.
# c) Implicit-assumption audit of the original main()
#    - The ray loop assumed "first visible red == candidate". This no longer holds.
#    - The scoring used only hit count and distance, so any visible red could win.
#    - The final shift assumed the chosen component had the correct extent for the
#      socket; previously harmless because the generator guaranteed it.
#    The new main() therefore parses each structure, computes its expected docked
#    rectangle, and selects reds by exact shape + side alignment, not by first hit.
# d) Placement legality
#    Structures are still placed with padding and never overlap. Visible decoys are
#    placed strictly outside the structure bbox, on the opening side, with a clear
#    background corridor from the slot to the decoy so that the line-of-sight is
#    visually genuine. Remaining distractors keep the old non-LOS rule to avoid
#    accidental extra candidates.
# e) Visual observability
#    Each sample contains at least one red rectangle sitting directly in front of a
#    structure opening yet disappearing in the output without docking. That makes
#    the new rule visible: some aligned reds move and leave black trails, while
#    other aligned reds are ignored because their rectangles have the wrong size.
# f) Visual cleanliness
#    The required visible decoy is limited to one per unmatched structure and is
#    separated from other reds by 4-neighborhood padding. Other distractors still
#    avoid the structure rays, preventing dense clutter around the openings.
# g) Color reasoning
#    No new color mapping is introduced. Red remains the movable/vanishing color,
#    black remains the trail, and structure colors remain unchanged.
# h) New solver assumptions and generator guarantees
#    - Structures are rotations of the known comb template. The generator preserves
#      that exactly, so local parsing from the full back wall is reliable.
#    - A true match is always a solid rectangle whose bbox equals the inferred slot.
#      The generator continues to build matched reds that way.
#    - At most one red rectangle shape-aligns with each structure. The generator
#      validates this by oracle checking `main(input) == output` and retries if not.


GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 300
VISIBLE_DECOYS_PER_GRID = 1

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox_hw(b: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(b[2] - b[0] + 1), int(b[3] - b[1] + 1)


def _shift_bbox(b: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    return int(b[0] + dr), int(b[1] + dc), int(b[2] + dr), int(b[3] + dc)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = g[1, width - 2] = sc
    g[2:4, 1] = g[2:4, width - 2] = sc
    g[1, start:start + teeth] = sc
    return g


def _analyze_local_structure(local_mask: np.ndarray) -> dict:
    top_full = bool(local_mask[0, :].all())
    bottom_full = bool(local_mask[-1, :].all())
    left_full = bool(local_mask[:, 0].all())
    right_full = bool(local_mask[:, -1].all())

    if top_full:
        open_dir = (1, 0)
        rot_to_base = 0
    elif left_full:
        open_dir = (0, 1)
        rot_to_base = 3
    elif bottom_full:
        open_dir = (-1, 0)
        rot_to_base = 2
    elif right_full:
        open_dir = (0, -1)
        rot_to_base = 1
    else:
        raise AssertionError("structure does not match the comb template")

    canonical = np.rot90(local_mask, k=rot_to_base)
    if canonical.shape[0] != 4:
        raise AssertionError("unexpected comb depth")

    tooth_row = canonical[1].copy()
    tooth_row[1] = False
    tooth_row[-2] = False
    tooth_cols = np.flatnonzero(tooth_row)
    if len(tooth_cols) == 0 or np.any(np.diff(tooth_cols) != 1):
        raise AssertionError("teeth must form one contiguous block")

    teeth = int(len(tooth_cols))
    start = int(tooth_cols[0])
    red_h = int(min(teeth, 2))

    dock_canonical = np.zeros_like(canonical, dtype=bool)
    dock_canonical[2:2 + red_h, start:start + teeth] = True
    dock_local = np.rot90(dock_canonical, k=(4 - rot_to_base) % 4)
    dock_bbox = tuple(map(int, bbox_coords(np.argwhere(dock_local))))

    return {
        "dir": open_dir,
        "teeth": teeth,
        "shape": _bbox_hw(dock_bbox),
        "dock_bbox_local": dock_bbox,
    }


def _red_infos(g: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    red = int(Color.RED)
    infos: list[dict] = []
    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(components_mask(g == red, connectivity=4)):
        pts = [(int(r), int(c)) for r, c in coords]
        box = tuple(map(int, bbox_coords(coords)))
        h, w = _bbox_hw(box)
        info = {
            "coords": pts,
            "bbox": box,
            "shape": (h, w),
            "is_rect": len(pts) == h * w,
        }
        infos.append(info)
        for rc in pts:
            red_at[rc] = int(rid)
    return infos, red_at


def _structure_infos(g: np.ndarray, bg: int) -> list[dict]:
    red = int(Color.RED)
    struct_mask = (g != int(bg)) & (g != red)
    infos: list[dict] = []
    for sid, coords in enumerate(components_mask(struct_mask, connectivity=4)):
        pts = [(int(r), int(c)) for r, c in coords]
        box = tuple(map(int, bbox_coords(coords)))
        r0, c0, r1, c1 = box
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local_mask[r - r0, c - c0] = True
        meta = _analyze_local_structure(local_mask)
        infos.append(
            {
                "sid": int(sid),
                "coords": set(pts),
                "bbox": box,
                "dir": meta["dir"],
                "shape": meta["shape"],
                "teeth": meta["teeth"],
                "dock_bbox": _shift_bbox(meta["dock_bbox_local"], r0, c0),
            }
        )
    return infos


def _candidate_distance(
    grid: np.ndarray,
    bg: int,
    struct_info: dict,
    red_bbox: tuple[int, int, int, int],
) -> int | None:
    sr0, sc0, sr1, sc1 = struct_info["bbox"]
    tr0, tc0, tr1, tc1 = struct_info["dock_bbox"]
    rr0, rc0, rr1, rc1 = red_bbox
    dr, dc = struct_info["dir"]
    red = int(Color.RED)

    if dr == 1:
        if rc0 != tc0 or rc1 != tc1 or rr0 <= sr1:
            return None
        corridor = grid[tr1 + 1:rr0, tc0:tc1 + 1]
    elif dr == -1:
        if rc0 != tc0 or rc1 != tc1 or rr1 >= sr0:
            return None
        corridor = grid[rr1 + 1:tr0, tc0:tc1 + 1]
    elif dc == 1:
        if rr0 != tr0 or rr1 != tr1 or rc0 <= sc1:
            return None
        corridor = grid[tr0:tr1 + 1, tc1 + 1:rc0]
    else:
        if rr0 != tr0 or rr1 != tr1 or rc1 >= sc0:
            return None
        corridor = grid[tr0:tr1 + 1, rc1 + 1:tc0]

    if corridor.size and np.any((corridor != int(bg)) & (corridor != red)):
        return None
    return int(abs(rr0 - tr0) + abs(rc0 - tc0))


def _solve_assignments(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, tuple[int, int], int]], list[dict], list[dict]]:
    red_infos, _ = _red_infos(grid)
    struct_infos = _structure_infos(grid, bg)

    best_per_struct: list[tuple[int, int, int] | None] = []
    for sinfo in struct_infos:
        best: tuple[int, int, int] | None = None
        for rid, rinfo in enumerate(red_infos):
            if not rinfo["is_rect"] or rinfo["shape"] != sinfo["shape"]:
                continue
            dist = _candidate_distance(grid, bg, sinfo, rinfo["bbox"])
            if dist is None:
                continue
            cand = (int(dist), int(rid), int(sinfo["sid"]))
            if best is None or cand < best:
                best = cand
        best_per_struct.append(best)

    ranked = [cand for cand in best_per_struct if cand is not None]
    ranked.sort()
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], int]] = []
    for _dist, rid, sid in ranked:
        if rid in used_red:
            continue
        used_red.add(rid)
        red_box = red_infos[rid]["bbox"]
        dock_box = struct_infos[sid]["dock_bbox"]
        shift = (int(dock_box[0] - red_box[0]), int(dock_box[1] - red_box[1]))
        assignments.append((int(rid), shift, int(sid)))
    return assignments, red_infos, struct_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only correctly sized red rectangles into the cavities of comb-like structures, leaving black trails.

    Input structure:
    - The background is the dominant color. Large non-background, non-red components are comb/bracket structures.
    - Red connected components are rectangles outside the structures; some are valid movers and some are aligned decoys.

    Rule:
    - Parse each structure to recover its opening direction and the exact rectangle size implied by its teeth.
    - A red component docks only if it is a solid rectangle of that exact size and is positioned on the opening side with the same cross-axis span.
    - Move each matched red straight into the interior slot, paint its swept path black, and erase every unmatched red.

    Key insight:
    Alignment to an opening is no longer enough: a visible red may sit on the correct ray yet still be rejected because its rectangle does not match the slot shape.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    red = int(Color.RED)
    black = int(Color.BLACK)

    assert red in set(map(int, np.unique(g)))
    assert components_mask(g == red, connectivity=4)

    assignments, red_infos, _struct_infos = _solve_assignments(g, bg)

    out = g.copy()
    out[out == red] = int(bg)

    for rid, (shift_r, shift_c), _sid in assignments:
        coords = red_infos[rid]["coords"]
        steps = int(max(abs(shift_r), abs(shift_c)))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(black)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(red)

    return out


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    red = int(Color.RED)
    black = int(Color.BLACK)
    red_h = min(teeth, 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    for off in range(1, slide + 1):
        output_patch[2 + off:2 + off + red_h, start:start + teeth] = black
    output_patch[2:2 + red_h, start:start + teeth] = red
    input_patch[2 + slide:2 + slide + red_h, start:start + teeth] = red

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "matched": True,
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {
        "structure": structure,
        "input": structure.copy(),
        "output": structure.copy(),
        "matched": False,
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
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
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _shape_aware_match_count(grid: np.ndarray, bg: int) -> int:
    assignments, _red_infos_unused, _struct_infos_unused = _solve_assignments(grid, bg)
    return len(assignments)


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
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

        lr0, lc0, lr1, lc1 = map(int, bbox_coords(coords_arr))
        local_mask = spec["structure"][lr0:lr1 + 1, lc0:lc1 + 1] == sc
        meta = _analyze_local_structure(local_mask)

        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "dir": meta["dir"],
                "shape": meta["shape"],
                "teeth": meta["teeth"],
                "dock_bbox": _shift_bbox(meta["dock_bbox_local"], r0 + lr0, c0 + lc0),
                "matched": bool(spec["matched"]),
            }
        )

    return inp, out, struct_grid, placed


def _visible_decoy_shapes(item: dict) -> list[tuple[int, int]]:
    dh, dw = item["shape"]
    dr, dc = item["dir"]
    out = []
    for rh, rw in DISTRACTOR_SHAPES:
        if (rh, rw) == (dh, dw):
            continue
        if dr != 0 and rw <= dw:
            out.append((int(rh), int(rw)))
        if dc != 0 and rh <= dh:
            out.append((int(rh), int(rw)))
    return out


def _strict_bg_path_ok(
    inp: np.ndarray,
    item: dict,
    top: int,
    left: int,
    rh: int,
    rw: int,
    bg: int,
) -> bool:
    tr0, tc0, tr1, tc1 = item["dock_bbox"]
    dr, dc = item["dir"]

    if dr == 1:
        path = inp[tr1 + 1:top, left:left + rw]
    elif dr == -1:
        path = inp[top + rh:tr0, left:left + rw]
    elif dc == 1:
        path = inp[top:top + rh, tc1 + 1:left]
    else:
        path = inp[top:top + rh, left + rw:tc0]
    return (not path.size) or bool(np.all(path == bg))


def _would_match_any_structure(
    structures: list[dict],
    red_bbox: tuple[int, int, int, int],
    red_shape: tuple[int, int],
    grid: np.ndarray,
    bg: int,
) -> bool:
    for item in structures:
        if red_shape != item["shape"]:
            continue
        if _candidate_distance(grid, bg, item, red_bbox) is not None:
            return True
    return False


def _place_visible_decoy(inp: np.ndarray, placed: list[dict], bg: int) -> bool:
    red_mask = inp == int(Color.RED)
    blocked_adj = _dilate4(red_mask)
    h, w = inp.shape

    unmatched = [item for item in placed if not item["matched"]]
    random.shuffle(unmatched)

    for item in unmatched:
        sr0, sc0, sr1, sc1 = item["bbox"]
        tr0, tc0, tr1, tc1 = item["dock_bbox"]
        shapes = _visible_decoy_shapes(item)
        random.shuffle(shapes)
        options: list[tuple[int, int, int, int]] = []

        for rh, rw in shapes:
            if item["dir"][0] != 0:
                for left in range(tc0, tc1 - rw + 2):
                    if item["dir"][0] == 1:
                        rows = range(sr1 + 2, h - rh + 1)
                    else:
                        rows = range(0, sr0 - rh)
                    for top in rows:
                        if top < 0 or top + rh > h:
                            continue
                        if np.any(inp[top:top + rh, left:left + rw] != bg):
                            continue
                        if blocked_adj[top:top + rh, left:left + rw].any():
                            continue
                        if not _strict_bg_path_ok(inp, item, top, left, rh, rw, bg):
                            continue
                        box = (top, left, top + rh - 1, left + rw - 1)
                        if _would_match_any_structure(placed, box, (rh, rw), inp, bg):
                            continue
                        options.append((top, left, rh, rw))
            else:
                for top in range(tr0, tr1 - rh + 2):
                    if item["dir"][1] == 1:
                        cols = range(sc1 + 2, w - rw + 1)
                    else:
                        cols = range(0, sc0 - rw)
                    for left in cols:
                        if left < 0 or left + rw > w:
                            continue
                        if np.any(inp[top:top + rh, left:left + rw] != bg):
                            continue
                        if blocked_adj[top:top + rh, left:left + rw].any():
                            continue
                        if not _strict_bg_path_ok(inp, item, top, left, rh, rw, bg):
                            continue
                        box = (top, left, top + rh - 1, left + rw - 1)
                        if _would_match_any_structure(placed, box, (rh, rw), inp, bg):
                            continue
                        options.append((top, left, rh, rw))

        if options:
            top, left, rh, rw = random.choice(options)
            patch = np.full((rh, rw), int(Color.RED), dtype=int)
            paste(inp, patch, top, left, transparent=bg)
            return True
    return False


def _place_background_distractors(
    inp: np.ndarray, struct_grid: np.ndarray, placed: list[dict], bg: int, sc: int, count: int
) -> bool:
    if count <= 0:
        return True

    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0:r1 + 1, c0:c1 + 1] = True

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
            if loc is None:
                continue
            top, left = loc
            box = (top, left, top + rh - 1, left + rw - 1)
            if _would_match_any_structure(placed, box, (rh, rw), inp, bg):
                continue
            paste(inp, patch, top, left, transparent=bg)
            red_mask[top:top + rh, left:left + rw] = True
            ok = True
            break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(Color.RED), int(Color.BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    min_dist = max(DISTRACTOR_MIN, TOTAL_RED_MIN - matched)
    max_dist = min(DISTRACTOR_MAX, TOTAL_RED_MAX - matched)
    if min_dist > max_dist or min_dist < VISIBLE_DECOYS_PER_GRID:
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
    inp, out, struct_grid, placed = result

    if _shape_aware_match_count(inp, bg) != matched:
        return None

    visible_placed = 0
    for _ in range(VISIBLE_DECOYS_PER_GRID):
        if not _place_visible_decoy(inp, placed, bg):
            return None
        visible_placed += 1
    if visible_placed != VISIBLE_DECOYS_PER_GRID:
        return None

    if not _place_background_distractors(inp, struct_grid, placed, bg, sc, distractor_count - visible_placed):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D11")
