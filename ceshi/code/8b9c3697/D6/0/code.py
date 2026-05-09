# Variant of 8b9c3697
# Dimension: D6 — 网格形状：输出框架 (grid_shape)
# Change: 输出与输入网格尺寸完全相同（19-23行 x 22-27列），仅改变部分像素值... → 紧凑裁剪输出

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis for the D6 variant
# ----------------------------------
# Reference baseline:
# - Source task 20270e3b uses the same complexity level we want here: perform the
#   original transformation first, then apply a tight crop around the final
#   foreground. The crop is not a new object interaction; it is a new output
#   framing rule.
#
# a) Concrete behavior trace
# - The visible scene before cropping is still the original 8b9c3697 scene:
#   2-3 comb/bracket structures stay in place, matched red rectangles slide into
#   sockets, black trails mark intermediate positions, unmatched red distractors
#   disappear.
# - The D6 change happens only after that full-size scene has been produced.
#   Instead of returning the whole input-sized canvas, we return the minimal
#   rectangle containing every non-background cell in the final scene.
# - Small trace:
#   original logic: input has a comb near the middle and a matched red block 6
#   cells below it; output keeps the whole 21x24 canvas and only edits pixels.
#   variant logic: the same sliding/trail output is built first, then rows/cols
#   containing only background are removed from all four outer sides; the solver
#   must infer the crop from the transformed foreground, not from the input bbox.
#
# b) Complexity calibration
# - We keep the original comb geometry, directions, slide lengths, distractor
#   rules, and greedy assignment. This preserves ambiguity control and prevents
#   the task from collapsing into a trivial "crop the input foreground" task.
# - To keep the D6 change visible, the generator guarantees at least a 1-cell
#   border around every structure/trail patch in the pre-crop output, so the
#   output shape is strictly smaller than the input in both dimensions.
#
# c) Implicit-assumption review of main()
# - All ray-casting and sliding code assumes input and working output share the
#   same coordinates. That remains true; cropping must happen only after all
#   assignments are rendered.
# - `out[out == RED] = bg` is still safe because matched reds are redrawn before
#   cropping and unmatched reds must vanish.
# - Every structure remains non-background, so the final scene is never empty;
#   tight-crop therefore always has a valid bbox to crop to.
#
# d) Placement legality
# - Structures are still placed with non-overlap and padding.
# - For D6, structure/output patches are also placed with `border=1`, which keeps
#   all final non-background pixels away from the canvas edge.
# - Distractors may sit near the edge in the input; this is legal because they
#   disappear before cropping and therefore do not affect the output frame.
#
# e) Visual observability
# - Across examples, the internal transformation stays recognizable while the
#   outer blank border disappears. Because the crop is driven by visible
#   structures and trails, a human can infer it from 2-3 examples.
# - No hidden ordering by color id or other non-visual property is introduced.
#
# f) Visual cleanliness
# - The crop removes only uniform background on the outside; it does not add new
#   overlap or symbolic markers. The final grids remain the same clean scenes,
#   only framed more tightly.
#
# g) Color inference
# - Color semantics are unchanged: red = moving blocks, black = trail, one
#   structure color, one background color. D6 changes shape only, not the palette.
#
# h) Variant validity metric
# - The core metric is shrinkage of the final frame:
#   `(input_h - output_h, input_w - output_w)`.
# - A generated sample is considered D6-valid only if the pre-crop output has a
#   positive background margin on all four sides, which guarantees non-zero
#   shrinkage and prevents degeneration to the original full-frame behavior.
#
# i) New assumptions introduced by cropping
# - `main()` now assumes the final foreground bbox is the intended output frame.
# - `generate_input()` enforces this by placing every structure/trail patch away
#   from the border and by oracle-checking against the cropped expected output.


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    """Return the minimal crop containing all non-background cells."""
    nz = np.argwhere(grid != int(bg))
    assert nz.size > 0, "expected non-background output"
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide red components into comb sockets, paint the traversed path black, then tight-crop.

    Input structure:
    - The dominant color is background; large non-background, non-red components are comb-like structures.
    - Red rectangles sit outside the structures, and only some are aligned with an interior socket.

    Rule:
    - Detect interior socket cells as background cells inside a structure bbox that touch the structure.
    - Ray-cast outward from sockets to find which red component is visible from each structure, then slide that red into the socket and color intermediate positions black.
    - Remove unmatched reds, keep structures fixed, and finally crop the result to the smallest rectangle containing every non-background output cell.

    Key insight: The docking/motion rule is unchanged from the original task; the new deduction is that the output frame is the final foreground bbox, not the input canvas.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    RED = int(Color.RED)
    BLACK = int(Color.BLACK)

    assert not (int(RED) not in set(map(int, np.unique(g))))

    # --- connected components: red blobs ---
    red_comps = components_mask(g == int(RED), connectivity=4)
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

    # --- connected components: structures (non-bg, non-red) ---
    struct_mask = (g != int(bg)) & (g != int(RED))
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # For each structure, find its best (red, dir, anchor) triple.
    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []
    # entries: ((hit_count, -min_dist), rid, dir, anchor, sid)

    for sid, S in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best = None  # (hit_count, -min_dist, rid, dir, anchor)

        for dr, dc in dirs:
            # rid -> [count, min_dist, best_anchor]
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            for r, c in S:
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

                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(RED):
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
            struct_best.append(((int(cnt), int(neg_mind)), int(rid), (int(dirv[0]), int(dirv[1])), anchor, int(sid)))

    # Assign each red component to at most one structure (greedy best-first).
    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for _score, rid, (dr, dc), anchor, _sid in struct_best:
        if int(rid) in used_red or anchor is None:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dr), int(dc)), (int(anchor[0]), int(anchor[1]))))

    # --- build output on the original canvas, crop only at the end ---
    out = g.copy()
    out[out == int(RED)] = int(bg)

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
                    out[rr, cc] = int(BLACK)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(RED)

    return _tight_crop_non_bg(out, bg)


# --------------- generator (output-first + final tight crop) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

# ── CFG: structures ─────────────────────────────────────────────────
STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4
OUTPUT_MARGIN = 1

# ── CFG: distractors ────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

# ── CFG: retry ──────────────────────────────────────────────────────
MAX_RETRIES = 300

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
    red, black = int(Color.RED), int(Color.BLACK)
    red_h = min(teeth, 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    for off in range(1, slide + 1):
        output_patch[2 + off : 2 + off + red_h, start : start + teeth] = black
    output_patch[2 : 2 + red_h, start : start + teeth] = red
    input_patch[2 + slide : 2 + slide + red_h, start : start + teeth] = red

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


def _place_patch(
    canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int, border: int = 0
) -> tuple[int, int] | None:
    """Try to place patch on canvas; return (r, c) or None."""
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding, border=border)
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


def _output_has_margin(out: np.ndarray, bg: int, margin: int) -> bool:
    """Check that final non-background output stays inset from the canvas border."""
    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return False
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    h, w = out.shape
    return r0 >= margin and c0 >= margin and r1 < h - margin and c1 < w - margin


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    """Place all structure patches on fresh canvases. Returns (inp, out, struct_grid, placed) or None."""
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(out, spec["output"], bg, padding=STRUCT_PADDING, border=OUTPUT_MARGIN)
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

    if not _output_has_margin(out, bg, OUTPUT_MARGIN):
        return None
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
    inp, out, struct_grid, placed = result

    if not _unique_matched_visibility(inp, bg, matched):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, sc, distractor_count):
        return None

    expected = _tight_crop_non_bg(out, bg)
    if expected.shape == inp.shape:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D6 variant")
