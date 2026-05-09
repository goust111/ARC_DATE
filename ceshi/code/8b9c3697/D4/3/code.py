# Variant of 8b9c3697
# Dimension: D4 — 变换：轨迹涂色 (trajectory)
# Change: 红色色块滑入时，所有经过的中间位置涂黑（continuous_trail），形成从原始位置到最终停靠位置的黑色拖尾... → 洪水填充轨迹

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D4 variant
# ----------------------------------
# spec_variant:
#   unchanged_core:
#     - The scene still contains 2-3 comb/bracket structures plus several red rectangles.
#     - A matched red is identified exactly as in the base task: a ray shot from an internal
#       socket, through the opening direction, lands on one red component.
#     - Each red can dock into at most one structure; unmatched red distractors still disappear.
#   d4_change:
#     - Base task: black marks the union of intermediate translated placements of the moving red.
#     - Variant: the matched red is intentionally narrower than the socket, so the old sweep would
#       only paint a thin strip. After docking, we instead flood the entire socket-width corridor
#       from the outer face of the parked red back to the original red position, then redraw the
#       final red on top.
#   concrete_trace:
#     - Normalized opening-down case, teeth=3, matched red shape=2x1, slide=4.
#     - Base output would paint a 1-column black strip below that red.
#     - Variant output paints a 3-column black rectangle below it, because corridor width is
#       inherited from the whole socket, not from the red blob itself.
#     - The generator therefore keeps matched sockets at width 2 or 3 and uses a thin 2x1 red so
#       the difference is always visible.
#   hidden_assumption_audit:
#     - The original matching code assumes the docking shift is axis-aligned and does not require
#       any perpendicular re-alignment. This still holds and is preserved.
#     - The original trail painter implicitly relied on "painting translated copies" being the
#       intended behavior. That assumption is no longer valid; the new code derives a rectangle
#       from the original bbox and final bbox instead.
#     - The corridor-width rule no longer comes from the red bbox. The solver must recover the
#       socket span from the structure cells that can see the matched red, and the generator must
#       keep those visibility rays unambiguous.
#   placement_legality:
#     - Output-first generation remains valid because the new black corridor stays inside the same
#       local patch footprint: it extends from the docked red to the original red and never past it.
#     - Distractors are still forbidden from socket line-of-sight, so solver matching remains
#       unambiguous after the D4 change.
#   visual_readability:
#     - Black only appears in opening corridors; structures keep a single non-red color; distractors
#       still vanish. This keeps the scene readable.
#     - Restricting matched structures to teeth 2/3 avoids degenerate cases where flood-fill looks
#       identical to the old narrow trail.


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _socket_cells_for_direction(
    structure_cells: set[tuple[int, int]],
    structure_bbox: tuple[int, int, int, int],
    grid: np.ndarray,
    bg: int,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    """All interior socket cells for one structure when viewed from one opening direction."""
    sr0, sc0, sr1, sc1 = map(int, structure_bbox)
    dr, dc = map(int, direction)
    cells = set()
    for r, c in structure_cells:
        if (dr, dc) == (1, 0) and int(r) == sr0:
            continue
        if (dr, dc) == (-1, 0) and int(r) == sr1:
            continue
        if (dr, dc) == (0, 1) and int(c) == sc0:
            continue
        if (dr, dc) == (0, -1) and int(c) == sc1:
            continue
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            continue
        if int(grid[nr, nc]) != int(bg):
            continue
        if not (sr0 < nr < sr1 and sc0 < nc < sc1):
            continue
        cells.add((int(nr), int(nc)))
    return sorted(cells)


def _flood_corridor_bounds(
    original_bbox: tuple[int, int, int, int],
    final_bbox: tuple[int, int, int, int],
    socket_cells: list[tuple[int, int]],
    direction: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return the full socket-width opening-side rectangle, excluding the final red placement."""
    or0, oc0, or1, oc1 = map(int, original_bbox)
    fr0, fc0, fr1, fc1 = map(int, final_bbox)
    dr, dc = map(int, direction)
    rows = [int(r) for r, _ in socket_cells]
    cols = [int(c) for _, c in socket_cells]

    if (dr, dc) == (1, 0):
        r0, r1 = int(fr1 + 1), int(or1)
        c0, c1 = min(cols), max(cols)
    elif (dr, dc) == (-1, 0):
        r0, r1 = int(or0), int(fr0 - 1)
        c0, c1 = min(cols), max(cols)
    elif (dr, dc) == (0, 1):
        r0, r1 = min(rows), max(rows)
        c0, c1 = int(fc1 + 1), int(oc1)
    else:
        r0, r1 = min(rows), max(rows)
        c0, c1 = int(oc0), int(fc0 - 1)

    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _paint_rect(out: np.ndarray, rect: tuple[int, int, int, int] | None, color: int) -> None:
    """Paint an inclusive rectangle if it exists."""
    if rect is None:
        return
    r0, c0, r1, c1 = map(int, rect)
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock outside red blocks into the hollow sockets of large comb structures and paint a filled corridor.

    Input structure:
    - The background is the dominant color.
    - Large non-background, non-red bracket/comb shapes contain interior socket cells.
    - Red components sit outside the structures; some are aligned with one structure opening.

    Rule:
    - For each structure, ray-cast from interior socket cells to determine which red component is seen
      through the opening direction.
    - Slide that red component inward so its nearest face parks in the socket.
    - Instead of painting only the translated intermediate placements, fill the entire opening-side
      rectangle from the docked red back out to the original red position with black.
    - Red components that are not matched to any structure disappear.

    Key insight: The hard part is still finding the docking direction by visibility from socket cells;
    once docked, the new trail uses the socket span from the structure itself and extends from the
    parked red back to the original red location.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    red = int(Color.RED)
    black = int(Color.BLACK)

    assert red in set(map(int, np.unique(g)))

    red_comps = components_mask(g == red, connectivity=4)
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

    struct_mask = (g != bg) & (g != red)
    struct_comps = components_mask(struct_mask, connectivity=4)
    assert struct_comps, "expected at least one non-bg non-red structure to dock into"

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []
    # entries: ((hit_count, -min_dist), rid, dir, anchor, sid)

    for sid, coords in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best_key = None
        best_payload = None

        for dr, dc in _DIRS:
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            for r, c in coords:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != bg:
                    continue
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == bg:
                    rr += int(dr)
                    cc += int(dc)

                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == red:
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
            cand_key = (int(cnt), -int(mind), int(rid_best), (int(dr), int(dc)), anchor)
            if best_key is None or cand_key > best_key:
                best_key = cand_key
                best_payload = cand_key

        if best_payload is not None:
            cnt, neg_mind, rid, dirv, anchor = best_payload
            struct_best.append(
                (
                    (int(cnt), int(neg_mind)),
                    int(rid),
                    (int(dirv[0]), int(dirv[1])),
                    anchor,
                    int(sid),
                )
            )

    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    for _score, rid, direction, anchor, sid in struct_best:
        if int(rid) in used_red or anchor is None:
            continue
        used_red.add(int(rid))
        assignments.append(
            (
                int(rid),
                (int(direction[0]), int(direction[1])),
                (int(anchor[0]), int(anchor[1])),
                int(sid),
            )
        )

    out = g.copy()
    out[out == red] = int(bg)

    for rid, (dr, dc), (ar, ac), sid in assignments:
        coords = red_coords[int(rid)]
        or0, oc0, or1, oc1 = map(int, red_bbox[int(rid)])
        tr, tc = int(ar + dr), int(ac + dc)

        if (dr, dc) == (0, 1):
            shift_r, shift_c = 0, int(tc - oc0)
        elif (dr, dc) == (0, -1):
            shift_r, shift_c = 0, int(tc - oc1)
        elif (dr, dc) == (1, 0):
            shift_r, shift_c = int(tr - or0), 0
        else:
            shift_r, shift_c = int(tr - or1), 0

        final_coords = [(int(r + shift_r), int(c + shift_c)) for r, c in coords]
        final_bbox = (
            int(or0 + shift_r),
            int(oc0 + shift_c),
            int(or1 + shift_r),
            int(oc1 + shift_c),
        )

        socket_cells = _socket_cells_for_direction(struct_coords[int(sid)], struct_bbox[int(sid)], g, bg, (dr, dc))
        corridor = _flood_corridor_bounds((or0, oc0, or1, oc1), final_bbox, socket_cells, (dr, dc))
        _paint_rect(out, corridor, black)

        for rr, cc in final_coords:
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = red

    return out


# --------------- generator (D4 flood-corridor variant) ---------------

# -- canvas & grid --
GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

# -- structures --
# Matched structures use teeth 2 or 3 so the flood corridor is visibly thicker
# than the base task's translated sweep.
MATCHED_STRUCT_WIDTH_TEETH = ((8, 2), (9, 3))
UNMATCHED_STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 1), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4

# -- distractors --
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

# -- retry --
MAX_RETRIES = 300


def _dilate4(mask: np.ndarray) -> np.ndarray:
    """4-connected dilation of a boolean mask."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    """Build a 4-row normalized comb with opening facing down."""
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = sc
    g[1, 1] = sc
    g[1, width - 2] = sc
    g[2:4, 1] = sc
    g[2:4, width - 2] = sc
    g[1, start : start + teeth] = sc
    return g


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build input/output/structure patches for one matched structure+red pair."""
    red = int(Color.RED)
    black = int(Color.BLACK)
    red_h = 2
    red_w = 1
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + red_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    lane = start + (teeth // 2)
    output_patch[2 + red_h : 2 + slide + red_h, start : start + teeth] = black
    output_patch[2 : 2 + red_h, lane : lane + red_w] = red
    input_patch[2 + slide : 2 + slide + red_h, lane : lane + red_w] = red

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "matched": True,
        "teeth": int(teeth),
        "slide": int(slide),
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int) -> dict:
    """Build patches for an unmatched structure with no aligned red component."""
    structure = np.rot90(_structure_array(width, teeth, bg, sc), k=rot_k)
    return {
        "structure": structure,
        "input": structure.copy(),
        "output": structure.copy(),
        "matched": False,
        "teeth": int(teeth),
        "slide": 0,
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    """Return a mask of cells lying on any socket's straight opening ray."""
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
    """Try to place a patch and return its top-left corner."""
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _unique_matched_visibility(grid: np.ndarray, bg: int, expected_matched: int) -> bool:
    """Each matched red must be visible from exactly one structure, and vice versa."""
    h, w = grid.shape
    red_val = int(Color.RED)
    red_comps = components_mask(grid == red_val, connectivity=4)
    if len(red_comps) != expected_matched:
        return False

    red_at: dict[tuple[int, int], int] = {}
    for rid, coords in enumerate(red_comps):
        for r, c in coords:
            red_at[(int(r), int(c))] = int(rid)

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
                    visible_rids.add(int(red_at[(rr, cc)]))
        if len(visible_rids) > 1:
            return False
        if visible_rids:
            struct_hits += 1
            rid = next(iter(visible_rids))
            red_to_structs.setdefault(rid, set()).add(int(sid))

    if struct_hits != expected_matched:
        return False
    return len(red_to_structs) == expected_matched and all(len(sids) == 1 for sids in red_to_structs.values())


def _place_structures(
    specs: list[dict], h: int, w: int, bg: int, sc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    """Place all structure patches on fresh canvases."""
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
    """Scatter distractor red rectangles away from all structure sight lines."""
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


def _variant_visible(specs: list[dict]) -> bool:
    """All matched reds must be thinner than their sockets so the D4 change is observable."""
    return any(spec["matched"] and spec["teeth"] > 1 for spec in specs)


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
        rot_k = random.randrange(4)
        if idx < matched:
            width, teeth = random.choice(MATCHED_STRUCT_WIDTH_TEETH)
            slide = random.randint(*SLIDE_RANGE)
            specs.append(_matched_patches(width, teeth, slide, rot_k, bg, sc))
        else:
            width, teeth = random.choice(UNMATCHED_STRUCT_WIDTH_TEETH)
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

    if not _variant_visible(specs):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D4 flood-corridor variant")
