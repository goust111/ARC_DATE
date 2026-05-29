# Combo variant of 8b9c3697 (index 7)
# combo_spec: D1/0+D12/0
# Name: 多色结构颜色路由

from __future__ import annotations

# Design analysis for this combo variant
# --------------------------------------
# 1. Concrete visual change and a small behavior trace
#    - Original grids used 2-3 large monochrome comb structures plus pure-red rectangles.
#      A structure chose whichever red rectangle was visible on a socket ray, regardless of color.
#    - This combo makes every structure a different solid color, and the movable rectangles inherit
#      one of those structure colors. Example: a blue comb may have a teal rectangle aligned on its
#      opening ray and a separate blue rectangle aligned to a different blue comb elsewhere.
#    - Old behavior on the teal-on-blue case: the blue comb would still pull in the teal rectangle
#      because it is ray-visible. New behavior: that teal rectangle is rejected by the blue comb and
#      disappears unless a teal comb can reach it. Geometry still matters because only same-color
#      rectangles on a clear row/column ray can slide.
#
# 2. Complexity calibration
#    - Structure colors are unique within each grid, so "which structure owns this block" is visible.
#    - Teeth counts stay at 2 or 3. That keeps matched blocks wide enough to notice and avoids the
#      tiny 1x1 case, which would make the color-routing effect easy to miss.
#    - Every generated grid contains at least one wrong-color but ray-aligned decoy structure. That
#      makes the color-routing rule observable instead of degenerating into the original task.
#
# 3. Broken old assumptions and the resulting solver changes
#    - Old code assumed movable objects were the only red components. That is no longer true because
#      the moving pieces are no longer red at all. The new solver therefore classifies non-background
#      components by geometry: large 4x7-9 / 7-9x4 combs are structures, small solid rectangles are
#      pieces.
#    - Old matching logic treated "first visible red on a socket ray" as sufficient. The new logic
#      keeps the same socket-ray scan but filters hits by structure color.
#    - Old redraw logic always placed red in the final socket. The new redraw preserves each piece's
#      own color and still paints only the traversed intermediate cells black.
#
# 4. Placement legality
#    - Generation is still output-first, but placement uses a dedicated occupancy patch that is the
#      union of input and output occupied cells. This matters for wrong-color decoy structures whose
#      input contains a piece that vanishes in the output.
#    - Extra distractors are forbidden from all socket rays and all structure bounding boxes, so they
#      cannot accidentally become additional legal matches.
#
# 5. Visual observability and cleanliness
#    - The key visible cue is now "same color + same ray" rather than "same ray alone". Each grid
#      includes repeated evidence across multiple colors, while the total palette stays small.
#    - Structures are spaced with padding, and distractors stay off the ray corridors, so the black
#      trails, large combs, and small colored rectangles remain visually separable.
#
# 6. New solver assumptions that the generator must guarantee
#    - Every structure is a single-color comb-like component with one long bbox dimension >= 7 and
#      one short bbox dimension >= 4.
#    - Every movable object is a single-color rectangle smaller than the combs.
#    - For each matched structure, at most one same-color piece is socket-visible; for at least one
#      unmatched structure, one or more wrong-color pieces are visible while no same-color piece is.

from collections import defaultdict
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


BLACK = int(Color.BLACK)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _component_info(g: np.ndarray, coords: np.ndarray) -> dict:
    pts = [(int(r), int(c)) for r, c in coords]
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    colors = {int(g[int(r), int(c)]) for r, c in coords}
    return {
        "coords": pts,
        "coord_set": set(pts),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "area": int(len(pts)),
        "color": int(next(iter(colors))) if len(colors) == 1 else None,
        "colors": colors,
    }


def _is_structure(info: dict) -> bool:
    return (
        len(info["colors"]) == 1
        and int(max(info["h"], info["w"])) >= 7
        and int(min(info["h"], info["w"])) >= 4
        and int(info["area"]) >= 10
    )


def _parse_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    structures: list[dict] = []
    pieces: list[dict] = []
    for coords in components_mask(grid != int(bg), connectivity=4):
        info = _component_info(grid, coords)
        if _is_structure(info):
            structures.append(info)
        else:
            pieces.append(info)
    return structures, pieces


def _visible_piece_hits(
    g: np.ndarray,
    struct: dict,
    piece_at: dict[tuple[int, int], int],
    bg: int,
) -> dict[tuple[int, int], dict[int, tuple[int, int, tuple[int, int] | None]]]:
    h, w = g.shape
    r0, c0, r1, c1 = map(int, struct["bbox"])
    hits_by_dir: dict[tuple[int, int], dict[int, list]] = {}

    for dr, dc in _DIRS:
        hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
        for r, c in struct["coords"]:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(g[nr, nc]) != int(bg):
                continue
            if not (r0 < nr < r1 and c0 < nc < c1):
                continue

            rr, cc = int(nr), int(nc)
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                rr += int(dr)
                cc += int(dc)

            pid = piece_at.get((int(rr), int(cc)))
            if pid is None:
                continue

            rec = hits[int(pid)]
            rec[0] += 1
            dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
            if int(dist) < int(rec[1]):
                rec[1] = int(dist)
                rec[2] = (int(r), int(c))
        if hits:
            hits_by_dir[(int(dr), int(dc))] = {
                int(pid): (int(vals[0]), int(vals[1]), vals[2]) for pid, vals in hits.items()
            }
    return hits_by_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route small colored blocks into same-colored comb structures, leaving a black slide trail.

    Input structure:
    - The background is the most common color.
    - There are 2-3 large comb/bracket structures, each a different solid color.
    - Small solid rectangles of various structure colors sit outside the combs; some are aligned on a clear socket ray,
      others are distractors or wrong-color decoys.

    Rule:
    - For each structure, inspect the interior socket rays exactly as in the base task.
    - A visible rectangle only counts if its color matches the structure color; then it slides inward along that ray and
      every intermediate occupied cell becomes black.
    - Rectangles with no same-color reachable structure vanish, including wrong-color pieces that are aligned to a comb.

    Key insight:
    - Geometry still determines whether a block is reachable and from which side it moves, but color now decides ownership:
      alignment alone is not enough anymore.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    structures, pieces = _parse_objects(g, bg)
    assert structures, "expected at least one large monochrome comb structure"
    assert pieces, "expected at least one small colored piece"

    piece_at: dict[tuple[int, int], int] = {}
    for pid, piece in enumerate(pieces):
        for rc in piece["coords"]:
            piece_at[rc] = int(pid)

    scored: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []
    for sid, struct in enumerate(structures):
        struct_color = int(struct["color"])
        hits_by_dir = _visible_piece_hits(g, struct, piece_at, bg)
        best = None
        for dirv, hits in hits_by_dir.items():
            same_hits = {pid: rec for pid, rec in hits.items() if int(pieces[int(pid)]["color"]) == int(struct_color)}
            if not same_hits:
                continue
            pid_best = max(same_hits.keys(), key=lambda pid: (int(same_hits[pid][0]), -int(same_hits[pid][1])))
            cnt, mind, anchor = same_hits[pid_best]
            cand = (int(cnt), -int(mind), int(pid_best), (int(dirv[0]), int(dirv[1])), anchor)
            if best is None or cand > best:
                best = cand
        if best is not None:
            cnt, neg_mind, pid, dirv, anchor = best
            scored.append(((int(cnt), int(neg_mind)), int(pid), (int(dirv[0]), int(dirv[1])), anchor, int(sid)))

    scored.sort(reverse=True, key=lambda item: item[0])
    used_pieces: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for _score, pid, (dr, dc), anchor, _sid in scored:
        if int(pid) in used_pieces or anchor is None:
            continue
        used_pieces.add(int(pid))
        assignments.append((int(pid), (int(dr), int(dc)), (int(anchor[0]), int(anchor[1]))))

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)

    for pid, (dr, dc), (ar, ac) in assignments:
        piece = pieces[int(pid)]
        coords = piece["coords"]
        color = int(piece["color"])
        r0, c0, r1, c1 = map(int, piece["bbox"])
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
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(BLACK)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(color)

    return out


# --------------- generator (combo D1/0 + D12/0) ---------------

GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 5, 6, 7, 8, 9)

STRUCT_WIDTH_TEETH = ((8, 2), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
SLIDE_RANGE = (4, 8)
STRUCT_PADDING = 4
DECOY_STRUCTURES = 1

DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_PIECES_MIN = 5
TOTAL_PIECES_MAX = 11
DISTRACTOR_MIN = 2
DISTRACTOR_MAX = 8

MAX_RETRIES = 500


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
    g[0, :] = int(sc)
    g[1, 1] = int(sc)
    g[1, width - 2] = int(sc)
    g[2:4, 1] = int(sc)
    g[2:4, width - 2] = int(sc)
    g[1, start : start + teeth] = int(sc)
    return g


def _piece_array(teeth: int, color: int) -> np.ndarray:
    return np.full((min(teeth, 2), teeth), int(color), dtype=int)


def _union_patch(a: np.ndarray, b: np.ndarray, bg: int, fill: int) -> np.ndarray:
    out = np.full(a.shape, int(bg), dtype=int)
    out[a != int(bg)] = int(fill)
    out[b != int(bg)] = int(fill)
    return out


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int) -> dict:
    piece = _piece_array(teeth, sc)
    piece_h = int(piece.shape[0])
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + piece_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)

    input_patch = structure.copy()
    output_patch = structure.copy()
    start = (width - teeth) // 2

    paste(input_patch, piece, 2 + slide, start, transparent=bg)
    for off in range(1, slide + 1):
        output_patch[2 + off : 2 + off + piece_h, start : start + teeth] = int(BLACK)
    paste(output_patch, piece, 2, start, transparent=bg)

    placement = _union_patch(input_patch, output_patch, bg, sc)
    return {
        "kind": "matched",
        "struct_color": int(sc),
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "placement": np.rot90(placement, k=rot_k),
    }


def _decoy_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int, decoy_color: int) -> dict:
    piece = _piece_array(teeth, decoy_color)
    piece_h = int(piece.shape[0])
    base = _structure_array(width, teeth, bg, sc)
    patch_h = 2 + slide + piece_h

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)

    input_patch = structure.copy()
    output_patch = structure.copy()
    start = (width - teeth) // 2
    paste(input_patch, piece, 2 + slide, start, transparent=bg)

    placement = _union_patch(input_patch, output_patch, bg, sc)
    return {
        "kind": "decoy",
        "struct_color": int(sc),
        "decoy_color": int(decoy_color),
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "placement": np.rot90(placement, k=rot_k),
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        for r, c in item["coords"]:
            for dr, dc in _DIRS:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(struct_grid[nr, nc]) != int(bg):
                    continue
                if not (r0 < nr < r1 and c0 < nc < c1):
                    continue
                rr, cc = int(nr), int(nc)
                while 0 <= rr < h and 0 <= cc < w and int(struct_grid[rr, cc]) == int(bg):
                    forbidden[rr, cc] = True
                    rr += int(dr)
                    cc += int(dc)
    return forbidden


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(specs: list[dict], h: int, w: int, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    struct_grid = np.full((h, w), bg, dtype=int)
    occ = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        loc = _place_patch(occ, spec["placement"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(occ, spec["placement"], r0, c0, transparent=bg)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        coords_arr = np.argwhere(spec["structure"] != bg)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "color": int(spec["struct_color"]),
                "kind": spec["kind"],
            }
        )

    return inp, out, struct_grid, placed


def _analyze_visibility(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict], list[dict]]:
    structures, pieces = _parse_objects(grid, bg)
    piece_at: dict[tuple[int, int], int] = {}
    for pid, piece in enumerate(pieces):
        for rc in piece["coords"]:
            piece_at[rc] = int(pid)

    analysis: list[dict] = []
    for sid, struct in enumerate(structures):
        hits_by_dir = _visible_piece_hits(grid, struct, piece_at, bg)
        visible_pids: set[int] = set()
        for hits in hits_by_dir.values():
            visible_pids.update(int(pid) for pid in hits.keys())
        same = sorted(int(pid) for pid in visible_pids if int(pieces[int(pid)]["color"]) == int(struct["color"]))
        wrong = sorted(int(pid) for pid in visible_pids if int(pieces[int(pid)]["color"]) != int(struct["color"]))
        analysis.append(
            {
                "sid": int(sid),
                "struct_color": int(struct["color"]),
                "same": same,
                "wrong": wrong,
            }
        )
    return structures, pieces, analysis


def _valid_color_routing(grid: np.ndarray, bg: int, expected_structs: int, expected_matched: int, expected_decoys: int) -> bool:
    structures, pieces, analysis = _analyze_visibility(grid, bg)
    if len(structures) != expected_structs:
        return False
    if len({int(s["color"]) for s in structures}) != len(structures):
        return False
    if not pieces:
        return False

    matched = [rec for rec in analysis if len(rec["same"]) == 1]
    if len(matched) != expected_matched:
        return False
    if any(len(rec["same"]) > 1 for rec in analysis):
        return False

    decoys = [rec for rec in analysis if len(rec["same"]) == 0 and len(rec["wrong"]) >= 1]
    if len(decoys) < expected_decoys:
        return False

    seen_same: set[int] = set()
    for rec in matched:
        pid = int(rec["same"][0])
        if pid in seen_same:
            return False
        seen_same.add(pid)
    return True


def _place_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    palette: list[int],
    count: int,
) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    piece_mask = inp != bg
    shapes = list(DISTRACTOR_SHAPES)
    for _ in range(count):
        blocked = forbidden | bbox_blocked | (inp != bg) | _dilate4(piece_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = int(palette[0])
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            color = random.choice(palette)
            patch = np.full((rh, rw), int(color), dtype=int)
            loc = _place_patch(canvas, patch, bg, padding=0)
            if loc is None:
                continue
            paste(inp, patch, loc[0], loc[1], transparent=bg)
            piece_mask[loc[0] : loc[0] + rh, loc[1] : loc[1] + rw] = True
            ok = True
            break
        if not ok:
            return False
    return True


def _gen_attempt(target_structs: int | None = None) -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    n_structs = int(target_structs) if target_structs is not None else random.randint(*STRUCT_COUNT_RANGE)
    if int(n_structs) == int(STRUCT_COUNT_RANGE[1]):
        # Larger canvases keep the 3-structure branch from being acceptance-starved.
        h = random.randint(max(GRID_H_RANGE[0], GRID_H_RANGE[1] - 3), GRID_H_RANGE[1])
        w = random.randint(max(GRID_W_RANGE[0], GRID_W_RANGE[1] - 4), GRID_W_RANGE[1])
    else:
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
    if int(DECOY_STRUCTURES) >= int(n_structs):
        return None

    palette = [int(c) for c in STRUCT_COLORS if c not in (int(bg), int(BLACK))]
    if len(palette) < n_structs:
        return None

    random.shuffle(palette)
    struct_colors = palette[:n_structs]
    matched = int(n_structs - DECOY_STRUCTURES)

    base_pieces = int(matched + DECOY_STRUCTURES)
    min_distractors = max(DISTRACTOR_MIN, TOTAL_PIECES_MIN - base_pieces)
    max_distractors = min(DISTRACTOR_MAX, TOTAL_PIECES_MAX - base_pieces)
    if min_distractors > max_distractors:
        return None
    distractor_count = random.randint(min_distractors, max_distractors)

    specs: list[dict] = []
    for idx, sc in enumerate(struct_colors):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        slide = random.randint(*SLIDE_RANGE)
        if idx < matched:
            specs.append(_matched_patches(width, teeth, slide, rot_k, bg, sc))
        else:
            other_choices = [int(c) for c in struct_colors if int(c) != int(sc)]
            decoy_color = random.choice(other_choices)
            specs.append(_decoy_patches(width, teeth, slide, rot_k, bg, sc, decoy_color))

    random.shuffle(specs)
    specs.sort(key=lambda spec: int(np.count_nonzero(spec["placement"] != bg)), reverse=True)

    placed = _place_structures(specs, h, w, bg)
    if placed is None:
        return None
    inp, out, struct_grid, placed_info = placed

    if not _valid_color_routing(inp, bg, n_structs, matched, DECOY_STRUCTURES):
        return None

    if not _place_distractors(inp, struct_grid, placed_info, bg, struct_colors, distractor_count):
        return None

    if not _valid_color_routing(inp, bg, n_structs, matched, DECOY_STRUCTURES):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    target_structs = random.randint(*STRUCT_COUNT_RANGE)
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(target_structs=target_structs)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo variant 8b9c3697/7")
