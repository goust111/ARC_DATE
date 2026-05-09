# Variant of 8b9c3697
# Dimension: D1 — 匹配条件：行列对齐射线投射 (match_cond)
# Change: 从结构内部插槽（socket）向四个基本方向发射射线，命中红色色块即确定匹配关系和滑入方向（s.col/row == d... → 颜色匹配替代对齐

from __future__ import annotations

# Design analysis for this D1 variant
# -----------------------------------
# 1. Concrete visual change
#    - The original comb/bracket geometry stays intact.
#    - Each structure now has two colors: a shared wall color and a per-structure tooth color.
#    - Each movable block is still mostly red, but its contact face is recolored to the matching tooth color.
#    - A matched block docks exactly as before and still leaves a black swept trail, but the final docked block keeps
#      its colored contact stripe.
#
# 2. Behavior trace on the changed step
#    - Original: a structure shoots rays out of interior sockets; the red blob hit on that ray becomes the docking blob.
#    - Variant: the structure first reads its tooth color and its opening direction from geometry, then selects the blob
#      whose non-red stripe has the same color. The opening side still controls where the block ends up.
#    - To make the change observable, every structure gets a wrong-color blob placed on its opening ray. Spatially that
#      wrong blob is a plausible candidate, but it must disappear because its color does not match the teeth.
#
# 3. Complexity calibration
#    - Tooth colors are unique within a grid, so matching is one-to-one and human-visible.
#    - Teeth counts are restricted to 2 or 3 so the color stripe is always visible; the original teeth=1 case would
#      collapse the marker and make the D1 change hard to see.
#    - The generator still uses 2-3 structures and several distractors, so the task does not degenerate into a single
#      trivial color lookup.
#
# 4. Old implicit assumptions that no longer hold
#    - Old code assumed every movable object was a pure red component. That breaks once blocks contain a colored stripe.
#      The new solver therefore extracts connected components from all non-background cells and classifies components by
#      whether they contain red (movable block) or not (structure).
#    - Old code inferred docking direction by finding which socket-ray hit red. That is replaced by reading the tooth
#      segment itself: the rare color inside a structure is the tooth color, and the bg cells directly adjacent to those
#      teeth inside the bbox reveal the opening direction and socket location.
#    - Old output redraw always used solid red. That would erase the new marker information, so assigned blocks are now
#      redrawn with their original internal colors.
#
# 5. Placement legality and cleanliness
#    - Structure patches are still output-first and are placed with padding, so structures and their trails do not
#      overlap.
#    - Wrong-color on-ray distractors are embedded inside the same local patch as each structure, guaranteeing they stay
#      legal while remaining visually close enough to demonstrate that color, not ray-hit order, controls the result.
#    - Extra global distractors are kept off structure rays to avoid accidental extra ambiguity and visual clutter.
#
# 6. New solver assumptions and generator guarantees
#    - Each structure has exactly one tooth color distinct from the shared wall color.
#    - Each matched tooth color appears on exactly one movable block; fake markers use colors not used by any teeth.
#    - Matched blocks keep the original axis-aligned slide geometry, so the black trail remains easy to read.

from collections import Counter
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


RED = int(Color.RED)
BLACK = int(Color.BLACK)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _component_colors(g: np.ndarray, coords: np.ndarray) -> set[int]:
    return {int(g[int(r), int(c)]) for r, c in coords}


def _piece_marker_color(g: np.ndarray, coords: np.ndarray, bg: int) -> int | None:
    cols = _component_colors(g, coords) - {int(bg), int(RED), int(BLACK)}
    if len(cols) != 1:
        return None
    return next(iter(cols))


def _piece_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    pts = [(int(r), int(c)) for r, c in coords]
    marker = _piece_marker_color(g, coords, bg)
    info = {
        "coords": pts,
        "bbox": tuple(map(int, bbox_coords(coords))),
        "marker_color": None if marker is None else int(marker),
    }
    if marker is not None:
        marker_pts = np.array([(r, c) for r, c in pts if int(g[r, c]) == int(marker)], dtype=int)
        info["marker_bbox"] = tuple(map(int, bbox_coords(marker_pts)))
    else:
        info["marker_bbox"] = None
    return info


def _structure_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    pts = [(int(r), int(c)) for r, c in coords]
    if len(_component_colors(g, coords)) < 2:
        return None

    counts = Counter(int(g[r, c]) for r, c in pts)
    tooth_color = min(counts.items(), key=lambda kv: int(kv[1]))[0]
    tooth_cells = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == int(tooth_color)]
    if not tooth_cells:
        return None

    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    dir_hits: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in tooth_cells:
        for dr, dc in _DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                continue
            if int(g[nr, nc]) != int(bg):
                continue
            if not (r0 < nr < r1 and c0 < nc < c1):
                continue
            dir_hits.setdefault((int(dr), int(dc)), []).append((int(r), int(c)))

    if not dir_hits:
        return None

    direction = max(dir_hits.keys(), key=lambda d: len(dir_hits[d]))
    anchors = dir_hits[direction]
    tooth_bbox = tuple(map(int, bbox_coords(np.array(tooth_cells, dtype=int))))
    dr, dc = direction
    target_marker_bbox = (
        int(tooth_bbox[0] + dr),
        int(tooth_bbox[1] + dc),
        int(tooth_bbox[2] + dr),
        int(tooth_bbox[3] + dc),
    )
    return {
        "coords": set(pts),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "tooth_color": int(tooth_color),
        "direction": (int(dr), int(dc)),
        "anchor": anchors[0],
        "target_marker_bbox": target_marker_bbox,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert color-tagged red components into tooth-matched interior sockets, leaving a black trail.

    Input structure:
    - bg = most common. Large comb-like structures use one wall color plus a smaller tooth segment in a distinct color.
    - Movable components are red rectangles whose contact face is marked with one non-red color.

    Rule:
    - For each structure, identify its tooth color and the socket direction indicated by the bg cells just inside those teeth.
    - Find the movable component whose colored marker matches that tooth color.
    - Slide that component into the socket and paint every intermediate placement black; components with no color match disappear.

    Key insight: The docking relation is determined by tooth-color to marker-color pairing, while the structure geometry still
    determines the side from which the matched component slides in.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    assert int(RED) in set(map(int, np.unique(g)))

    comps = components_mask(g != int(bg), connectivity=4)
    assert comps

    pieces: list[dict] = []
    structures: list[dict] = []
    for coords in comps:
        cols = _component_colors(g, coords)
        if int(RED) in cols:
            pieces.append(_piece_info(g, coords, bg))
        else:
            info = _structure_info(g, coords, bg)
            if info is not None:
                structures.append(info)

    assert pieces
    assert structures, "expected at least one colored comb structure"

    pieces_by_marker: dict[int, list[int]] = {}
    for pid, piece in enumerate(pieces):
        marker = piece["marker_color"]
        if marker is None:
            continue
        pieces_by_marker.setdefault(int(marker), []).append(int(pid))

    assignments: list[tuple[int, dict]] = []
    for struct in structures:
        marker = int(struct["tooth_color"])
        candidates = pieces_by_marker.get(marker, [])
        if len(candidates) != 1:
            continue
        pid = int(candidates[0])
        piece = pieces[pid]
        if piece["marker_bbox"] is None:
            continue
        assignments.append((pid, struct))

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)

    used_pids: set[int] = set()
    for pid, struct in assignments:
        if pid in used_pids:
            continue
        used_pids.add(pid)

        piece = pieces[pid]
        marker_bbox = piece["marker_bbox"]
        assert marker_bbox is not None
        tr0, tc0, _tr1, _tc1 = map(int, struct["target_marker_bbox"])
        mr0, mc0, _mr1, _mc1 = map(int, marker_bbox)
        shift_r = int(tr0 - mr0)
        shift_c = int(tc0 - mc0)
        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))

        for step in range(0, steps):
            if steps == 0:
                off_r = off_c = 0
            else:
                off_r = int(round(step * shift_r / steps))
                off_c = int(round(step * shift_c / steps))
            for r, c in piece["coords"]:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = int(BLACK)

        for r, c in piece["coords"]:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator (codex-att2, D1 color-match variant) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────────────
GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
WALL_COLORS = (1, 8, 9)
MARKER_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

# ── CFG: structures ─────────────────────────────────────────────────────────
STRUCT_WIDTH_TEETH = ((8, 2), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
TRUE_SLIDE_RANGE = (5, 8)
FALSE_SLIDE_MIN = 2
STRUCT_PADDING = 4

# ── CFG: distractors ────────────────────────────────────────────────────────
DISTRACTOR_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
TOTAL_PIECES_MIN = 5
TOTAL_PIECES_MAX = 11

# ── CFG: retry ──────────────────────────────────────────────────────────────
MAX_RETRIES = 400


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _structure_array(width: int, teeth: int, bg: int, wall_color: int, tooth_color: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = int(wall_color)
    g[1, 1] = g[1, width - 2] = int(wall_color)
    g[2:4, 1] = g[2:4, width - 2] = int(wall_color)
    g[1, start : start + teeth] = int(tooth_color)
    return g


def _piece_array(teeth: int, bg: int, marker_color: int) -> np.ndarray:
    piece = np.full((2, teeth), bg, dtype=int)
    piece[0, :] = int(marker_color)
    piece[1, :] = int(RED)
    return piece


def _matched_patches(
    width: int,
    teeth: int,
    true_slide: int,
    false_slide: int,
    rot_k: int,
    bg: int,
    wall_color: int,
    tooth_color: int,
    fake_color: int,
) -> dict:
    base = _structure_array(width, teeth, bg, wall_color, tooth_color)
    true_piece = _piece_array(teeth, bg, tooth_color)
    fake_piece = _piece_array(teeth, bg, fake_color)
    patch_h = 2 + true_slide + true_piece.shape[0]
    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)

    input_patch = structure.copy()
    output_patch = structure.copy()
    start = (width - teeth) // 2

    paste(input_patch, true_piece, 2 + true_slide, start, transparent=bg)
    paste(input_patch, fake_piece, 2 + false_slide, start, transparent=bg)

    for off in range(1, true_slide + 1):
        output_patch[2 + off : 2 + off + true_piece.shape[0], start : start + teeth] = int(BLACK)
    paste(output_patch, true_piece, 2, start, transparent=bg)

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _unmatched_patches(
    width: int,
    teeth: int,
    false_slide: int,
    rot_k: int,
    bg: int,
    wall_color: int,
    tooth_color: int,
    fake_color: int,
) -> dict:
    base = _structure_array(width, teeth, bg, wall_color, tooth_color)
    fake_piece = _piece_array(teeth, bg, fake_color)
    patch_h = 2 + false_slide + fake_piece.shape[0]
    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)

    input_patch = structure.copy()
    paste(input_patch, fake_piece, 2 + false_slide, (width - teeth) // 2, transparent=bg)
    output_patch = structure.copy()

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        tooth_color = int(item["tooth_color"])
        for r, c in item["coords"]:
            if int(struct_grid[r, c]) != int(tooth_color):
                continue
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


def _parse_input_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    structures: list[dict] = []
    pieces: list[dict] = []
    for coords in components_mask(grid != int(bg), connectivity=4):
        cols = _component_colors(grid, coords)
        if int(RED) in cols:
            pieces.append(_piece_info(grid, coords, bg))
        else:
            info = _structure_info(grid, coords, bg)
            if info is not None:
                structures.append(info)
    return structures, pieces


def _valid_color_mapping(grid: np.ndarray, bg: int, expected_matched: int, expected_structs: int) -> bool:
    structures, pieces = _parse_input_objects(grid, bg)
    if len(structures) != expected_structs:
        return False

    tooth_colors = [int(s["tooth_color"]) for s in structures]
    if len(set(tooth_colors)) != len(tooth_colors):
        return False

    matched_colors = 0
    marker_count: Counter[int] = Counter()
    for piece in pieces:
        if piece["marker_color"] is not None:
            marker_count[int(piece["marker_color"])] += 1
    for col in tooth_colors:
        hits = int(marker_count.get(int(col), 0))
        if hits not in (0, 1):
            return False
        matched_colors += hits

    return matched_colors == expected_matched


def _place_structures(
    specs: list[dict],
    h: int,
    w: int,
    bg: int,
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

        coords_arr = np.argwhere(spec["structure"] != bg)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        tooth_vals = [int(spec["structure"][int(rr), int(cc)]) for rr, cc in coords_arr]
        counts = Counter(tooth_vals)
        tooth_color = min(counts.items(), key=lambda kv: int(kv[1]))[0]
        placed.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "tooth_color": int(tooth_color),
            }
        )

    return inp, out, struct_grid, placed


def _place_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    wall_color: int,
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
        canvas[blocked] = wall_color
        random.shuffle(shapes)
        ok = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(RED), dtype=int)
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


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    wall_choices = [c for c in WALL_COLORS if c not in (bg, RED, BLACK)]
    wall_color = random.choice(wall_choices)

    palette = [c for c in MARKER_COLORS if c not in (bg, wall_color, RED, BLACK)]
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    if len(palette) < 2 * n_structs:
        return None

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    random.shuffle(palette)
    tooth_colors = palette[:n_structs]
    fake_colors = palette[n_structs : 2 * n_structs]

    base_pieces = n_structs + matched
    extra_min = max(0, TOTAL_PIECES_MIN - base_pieces)
    extra_max = max(0, TOTAL_PIECES_MAX - base_pieces)
    if extra_min > extra_max:
        return None
    distractor_count = random.randint(extra_min, extra_max)

    specs = []
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        fake_max = TRUE_SLIDE_RANGE[1] - 3
        if idx < matched:
            true_slide = random.randint(*TRUE_SLIDE_RANGE)
            false_slide = random.randint(FALSE_SLIDE_MIN, min(true_slide - 3, fake_max))
            specs.append(
                _matched_patches(
                    width,
                    teeth,
                    true_slide,
                    false_slide,
                    rot_k,
                    bg,
                    wall_color,
                    tooth_colors[idx],
                    fake_colors[idx],
                )
            )
        else:
            false_slide = random.randint(FALSE_SLIDE_MIN + 1, TRUE_SLIDE_RANGE[1] - 1)
            specs.append(
                _unmatched_patches(
                    width,
                    teeth,
                    false_slide,
                    rot_k,
                    bg,
                    wall_color,
                    tooth_colors[idx],
                    fake_colors[idx],
                )
            )

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != bg), reverse=True)

    result = _place_structures(specs, h, w, bg)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not _valid_color_mapping(inp, bg, matched, n_structs):
        return None

    if not _place_distractors(inp, struct_grid, placed, bg, wall_color, distractor_count):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D1 color-match variant")
