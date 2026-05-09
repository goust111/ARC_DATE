# Combo variant of 8b9c3697 (index 1)
# combo_spec: D1/0+D2/1
# Name: 颜色匹配多色滑入

from __future__ import annotations

# Design analysis for this combo variant
# --------------------------------------
# 1. Concrete visual change
#    - The comb/bracket geometry is preserved, but each structure is now bicolored: a shared wall color plus a
#      per-structure tooth color. The tooth color is the structure's matching signature.
#    - Every movable object is a two-color sprite rather than a pure red block. The sprite is mostly one "main" color
#      with a small red accent, so humans can still group all movable objects together while noticing which color each
#      sprite belongs to.
#    - A matched sprite keeps its internal two-color pattern after docking; only the swept path becomes black.
#
# 2. Behavior trace on the modified step
#    - Original task: a structure ray-casts through its opening and whichever red rectangle lies on that row/column is
#      the one that slides inward.
#    - Fused task: the structure still reads its opening direction from geometry, but it no longer trusts the first
#      object on that ray. Instead it reads the tooth color and selects the sprite whose main color matches that tooth.
#    - Generator consequence: for every structure there is a closer wrong-color sprite on the opening ray. A ray-based
#      solver would choose the decoy; the fused solver must ignore it and pull in the farther color-matched sprite.
#
# 3. Complexity calibration
#    - Teeth counts are restricted to 2 or 3 so the tooth-color segment and the sprite face are both wide enough to be
#      visible. This avoids the teeth=1 case where the color signal becomes too weak.
#    - Tooth colors are unique inside one grid, which keeps the color mapping one-to-one and eliminates hidden tie
#      breaks.
#    - The sprite pattern is fixed across samples: a solid main-color contact face plus a small red accent on the
#      trailing row. That adds a second dimension of appearance without exploding the hypothesis space.
#
# 4. Old implicit assumptions that no longer hold
#    - Original code extracted movable components by searching for pure red connected components. That fails here because
#      sprites are multicolor. The new solver instead takes all non-background components and classifies them by whether
#      they contain red (sprite) or not (structure).
#    - Original matching depended on socket rays hitting red. That assumption is intentionally broken by the new decoys.
#      Matching now depends only on tooth color versus sprite main color.
#    - Original redraw used solid red at the final position. That would erase the new variant's visual identity, so the
#      final placement redraws the sprite's original two-color pixels.
#
# 5. Placement legality and cleanliness
#    - Structure patches are still generated output-first and placed with padding so combs, trails, and local decoys do
#      not overlap.
#    - Each local patch keeps the matched sprite and the wrong-color decoy on the same opening lane, with at least one
#      background gap between them and the structure. This preserves clean connected components and a readable slide path.
#    - Extra distractor sprites are placed away from socket rays and structure bounding boxes, preventing accidental
#      matches or visually messy overlaps.
#
# 6. Human observability
#    - In 2-4 examples a human can see the repeated pattern: each structure has a uniquely colored tooth segment, each
#      sprite has a dominant color plus a red accent, and only the same-colored sprite docks.
#    - The closer wrong-color on-ray decoys make the rule visibly different from the original spatial-alignment task.
#      If color were irrelevant, those nearer sprites should have moved instead, so the changed rule is inferable.
#
# 7. New solver assumptions and generator guarantees
#    - Every structure has exactly one minority tooth color and exactly one opening direction recoverable from the tooth
#      cells' interior-adjacent background cells.
#    - Every matched tooth color appears on exactly one sprite main color; fake colors never equal any tooth color.
#    - For each matched structure the first sprite on the opening ray is wrong-colored, while a farther sprite has the
#      correct main color. The generator checks for these override cases explicitly so the combo does not degenerate.

from collections import Counter
import random

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location


RED = int(Color.RED)
BLACK = int(Color.BLACK)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _component_colors(g: np.ndarray, coords: np.ndarray) -> set[int]:
    return {int(g[int(r), int(c)]) for r, c in coords}


def _piece_main_color(g: np.ndarray, coords: np.ndarray, bg: int) -> int | None:
    cols = _component_colors(g, coords) - {int(bg), int(RED), int(BLACK)}
    if len(cols) != 1:
        return None
    return next(iter(cols))


def _bbox_of_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    arr = np.array(points, dtype=int)
    return tuple(map(int, bbox_coords(arr)))


def _piece_face_bbox(
    g: np.ndarray,
    coords: list[tuple[int, int]],
    main_color: int,
    direction: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    dr, dc = map(int, direction)
    if (dr, dc) == (1, 0):
        side = "top"
    elif (dr, dc) == (-1, 0):
        side = "bottom"
    elif (dr, dc) == (0, 1):
        side = "left"
    else:
        side = "right"

    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    if side == "top":
        edge = min(rows)
        pts = [(int(r), int(c)) for r, c in coords if int(r) == int(edge) and int(g[int(r), int(c)]) == int(main_color)]
    elif side == "bottom":
        edge = max(rows)
        pts = [(int(r), int(c)) for r, c in coords if int(r) == int(edge) and int(g[int(r), int(c)]) == int(main_color)]
    elif side == "left":
        edge = min(cols)
        pts = [(int(r), int(c)) for r, c in coords if int(c) == int(edge) and int(g[int(r), int(c)]) == int(main_color)]
    else:
        edge = max(cols)
        pts = [(int(r), int(c)) for r, c in coords if int(c) == int(edge) and int(g[int(r), int(c)]) == int(main_color)]
    return _bbox_of_points(pts)


def _piece_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    pts = [(int(r), int(c)) for r, c in coords]
    return {
        "coords": pts,
        "bbox": tuple(map(int, bbox_coords(coords))),
        "main_color": _piece_main_color(g, coords, bg),
    }


def _structure_info(g: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    pts = [(int(r), int(c)) for r, c in coords]
    colors = _component_colors(g, coords)
    if len(colors) < 2:
        return None

    counts = Counter(int(g[int(r), int(c)]) for r, c in pts)
    tooth_color = min(counts.items(), key=lambda kv: int(kv[1]))[0]
    tooth_cells = [(int(r), int(c)) for r, c in pts if int(g[int(r), int(c)]) == int(tooth_color)]
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
    tooth_bbox = tuple(map(int, bbox_coords(np.array(tooth_cells, dtype=int))))
    dr, dc = map(int, direction)
    target_face_bbox = (
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
        "target_face_bbox": target_face_bbox,
    }


def _occupied_piece_cells(g: np.ndarray, pieces: list[dict]) -> dict[tuple[int, int], int]:
    at: dict[tuple[int, int], int] = {}
    for pid, piece in enumerate(pieces):
        for rc in piece["coords"]:
            at[(int(rc[0]), int(rc[1]))] = int(pid)
    return at


def _nearest_piece_on_ray(
    g: np.ndarray,
    struct: dict,
    bg: int,
    piece_at: dict[tuple[int, int], int],
) -> int | None:
    dr, dc = map(int, struct["direction"])
    tr0, tc0, tr1, tc1 = map(int, struct["target_face_bbox"])
    starts: list[tuple[int, int]] = []
    if dr != 0:
        for cc in range(int(tc0), int(tc1) + 1):
            starts.append((int(tr0), int(cc)))
    else:
        for rr in range(int(tr0), int(tr1) + 1):
            starts.append((int(rr), int(tc0)))

    best: tuple[int, int] | None = None
    for sr, sc in starts:
        rr, cc = int(sr + dr), int(sc + dc)
        dist = 1
        while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(bg):
            rr += int(dr)
            cc += int(dc)
            dist += 1
        pid = piece_at.get((int(rr), int(cc)))
        if pid is None:
            continue
        cand = (int(dist), int(pid))
        if best is None or cand < best:
            best = cand
    return None if best is None else int(best[1])


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multicolor sprites into tooth-colored comb sockets, leaving black sweep trails.

    Input structure:
    - bg is the most common color. Each large comb-like structure uses one wall color and one smaller tooth color.
    - Movable objects are two-color sprites: a dominant main color plus a small red accent, placed outside the structures.

    Rule:
    - Read each structure's tooth color and opening direction from the comb geometry.
    - Match that structure to the unique sprite whose main color equals the tooth color.
    - Move the matched sprite straight into the socket, color every intermediate placement black, and erase all unmatched sprites.

    Key insight:
    - Spatial line-of-sight no longer decides which object moves; it only provides the slide direction. The deciding cue is
      the color pairing between a structure's tooth segment and a sprite's dominant color.
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

    pieces_by_main: dict[int, list[int]] = {}
    for pid, piece in enumerate(pieces):
        main_color = piece["main_color"]
        if main_color is None:
            continue
        pieces_by_main.setdefault(int(main_color), []).append(int(pid))

    assignments: list[tuple[int, dict]] = []
    for struct in structures:
        color = int(struct["tooth_color"])
        candidates = pieces_by_main.get(color, [])
        if len(candidates) != 1:
            continue
        pid = int(candidates[0])
        piece = pieces[pid]
        if _piece_face_bbox(g, piece["coords"], int(color), struct["direction"]) is None:
            continue
        assignments.append((int(pid), struct))

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
        main_color = piece["main_color"]
        assert main_color is not None
        face_bbox = _piece_face_bbox(g, piece["coords"], int(main_color), struct["direction"])
        assert face_bbox is not None

        tr0, tc0, _tr1, _tc1 = map(int, struct["target_face_bbox"])
        fr0, fc0, _fr1, _fc1 = map(int, face_bbox)
        shift_r = int(tr0 - fr0)
        shift_c = int(tc0 - fc0)
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
                    out[int(rr), int(cc)] = int(BLACK)

        for r, c in piece["coords"]:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(g[int(r), int(c)])

    return out


# --------------- generator (codex-att2, combo D1/0 + D2/1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────────────
GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
WALL_COLORS = (1, 5, 8, 9)
PIECE_MAIN_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

# ── CFG: structures ─────────────────────────────────────────────────────────
STRUCT_WIDTH_TEETH = ((8, 2), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
TRUE_SLIDE_RANGE = (5, 8)
FALSE_SLIDE_MIN = 2
STRUCT_PADDING = 4

# ── CFG: distractors ────────────────────────────────────────────────────────
SPRITE_WIDTHS = (2, 3)
TOTAL_PIECES_MIN = 5
TOTAL_PIECES_MAX = 11

# ── CFG: retry ──────────────────────────────────────────────────────────────
MAX_RETRIES = 500


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


def _sprite_array(teeth: int, bg: int, main_color: int) -> np.ndarray:
    sprite = np.full((2, teeth), int(main_color), dtype=int)
    accent_col = 0 if int(teeth) == 2 else 1
    sprite[1, accent_col] = int(RED)
    return sprite


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
    true_piece = _sprite_array(teeth, bg, tooth_color)
    fake_piece = _sprite_array(teeth, bg, fake_color)
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
    fake_piece = _sprite_array(teeth, bg, fake_color)
    patch_h = 2 + false_slide + fake_piece.shape[0]

    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)
    input_patch = structure.copy()
    output_patch = structure.copy()

    paste(input_patch, fake_piece, 2 + false_slide, (width - teeth) // 2, transparent=bg)

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


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
        r0, c0 = map(int, loc)
        paste(out, spec["output"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(struct_grid, spec["structure"], r0, c0, transparent=bg)

        shifted = spec["structure"]
        coords_arr = np.argwhere(shifted != int(bg))
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_arr]
        rows = [int(r) for r, _ in coords]
        cols = [int(c) for _, c in coords]
        tooth_vals = [int(shifted[int(rr), int(cc)]) for rr, cc in coords_arr]
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


def _structure_rays(struct_grid: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    h, w = struct_grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        tooth_color = int(item["tooth_color"])
        for r, c in item["coords"]:
            if int(struct_grid[int(r), int(c)]) != int(tooth_color):
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
                    forbidden[int(rr), int(cc)] = True
                    rr += int(dr)
                    cc += int(dc)
    return forbidden


def _valid_color_mapping(grid: np.ndarray, bg: int, expected_matched: int, expected_structs: int) -> bool:
    structures, pieces = _parse_input_objects(grid, bg)
    if len(structures) != expected_structs:
        return False

    tooth_colors = [int(s["tooth_color"]) for s in structures]
    if len(set(tooth_colors)) != len(tooth_colors):
        return False

    main_count: Counter[int] = Counter()
    for piece in pieces:
        main_color = piece["main_color"]
        if main_color is not None:
            main_count[int(main_color)] += 1

    matched_colors = 0
    for color in tooth_colors:
        hits = int(main_count.get(int(color), 0))
        if hits not in (0, 1):
            return False
        matched_colors += hits
    return matched_colors == expected_matched


def _override_case_count(grid: np.ndarray, bg: int) -> int:
    structures, pieces = _parse_input_objects(grid, bg)
    piece_at = _occupied_piece_cells(grid, pieces)
    by_color: dict[int, list[int]] = {}
    for pid, piece in enumerate(pieces):
        main_color = piece["main_color"]
        if main_color is not None:
            by_color.setdefault(int(main_color), []).append(int(pid))

    count = 0
    for struct in structures:
        tooth_color = int(struct["tooth_color"])
        true_candidates = by_color.get(int(tooth_color), [])
        if len(true_candidates) != 1:
            continue
        near_pid = _nearest_piece_on_ray(grid, struct, bg, piece_at)
        if near_pid is None:
            continue
        if int(near_pid) != int(true_candidates[0]):
            count += 1
    return count


def _place_distractors(
    inp: np.ndarray,
    struct_grid: np.ndarray,
    placed: list[dict],
    bg: int,
    wall_color: int,
    colors: list[int],
    count: int,
) -> bool:
    h, w = inp.shape
    forbidden = _structure_rays(struct_grid, placed, bg)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        bbox_blocked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    piece_mask = inp != int(bg)

    for idx in range(count):
        blocked = forbidden | bbox_blocked | (inp != int(bg)) | _dilate4(piece_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = int(wall_color)
        teeth = int(random.choice(SPRITE_WIDTHS))
        main_color = int(colors[idx % len(colors)])
        patch = _sprite_array(teeth, bg, main_color)
        loc = _place_patch(canvas, patch, bg, padding=0)
        if loc is None:
            return False
        paste(inp, patch, int(loc[0]), int(loc[1]), transparent=bg)
        piece_mask[int(loc[0]) : int(loc[0]) + patch.shape[0], int(loc[1]) : int(loc[1]) + patch.shape[1]] |= patch != int(bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(BG_COLORS))
    wall_choices = [c for c in WALL_COLORS if c not in (bg, RED, BLACK)]
    wall_color = int(random.choice(wall_choices))

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    palette = [c for c in PIECE_MAIN_COLORS if c not in (bg, wall_color, RED, BLACK)]
    if len(palette) < 2 * n_structs:
        return None
    random.shuffle(palette)
    tooth_colors = palette[:n_structs]
    fake_colors = palette[n_structs : 2 * n_structs]
    distractor_palette = fake_colors + [c for c in palette[2 * n_structs :] if c not in tooth_colors]
    if not distractor_palette:
        distractor_palette = fake_colors[:]

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
        if idx < matched:
            true_slide = random.randint(*TRUE_SLIDE_RANGE)
            false_slide = random.randint(FALSE_SLIDE_MIN, int(true_slide - 3))
            specs.append(
                _matched_patches(
                    width,
                    teeth,
                    true_slide,
                    false_slide,
                    rot_k,
                    bg,
                    wall_color,
                    int(tooth_colors[idx]),
                    int(fake_colors[idx]),
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
                    int(tooth_colors[idx]),
                    int(fake_colors[idx]),
                )
            )

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["output"] != int(bg)), reverse=True)

    result = _place_structures(specs, h, w, bg)
    if result is None:
        return None
    inp, out, struct_grid, placed = result

    if not _valid_color_mapping(inp, bg, matched, n_structs):
        return None
    if _override_case_count(inp, bg) < matched:
        return None

    random.shuffle(distractor_palette)
    if not _place_distractors(inp, struct_grid, placed, bg, wall_color, distractor_palette, distractor_count):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo variant D1/0+D2/1")
